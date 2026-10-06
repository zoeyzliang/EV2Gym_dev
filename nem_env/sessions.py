"""
sessions.py
===========
Per-port EV charging sessions for the feeder environment
(docs/env_redesign_spec.md §4.4–4.6).

Session sampling mirrors EV2Gym's public scenario (ev2gym/utilities/utils.py
spawn_single_EV / EV_spawner), using the ElaadNL public distributions shipped
with EV2Gym:
  * arrivals: a free port receives an EV with a per-step probability shaped
    by the public arrival curve (weekday/weekend, 15-min), scaled so that
    `sessions_per_port_per_day` arrivals are offered per port per day;
  * energy demand ~ N(mean_public(arrival half-hour), 0.5·mean), min 5 kWh;
  * dwell time   ~ N(mean_public(arrival half-hour), 0.2·mean) hours;
  * EV model sampled by registrations from ev_specs_v2g_enabled2024.json
    (battery capacity, DC charge limit, V2G discharge limit ≈ 10–11 kW).

Additions for this study:
  * owners opt in to V2G once, at arrival, at the incentive offered at that
    moment (logistic participation model, nem_env/participation_model.py);
    the rate is contracted for the session and paid per kWh *discharged*;
  * participants are dispatchable within per-EV bounds that keep SoC in
    [soc_min, 1] and guarantee the departure target stays reachable
    (forced charging when an EV has no slack left);
  * non-participants charge immediately at full power until their target
    (uncontrolled load the aggregator cannot dispatch);
  * a hub-level setpoint is split across participants least-laxity-first.

Sign convention: EV/hub power > 0 is charging (import from the grid).
"""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd

_DATA = Path(__file__).resolve().parent.parent / "ev2gym" / "data"


@dataclass
class SessionConfig:
    sessions_per_port_per_day: float = 2.5   # offered arrivals per port per day
    soc_min: float = 0.20                     # V2G floor (fraction of capacity)
    eta: float = 0.95                         # one-way charge/discharge efficiency
    dt_hr: float = 5.0 / 60.0
    min_stay_steps: int = 3                   # 15 min


class SessionData:
    """ElaadNL public distributions and V2G EV specs (loaded once)."""

    def __init__(self):
        arr_wd = pd.read_csv(_DATA / "distribution-of-arrival.csv")["public"].to_numpy(float)
        arr_we = pd.read_csv(_DATA / "distribution-of-arrival-weekend.csv")["public"].to_numpy(float)
        self.arrival_shape = {False: arr_wd / arr_wd.sum(), True: arr_we / arr_we.sum()}  # per 15-min slot

        def by_half_hour(path):
            df = pd.read_csv(path)
            key = df["Arrival Time"].str.strip()
            return dict(zip(key, df["public"].astype(float)))
        self.mean_stay_h = by_half_hour(_DATA / "mean-session-length-per.csv")
        self.mean_demand_kwh = by_half_hour(_DATA / "mean-demand-per-arrival.csv")

        specs = json.load(open(_DATA / "ev_specs_v2g_enabled2024.json"))
        self.ev_names = list(specs)
        regs = np.array([specs[k]["number_of_registrations"] for k in self.ev_names], float)
        self.ev_p = regs / regs.sum()
        self.ev_cap = np.array([specs[k]["battery_capacity"] for k in self.ev_names], float)
        self.ev_pch = np.array([specs[k]["max_dc_charge_power"] for k in self.ev_names], float)
        self.ev_pdis = np.array([abs(specs[k]["max_dc_discharge_power"]) for k in self.ev_names], float)


class HubSessions:
    """
    All ports of all hubs, as flat arrays of length P (total ports).

    Parameters
    ----------
    hub_configs : list of HubConfig (n_chargers, charger_max_kw, distance_km)
    participation_model : ParticipationModel
    cfg : SessionConfig
    rng : np.random.Generator
    """

    def __init__(self, hub_configs, participation_model, cfg=None, rng=None, data=None):
        self.cfg = cfg or SessionConfig()
        self.rng = rng or np.random.default_rng()
        self.data = data or SessionData()
        self.pm = participation_model
        self.H = len(hub_configs)
        self.port_hub = np.concatenate([[h] * hc.n_chargers for h, hc in enumerate(hub_configs)]).astype(int)
        self.port_kw = np.concatenate([[hc.charger_max_kw] * hc.n_chargers for hc in hub_configs]).astype(float)
        self.ports_per_hub = np.bincount(self.port_hub, minlength=self.H)
        self.hub_distance = np.array([hc.distance_km for hc in hub_configs], float)
        self.P = len(self.port_hub)
        self.reset()

    # ------------------------------------------------------------------
    def reset(self):
        P = self.P
        self.occ = np.zeros(P, bool)
        self.part = np.zeros(P, bool)
        self.dep = np.zeros(P, int)          # departure step (exclusive)
        self.cap = np.zeros(P)               # battery capacity kWh
        self.E = np.zeros(P)                 # stored energy kWh
        self.target = np.zeros(P)            # energy required at departure kWh
        self.p_ch = np.zeros(P)              # max charge power kW
        self.p_dis = np.zeros(P)             # max discharge power kW
        self.rate = np.zeros(P)              # contracted incentive $/kWh discharged

    # ------------------------------------------------------------------
    # Arrivals and departures
    # ------------------------------------------------------------------
    def arrival_prob(self, ts: pd.Timestamp) -> float:
        """Probability that a free port receives an EV in this 5-min step."""
        shape = self.data.arrival_shape[ts.dayofweek >= 5]
        slot = ts.hour * 4 + ts.minute // 15
        return min(1.0, self.cfg.sessions_per_port_per_day * shape[slot] / 3.0)

    def expected_arrival_kw(self, ts: pd.Timestamp, p_part: np.ndarray) -> np.ndarray:
        """Expected uncontrolled charging (kW per hub) from EVs arriving this step."""
        pr = self.arrival_prob(ts)
        free = ~self.occ
        per_port = pr * free * self.port_kw * (1.0 - p_part[self.port_hub])
        return np.bincount(self.port_hub, weights=per_port, minlength=self.H)

    def depart(self, t: int) -> dict:
        """Release ports whose session ends at step t; return shortfalls (kWh)."""
        leaving = self.occ & (self.dep <= t)
        short = np.maximum(0.0, self.target - self.E) * leaving
        out = {
            "unmet_part_kwh": np.bincount(self.port_hub, weights=short * self.part, minlength=self.H),
            "unmet_nonpart_kwh": np.bincount(self.port_hub, weights=short * ~self.part, minlength=self.H),
            "departures": int(leaving.sum()),
        }
        self.occ[leaving] = False
        self.part[leaving] = False
        return out

    def arrive(self, t: int, ts: pd.Timestamp, price: float, doe_export_kw: np.ndarray,
               steps_per_day: int) -> dict:
        """
        Sample arrivals at free ports; each new owner decides once whether to
        opt in at the incentive `price` ($/kWh). Returns counts.
        """
        free = np.where(~self.occ)[0]
        arriving = free[self.rng.random(len(free)) < self.arrival_prob(ts)]
        n = len(arriving)
        if n == 0:
            return {"arrivals": 0, "opt_in": 0, "capped": 0, "capped_kwh": 0.0}
        d = self.data
        half = f"{ts.hour:02d}:{0 if ts.minute < 30 else 30:02d}"
        k = self.rng.choice(len(d.ev_names), size=n, p=d.ev_p)
        cap = d.ev_cap[k]
        demand = self.rng.normal(d.mean_demand_kwh[half], 0.5 * d.mean_demand_kwh[half], n)
        demand = np.where(demand < 5, self.rng.integers(5, 10, n), demand)
        stay_h = self.rng.normal(d.mean_stay_h[half], 0.2 * d.mean_stay_h[half], n)
        stay = np.maximum(self.cfg.min_stay_steps, np.round(stay_h * 60 / 5).astype(int))

        e_arr = np.clip(cap - demand, 0.1 * cap, 0.9 * cap)
        requested = np.minimum(cap, e_arr + demand)
        # Demand and dwell are sampled independently (as in EV2Gym), so some
        # owners would ask for more than their charger can deliver during the
        # stay. Cap the target at the deliverable energy: full power for every
        # step after arrival (a new participant is idle in its arrival step).
        # Without this, every policy is penalised for an unavoidable shortfall.
        p_ch = np.minimum(d.ev_pch[k], self.port_kw[arriving])
        deliverable = e_arr + p_ch * self.cfg.eta * self.cfg.dt_hr * np.maximum(stay - 1, 0)
        target = np.minimum(requested, deliverable)
        capped = target < requested - 1e-9

        # One-off opt-in decision at the offered incentive (spec §4.5).
        hubs = self.port_hub[arriving]
        n_conn = np.maximum(1, np.bincount(self.port_hub[self.occ], minlength=self.H)[hubs] + 1)
        prob = self.pm.participation_prob_vector(
            c_t=price * 1000.0,                                  # $/kWh -> $/MWh
            distances_km=self.hub_distance[hubs],
            mean_socs=e_arr / cap,
            doe_export_ws=doe_export_kw[hubs] * 1000.0,          # kW -> W
            n_connecteds=n_conn,
        )
        opt = self.rng.random(n) < prob

        self.occ[arriving] = True
        self.part[arriving] = opt
        self.dep[arriving] = t + stay
        self.cap[arriving] = cap
        self.E[arriving] = e_arr
        self.target[arriving] = target
        self.p_ch[arriving] = p_ch
        self.p_dis[arriving] = np.minimum(d.ev_pdis[k], self.port_kw[arriving])
        self.rate[arriving] = np.where(opt, price, 0.0)
        return {"arrivals": n, "opt_in": int(opt.sum()), "new_ports": arriving,
                "capped": int(capped.sum()), "capped_kwh": float((requested - target).sum())}

    # ------------------------------------------------------------------
    # Power bounds and dispatch
    # ------------------------------------------------------------------
    def bounds(self, t: int):
        """
        Per-port feasible grid power this step (kW, + = charging).

        lo: most negative allowed (max discharge, or a forced minimum charge
            if the EV has no slack left to reach its target by departure);
        hi: most positive allowed.
        Non-occupied ports have lo = hi = 0.
        """
        c, dt, eta = self.cfg, self.cfg.dt_hr, self.cfg.eta
        steps_left = np.maximum(1, self.dep - t)
        # Charge needed in the remaining steps after this one at full power
        e_max_after = self.p_ch * eta * dt * (steps_left - 1)
        need_now = np.maximum(0.0, self.target - self.E - e_max_after)       # kWh this step
        min_ch = np.minimum(self.p_ch, need_now / (eta * dt))
        hi = np.minimum(self.p_ch, np.maximum(0.0, self.cap - self.E) / (eta * dt))
        # Discharge: stay above soc_min and above what still lets the target be met
        floor = np.maximum(c.soc_min * self.cap, self.target - e_max_after)
        dis = np.minimum(self.p_dis, np.maximum(0.0, self.E - floor) * eta / dt)
        lo = np.where(min_ch > 0, min_ch, -dis)
        hi = np.maximum(hi, lo)
        lo = np.where(self.occ, lo, 0.0); hi = np.where(self.occ, hi, 0.0)
        return lo, hi

    def hub_bounds(self, t: int):
        """Fleet bounds per hub for participants and uncontrolled power of non-participants."""
        lo, hi = self.bounds(t)
        flex_lo = np.bincount(self.port_hub, weights=lo * self.part, minlength=self.H)
        flex_hi = np.bincount(self.port_hub, weights=hi * self.part, minlength=self.H)
        unctrl = np.bincount(self.port_hub, weights=hi * (self.occ & ~self.part), minlength=self.H)
        return flex_lo, flex_hi, unctrl, lo, hi

    def dispatch(self, t: int, q_hub: np.ndarray, lo: np.ndarray, hi: np.ndarray,
                 u_hub: np.ndarray = None) -> np.ndarray:
        """
        Per-port power (kW) for one step.

        Participants: hub setpoint q_hub (kW, within the fleet bounds) is
        split least-laxity-first: every EV starts at its lower bound and the
        remainder goes to the EVs with the least slack first.

        Non-participants: charge only (no V2G). The charger management caps
        their total at u_hub (kW per hub; default: everyone at full power),
        again least-laxity-first, so a DOE-limited site throttles them the
        way CSIP-AUS-compliant chargers would.
        """
        x = np.where(self.part, lo, 0.0)
        steps_left = np.maximum(1, self.dep - t)
        need_steps = np.maximum(0.0, self.target - self.E) / np.maximum(1e-9, self.p_ch * self.cfg.eta * self.cfg.dt_hr)
        laxity = steps_left - need_steps
        nonpart = self.occ & ~self.part
        if u_hub is None:
            u_hub = np.bincount(self.port_hub, weights=hi * nonpart, minlength=self.H)
        for h in range(self.H):
            on_hub = self.port_hub == h
            ports = np.where(on_hub & self.part)[0]
            rem = q_hub[h] - lo[ports].sum()
            for p in ports[np.argsort(laxity[ports], kind="stable")]:
                if rem <= 0:
                    break
                add = min(rem, hi[p] - lo[p])
                x[p] += add
                rem -= add
            ports = np.where(on_hub & nonpart)[0]
            rem = u_hub[h]
            for p in ports[np.argsort(laxity[ports], kind="stable")]:
                if rem <= 0:
                    break
                add = min(rem, hi[p])
                x[p] = add
                rem -= add
        return x

    def apply(self, x: np.ndarray) -> dict:
        """Update stored energy for per-port grid power x (kW) over one step."""
        dt, eta = self.cfg.dt_hr, self.cfg.eta
        dE = np.where(x >= 0, x * eta * dt, x * dt / eta)
        self.E = np.clip(self.E + dE * self.occ, 0.0, self.cap)
        dis_kwh = np.maximum(0.0, -x) * dt * self.part
        return {
            "discharged_kwh": np.bincount(self.port_hub, weights=dis_kwh, minlength=self.H),
            "incentive_paid": float(np.sum(dis_kwh * self.rate)),
            "flex_kw": np.bincount(self.port_hub, weights=x * self.part, minlength=self.H),
            "unctrl_kw": np.bincount(self.port_hub, weights=x * (self.occ & ~self.part), minlength=self.H),
        }

    # ------------------------------------------------------------------
    # Hub-level features
    # ------------------------------------------------------------------
    def hub_features(self, t: int) -> dict:
        part = self.part
        n_part = np.bincount(self.port_hub, weights=part, minlength=self.H)
        need = np.bincount(self.port_hub, weights=np.maximum(0, self.target - self.E) * part, minlength=self.H)
        stay_h = np.maximum(0, self.dep - t) * self.cfg.dt_hr
        mean_stay = np.bincount(self.port_hub, weights=stay_h * part, minlength=self.H) / np.maximum(1, n_part)
        soc = np.where(self.cap > 0, self.E / np.maximum(self.cap, 1e-9), 0.0)
        mean_soc = np.bincount(self.port_hub, weights=soc * part, minlength=self.H) / np.maximum(1, n_part)
        occ = np.bincount(self.port_hub, weights=self.occ, minlength=self.H)
        return {"n_part": n_part, "need_kwh": need, "mean_stay_h": mean_stay,
                "mean_soc": mean_soc, "occupancy": occ}
