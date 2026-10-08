"""
feeder_env.py
=============
NEMFeederEnv: public V2G hub dispatch on a distribution feeder with
power-flow-derived DOEs, EV sessions and session-level opt-in
(docs/env_redesign_spec.md). The legacy NEMDOEEnv is unchanged.

One step = one 5-min NEM dispatch interval; one episode = one trading day.

Within a step (t):
  1. Sessions ending at t depart; participants' shortfall is penalised.
  2. Per-port feasible bounds; participant fleet bounds per hub.
  3. Safety projection (the deployed clip): the participant setpoint is
     limited so that forecast net site flow stays inside the hub's limit
     (per-hub DOE, or individual hosting capacity in network mode), then
     to the fleet bounds (forced charging to reach targets wins).
  4. Setpoint split across participants (least-laxity-first).
  5. New EVs arrive *during* the step and decide once whether to opt in at
     the offered incentive; new non-participants start charging at once,
     so realised net site flow can differ from the forecast.
  6. Realised outcome: per-hub limit compliance on realised net site flow,
     and a power flow with the realised background (forecast × (1+ε)).

Action (H + 1): per-hub normalised participant setpoint in [-1, 1]
(+ = discharge, scaled by hub capacity) and one network-wide incentive
($/kWh discharged) offered to arriving owners.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .aemo_price_loader import PriceLoader
from .feeder import Feeder, FeederConfig
from .grid_profiles import GridProfiles
from .sessions import HubSessions, SessionConfig, SessionData


@dataclass
class FeederEnvConfig:
    doe_mode: str = "per_hub"        # "per_hub" | "network"   (spec E2)
    spatial: str = "feeder"          # "feeder" | "permuted"   (spec E2 ablation)
    forecast_sigma: float = 0.05     # background forecast error (spec E3)
    forecast_rho: float = 0.9        # AR(1) persistence of the error
    forecast_common: float = 0.5     # share of the error common to all buses
    price_min: float = 0.0           # incentive bounds ($/kWh discharged)
    price_max: float = 0.50
    lambda_unmet: float = 1.0        # $/kWh short of target at departure
    lambda_doe: float = 1.0          # $/kWh of limit violation (set by pilot)
    lambda_v: float = 100.0          # $ per pu·bus of voltage violation (network mode)
    # How a hub's normalised setpoint maps to kW (spec §4.6).
    #   "capacity": a·hub capacity (as built for the λ study and E1 v1). The
    #               participants' feasible range averages ~2% of capacity, so
    #               the projection clips almost the whole action range.
    #   "feasible": a ∈ [0, 1] → 0 … max participant discharge,
    #               a ∈ [-1, 0] → 0 … max participant charge (0 = idle).
    action_scale: str = "capacity"
    # Participants pay for the energy they asked for (up to what they got) at
    # the day's mean RRP, like the terminal-cost convention, so the
    # aggregator earns only from flexibility. False (v1): the aggregator buys
    # participants' charging and nobody pays it back (free charging).
    participant_billing: bool = False
    # Robustness check S1 (spec §6): costs the base formulation omits. Both
    # default to 0 (v1). deg_cost $/kWh of participants' discharged (grid-side)
    # energy; import_tariff $/MWh of participants' grid imports.
    deg_cost: float = 0.0
    import_tariff: float = 0.0
    # S1b (spec §6): pass the network tariff through on participants' requested
    # energy (bills and end-of-day purchases at mean RRP + tariff), so the
    # aggregator bears the tariff only on extra cycled energy. False = S1.
    tariff_passthrough: bool = False
    # Final RL stage (spec §4.10): 7 extra node features from AEMO predispatch
    # (set env.predispatch) and the next DOE window. False keeps 17 features.
    forecast_features: bool = False
    feeder: FeederConfig = field(default_factory=lambda: FeederConfig(kappa_load=0.7, pv_penetration=0.6))
    sessions: SessionConfig = field(default_factory=SessionConfig)


class NEMFeederEnv(gym.Env):
    metadata = {"render_modes": []}
    NODE_FEATURE_DIM = 17
    STEPS = PriceLoader.STEPS_PER_DAY
    DT_HR = 5.0 / 60.0

    def __init__(self, hub_configs, price_loader: PriceLoader, grid_profiles: GridProfiles,
                 participation_model, env_config: Optional[FeederEnvConfig] = None,
                 seed: Optional[int] = None, feeder: Optional[Feeder] = None):
        super().__init__()
        self.cfg = env_config or FeederEnvConfig()
        if self.cfg.doe_mode not in ("per_hub", "network"):
            raise ValueError(f"doe_mode must be 'per_hub' or 'network', got {self.cfg.doe_mode!r}")
        if self.cfg.action_scale not in ("capacity", "feasible"):
            raise ValueError(f"action_scale must be 'capacity' or 'feasible', got {self.cfg.action_scale!r}")
        if self.cfg.spatial not in ("feeder", "permuted"):
            raise ValueError(f"spatial must be 'feeder' or 'permuted', got {self.cfg.spatial!r}")
        self.hub_configs = hub_configs
        self.H = len(hub_configs)
        self.price_loader = price_loader
        self.grid_profiles = grid_profiles
        self.participation_model = participation_model
        self.feeder = feeder or Feeder(hub_configs, self.cfg.feeder)
        self.cap = self.feeder.hub_cap
        self._rng = np.random.default_rng(seed)
        self._session_data = SessionData()
        self.sessions = HubSessions(hub_configs, participation_model, self.cfg.sessions,
                                    rng=self._rng, data=self._session_data)
        self._day_cache = {}
        self.predispatch = None                  # nem_env.predispatch.Predispatch, if forecast_features
        self._fc = None
        self._node_dim = self.NODE_FEATURE_DIM + (7 if self.cfg.forecast_features else 0)

        self.action_space = spaces.Box(
            low=np.concatenate([-np.ones(self.H), [self.cfg.price_min]]).astype(np.float32),
            high=np.concatenate([np.ones(self.H), [self.cfg.price_max]]).astype(np.float32),
        )
        self.observation_space = spaces.Box(-np.inf, np.inf, (self.H * self._node_dim,), np.float32)

    # ------------------------------------------------------------------
    @property
    def n_hubs(self):
        return self.H

    @property
    def node_feature_dim(self):
        return self._node_dim

    @property
    def action_dim(self):
        return self.H + 1

    def obs_to_node_features(self, obs):
        return obs.reshape(self.H, self._node_dim)

    # ------------------------------------------------------------------
    def _day(self, index):
        """Forecast background, DOEs/hosting for a date (cached per date)."""
        key = str(index[0].date())
        if key not in self._day_cache:
            D, S = self.grid_profiles.day(index)
            P, Q = self.feeder.background(D, S)
            if self.cfg.doe_mode == "per_hub":
                imp, exp = self.feeder.doe_day(P, Q)
            else:
                imp, exp = self.feeder.hosting_day(P, Q)
            self._day_cache[key] = (P, Q, imp, exp)
        return self._day_cache[key]

    def _forecast_error(self):
        """AR(1) multiplicative background error, partly common to all buses."""
        c = self.cfg
        T, nb = self.STEPS, self.feeder.nb
        eps = np.zeros((T, nb))
        if c.forecast_sigma <= 0:
            return eps
        innov_scale = c.forecast_sigma * np.sqrt(1 - c.forecast_rho ** 2)
        e = self._rng.normal(0, c.forecast_sigma, nb)
        for t in range(T):
            z = (np.sqrt(c.forecast_common) * self._rng.normal()
                 + np.sqrt(1 - c.forecast_common) * self._rng.normal(size=nb))
            e = c.forecast_rho * e + innov_scale * z
            eps[t] = e
        return eps

    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            self.participation_model.rng = np.random.default_rng(seed)
            self.sessions.rng = self._rng
        opts = options or {}
        self._df = self.price_loader.sample_episode(
            date=opts.get("date"), episode=opts.get("episode"),
            total_episodes=opts.get("total_episodes"))
        self._idx = self._df.index
        self._rrp = self._df["spot_price"].to_numpy(dtype=float)
        P, Q, imp, exp = self._day(self._idx)
        self._P_fc, self._Q_fc = P, Q
        if self.cfg.spatial == "permuted":
            # Destroy spatial structure, keep each step's distribution of
            # limit fractions: permute hubs independently per DOE block.
            frac_i, frac_e = imp / self.cap, exp / self.cap
            B = self.feeder.cfg.doe_block
            imp, exp = imp.copy(), exp.copy()
            for s in range(0, self.STEPS, B):
                perm = self._rng.permutation(self.H)
                imp[s:s + B] = frac_i[s:s + B][:, perm] * self.cap
                exp[s:s + B] = frac_e[s:s + B][:, perm] * self.cap
        self._lim_imp, self._lim_exp = imp, exp
        self._eps = self._forecast_error()
        if self.cfg.forecast_features:
            if self.predispatch is None:
                raise RuntimeError("forecast_features needs env.predispatch")
            self._fc = np.clip(self.predispatch.day_features(self._idx) / 1000.0, -1.0, 20.3)
        self.sessions.reset()
        self._t = 0
        self._last_v = np.full(self.H, self.feeder.cfg.v_slack)
        return self._obs(), {"date": str(self._idx[0].date())}

    # ------------------------------------------------------------------
    def step(self, action):
        a = np.asarray(action, dtype=float)
        disp = np.clip(a[:self.H], -1.0, 1.0)
        price = float(np.clip(a[self.H], self.cfg.price_min, self.cfg.price_max))
        t, ts = self._t, self._idx[self._t]
        dt = self.DT_HR
        hs = self.sessions

        # 1. departures
        dep = hs.depart(t)
        # 2. bounds
        flex_lo, flex_hi, unctrl, lo, hi = hs.hub_bounds(t)
        p_part = self.participation_model.participation_prob_vector(
            c_t=price * 1000.0, distances_km=hs.hub_distance,
            mean_socs=np.full(self.H, 0.5),
            doe_export_ws=self._lim_exp[t] * 1000.0, n_connecteds=np.maximum(1, hs.ports_per_hub))
        arr_fc = hs.expected_arrival_kw(ts, p_part)
        # 3. safety projection (charge-positive participant setpoint q).
        #    Participants' setpoint is kept inside the export limit and the
        #    import limit net of expected in-step arrivals; non-participants
        #    then share the remaining import headroom (charger management
        #    throttles them). Forced charging to meet departure targets
        #    overrides the limit, which is then a genuine violation.
        lim_i, lim_e = self._lim_imp[t], self._lim_exp[t]
        if self.cfg.action_scale == "feasible":
            # + = discharge: scale by the participants' own range, so every
            # action value is reachable (forced charging, flex_lo > 0, is
            # still enforced by the clip to [flex_lo, flex_hi] below).
            q_req = np.where(disp >= 0, disp * np.minimum(flex_lo, 0.0), -disp * np.maximum(flex_hi, 0.0))
        else:
            q_req = -disp * self.cap
        q = np.clip(q_req, -lim_e, np.maximum(0.0, lim_i - arr_fc))
        q = np.clip(q, flex_lo, flex_hi)
        u = np.clip(lim_i - arr_fc - q, 0.0, unctrl)
        # 4. split across participants and non-participants
        x = hs.dispatch(t, q, lo, hi, u_hub=u)
        # 5. arrivals during the step; new non-participants charge at once
        arr = hs.arrive(t, ts, price, self._lim_exp[t], self.STEPS)
        if arr["arrivals"]:
            new = arr["new_ports"]
            lo2, hi2 = hs.bounds(t)
            x[new] = np.where(hs.part[new], 0.0, hi2[new])
        out = hs.apply(x)
        flex, unc = out["flex_kw"], out["unctrl_kw"]
        net = flex + unc                                         # realised net site flow (kW)

        # 6. realised outcome
        viol_kw = np.maximum(0.0, net - self._lim_imp[t]) + np.maximum(0.0, -net - self._lim_exp[t])
        P_real = self._P_fc[t] * (1.0 + self._eps[t])
        Q_real = self._Q_fc[t] * (1.0 + self._eps[t])
        phys = self.feeder.realised(P_real, Q_real, net)
        self._last_v = phys["hub_v"]

        rrp = self._rrp[t]
        r_wholesale = -rrp * flex.sum() * dt / 1000.0           # buy when charging, sell when discharging
        r_incentive = out["incentive_paid"]
        # Shortfall at departure for every customer: participants (V2G) and
        # non-participants (throttled to respect the limit) alike.
        p_unmet = self.cfg.lambda_unmet * (dep["unmet_part_kwh"].sum() + dep["unmet_nonpart_kwh"].sum())
        if self.cfg.doe_mode == "per_hub":
            p_limit = self.cfg.lambda_doe * viol_kw.sum() * dt
        else:
            p_limit = (self.cfg.lambda_doe * phys["overload_kw"] * dt
                       + self.cfg.lambda_v * phys["v_viol_pu"])
        mean_rrp = max(0.0, float(np.mean(self._rrp)))
        if self.cfg.tariff_passthrough:
            mean_rrp += self.cfg.import_tariff
        r_billing = (mean_rrp * dep["billed_part_kwh"] / hs.cfg.eta / 1000.0
                     if self.cfg.participant_billing else 0.0)
        r_costs = (self.cfg.deg_cost * float(out["discharged_kwh"].sum())
                   + self.cfg.import_tariff / 1000.0 * out["part_import_kwh"])
        reward = r_wholesale + r_billing - r_incentive - r_costs - p_unmet - p_limit

        self._t += 1
        terminated = self._t >= self.STEPS
        if terminated:
            # Energy participants still need is bought after the episode:
            # charge it at the day's mean price so draining batteries late
            # in the day is not free.
            still_needed = np.sum(np.maximum(0.0, hs.target - hs.E) * hs.part * hs.occ) / hs.cfg.eta
            p_terminal = mean_rrp * still_needed / 1000.0
            reward -= p_terminal
            if self.cfg.participant_billing:
                # participants still connected are billed for their whole request
                owed = np.sum(np.maximum(0.0, hs.target - hs.e_arr) * hs.part * hs.occ) / hs.cfg.eta
                r_end = mean_rrp * owed / 1000.0
                r_billing += r_end
                reward += r_end
        else:
            p_terminal = 0.0

        info = {
            "rrp": rrp, "incentive_price": price,
            "r_wholesale": r_wholesale, "r_incentive": r_incentive,
            "p_unmet": p_unmet, "p_limit": p_limit, "p_terminal": p_terminal,
            "r_billing": r_billing,
            "r_costs": r_costs,
            "arbitrage_profit": r_wholesale + r_billing - r_incentive - r_costs,
            "limit_viol_kwh": float(viol_kw.sum() * dt), "limit_compliant": bool(viol_kw.max() <= 1e-6),
            "v_min": phys["v_min"], "v_max": phys["v_max"], "v_viol_pu": phys["v_viol_pu"],
            "overload_kw": phys["overload_kw"],
            "unmet_part_kwh": float(dep["unmet_part_kwh"].sum()),
            "unmet_nonpart_kwh": float(dep["unmet_nonpart_kwh"].sum()),
            "arrivals": arr["arrivals"], "opt_in": arr["opt_in"],
            "targets_capped": arr["capped"], "capped_kwh": arr["capped_kwh"],
            "flex_kw": flex.tolist(), "net_kw": net.tolist(),
            "discharged_kwh": float(out["discharged_kwh"].sum()),
            "requested_kw": q_req.tolist(), "projected_kw": q.tolist(),
            "flex_lo_kw": flex_lo.tolist(), "flex_hi_kw": flex_hi.tolist(),
            "arr_fc_kw": arr_fc.tolist(), "lim_imp_kw": lim_i.tolist(), "lim_exp_kw": lim_e.tolist(),
            # Same keys as the legacy env, so training/eval logging carries over
            "rho_hat": arr["opt_in"] / arr["arrivals"] if arr["arrivals"] else 0.0,
            "doe_violations_kw": viol_kw.tolist(),
        }
        obs = self._obs() if not terminated else np.zeros(self.observation_space.shape, np.float32)
        return obs, float(reward), terminated, False, info

    # ------------------------------------------------------------------
    def _obs(self):
        t = min(self._t, self.STEPS - 1)
        ts = self._idx[t]
        hs = self.sessions
        flex_lo, flex_hi, unctrl, _, _ = hs.hub_bounds(t)
        f = hs.hub_features(t)
        ports = np.maximum(1, hs.ports_per_hub)
        rrp = self._rrp[t]
        rrp30 = float(np.mean(self._rrp[max(0, t - 5):t + 1]))
        hour = ts.hour + ts.minute / 60.0
        B = self.feeder.cfg.doe_block
        cap = self.cap
        node = np.stack([
            self._lim_imp[t] / cap,                       # 0 import limit (frac of capacity)
            self._lim_exp[t] / cap,                       # 1 export limit
            unctrl / cap,                                 # 2 uncontrolled load now
            f["n_part"] / ports,                          # 3 participating EVs per port
            flex_hi / cap,                                # 4 fleet max charge
            -flex_lo / cap,                               # 5 fleet max discharge (neg = forced charge)
            f["need_kwh"] / (cap * 1.0),                  # 6 energy still needed (h at full power)
            f["mean_stay_h"] / 24.0,                      # 7 mean hours to departure
            f["occupancy"] / ports,                       # 8 port occupancy
            cap / 200.0,                                  # 9 hub capacity
            (self._last_v - 1.0) * 10.0,                  # 10 last hub bus voltage deviation
            np.full(self.H, np.clip(rrp / 1000.0, -1.0, 20.3)),     # 11 RRP
            np.full(self.H, np.clip(rrp30 / 1000.0, -1.0, 20.3)),   # 12 mean RRP last 30 min
            np.full(self.H, np.sin(2 * np.pi * hour / 24)),         # 13
            np.full(self.H, np.cos(2 * np.pi * hour / 24)),         # 14
            np.full(self.H, float(ts.dayofweek >= 5)),              # 15 weekend
            np.full(self.H, (B - t % B) / B),                       # 16 time to next DOE update
        ], axis=1)
        if self.cfg.forecast_features:
            nxt = min(self.STEPS - 1, (t // B + 1) * B)
            node = np.concatenate([
                node,
                np.repeat(self._fc[t][None, :], self.H, axis=0),        # 17–21 predispatch: mean 1 h, mean 3 h,
                                                                        #   max 6 h, min 6 h, mean to day end
                (self._lim_imp[nxt] / cap)[:, None],                    # 22 next-window import limit
                (self._lim_exp[nxt] / cap)[:, None],                    # 23 next-window export limit
            ], axis=1)
        return node.astype(np.float32).reshape(-1)
