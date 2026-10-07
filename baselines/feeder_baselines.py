"""
feeder_baselines.py
===================
Non-learning baselines for NEMFeederEnv (docs/env_redesign_spec.md §5).

* NoV2GBaseline      — no V2G: participants charge as soon as possible, no
                       incentive offered. Reference.
* GreedyTOUBaseline  — price-threshold rule: discharge when RRP is high,
                       charge when low; thresholds fixed from the TRAINING
                       period's RRP quantiles (no look-ahead into test days).
* perfect_foresight_lp — upper bound: with the whole day known in advance
                       (prices, arrivals, opt-in decisions, limits), solve the
                       optimal dispatch as a linear program (HiGHS via SciPy),
                       for a constant incentive; grid-search the incentive.

Action convention (same as the env): per-hub setpoint in [-1, 1], + = discharge.
"""

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.optimize import linprog


class NoV2GBaseline:
    name = "NoV2G"
    needs_env = False

    def __init__(self, n_hubs: int):
        self.H = n_hubs

    def select_action(self, obs, deterministic=True):
        return np.append(-np.ones(self.H), 0.0).astype(np.float32)


class GreedyTOUBaseline:
    """
    Discharge at full setpoint when RRP ≥ hi, charge at full when RRP ≤ lo,
    otherwise idle; offer a fixed incentive. Reads RRP from node feature 11
    (RRP / 1000, clipped).
    """
    name = "GreedyTOU"
    needs_env = False

    def __init__(self, n_hubs: int, rrp_lo: float, rrp_hi: float, incentive: float = 0.15,
                 node_feature_dim: int = 17, rrp_feature: int = 11):
        self.H, self.lo, self.hi, self.c = n_hubs, rrp_lo, rrp_hi, incentive
        self.F, self.k = node_feature_dim, rrp_feature

    @classmethod
    def from_training_prices(cls, n_hubs, price_df, q_lo=0.25, q_hi=0.90, **kw):
        rrp = price_df["spot_price"].to_numpy(float)
        return cls(n_hubs, float(np.quantile(rrp, q_lo)), float(np.quantile(rrp, q_hi)), **kw)

    def select_action(self, obs, deterministic=True):
        rrp = float(np.asarray(obs).reshape(self.H, self.F)[0, self.k]) * 1000.0
        d = 1.0 if rrp >= self.hi else (-1.0 if rrp <= self.lo else 0.0)
        return np.append(np.full(self.H, d), self.c).astype(np.float32)


class RulePriceBaseline:
    """
    Port of the draft's RuleBasedPricing baseline (baselines/heuristics/
    rule_based_pricing.py) to the feeder env: offer a fixed share of the
    current spot price as the incentive, and discharge at full setpoint when
    RRP exceeds that incentive, otherwise charge at full setpoint. Reads RRP
    from node feature 11 (RRP / 1000).
    """
    name = "RulePrice"
    needs_env = False

    def __init__(self, n_hubs: int, price_fraction: float = 0.5, price_min: float = 0.0,
                 price_max: float = 0.50, node_feature_dim: int = 17, rrp_feature: int = 11):
        self.H, self.frac, self.pmin, self.pmax = n_hubs, price_fraction, price_min, price_max
        self.F, self.k = node_feature_dim, rrp_feature

    def select_action(self, obs, deterministic=True):
        rrp = float(np.asarray(obs).reshape(self.H, self.F)[0, self.k]) * 1000.0
        c = float(np.clip(self.frac * max(rrp, 0.0) / 1000.0, self.pmin, self.pmax))
        d = 1.0 if rrp > c * 1000.0 else -1.0
        return np.append(np.full(self.H, d), c).astype(np.float32)


# ----------------------------------------------------------------------
# Perfect-foresight LP
# ----------------------------------------------------------------------
@dataclass
class SessionRecord:
    port: int
    hub: int
    t_arr: int        # first step the EV can be dispatched (step after arrival)
    t_dep: int        # departure step (exclusive)
    cap: float
    e_arr: float
    target: float
    p_ch: float
    p_dis: float
    part: bool
    rate: float


def record_day(env, date: str, seed: int, incentive: float):
    """
    Realise one day's sessions and limits for a constant incentive.

    Runs the env with an idle setpoint while recording each EV's state at
    the moment it arrives. Arrival times, EV parameters and opt-in draws do
    not depend on the dispatch, so the recorded day is the same one any
    policy offering this constant incentive would face (same seed).

    Returns (sessions, rrp (T,), lim_imp (T,H), lim_exp (T,H)).
    """
    env.reset(seed=seed, options={"date": date})
    s = env.sessions
    sessions = []
    original_arrive = s.arrive

    def recording_arrive(t, ts, price, doe_export_kw, steps_per_day):
        out = original_arrive(t, ts, price, doe_export_kw, steps_per_day)
        for p in out.get("new_ports", []):
            part = bool(s.part[p])
            sessions.append(SessionRecord(
                port=int(p), hub=int(s.port_hub[p]),
                # env: new participants idle in their arrival step, new
                # non-participants may charge in it
                t_arr=t + 1 if part else t, t_dep=int(s.dep[p]),
                cap=float(s.cap[p]), e_arr=float(s.E[p]), target=float(s.target[p]),
                p_ch=float(s.p_ch[p]), p_dis=float(s.p_dis[p]),
                part=part, rate=float(s.rate[p])))
        return out

    s.arrive = recording_arrive
    try:
        done = False
        while not done:
            _, _, done, _, _ = env.step(np.append(np.zeros(env.H), incentive))
    finally:
        s.arrive = original_arrive
    return sessions, env._rrp.copy(), env._lim_imp.copy(), env._lim_exp.copy()


def perfect_foresight_lp(sessions, rrp, lim_imp, lim_exp, *, eta=0.95, dt=5 / 60,
                         soc_min=0.2, lambda_unmet=1.0, T=288, participant_billing=False,
                         network=None, return_flows=False):
    """
    Optimal day-ahead dispatch with full knowledge (spec §5).

    Variables per session i and step t in [t_arr, min(t_dep, T)):
      c_it ≥ 0 charge kW, d_it ≥ 0 discharge kW (participants only),
      plus u_i ≥ 0 unmet energy at departure (kWh).
    Energy: E_i(t) = e_arr + Σ_{τ<t} (η c − d/η) Δt  within [floor_i, cap_i],
      floor = soc_min·cap for participants, 0 otherwise.
    Departure (if within the day): E_i(t_dep) + u_i ≥ target.
    Sessions still connected at the end: remaining need bought at the day's
      mean price (same terminal rule as the env).
    Hub limits each step: −lim_exp ≤ Σ_i (c − d) ≤ lim_imp.
    Objective (max): Σ_t RRP_t/1000 · Δt · Σ_part (d − c) − Σ_part rate·d·Δt
                     − λ_u Σ u_i − terminal cost.
    Returns (objective $, status message).
    """
    H = lim_imp.shape[1]
    cols = []                     # (session, t, kind) kind 0=c, 1=d
    for i, s in enumerate(sessions):
        t_end = min(s.t_dep, T)
        for t in range(s.t_arr, t_end):
            cols.append((i, t, 0))
            if s.part:
                cols.append((i, t, 1))
    n_x = len(cols)
    n_u = len(sessions)
    n = n_x + n_u
    if n_x == 0:
        return 0.0, "no sessions"
    ci = np.array([c[0] for c in cols]); ct = np.array([c[1] for c in cols]); ck = np.array([c[2] for c in cols])
    part = np.array([sessions[i].part for i in ci])
    rrp_t = rrp[ct]

    # Objective (minimise negative profit)
    cost = np.zeros(n)
    wholesale = rrp_t / 1000.0 * dt
    rate = np.array([sessions[i].rate for i in ci])
    cost[:n_x] = np.where(ck == 0, wholesale * part, -wholesale * part + rate * dt * part)
    cost[n_x:] = lambda_unmet
    mean_rrp = max(0.0, float(np.mean(rrp)))
    # Terminal cost for sessions still connected at T: (target − E_T)+ / η · mean_rrp/1000.
    # Handled with an extra slack per such session: v_i ≥ target − E_T, v_i ≥ 0.
    open_ids = [i for i, s in enumerate(sessions) if s.t_dep > T and s.part]
    # Participant billing (env option): each participant pays mean_rrp/η per
    # kWh of its request, less any unmet energy -> a constant plus a cost on u.
    bill_const = 0.0
    if participant_billing:
        p_kwh = mean_rrp / 1000.0 / eta
        for i, s in enumerate(sessions):
            if s.part:
                bill_const += p_kwh * max(0.0, s.target - s.e_arr)
                if s.t_dep <= T:
                    cost[n_x + i] += p_kwh
    n_v = len(open_ids)
    cost = np.concatenate([cost, np.full(n_v, mean_rrp / 1000.0 / eta)])
    n_tot = n + n_v

    bounds = []
    for (i, t, k) in cols:
        s = sessions[i]
        bounds.append((0.0, s.p_ch if k == 0 else s.p_dis))
    bounds += [(0.0, None)] * (n_u + n_v)

    rows, data, rhs_ub = [], [], []
    rid = 0
    A_rows, A_cols, A_vals = [], [], []

    def add_row(idx, vals, b):
        nonlocal rid
        A_rows.extend([rid] * len(idx)); A_cols.extend(idx); A_vals.extend(vals); rhs_ub.append(b); rid += 1

    # Energy limits for every session at every step after arrival (cumulative)
    by_session = {}
    for j, (i, t, k) in enumerate(cols):
        by_session.setdefault(i, []).append((t, k, j))
    for i, s in enumerate(sessions):
        entries = sorted(by_session.get(i, []))
        # The env only blocks discharge below soc_min, so an EV that arrives
        # below it has its arrival energy as the effective floor.
        floor = min(soc_min * s.cap, s.e_arr) if s.part else 0.0
        acc_idx, acc_val = [], []
        steps = sorted({t for t, _, _ in entries})
        e_of = {t: [(k, j) for (tt, k, j) in entries if tt == t] for t in steps}
        for t in steps:
            for k, j in e_of[t]:
                acc_idx.append(j); acc_val.append(eta * dt if k == 0 else -dt / eta)
            # E after step t: e_arr + Σ ≤ cap ;  ≥ floor
            add_row(list(acc_idx), list(acc_val), s.cap - s.e_arr)
            add_row(list(acc_idx), [-v for v in acc_val], s.e_arr - floor)
        if s.t_dep <= T:
            # e_arr + Σ + u ≥ target  ->  −Σ − u ≤ e_arr − target
            add_row(list(acc_idx) + [n_x + i], [-v for v in acc_val] + [-1.0], s.e_arr - s.target)
        elif s.part:
            vj = n + open_ids.index(i)
            add_row(list(acc_idx) + [vj], [-v for v in acc_val] + [-1.0], s.e_arr - s.target)

    hub_of = np.array([sessions[i].hub for i in ci])
    sgn = np.where(ck == 0, 1.0, -1.0)
    if network is None:
        # Hub limits per step (per-hub DOEs)
        for h in range(H):
            on_h = hub_of == h
            for t in range(T):
                m = np.where(on_h & (ct == t))[0]
                if len(m) == 0:
                    continue
                sign = sgn[m]
                add_row(list(m), list(sign), float(lim_imp[t, h]))
                add_row(list(m), list(-sign), float(lim_exp[t, h]))
    else:
        # Joint feeder constraints instead of per-hub DOEs (coordination
        # study, spec §6): hub capacity, linearised bus voltages and exact
        # (radial, lossless) section flows, per 30-min window at the
        # window's worst-case forecast background. y_h,t = Σ_{i on h} (c − d).
        B = network["block"]
        cap_h = network["hub_cap"]
        for h in range(H):
            on_h = hub_of == h
            for t in range(T):
                m = np.where(on_h & (ct == t))[0]
                if len(m):
                    add_row(list(m), list(sgn[m]), float(cap_h[h]))
                    add_row(list(m), list(-sgn[m]), float(cap_h[h]))
        for t in range(T):
            m = np.where(ct == t)[0]
            if len(m) == 0:
                continue
            w = t // B
            for key_S, key_rhs, flip in (("S_lo", "rhs_lo", -1.0), ("S_hi", "rhs_hi", 1.0)):
                coef = flip * network[key_S][w][:, hub_of[m]] * sgn[m][None, :]     # (nb, |m|)
                for b in range(coef.shape[0]):
                    add_row(list(m), list(coef[b]), float(network[key_rhs][w][b]))
            Mh = network["Mh"][:, hub_of[m]] * sgn[m][None, :]                      # (sections, |m|)
            for l in range(Mh.shape[0]):
                if np.any(Mh[l]):
                    add_row(list(m), list(Mh[l]), float(network["f_rhs_hi"][w][l]))
                    add_row(list(m), list(-Mh[l]), float(network["f_rhs_lo"][w][l]))

    A = sparse.csr_matrix((A_vals, (A_rows, A_cols)), shape=(rid, n_tot))
    res = linprog(cost, A_ub=A, b_ub=np.array(rhs_ub), bounds=bounds, method="highs")
    if res.status != 0:
        return (float("nan"), res.message, None) if return_flows else (float("nan"), res.message)
    if return_flows:
        flows = np.zeros((T, H))
        np.add.at(flows, (ct, hub_of), sgn * res.x[:n_x])
        return float(-res.fun) + bill_const, "optimal", flows
    return float(-res.fun) + bill_const, "optimal"


def perfect_foresight_bound(env, date: str, seed: int, incentives=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5)):
    """Best LP objective over constant incentives. Returns (value, best_incentive)."""
    best = (-np.inf, None)
    for c in incentives:
        sessions, rrp, li, le = record_day(env, date, seed, c)
        v, status = perfect_foresight_lp(sessions, rrp, li, le,
                                         eta=env.sessions.cfg.eta, dt=env.DT_HR,
                                         soc_min=env.sessions.cfg.soc_min,
                                         lambda_unmet=env.cfg.lambda_unmet, T=env.STEPS,
                                         participant_billing=env.cfg.participant_billing)
        if np.isfinite(v) and v > best[0]:
            best = (v, c)
    return best
