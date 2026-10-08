"""
mpc.py
======
Forecast-based model predictive control baseline for NEMFeederEnv
(spec §5, referee M7): what a real VIC1 aggregator could run without RL.

Every `replan_every` steps (default 6 = 30 min, the predispatch cadence) the
controller solves a shrinking-horizon LP to the end of the day for the
participants currently connected, then follows that plan until the next
re-plan. Information used at the decision for the interval ending ts:
  - current interval RRP (agents observe it too);
  - for later intervals, the latest AEMO predispatch run published at or
    before ts − 5 min (30-min periods, held over their six 5-min intervals),
    or the realised prices for the perfect-price diagnostic variant;
  - connected participants' energy, departure step and target (declared at
    plug-in), power limits and contracted incentive;
  - the day's DOE schedule (published day-ahead, from forecast background).
It does not know future arrivals or opt-in decisions. It offers a constant
incentive, selected on the validation days (never the test days).

LP (per re-plan at step t0, participants i, steps t ∈ [t0, min(dep_i, T))):
  c_it, d_it ≥ 0 charge / discharge kW,  E_i,t+1 = E_it + (η c − d/η)Δt,
  floor_i ≤ E_it ≤ cap_i,  E_i,dep + u_i ≥ target_i,
  sessions still connected at T: v_i ≥ target_i − E_i,T,
  per hub and step: −lim_exp ≤ Σ_i (c − d) ≤ lim_imp − reserve,
  max  Σ RRP_t/1000·Δt·Σ_i (d − c) − Σ rate_i·d·Δt − λ_u Σ u_i − mean(RRP)/1000/η Σ v_i.
Same objective terms as baselines/feeder_baselines.perfect_foresight_lp, but
with explicit energy states so it stays sparse enough to re-solve 48× a day.
"""

import copy
import logging

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linprog

logger = logging.getLogger(__name__)


def plan_lp(sess, rrp, lim_imp, lim_exp, t0, *, H, eta, dt, soc_min, lambda_unmet, T,
            participant_billing=False, deg_cost=0.0, import_tariff=0.0, tariff_passthrough=False,
            discharge_price_factor=1.0):
    """
    Solve the shrinking-horizon LP. `sess` is a dict of arrays for connected
    participants: hub, dep, cap, E, target, p_ch, p_dis, rate.
    Returns (hub_q (T-t0, H) planned participant net kW, + = charging, status).
    """
    n_s = len(sess["hub"])
    steps = T - t0
    if n_s == 0 or steps <= 0:
        return np.zeros((max(steps, 0), H)), "no sessions"
    # Per session, the steps it can act in
    t_end = np.minimum(sess["dep"], T)
    lens = np.maximum(t_end - t0, 0)
    off = np.concatenate([[0], np.cumsum(lens)])        # offsets of each session's step block
    n_x = int(off[-1])                                  # per (session, step): c and d -> 2 n_x
    # Variable layout: [c (n_x) | d (n_x) | E (n_x + n_s): E at the start of each step + final | u (n_s) | v (n_s)]
    iC, iD = 0, n_x
    iE = 2 * n_x
    n_E = n_x + n_s
    iU = iE + n_E
    iV = iU + n_s
    n = iV + n_s

    sid = np.repeat(np.arange(n_s), lens)               # session of each (session, step) pair
    tt = t0 + np.concatenate([np.arange(L) for L in lens]) if n_x else np.zeros(0, int)
    hub = sess["hub"][sid]

    cost = np.zeros(n)
    w = rrp[tt] / 1000.0 * dt
    cost[iC:iC + n_x] = w
    cost[iD:iD + n_x] = -discharge_price_factor * w + sess["rate"][sid] * dt + deg_cost * dt
    cost[iC:iC + n_x] += import_tariff / 1000.0 * dt
    cost[iU:iU + n_s] = lambda_unmet
    p_mean = max(0.0, float(np.mean(rrp))) + (import_tariff if tariff_passthrough else 0.0)
    if participant_billing:   # unmet energy is also not billed (the billed constant does not affect the plan)
        cost[iU:iU + n_s] += p_mean / 1000.0 / eta
    still = sess["dep"] > T
    cost[iV:iV + n_s] = np.where(still, p_mean / 1000.0 / eta, 0.0)

    floor = np.minimum(soc_min * sess["cap"], sess["E"])
    lb = np.zeros(n); ub = np.full(n, np.inf)
    ub[iC:iC + n_x] = sess["p_ch"][sid]
    ub[iD:iD + n_x] = sess["p_dis"][sid]
    # E index of session s at local step k: iE + off[s] + s + k  (k = 0..lens[s])
    e_idx = lambda s, k: iE + off[s] + s + k
    for s in range(n_s):
        lb[e_idx(s, 0):e_idx(s, lens[s]) + 1] = floor[s]
        ub[e_idx(s, 0):e_idx(s, lens[s]) + 1] = sess["cap"][s]
    ub[iV:iV + n_s] = np.where(still, np.inf, 0.0)

    # Equalities: E_s,0 = E_now ; E_s,k+1 − E_s,k − η dt c + dt/η d = 0
    r, c_, v, b = [], [], [], []
    row = 0
    for s in range(n_s):
        r.append(row); c_.append(e_idx(s, 0)); v.append(1.0); b.append(sess["E"][s]); row += 1
    j = np.arange(n_x)
    k_local = np.concatenate([np.arange(L) for L in lens]) if n_x else np.zeros(0, int)
    rows_dyn = row + j
    e_next = np.array([e_idx(s, k + 1) for s, k in zip(sid, k_local)], int)
    e_now = e_next - 1
    r += list(np.repeat(rows_dyn, 4));
    c_ += list(np.column_stack([e_next, e_now, iC + j, iD + j]).ravel())
    v += list(np.tile([1.0, -1.0, -eta * dt, dt / eta], n_x))
    b += [0.0] * n_x
    row += n_x
    A_eq = sparse.csr_matrix((v, (r, c_)), shape=(row, n))
    b_eq = np.array(b)

    # Inequalities: departure targets, end-of-day need, hub limits
    r, c_, v, b = [], [], [], []
    row = 0
    for s in range(n_s):
        e_last = e_idx(s, lens[s])
        if not still[s]:   # −E_dep − u ≤ −target
            r += [row, row]; c_ += [e_last, iU + s]; v += [-1.0, -1.0]; b.append(-sess["target"][s])
        else:              # −E_T − v ≤ −target
            r += [row, row]; c_ += [e_last, iV + s]; v += [-1.0, -1.0]; b.append(-sess["target"][s])
        row += 1
    if n_x:
        key = (tt - t0) * H + hub
        uniq, inv = np.unique(key, return_inverse=True)
        n_hr = len(uniq)
        # Σ (c − d) ≤ lim_imp   and   Σ (d − c) ≤ lim_exp
        r += list(row + inv) + list(row + inv)
        c_ += list(iC + j) + list(iD + j)
        v += [1.0] * n_x + [-1.0] * n_x
        ut, uh = uniq // H + t0, uniq % H
        b += list(lim_imp[ut, uh])
        row += n_hr
        r += list(row + inv) + list(row + inv)
        c_ += list(iC + j) + list(iD + j)
        v += [-1.0] * n_x + [1.0] * n_x
        b += list(lim_exp[ut, uh])
        row += n_hr
    A_ub = sparse.csr_matrix((v, (r, c_)), shape=(row, n))
    b_ub = np.array(b, float)

    res = linprog(cost, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=np.column_stack([lb, ub]), method="highs")
    if res.status != 0:
        return np.zeros((steps, H)), res.message
    x = res.x
    q = x[iC:iC + n_x] - x[iD:iD + n_x]
    if "exec" in sess:                     # VEM-1 O2: only real participants follow the plan
        q = q * sess["exec"][sid]
    hub_q = np.zeros((steps, H))
    np.add.at(hub_q, (tt - t0, hub), q)
    return hub_q, "optimal"


class ForecastMPC:
    """Rolling LP with AEMO predispatch prices (or realised prices, diagnostic)."""
    needs_env = True

    def __init__(self, env, incentive: float, predispatch=None, replan_every: int = 6,
                 name: str = None, assume=()):
        self.env = env
        self.c = float(incentive)
        self.pd = predispatch                  # None -> realised prices (perfect-price diagnostic)
        self.k = replan_every
        self.name = name or ("MPC-Predispatch" if predispatch is not None else "MPC-PerfectPrice")
        self._plan, self._t_plan = None, None
        self.solve_failures = 0
        # VEM-1 (spec §6): plan with old-model assumptions, executed in the real env.
        #   O1 no DOEs (limits = hub capacity), O2 every connected EV dispatchable,
        #   O3 discharge paid 1.2× RRP, O4 soft departure (λ_unmet 0.1, no billing on u).
        self.assume = set(assume)
        bad = self.assume - {"O1", "O2", "O3", "O4"}
        if bad:
            raise ValueError(f"unknown MPC assumptions {bad}")

    # --------------------------------------------------------------
    def _prices(self, t):
        env = self.env
        rrp = env._rrp.astype(float).copy()
        if self.pd is None:
            return rrp
        idx = env._idx
        decided_at = idx[t] - pd.Timedelta(minutes=5)
        fc = self.pd.forecast_at(decided_at)
        periods = idx[t + 1:].ceil("30min")
        f = fc.reindex(periods).to_numpy(float)
        if np.isnan(f).any():                  # periods the run does not cover: hold the last forecast
            f = pd.Series(f).ffill().bfill().fillna(rrp[t]).to_numpy()
        rrp[t + 1:] = f                        # current interval: observed price
        return rrp

    def _replan(self, t):
        env, hs = self.env, self.env.sessions
        A = self.assume
        m = hs.occ & (hs.part | ("O2" in A)) & (hs.dep > t)
        sess = {"hub": hs.port_hub[m], "dep": hs.dep[m], "cap": hs.cap[m], "E": hs.E[m],
                "target": hs.target[m], "p_ch": hs.p_ch[m], "p_dis": hs.p_dis[m],
                "rate": hs.rate[m] * hs.part[m]}
        if "O2" in A:
            sess["exec"] = hs.part[m].astype(float)
        cap = np.broadcast_to(env.cap, env._lim_imp.shape)
        lim_i = cap if "O1" in A else env._lim_imp
        lim_e = cap if "O1" in A else env._lim_exp
        plan, status = plan_lp(sess, self._prices(t), lim_i, lim_e, t,
                               H=env.H, eta=hs.cfg.eta, dt=env.DT_HR, soc_min=hs.cfg.soc_min,
                               lambda_unmet=0.1 if "O4" in A else env.cfg.lambda_unmet, T=env.STEPS,
                               participant_billing=env.cfg.participant_billing and "O4" not in A,
                               deg_cost=env.cfg.deg_cost, import_tariff=env.cfg.import_tariff,
                               tariff_passthrough=env.cfg.tariff_passthrough,
                               discharge_price_factor=(1.2 if "O3" in A else 1.0) * env.cfg.discharge_price_factor)
        if status not in ("optimal", "no sessions"):
            self.solve_failures += 1
            logger.warning(f"MPC LP at step {t}: {status}; holding idle")
        self._plan, self._t_plan = plan, t

    def _to_action(self, q, t):
        """Requested participant net kW per hub -> env's normalised setpoint."""
        env = self.env
        if env.cfg.action_scale == "capacity":
            disp = -q / env.cap
        else:
            sim = copy.deepcopy(env.sessions)
            sim.depart(t)
            flex_lo, flex_hi = sim.hub_bounds(t)[:2]
            disp = np.zeros(env.H)
            dis = q < 0
            disp[dis] = np.where(flex_lo[dis] < -1e-9, q[dis] / np.minimum(flex_lo[dis], -1e-9), 0.0)
            disp[~dis] = np.where(flex_hi[~dis] > 1e-9, -q[~dis] / np.maximum(flex_hi[~dis], 1e-9), 0.0)
        return np.clip(disp, -1.0, 1.0)

    def select_action(self, obs, deterministic=True):
        t = self.env._t
        if t == 0 or self._plan is None or self._t_plan is None or t < self._t_plan \
                or t - self._t_plan >= self.k or t - self._t_plan >= len(self._plan):
            self._replan(t)
        q = self._plan[t - self._t_plan] if len(self._plan) else np.zeros(self.env.H)
        return np.append(self._to_action(q, t), self.c).astype(np.float32)
