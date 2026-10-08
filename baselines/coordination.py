"""
coordination.py
===============
Value of coordination under perfect foresight (spec §6, claim C2 answered
with optimisation): the perfect-foresight LP with per-hub DOEs (each hub
limited to its pre-allocated κ*·h_i) against the same LP with the feeder's
joint constraints (hub capacity, linearised bus voltages, exact section
flows). Their difference is what coordinating hubs through the network is
worth, independent of whether an RL agent learns it.

Both use the same forecast background and the same 30-minute windows as the
DOE computation: within a window the joint constraints hold at the window's
worst-case background (per bus / per section), so the comparison isolates
coordination from time resolution. As in the DOE rule, a hub may not make a
pre-existing background violation worse.
"""

import numpy as np

from baselines.feeder_baselines import perfect_foresight_lp, record_day


def network_constraints(feeder, P_fc, Q_fc, delta_kw: float = 10.0):
    """
    Linearised joint constraints per DOE window.

    P_fc, Q_fc : (T, nb) forecast background. Returns a dict for
    perfect_foresight_lp(network=...): per window w,
      S_lo[w], S_hi[w] : (nb, H) dV/dy (pu per kW of hub import) at the
                         window step with the lowest / highest voltage
      rhs_lo[w] = V0_min − min(v_min, V0_min)   (−S_lo·y ≤ rhs_lo)
      rhs_hi[w] = max(v_max, V0_max) − V0_max   ( S_hi·y ≤ rhs_hi)
      f_rhs_hi[w] = max(rating, F0_max) − F0_max ;  f_rhs_lo[w] = F0_min − min(−rating, F0_min)
    plus Mh (sections, H) and hub_cap.
    """
    cfg = feeder.cfg
    T, nb = P_fc.shape
    B = cfg.doe_block
    H = feeder.H
    out = {k: [] for k in ("S_lo", "S_hi", "rhs_lo", "rhs_hi", "f_rhs_hi", "f_rhs_lo")}
    for s in range(0, T, B):
        Pw, Qw = P_fc[s:s + B], Q_fc[s:s + B]
        V = feeder.voltages(Pw, Qw)                                  # (B, nb)
        v0_min, v0_max = V.min(axis=0), V.max(axis=0)
        sens = []
        for k in (int(np.argmin(V.min(axis=1))), int(np.argmax(V.max(axis=1)))):
            P = np.repeat(Pw[k][None, :], H + 1, axis=0)
            P[np.arange(1, H + 1), feeder.hub_bus] += delta_kw
            Vp = feeder.voltages(P, np.repeat(Qw[k][None, :], H + 1, axis=0))
            sens.append(((Vp[1:] - Vp[0]) / delta_kw).T)             # (nb, H)
        out["S_lo"].append(sens[0]); out["S_hi"].append(sens[1])
        out["rhs_lo"].append(v0_min - np.minimum(cfg.v_min, v0_min))
        out["rhs_hi"].append(np.maximum(cfg.v_max, v0_max) - v0_max)
        F = feeder.section_flows(Pw)                                 # (B, sections)
        f_max, f_min = F.max(axis=0), F.min(axis=0)
        out["f_rhs_hi"].append(np.maximum(feeder.rating, f_max) - f_max)
        out["f_rhs_lo"].append(f_min - np.minimum(-feeder.rating, f_min))
    out = {k: np.array(v) for k, v in out.items()}
    out["Mh"] = feeder.M[:, feeder.hub_bus]
    out["hub_cap"] = feeder.hub_cap.copy()
    out["block"] = B
    return out


def check_flows(feeder, P_bg, Q_bg, flows):
    """Full power flow of planned hub flows (T, H) on a background: worst violations."""
    P = feeder.with_hubs(P_bg, flows)
    V = feeder.voltages(P, Q_bg)
    F = feeder.section_flows(P)
    Vb = feeder.voltages(P_bg, Q_bg)
    Fb = feeder.section_flows(P_bg)
    cfg = feeder.cfg
    # violations beyond what the background alone already has
    v_lo = np.maximum(0.0, np.minimum(cfg.v_min, Vb) - V).max()
    v_hi = np.maximum(0.0, V - np.maximum(cfg.v_max, Vb)).max()
    f_hi = np.maximum(0.0, F - np.maximum(feeder.rating, Fb)).max()
    f_lo = np.maximum(0.0, np.minimum(-feeder.rating, Fb) - F).max()
    return {"v_under_pu": float(v_lo), "v_over_pu": float(v_hi),
            "overload_kw": float(max(f_hi, f_lo))}


def coordination_value(env, date: str, seed: int, incentives=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5)):
    """
    Best LP value over constant incentives with per-hub DOEs and with joint
    feeder constraints, plus a full-power-flow check of the network plan on
    the forecast background (linearisation) and the realised one.
    """
    kw = dict(eta=env.sessions.cfg.eta, dt=env.DT_HR, soc_min=env.sessions.cfg.soc_min,
              lambda_unmet=env.cfg.lambda_unmet, T=env.STEPS,
              participant_billing=env.cfg.participant_billing,
              deg_cost=env.cfg.deg_cost, import_tariff=env.cfg.import_tariff,
              tariff_passthrough=env.cfg.tariff_passthrough)
    best = {"perhub": (-np.inf, None, None), "network": (-np.inf, None, None)}
    net = None
    for c in incentives:
        sessions, rrp, li, le = record_day(env, date, seed, c)
        if net is None:
            net = network_constraints(env.feeder, env._P_fc, env._Q_fc)
            P_real = env._P_fc * (1.0 + env._eps); Q_real = env._Q_fc * (1.0 + env._eps)
        v, _, f = perfect_foresight_lp(sessions, rrp, li, le, return_flows=True, **kw)
        if np.isfinite(v) and v > best["perhub"][0]:
            best["perhub"] = (v, c, f)
        v, _, f = perfect_foresight_lp(sessions, rrp, li, le, network=net, return_flows=True, **kw)
        if np.isfinite(v) and v > best["network"][0]:
            best["network"] = (v, c, f)
    row = {}
    for mode in ("perhub", "network"):
        v, c, f = best[mode]
        row[f"lp_{mode}"] = v
        row[f"c_{mode}"] = c
        if f is not None:
            for bg, (P, Q) in (("fc", (env._P_fc, env._Q_fc)), ("real", (P_real, Q_real))):
                for k, x in check_flows(env.feeder, P, Q, f).items():
                    row[f"{mode}_{bg}_{k}"] = x
    row["value_of_coordination"] = row["lp_network"] - row["lp_perhub"]
    return row


def shared_limit_groups(feeder, k_hops: int = 3):
    """Groups of hubs within k electrical hops of each other (graph components)."""
    import networkx as nx
    bus = [int(feeder.bus_ids[b]) for b in feeder.hub_bus]
    g = nx.Graph(); g.add_nodes_from(range(feeder.H))
    for i in range(feeder.H):
        for j in range(i + 1, feeder.H):
            if feeder.hops[bus[i]][bus[j]] <= k_hops:
                g.add_edge(i, j)
    return [sorted(c) for c in nx.connected_components(g)]


def shared_limit_value(env, date: str, seed: int, frac: float,
                       incentives=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5)):
    """
    Robustness check S2a: each group of hubs sits behind one shared connection
    limit L_g = frac × Σ hub capacity. Coordinated: Σ_group flows within ±L_g
    (plus each hub's own DOE). Uncoordinated: L_g split across the group's hubs
    in proportion to capacity, each hub limited to min(DOE, its share).
    Returns best LP values over constant incentives and their difference.
    """
    kw = dict(eta=env.sessions.cfg.eta, dt=env.DT_HR, soc_min=env.sessions.cfg.soc_min,
              lambda_unmet=env.cfg.lambda_unmet, T=env.STEPS,
              participant_billing=env.cfg.participant_billing,
              deg_cost=env.cfg.deg_cost, import_tariff=env.cfg.import_tariff,
              tariff_passthrough=env.cfg.tariff_passthrough)
    groups = shared_limit_groups(env.feeder)
    cap = env.feeder.hub_cap
    limits = [(g, frac * cap[g].sum()) for g in groups]
    share = np.zeros(env.H)
    for g, L in limits:
        share[g] = L * cap[g] / cap[g].sum()
    best = {"coord": (-np.inf, None), "split": (-np.inf, None)}
    for c in incentives:
        sessions, rrp, li, le = record_day(env, date, seed, c)
        v, _ = perfect_foresight_lp(sessions, rrp, li, le, group_limits=limits, **kw)
        if np.isfinite(v) and v > best["coord"][0]:
            best["coord"] = (v, c)
        v, _ = perfect_foresight_lp(sessions, rrp, np.minimum(li, share), np.minimum(le, share), **kw)
        if np.isfinite(v) and v > best["split"][0]:
            best["split"] = (v, c)
    return {"frac": frac, "n_groups": len(groups), "group_sizes": [len(g) for g in groups],
            "lp_coord": best["coord"][0], "c_coord": best["coord"][1],
            "lp_split": best["split"][0], "c_split": best["split"][1],
            "value_of_coordination": best["coord"][0] - best["split"][0]}
