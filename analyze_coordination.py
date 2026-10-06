"""
analyze_coordination.py
=======================
Does a trained policy use information from other hubs, and when?
(docs/env_redesign_spec.md §6, diagnostics for claim C2.)

For each checkpoint:
  1. Collect states the policy itself visits on held-out days (deterministic).
  2. Local perturbation: for a hub j, replace its *dynamic local* features
     (limits, uncontrolled load, fleet state, energy need, occupancy,
     voltage; not its static capacity, not the global price/time features)
     with those of a randomly drawn real (state, hub). Measure the change in
     the normalised dispatch action (units of hub capacity, range [-1, 1]) at
     hub j itself and at other hubs grouped by hop distance on the ELECTRICAL
     feeder graph (1, 2, ≥3 / unconnected).
  3. Stratify by whether hub j's limit was binding in the original state.
  4. Global perturbation: swap price/time features with another state's.
  5. GAT attention: entropy (normalised by log of in-degree incl. self-loop)
     and self-attention weight, layer 1.

Neighbours are always defined on the electrical graph, whatever graph the
agent used, so an edgeless agent is a valid control (neighbour response ≈ 0).

Usage:
  python analyze_coordination.py \
      --agent SAC-GNN=sac_gnn:<ckpt> --agent SAC-GNN-NoEdge=sac_gnn:<ckpt> \
      --doe_mode per_hub --n_days 6 --results_dir <dir>
"""

import argparse
import json
import logging
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

from train_sac_gnn import DEFAULT_CONFIG, make_env
from evaluate_feeder import ENV_KEYS, select_eval_days, load_learned

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

LOCAL_DYNAMIC = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10]   # node features perturbed locally (not 9 = capacity)
GLOBAL = [11, 12, 13, 14, 15, 16]                 # price / time features


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--agent", action="append", required=True, help="NAME=TYPE:CHECKPOINT")
    p.add_argument("--doe_mode", default="per_hub", choices=["per_hub", "network"])
    p.add_argument("--spatial", default="feeder", choices=["feeder", "permuted"])
    p.add_argument("--forecast_sigma", type=float, default=DEFAULT_CONFIG["forecast_sigma"])
    p.add_argument("--pv_penetration", type=float, default=DEFAULT_CONFIG["pv_penetration"])
    p.add_argument("--n_days", type=int, default=6, help="held-out days used to collect states")
    p.add_argument("--state_every", type=int, default=8, help="keep every k-th step's state")
    p.add_argument("--hubs_per_state", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--results_dir", required=True)
    p.add_argument("--skip_config_check", action="store_true")
    return p.parse_args()


def hop_matrix(edge_index: np.ndarray, H: int) -> np.ndarray:
    """All-pairs hop distance on an undirected graph (inf if unconnected)."""
    adj = [[] for _ in range(H)]
    for s, d in edge_index.T:
        if s != d:
            adj[s].append(d); adj[d].append(s)
    D = np.full((H, H), np.inf)
    for s in range(H):
        D[s, s] = 0; q = deque([s])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if D[s, v] == np.inf:
                    D[s, v] = D[s, u] + 1; q.append(v)
    return D


def collect_states(env, agent, days, every, seed):
    states = []
    for i, date in enumerate(days):
        obs, _ = env.reset(seed=seed + i, options={"date": date})
        done, t = False, 0
        while not done:
            if t % every == 0:
                states.append(env.obs_to_node_features(obs).copy())
            obs, _, done, _, _ = env.step(agent.select_action(obs, deterministic=True))
            t += 1
    return np.array(states)                                   # (N, H, F)


def dispatch(agent, X):
    return agent.select_action(X.reshape(-1), deterministic=True)[:X.shape[0]]


def attention_stats(agent, X):
    """Layer-1 GAT attention entropy (normalised) and self-weight; None if not a GAT agent."""
    enc = getattr(getattr(agent, "actor", None), "encoder", None)
    gat1 = getattr(enc, "gat1", None)
    if gat1 is None:
        return None
    import torch
    agent.actor.eval()
    ents, selfw = [], []
    with torch.no_grad():
        for x in X:
            _, (ei, alpha) = gat1(torch.tensor(x, dtype=torch.float32, device=agent.device),
                                  agent._edge_index_t, return_attention_weights=True)
            ei = ei.cpu().numpy(); al = alpha.cpu().numpy().mean(1)
            for i in range(x.shape[0]):
                m = ei[1] == i; w = al[m]; k = int(m.sum())
                if k > 1:
                    ents.append(float(-(w * np.log(w + 1e-12)).sum() / np.log(k)))
                selfw.append(float(w[ei[0][m] == i].sum()))
    return {"attention_entropy_norm": float(np.mean(ents)) if ents else float("nan"),
            "attention_self_weight": float(np.mean(selfw))}


def analyse(agent, states, D, cfg, rng):
    N, H, F = states.shape
    pool = states[:, :, LOCAL_DYNAMIC].reshape(-1, len(LOCAL_DYNAMIC))
    rows = []
    for n in range(N):
        X = states[n]
        base = dispatch(agent, X)
        for j in rng.choice(H, size=min(cfg.hubs_per_state, H), replace=False):
            X2 = X.copy(); X2[j, LOCAL_DYNAMIC] = pool[rng.integers(len(pool))]
            diff = np.abs(dispatch(agent, X2) - base)
            binding = bool(min(X[j, 0], X[j, 1]) < 0.999)
            for i in range(H):
                h = D[j, i]
                group = "self" if i == j else ("hop1" if h == 1 else "hop2" if h == 2 else "hop3plus_or_unconnected")
                rows.append({"state": n, "j": int(j), "i": i, "group": group, "binding_j": binding, "delta": float(diff[i])})
    df = pd.DataFrame(rows)
    gl = []
    for n in range(N):
        X2 = states[n].copy(); X2[:, GLOBAL] = states[rng.integers(N)][:, GLOBAL]
        gl.append(float(np.abs(dispatch(agent, X2) - dispatch(agent, states[n])).mean()))
    return df, float(np.mean(gl))


def main():
    args = parse_args()
    out = Path(args.results_dir); out.mkdir(parents=True, exist_ok=True)
    cfg = dict(DEFAULT_CONFIG)
    cfg.update(env="feeder", doe_mode=args.doe_mode, spatial=args.spatial,
               forecast_sigma=args.forecast_sigma, pv_penetration=args.pv_penetration, graph="electrical")
    env, _, _ = make_env(cfg, split="eval", seed=0)
    from nem_env.spatial_graph import HubGraphBuilder
    road_graph, _ = HubGraphBuilder.load(cfg["graph_path"])
    elec = env.feeder.electrical_graph(road_graph)
    D = hop_matrix(elec.edge_index, env.H)

    sel = select_eval_days(env.price_loader._price_df)
    days = list(sel[sel["set"] == "representative"].index)     # typical days only
    rng = np.random.default_rng(args.seed)
    days = [days[i] for i in sorted(rng.choice(len(days), size=min(args.n_days, len(days)), replace=False))]
    expected = {k: cfg[k] for k in ENV_KEYS}

    summary = []
    for spec in args.agent:
        name, agent = load_learned(spec, env, road_graph, expected, args.skip_config_check)
        states = collect_states(env, agent, days, args.state_every, args.seed)
        df, glob = analyse(agent, states, D, args, np.random.default_rng(args.seed))
        df.to_csv(out / f"perturbation_{name}.csv", index=False)
        g = df.groupby("group")["delta"].mean()
        gb = df.groupby(["binding_j", "group"])["delta"].mean()
        rec = {"agent": name, "states": int(len(states)), "days": days,
               "self": g.get("self"), "hop1": g.get("hop1"), "hop2": g.get("hop2"),
               "hop3plus_or_unconnected": g.get("hop3plus_or_unconnected"),
               "hop1_over_self": g.get("hop1") / g.get("self") if g.get("self") else float("nan"),
               "hop1_when_j_binding": gb.get((True, "hop1"), float("nan")),
               "hop1_when_j_not_binding": gb.get((False, "hop1"), float("nan")),
               "self_when_j_binding": gb.get((True, "self"), float("nan")),
               "global_price_time": glob,
               "share_binding_states": float(df[df.group == "self"].binding_j.mean())}
        att = attention_stats(agent, states[:: max(1, len(states) // 60)])
        if att:
            rec.update(att)
        summary.append(rec)
        logger.info({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items() if k != "days"})

    pd.DataFrame(summary).drop(columns=["days"]).to_csv(out / "coordination_summary.csv", index=False)
    json.dump({"args": vars(args), "days": days}, open(out / "coordination_args.json", "w"), indent=2)


if __name__ == "__main__":
    main()
