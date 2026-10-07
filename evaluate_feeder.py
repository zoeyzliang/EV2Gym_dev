"""
evaluate_feeder.py
==================
Evaluation for NEMFeederEnv (docs/env_redesign_spec.md §6).

* Held-out 2024 days, stratified: for each month, the days at the 20th, 50th
  and 90th percentile of that month's daily RRP volatility (36 days), never
  including the training VALIDATION_DAYS used for checkpoint selection. Each
  day is labelled with its volatility tier and weekday/weekend.
* Paired runs: repetition k of a day uses the same env seed for every agent.
* Metrics per run: reward and its components (wholesale, incentive, unmet,
  limit, terminal), limit compliance on realised net site flow, realised
  voltage and section-overload outcomes from power flow, unmet energy,
  opt-in rate, discharged energy.
* Optional perfect-foresight LP bound per day (best constant incentive).
* Each learned checkpoint's config.json must match the evaluation env
  (env, doe_mode, spatial, forecast_sigma, pv_penetration, graph); a
  mismatch aborts unless --skip_config_check.

Usage:
  python evaluate_feeder.py \
      --agent SAC-GNN=sac_gnn:results/x/checkpoints/best.pt \
      --agent SAC-GNN-NoEdge=sac_gnn:results/y/checkpoints/best.pt \
      --doe_mode per_hub --n_reps 3 --lp --results_dir results/eval_feeder_x
"""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from train_sac_gnn import DEFAULT_CONFIG, make_env, VALIDATION_DAYS
from baselines.gnn_rl.agent import SACGNNAgent
from baselines.gnn_rl.sac_gcn import SACGCNAgent
from baselines.flat_mlp.sac_flat import SACFlatAgent
from baselines.gnn_rl.networks import NetworkConfig
from baselines.feeder_baselines import NoV2GBaseline, GreedyTOUBaseline, RulePriceBaseline, perfect_foresight_bound

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

ENV_KEYS = ("env", "doe_mode", "spatial", "forecast_sigma", "pv_penetration", "kappa_load", "graph",
            "action_scale", "participant_billing")


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate agents on NEMFeederEnv")
    p.add_argument("--agent", action="append", default=[],
                   help="NAME=TYPE:CHECKPOINT, TYPE in sac_gnn|sac_gcn|sac_flat (repeatable)")
    p.add_argument("--doe_mode", default="per_hub", choices=["per_hub", "network"])
    p.add_argument("--spatial", default="feeder", choices=["feeder", "permuted"])
    p.add_argument("--participant_billing", action="store_true",
                   help="bill participants for their energy (must match training)")
    p.add_argument("--action_scale", default="capacity", choices=["capacity", "feasible"],
                   help="must match training (checked against each checkpoint's config)")
    p.add_argument("--forecast_sigma", type=float, default=DEFAULT_CONFIG["forecast_sigma"])
    p.add_argument("--pv_penetration", type=float, default=DEFAULT_CONFIG["pv_penetration"])
    p.add_argument("--graph", default="electrical", choices=["electrical", "road"])
    p.add_argument("--n_reps", type=int, default=3, help="repetitions per day (paired seeds)")
    p.add_argument("--max_days", type=int, default=None, help="limit days (smoke tests)")
    p.add_argument("--lp", action="store_true", help="compute the perfect-foresight LP bound")
    p.add_argument("--lp_reps", type=int, default=1, help="repetitions per day for the LP bound")
    p.add_argument("--mpc_incentive", type=float, default=None,
                   help="add the forecast MPC baseline (AEMO predispatch prices) with this constant "
                        "incentive ($/kWh, chosen on validation days, spec §5)")
    p.add_argument("--mpc_perfect", action="store_true",
                   help="also add the perfect-price MPC diagnostic (realised prices, same incentive)")
    p.add_argument("--results_dir", required=True)
    p.add_argument("--skip_config_check", action="store_true")
    # E4 (spec §6): evaluate trained policies when owners respond differently
    # from the participation model they were trained under.
    p.add_argument("--beta1_scale", type=float, default=1.0, help="multiplier on price sensitivity β1")
    p.add_argument("--beta3_scale", type=float, default=1.0, help="multiplier on SoC effect β3")
    p.add_argument("--gamma_scale", type=float, default=1.0, help="multiplier on degradation cost γ")
    return p.parse_args()


# ----------------------------------------------------------------------
def select_eval_days(price_df: pd.DataFrame, year: int = 2024,
                     stress_threshold: float = 500.0, stress_max: int = 8) -> pd.DataFrame:
    """
    Held-out evaluation days, pre-registered in docs/env_redesign_spec.md §6
    (never including the training VALIDATION_DAYS):

    * representative set: for each month, the days at the 20th / 50th / 90th
      percentile of that month's daily RRP standard deviation (main results);
    * stress set: every other day with daily RRP std ≥ stress_threshold
      ($/MWh, the volatile/extreme curriculum tiers), at most stress_max of
      them (highest std first). Reported separately, never pooled.

    Returns a frame indexed by "YYYY-MM-DD" with columns std, min, max, size,
    set ("representative" | "stress"), tier and weekend.
    """
    df = price_df[price_df.index.year == year]
    daily = df.groupby(df.index.date)["spot_price"].agg(["std", "min", "max", "size"])
    # A day needs all 288 intervals: PriceLoader silently substitutes a random
    # day for a shorter one (e.g. 1 Jan, which lacks its 00:00 interval).
    daily = daily[daily["size"] >= 288]
    daily.index = pd.to_datetime(daily.index)
    daily = daily[~daily.index.strftime("%Y-%m-%d").isin(VALIDATION_DAYS)]
    rep = []
    for m, g in daily.groupby(daily.index.month):
        g = g.sort_values("std")
        for q in (0.2, 0.5, 0.9):
            rep.append(g.index[int(round(q * (len(g) - 1)))])
    rep = sorted(set(rep))
    rest = daily.drop(index=rep)
    stress = rest[rest["std"] >= stress_threshold].sort_values("std", ascending=False).index[:stress_max]
    out = pd.concat([daily.loc[rep].assign(set="representative"),
                     daily.loc[sorted(stress)].assign(set="stress")])
    out["tier"] = pd.cut(out["std"], [-np.inf, 100, 500, 2000, np.inf],
                         labels=["calm", "normal", "volatile", "extreme"])
    out["weekend"] = out.index.dayofweek >= 5
    out.index = out.index.strftime("%Y-%m-%d")
    return out


def episode_seed(date: str, rep: int) -> int:
    """Seed tied to the date (not its position in the list), so a (date, rep)
    pair always gets the same environment realisation in every evaluation."""
    return 10_000 + int(pd.Timestamp(date).strftime("%Y%m%d")) % 1_000_000 * 10 + rep


def check_config(ckpt: str, expected: dict, skip: bool) -> dict:
    path = Path(ckpt).parent.parent / "config.json"
    if not path.exists():
        msg = f"No config.json next to {ckpt}"
        if skip:
            logger.warning(msg); return {}
        raise SystemExit(msg)
    cfg = json.load(open(path))
    bad = [f"{k}: trained {cfg.get(k)!r} vs eval {expected[k]!r}" for k in ENV_KEYS
           if k in expected and cfg.get(k, DEFAULT_CONFIG.get(k)) != expected[k]]
    if bad:
        msg = f"Checkpoint/env mismatch for {ckpt}: " + "; ".join(bad)
        if skip:
            logger.warning(msg)
        else:
            raise SystemExit(msg)
    return cfg


def load_learned(spec: str, env, road_graph, expected, skip):
    name, rest = spec.split("=", 1)
    kind, ckpt = rest.split(":", 1)
    cfg = check_config(ckpt, expected, skip)
    H = env.H
    if kind == "sac_flat":
        agent = SACFlatAgent(obs_dim=env.observation_space.shape[0], action_dim=H + 1, seed=0,
                             node_feature_dim=env.node_feature_dim, cap_feature=None, cap_const=1.0)
    else:
        graph = env.feeder.electrical_graph(road_graph) if cfg.get("graph", "electrical") == "electrical" else road_graph
        if cfg.get("no_edges"):
            graph = graph.self_loops_only()
        net = NetworkConfig(node_feature_dim=env.node_feature_dim, equipment_cap_kw=1.0)
        cls = SACGCNAgent if kind == "sac_gcn" else SACGNNAgent
        agent = cls(n_hubs=H, graph_data=graph, obs_dim=env.observation_space.shape[0], net_cfg=net, seed=0)
    agent.load(ckpt)
    logger.info(f"Loaded {name} ({kind}, no_edges={cfg.get('no_edges', False)}) from {ckpt}")
    return name, agent


def run_episode(env, agent, date, seed):
    obs, info = env.reset(seed=seed, options={"date": date})
    if info.get("date") != date:
        raise RuntimeError(f"requested {date} but the env loaded {info.get('date')} "
                           "(PriceLoader falls back to a random day for incomplete dates)")
    keys = ["r_wholesale", "r_billing", "r_incentive", "p_unmet", "p_limit", "p_terminal", "arbitrage_profit",
            "limit_viol_kwh", "v_viol_pu", "overload_kw", "unmet_part_kwh", "unmet_nonpart_kwh",
            "arrivals", "opt_in", "discharged_kwh"]
    acc = {k: 0.0 for k in keys}
    reward, compliant, vmin, vmax, done = 0.0, 0, 9.0, 0.0, False
    imp_kwh = exp_kwh = 0.0                  # participants' grid-side energy (M4 accounting)
    prices, rrps, arrivals = [], [], []      # incentive trajectory (M9)
    while not done:
        obs, r, done, _, info = env.step(agent.select_action(obs, deterministic=True))
        reward += r
        for k in keys:
            acc[k] += info[k]
        flex = np.asarray(info["flex_kw"])
        imp_kwh += np.clip(flex, 0, None).sum() * env.DT_HR
        exp_kwh += np.clip(-flex, 0, None).sum() * env.DT_HR
        prices.append(info["incentive_price"]); rrps.append(info["rrp"]); arrivals.append(info["arrivals"])
        compliant += info["limit_compliant"]
        vmin, vmax = min(vmin, info["v_min"]), max(vmax, info["v_max"])
    acc["overload_kwh"] = acc.pop("overload_kw") * env.DT_HR
    # Main economic metric (spec §6): pre-penalty profit net of the energy
    # still owed to participants at the end of the day (a purchase, not a penalty).
    acc["net_economic"] = acc["arbitrage_profit"] - acc["p_terminal"]
    acc["part_import_kwh"], acc["part_export_kwh"] = imp_kwh, exp_kwh
    prices, rrps, arrivals = map(np.asarray, (prices, rrps, arrivals))
    acc["incentive_mean"] = float(prices.mean())
    # the offer that counts is the one made when EVs arrive (opt-in is per session)
    acc["incentive_at_arrival"] = float((prices * arrivals).sum() / arrivals.sum()) if arrivals.sum() else np.nan
    acc["incentive_rrp_corr"] = (float(np.corrcoef(prices, rrps)[0, 1])
                                 if prices.std() > 1e-9 and rrps.std() > 1e-9 else np.nan)
    return {"reward": reward, **acc, "compliance": compliant / env.STEPS,
            "opt_in_rate": acc["opt_in"] / max(1.0, acc["arrivals"]), "v_min": vmin, "v_max": vmax}


def main():
    args = parse_args()
    out = Path(args.results_dir); out.mkdir(parents=True, exist_ok=True)
    cfg = dict(DEFAULT_CONFIG)
    cfg.update(env="feeder", doe_mode=args.doe_mode, spatial=args.spatial, action_scale=args.action_scale,
               participant_billing=args.participant_billing,
               forecast_sigma=args.forecast_sigma, pv_penetration=args.pv_penetration, graph=args.graph)
    env, road_graph, hubs = make_env(cfg, split="eval", seed=0)
    pm = env.participation_model
    pm.beta_1 *= args.beta1_scale
    pm.beta_3 *= args.beta3_scale
    pm.gamma *= args.gamma_scale
    if (args.beta1_scale, args.beta3_scale, args.gamma_scale) != (1.0, 1.0, 1.0):
        logger.info(f"E4 participation perturbation: beta_1={pm.beta_1:.4f}, "
                    f"beta_3={pm.beta_3:.3f}, gamma={pm.gamma:.4f}")
    # make_env returns the encoder graph; learned agents rebuild theirs from config
    from nem_env.spatial_graph import HubGraphBuilder
    road_graph, _ = HubGraphBuilder.load(cfg["graph_path"])

    days = select_eval_days(env.price_loader._price_df)
    if args.max_days:
        days = days.iloc[:args.max_days]
    days.to_csv(out / "eval_days.csv")
    logger.info(f"{len(days)} held-out days: {days['set'].value_counts().to_dict()}; "
                f"tiers: {days['tier'].value_counts().to_dict()}")

    train_prices = pd.read_parquet(f"{cfg['cache_dir']}/{cfg['region']}_{cfg['price_start']}_{cfg['price_end']}.parquet")
    agents = [("NoV2G", NoV2GBaseline(env.H)),
              ("GreedyTOU", GreedyTOUBaseline.from_training_prices(env.H, train_prices)),
              ("RulePrice", RulePriceBaseline(env.H))]
    if args.mpc_incentive is not None:
        from baselines.forecast_mpc import ForecastMPC
        from nem_env.predispatch import Predispatch
        pdx = Predispatch.load_cache(f"{cfg['cache_dir']}/{cfg['region']}_predispatch_2024.parquet")
        agents.append(("MPC-Predispatch", ForecastMPC(env, args.mpc_incentive, pdx)))
        if args.mpc_perfect:
            agents.append(("MPC-PerfectPrice", ForecastMPC(env, args.mpc_incentive, None)))
    expected = {k: cfg[k] for k in ENV_KEYS}
    for spec in args.agent:
        agents.append(load_learned(spec, env, road_graph, expected, args.skip_config_check))

    rows, t0 = [], time.time()
    for di, (date, meta) in enumerate(days.iterrows()):
        for k in range(args.n_reps):
            seed = episode_seed(date, k)
            for name, agent in agents:
                m = run_episode(env, agent, date, seed)
                rows.append({"agent": name, "date": date, "rep": k, "seed": seed, "set": meta["set"],
                             "tier": meta["tier"], "weekend": meta["weekend"], **m})
            if args.lp and k < args.lp_reps:
                v, c = perfect_foresight_bound(env, date, seed)
                rows.append({"agent": "PF-LP-bound", "date": date, "rep": k, "seed": seed, "set": meta["set"],
                             "tier": meta["tier"], "weekend": meta["weekend"], "reward": v, "lp_incentive": c})
        logger.info(f"day {di + 1}/{len(days)} {date} ({meta['set']}, {meta['tier']}) done, {time.time() - t0:.0f}s")
        pd.DataFrame(rows).to_csv(out / "per_run.csv", index=False)

    df = pd.DataFrame(rows)
    df.to_csv(out / "per_run.csv", index=False)
    # Representative and stress days are summarised separately, never pooled.
    summary = df.groupby(["set", "agent"]).agg(
        reward=("reward", "mean"), net_economic=("net_economic", "mean"),
        arbitrage_profit=("arbitrage_profit", "mean"), p_terminal=("p_terminal", "mean"),
        compliance=("compliance", "mean"), limit_viol_kwh=("limit_viol_kwh", "mean"),
        v_viol_pu=("v_viol_pu", "mean"), overload_kwh=("overload_kwh", "mean"),
        unmet_part_kwh=("unmet_part_kwh", "mean"), unmet_nonpart_kwh=("unmet_nonpart_kwh", "mean"),
        opt_in_rate=("opt_in_rate", "mean"), incentive_at_arrival=("incentive_at_arrival", "mean"),
        part_import_kwh=("part_import_kwh", "mean"), part_export_kwh=("part_export_kwh", "mean"),
        runs=("reward", "size"))
    summary.to_csv(out / "summary.csv")
    json.dump(vars(args), open(out / "eval_args.json", "w"), indent=2)
    logger.info("\n" + summary.round(3).to_string())


if __name__ == "__main__":
    main()
