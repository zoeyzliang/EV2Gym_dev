"""
coordination_study.py
=====================
Value of coordination (spec §6, C2): perfect-foresight LP with per-hub DOEs
vs with joint feeder constraints, on the pre-registered evaluation days and
paired repetitions. See baselines/coordination.py.

Usage:
  python coordination_study.py --pv_penetration 0.9 --participant_billing --results_dir results/coord_pv0.9
"""

import argparse
import json
import logging
import time
from pathlib import Path

import pandas as pd

from baselines.coordination import coordination_value, shared_limit_value
from evaluate_feeder import episode_seed, select_eval_days
from train_sac_gnn import DEFAULT_CONFIG, make_env

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pv_penetration", type=float, default=DEFAULT_CONFIG["pv_penetration"])
    p.add_argument("--forecast_sigma", type=float, default=DEFAULT_CONFIG["forecast_sigma"])
    p.add_argument("--participant_billing", action="store_true")
    p.add_argument("--mode", default="network", choices=["network", "shared"],
                   help="network: per-hub DOEs vs joint feeder constraints; shared: S2a shared group limit")
    p.add_argument("--shared_frac", type=float, default=0.3, help="S2a group limit as a share of group capacity")
    p.add_argument("--thermal_margin", type=float, default=1.0, help="S2b section rating margin")
    p.add_argument("--feeder_network", default="node_34", help="node_34 (default) or node_cre21 (CRE21 study)")
    p.add_argument("--feeder_v_base", type=float, default=11.0, help="kV of the network files (CRE21: 22)")
    p.add_argument("--kappa_load", type=float, default=None, help="background load scale (CRE21: 1.0, no retuning)")
    p.add_argument("--siting", default="base", choices=["base", "constrained"], help="CRE21 hub siting")
    p.add_argument("--n_reps", type=int, default=3)
    p.add_argument("--max_days", type=int, default=None)
    p.add_argument("--results_dir", required=True)
    args = p.parse_args()

    out = Path(args.results_dir); out.mkdir(parents=True, exist_ok=True)
    cfg = dict(DEFAULT_CONFIG)
    cfg.update(env="feeder", doe_mode="per_hub", pv_penetration=args.pv_penetration,
               forecast_sigma=args.forecast_sigma, participant_billing=args.participant_billing,
               thermal_margin=args.thermal_margin, feeder_network=args.feeder_network, feeder_v_base=args.feeder_v_base, siting=args.siting,
               **({"kappa_load": args.kappa_load} if args.kappa_load is not None else {}))
    env, _, _ = make_env(cfg, split="eval", seed=0)
    days = select_eval_days(env.price_loader._price_df)
    if args.max_days:
        days = days.iloc[:args.max_days]

    rows, t0 = [], time.time()
    for di, (date, meta) in enumerate(days.iterrows()):
        for k in range(args.n_reps):
            seed = episode_seed(date, k)
            r = (coordination_value(env, date, seed) if args.mode == "network"
                 else shared_limit_value(env, date, seed, args.shared_frac))
            rows.append({"date": date, "rep": k, "seed": seed, "set": meta["set"], "tier": meta["tier"],
                         "pv_penetration": args.pv_penetration, "mode": args.mode,
                         "thermal_margin": args.thermal_margin, **r})
        logger.info(f"day {di + 1}/{len(days)} {date} ({meta['set']}) done, {time.time() - t0:.0f}s; "
                    f"value of coordination rep0 {rows[-args.n_reps]['value_of_coordination']:.3f}")
        pd.DataFrame(rows).to_csv(out / "per_run.csv", index=False)
    json.dump(vars(args), open(out / "args.json", "w"), indent=2)
    df = pd.DataFrame(rows)
    logger.info("\n" + df.groupby("set")[["value_of_coordination"]].describe().round(3).T.to_string())


if __name__ == "__main__":
    main()
