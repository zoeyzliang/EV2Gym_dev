"""
select_mpc_incentive.py
=======================
Choose the forecast MPC's constant incentive on the validation days only
(spec §5), never on the test days.

Grid: 0, 0.1, ..., 0.5 $/kWh. Each value is run on train_sac_gnn's
VALIDATION_DAYS with the same seeds as checkpoint selection
(VALIDATION_SEED + i). The chosen value maximises mean reward over the
normal validation days, excluding the stress-test day (the rule best.pt uses).
The evaluation environment is evaluate_feeder's (same settings as the test
evaluation).

Usage:
  python select_mpc_incentive.py [--doe_mode per_hub] [--action_scale capacity] [--out mpc_incentive.json]
"""

import argparse
import json
import logging

import numpy as np

from baselines.forecast_mpc import ForecastMPC
from evaluate_feeder import run_episode
from nem_env.predispatch import Predispatch
from train_sac_gnn import (DEFAULT_CONFIG, STRESS_TEST_DAY_INDEX, VALIDATION_DAYS,
                           VALIDATION_SEED, make_env)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

GRID = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--doe_mode", default="per_hub", choices=["per_hub", "network"])
    p.add_argument("--action_scale", default="capacity", choices=["capacity", "feasible"])
    p.add_argument("--participant_billing", action="store_true")
    p.add_argument("--import_tariff", type=float, default=0.0)
    p.add_argument("--tariff_passthrough", action="store_true")
    p.add_argument("--feeder_network", default="node_34", help="node_34 (default) or node_cre21 (CRE21 study)")
    p.add_argument("--feeder_v_base", type=float, default=11.0, help="kV of the network files (CRE21: 22)")
    p.add_argument("--kappa_load", type=float, default=None, help="background load scale (CRE21: 1.0, no retuning)")
    p.add_argument("--siting", default="base", choices=["base", "constrained"], help="CRE21 hub siting")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    cfg = dict(DEFAULT_CONFIG)
    cfg.update(env="feeder", doe_mode=args.doe_mode, action_scale=args.action_scale,
               participant_billing=args.participant_billing, import_tariff=args.import_tariff,
               tariff_passthrough=args.tariff_passthrough,
               feeder_network=args.feeder_network, feeder_v_base=args.feeder_v_base, siting=args.siting,
               **({"kappa_load": args.kappa_load} if args.kappa_load is not None else {}))
    env, _, _ = make_env(cfg, split="eval", seed=0)
    pdx = Predispatch.load_cache(f"{cfg['cache_dir']}/{cfg['region']}_predispatch_2024.parquet")

    rows = {}
    for c in GRID:
        mpc = ForecastMPC(env, c, pdx)
        per_day = [run_episode(env, mpc, d, VALIDATION_SEED + i)["reward"] for i, d in enumerate(VALIDATION_DAYS)]
        normal = [r for i, r in enumerate(per_day) if i != STRESS_TEST_DAY_INDEX]
        rows[c] = {"normal_mean": float(np.mean(normal)), "per_day": per_day,
                   "solve_failures": mpc.solve_failures}
        logger.info(f"incentive {c:.1f}: normal-day mean reward {np.mean(normal):.2f}  per day {np.round(per_day, 1)}")
    best = max(rows, key=lambda c: rows[c]["normal_mean"])
    logger.info(f"Chosen MPC incentive: {best} $/kWh")
    if args.out:
        json.dump({"grid": list(GRID), "validation_days": VALIDATION_DAYS,
                   "stress_day_excluded": VALIDATION_DAYS[STRESS_TEST_DAY_INDEX],
                   "results": {str(k): v for k, v in rows.items()}, "chosen": best,
                   "doe_mode": args.doe_mode, "participant_billing": args.participant_billing,
                   "import_tariff": args.import_tariff, "tariff_passthrough": args.tariff_passthrough,
                   "feeder_network": args.feeder_network, "siting": args.siting}, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
