"""
doe_stats.py
============
DOE statistics on the pre-registered evaluation days (spec §6, CRE21 study
(a)): how often each hub's import / export limit binds (below hub capacity),
the limit as a share of capacity, and the background-only voltage range.

Usage:
  python tools/doe_stats.py --feeder_network node_cre21 --feeder_v_base 22 --kappa_load 1.0 \\
      --siting base --out results/cre21_doe_stats_base.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # run as a script from the repo root

from evaluate_feeder import select_eval_days
from train_sac_gnn import DEFAULT_CONFIG, make_env


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--feeder_network", default="node_34")
    p.add_argument("--feeder_v_base", type=float, default=11.0)
    p.add_argument("--kappa_load", type=float, default=None)
    p.add_argument("--siting", default="base")
    p.add_argument("--pv_penetration", type=float, default=DEFAULT_CONFIG["pv_penetration"])
    p.add_argument("--out", required=True)
    a = p.parse_args()
    cfg = dict(DEFAULT_CONFIG)
    cfg.update(env="feeder", doe_mode="per_hub", feeder_network=a.feeder_network,
               feeder_v_base=a.feeder_v_base, siting=a.siting, pv_penetration=a.pv_penetration,
               **({"kappa_load": a.kappa_load} if a.kappa_load is not None else {}))
    env, _, _ = make_env(cfg, split="eval", seed=0)
    days = select_eval_days(env.price_loader._price_df)
    rows = []
    for date, meta in days.iterrows():
        env.reset(seed=0, options={"date": date})
        imp, exp, cap = env._lim_imp, env._lim_exp, env.cap
        V = env.feeder.voltages(env._P_fc, env._Q_fc)
        rows.append({"date": date, "set": meta["set"],
                     "imp_binding_pct": 100 * float(np.mean(imp < cap - 1e-6)),
                     "exp_binding_pct": 100 * float(np.mean(exp < cap - 1e-6)),
                     "imp_frac_mean": float(np.mean(imp / cap)), "imp_frac_min": float(np.min(imp / cap)),
                     "exp_frac_mean": float(np.mean(exp / cap)), "exp_frac_min": float(np.min(exp / cap)),
                     "v_min": float(V.min()), "v_max": float(V.max())})
    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)
    print(df.drop(columns=["date"]).groupby("set").mean().round(3).to_string())


if __name__ == "__main__":
    main()
