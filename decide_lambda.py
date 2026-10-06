"""
decide_lambda.py
================
Apply the pre-registered λ selection criterion (docs/env_redesign_spec.md
§10) to evaluations of the λ study, mechanically.

Input: one or more per_run.csv files from evaluate_feeder.py (one per
training seed), each containing NoV2G and SAC-GNN-lc<λ> agents evaluated on
identical paired (date, rep) episodes.

Criterion (representative days only):
  d = limit_viol_kwh(λ-agent) − limit_viol_kwh(NoV2G) per paired episode.
  Mean d = mean over seeds of each seed's mean d; 95% CI from a hierarchical
  bootstrap (resample seeds, then paired episodes within each seed; 10,000
  resamples). "No worse than NoV2G" ⇔ CI upper bound ≤ 0.5 kWh/day.
  Choose the smallest passing λ; if none passes, the λ with the smallest
  mean d (and report that compliance is not matched). Pre-penalty profit
  (arbitrage_profit) differences are reported with the same CI but do not
  decide λ.

Usage:
  python decide_lambda.py <per_run.csv> [<per_run.csv> ...] [--out decision.json]
"""

import argparse
import json
import re

import numpy as np
import pandas as pd

MARGIN = 0.5          # kWh/day, pre-registered
N_BOOT = 10_000


def paired_diffs(df: pd.DataFrame, agent: str, metric: str) -> np.ndarray:
    base = df[df.agent == "NoV2G"].set_index(["date", "rep"])[metric]
    x = df[df.agent == agent].set_index(["date", "rep"])[metric]
    common = base.index.intersection(x.index)
    if len(common) != len(base) or len(common) != len(x):
        raise ValueError(f"{agent}: episodes not fully paired with NoV2G ({len(x)} vs {len(base)})")
    return (x.loc[common] - base.loc[common]).to_numpy()


def hierarchical_ci(per_seed: list, rng) -> tuple:
    """Mean over seeds of per-seed mean, with seeds-then-episodes bootstrap CI."""
    point = float(np.mean([d.mean() for d in per_seed]))
    S = len(per_seed)
    boots = np.empty(N_BOOT)
    for b in range(N_BOOT):
        seeds = rng.integers(S, size=S)
        boots[b] = np.mean([per_seed[s][rng.integers(len(per_seed[s]), size=len(per_seed[s]))].mean()
                            for s in seeds])
    return point, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("per_run", nargs="+", help="per_run.csv, one per training seed")
    p.add_argument("--out", default=None, help="write the decision as JSON")
    p.add_argument("--seed", type=int, default=0, help="bootstrap RNG seed")
    args = p.parse_args()

    runs = [pd.read_csv(f) for f in args.per_run]
    runs = [r[r["set"] == "representative"] if "set" in r.columns else r for r in runs]
    lambdas = sorted({float(m.group(1)) for r in runs for a in r.agent.unique()
                      if (m := re.fullmatch(r"SAC-GNN-lc([0-9.]+)", a))})
    if not lambdas:
        raise SystemExit("no SAC-GNN-lc<λ> agents found")

    rng = np.random.default_rng(args.seed)
    rows = []
    for lam in lambdas:
        name = f"SAC-GNN-lc{lam:g}"
        viol = [paired_diffs(r, name, "limit_viol_kwh") for r in runs if name in set(r.agent)]
        prof = [paired_diffs(r, name, "arbitrage_profit") for r in runs if name in set(r.agent)]
        v, vlo, vhi = hierarchical_ci(viol, rng)
        pr, plo, phi = hierarchical_ci(prof, rng)
        rows.append({"lambda": lam, "seeds": len(viol), "episodes_per_seed": [len(d) for d in viol],
                     "viol_diff": v, "viol_ci": [vlo, vhi], "passes": bool(vhi <= MARGIN),
                     "profit_diff": pr, "profit_ci": [plo, phi]})

    passing = [r for r in rows if r["passes"]]
    if passing:
        chosen, note = min(passing, key=lambda r: r["lambda"]), "smallest λ meeting the criterion"
    else:
        chosen, note = min(rows, key=lambda r: r["viol_diff"]), "no λ meets the criterion: smallest mean d (compliance NOT matched)"

    print(f"Pre-registered criterion: CI upper bound of paired violation difference vs NoV2G ≤ {MARGIN} kWh/day")
    for r in rows:
        print(f"  λ={r['lambda']:<5g} seeds={r['seeds']}  violations {r['viol_diff']:+.2f} "
              f"[{r['viol_ci'][0]:+.2f}, {r['viol_ci'][1]:+.2f}] kWh/day  {'PASS' if r['passes'] else 'fail'}   "
              f"profit {r['profit_diff']:+.2f} [{r['profit_ci'][0]:+.2f}, {r['profit_ci'][1]:+.2f}] $/day")
    print(f"Decision: λ = {chosen['lambda']:g} ({note})")
    if args.out:
        json.dump({"criterion_margin_kwh": MARGIN, "n_boot": N_BOOT, "inputs": args.per_run,
                   "results": rows, "chosen_lambda": chosen["lambda"], "note": note}, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
