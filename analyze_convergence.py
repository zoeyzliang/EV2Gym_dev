"""
analyze_convergence.py
========================
Post-hoc, outlier-robust final-performance reporting for a completed
training run's eval_log.csv.

Motivation
----------
A naive "mean ± std over the last N checkpoints" statistic is itself
vulnerable to distortion by a single anomalous checkpoint — e.g. one
unlucky stochastic participation draw at one eval point inflating the
apparent std/CV to a level that looks like genuine training instability,
when the underlying policy is actually well-converged (see session
analysis: sac_gcn_real_seed1 showed CV=44.4% using a raw last-5-mean
statistic, driven entirely by a single episode-1450 outlier checkpoint;
excluding it, the same window's CV drops to ~3%, consistent with the
genuinely converged neighbouring checkpoints).

This script computes BOTH the naive statistic and an outlier-robust one
(median-absolute-deviation-based outlier detection, applied over the
last N checkpoints), so a single bad draw doesn't silently produce a
misleading "unstable" verdict — and so a genuinely unstable run (e.g.
multiple large swings, not just one isolated point) is still correctly
flagged as such, not laundered away.

Usage
-----
    # Single run
    python analyze_convergence.py --eval_log path/to/eval_log.csv

    # All runs in a directory of pulled results (looks for */logs/eval_log.csv)
    python analyze_convergence.py --results_dir pulled_results/ --last_n 5

Output
------
Prints one row per run: raw last-N mean/std/CV, outlier-robust
mean/std/CV, any detected outlier episodes, and a plain-language
verdict (converged / converged-with-outlier / needs-investigation).
"""

import argparse
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd


def detect_outliers_mad(values: np.ndarray, threshold: float = 3.5) -> np.ndarray:
    """
    Modified z-score outlier detection using median absolute deviation
    (MAD), following Iglewicz & Hoaglin's standard formulation.

    Robust to the exact failure mode this script exists to catch: a
    single extreme point does not itself distort the median or MAD the
    way it would distort a mean/std-based z-score, so this correctly
    flags one bad checkpoint as an outlier rather than having that same
    checkpoint inflate the "normal" spread used to judge it.

    Returns
    -------
    np.ndarray of bool, True where the corresponding value is an outlier.
    """
    if len(values) < 3:
        return np.zeros(len(values), dtype=bool)
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    if mad < 1e-9:
        # All values identical (or MAD degenerate) — nothing is an outlier
        return np.zeros(len(values), dtype=bool)
    modified_z = 0.6745 * (values - median) / mad
    return np.abs(modified_z) > threshold


def analyze_run(eval_log_path: str, last_n: int = 5) -> dict:
    """
    Compute raw and outlier-robust final-performance statistics for one
    completed run's eval_log.csv.
    """
    df = pd.read_csv(eval_log_path)
    if "mean_net_profit_normal" not in df.columns:
        raise ValueError(f"{eval_log_path}: missing mean_net_profit_normal column")

    df = df.sort_values("eval_episode")
    last = df.tail(last_n)
    episodes = last["eval_episode"].values
    values = last["mean_net_profit_normal"].values

    if len(values) == 0:
        return {"error": "no eval checkpoints found"}

    # Raw statistic — naive, vulnerable to single-outlier distortion
    raw_mean = float(np.mean(values))
    raw_std = float(np.std(values))
    raw_cv = (raw_std / abs(raw_mean) * 100) if raw_mean != 0 else float("nan")

    # Outlier-robust statistic
    is_outlier = detect_outliers_mad(values)
    outlier_episodes = episodes[is_outlier].tolist()
    clean_values = values[~is_outlier]

    if len(clean_values) >= 2:
        robust_mean = float(np.mean(clean_values))
        robust_std = float(np.std(clean_values))
        robust_cv = (robust_std / abs(robust_mean) * 100) if robust_mean != 0 else float("nan")
    else:
        # Everything flagged as an outlier, or too few points left —
        # fall back to the raw statistic rather than reporting nothing
        robust_mean, robust_std, robust_cv = raw_mean, raw_std, raw_cv

    # Verdict — plain-language summary of what the numbers mean
    if len(outlier_episodes) == 0 and raw_cv < 5.0:
        verdict = "converged (stable, no outliers)"
    elif len(outlier_episodes) > 0 and robust_cv < 5.0:
        verdict = f"converged (outlier at ep {outlier_episodes}, excluded)"
    elif robust_cv < 10.0:
        verdict = "likely converged (moderate residual noise)"
    else:
        verdict = "NEEDS INVESTIGATION (genuine instability, not just an outlier)"

    return {
        "path": eval_log_path,
        "last_n_episodes": episodes.tolist(),
        "raw_mean": round(raw_mean, 1),
        "raw_std": round(raw_std, 1),
        "raw_cv_pct": round(raw_cv, 1),
        "outlier_episodes": outlier_episodes,
        "robust_mean": round(robust_mean, 1),
        "robust_std": round(robust_std, 1),
        "robust_cv_pct": round(robust_cv, 1),
        "verdict": verdict,
    }


def find_eval_logs(results_dir: str) -> list:
    """Find every */logs/eval_log.csv under results_dir."""
    pattern = os.path.join(results_dir, "*", "logs", "eval_log.csv")
    return sorted(glob.glob(pattern))


def print_summary_table(results: list) -> None:
    print()
    print("=" * 130)
    print(f"{'Run':<50} {'Raw mean':>10} {'Raw CV%':>8} {'Robust mean':>12} {'Robust CV%':>10} {'Outliers':>12}  Verdict")
    print("-" * 130)
    for r in results:
        if "error" in r:
            print(f"{Path(r.get('path', '?')).parent.parent.name:<50} ERROR: {r['error']}")
            continue
        name = Path(r["path"]).parent.parent.name
        outliers_str = str(r["outlier_episodes"]) if r["outlier_episodes"] else "-"
        print(
            f"{name:<50} "
            f"${r['raw_mean']:>9,.0f} "
            f"{r['raw_cv_pct']:>7.1f}% "
            f"${r['robust_mean']:>10,.0f} "
            f"{r['robust_cv_pct']:>9.1f}% "
            f"{outliers_str:>12}  {r['verdict']}"
        )
    print("=" * 130)
    print()
    print("Recommended reporting value: 'Robust mean' column (outlier-excluded),")
    print("not the single best.pt checkpoint value, for the final results table.")


def main():
    parser = argparse.ArgumentParser(
        description="Outlier-robust final-performance analysis for completed training runs."
    )
    parser.add_argument("--eval_log", type=str, default=None,
                         help="Path to a single eval_log.csv")
    parser.add_argument("--results_dir", type=str, default=None,
                         help="Directory containing */logs/eval_log.csv (e.g. pulled_results/)")
    parser.add_argument("--last_n", type=int, default=5,
                         help="Number of most recent eval checkpoints to analyze (default 5)")
    parser.add_argument("--outlier_threshold", type=float, default=3.5,
                         help="Modified z-score threshold for outlier detection (default 3.5, "
                              "the standard Iglewicz & Hoaglin recommendation)")
    args = parser.parse_args()

    if not args.eval_log and not args.results_dir:
        print("Error: provide either --eval_log or --results_dir")
        return

    paths = [args.eval_log] if args.eval_log else find_eval_logs(args.results_dir)
    if not paths:
        print(f"No eval_log.csv files found under {args.results_dir}")
        return

    results = []
    for p in paths:
        try:
            r = analyze_run(p, last_n=args.last_n)
            results.append(r)
        except Exception as e:
            results.append({"path": p, "error": str(e)})

    print_summary_table(results)


if __name__ == "__main__":
    main()
