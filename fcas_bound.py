"""
fcas_bound.py
=============
Indicative FCAS availability bound (spec §6, S4). Runs MPC-Predispatch on the
pre-registered evaluation days and paired reps and, every 5-min step, records
the participants' headroom around their set-point:
  raise = Σ_h max(0, q_h − max(flex_lo_h, −lim_exp_h))
  lower = Σ_h max(0, min(flex_hi_h, lim_imp_h − arrivals_h) − q_h)
valued at the 2024 VIC1 FCAS enablement prices ($/MW/h):
  low  = raise·max(raise prices) + lower·max(lower prices)   (one service each way)
  high = raise·Σ raise prices    + lower·Σ lower prices       (all services co-enabled)
Missing 1-second prices (files before Aug 2024) count as 0. Ignores the 1 MW
minimum aggregation, telemetry costs and FCAS trapezium limits.

Usage:
  python fcas_bound.py --mpc_incentive 0.2 --participant_billing --out results/new/fcas_bound
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from baselines.forecast_mpc import ForecastMPC
from evaluate_feeder import episode_seed, select_eval_days
from nem_env.fcas_prices import SERVICES, fetch_fcas
from nem_env.predispatch import Predispatch
from train_sac_gnn import DEFAULT_CONFIG, make_env

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

RAISE = [s for s in SERVICES if s.startswith("RAISE")]
LOWER = [s for s in SERVICES if s.startswith("LOWER")]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mpc_incentive", type=float, default=0.2)
    p.add_argument("--participant_billing", action="store_true")
    p.add_argument("--n_reps", type=int, default=3)
    p.add_argument("--max_days", type=int, default=None)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    cfg = dict(DEFAULT_CONFIG)
    cfg.update(env="feeder", doe_mode="per_hub", participant_billing=args.participant_billing)
    env, _, _ = make_env(cfg, split="eval", seed=0)
    fcas = fetch_fcas(cfg["region"], 2024, cfg["cache_dir"]).fillna(0.0)
    pdx = Predispatch.load_cache(f"{cfg['cache_dir']}/{cfg['region']}_predispatch_2024.parquet")
    mpc = ForecastMPC(env, args.mpc_incentive, pdx)
    days = select_eval_days(env.price_loader._price_df)
    if args.max_days:
        days = days.iloc[:args.max_days]

    rows = []
    for date, meta in days.iterrows():
        for k in range(args.n_reps):
            obs, info = env.reset(seed=episode_seed(date, k), options={"date": date})
            assert info["date"] == date
            idx = env._idx
            pr = fcas.reindex(idx)
            if pr.isna().any().any():
                raise RuntimeError(f"FCAS prices missing for {date}")
            raise_mw, lower_mw, arb, done = [], [], 0.0, False
            while not done:
                obs, r, done, _, inf = env.step(mpc.select_action(obs))
                q = np.asarray(inf["projected_kw"]); lo = np.asarray(inf["flex_lo_kw"]); hi = np.asarray(inf["flex_hi_kw"])
                le = np.asarray(inf["lim_exp_kw"]); li = np.asarray(inf["lim_imp_kw"]); af = np.asarray(inf["arr_fc_kw"])
                raise_mw.append(np.maximum(0.0, q - np.maximum(lo, -le)).sum() / 1000.0)
                lower_mw.append(np.maximum(0.0, np.minimum(hi, li - af) - q).sum() / 1000.0)
                arb += inf["arbitrage_profit"] - inf["p_terminal"] - inf["p_unmet"]
            R, L = np.array(raise_mw), np.array(lower_mw)
            dt = env.DT_HR
            low = float(np.sum(R * pr[RAISE].max(axis=1).to_numpy() + L * pr[LOWER].max(axis=1).to_numpy()) * dt)
            high = float(np.sum(R * pr[RAISE].sum(axis=1).to_numpy() + L * pr[LOWER].sum(axis=1).to_numpy()) * dt)
            rows.append({"date": date, "rep": k, "set": meta["set"], "tier": meta["tier"],
                         "raise_mw_mean": float(R.mean()), "lower_mw_mean": float(L.mean()),
                         "raise_mw_max": float(R.max()), "fcas_low": low, "fcas_high": high,
                         "mpc_econ_minus_unmet": arb})
        logger.info(f"{date} ({meta['set']}): raise {rows[-1]['raise_mw_mean']*1000:.0f} kW mean, "
                    f"FCAS {rows[-1]['fcas_low']:.2f}–{rows[-1]['fcas_high']:.2f} $/day, arbitrage {arb:.2f}")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows); df.to_csv(out / "per_run.csv", index=False)
    json.dump(vars(args), open(out / "args.json", "w"), indent=2)
    logger.info("\n" + df.groupby("set")[["raise_mw_mean", "lower_mw_mean", "fcas_low", "fcas_high",
                                          "mpc_econ_minus_unmet"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
