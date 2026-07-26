"""
evaluate.py
===========
Post-training evaluation script for the SAC-GNN V2G Hub thesis.

Loads trained agent checkpoints and evaluates all agents across the
five thesis case studies, producing Table 3 and associated figures.

This is the FINAL evaluation script — run after all training is complete.
It is separate from the evaluate() function inside train_sac_gnn.py,
which is a lightweight checkpoint-comparison function used during training.

Usage
-----
    # Evaluate all available agents on all case studies
    python evaluate.py

    # Evaluate specific checkpoint only
    python evaluate.py --sac_gnn_checkpoint results/sac_gnn_real_seed42/checkpoints/best.pt

    # Quick run with fewer repetitions
    python evaluate.py --n_runs 5

    # Use synthetic prices (for offline testing without NEMOSIS)
    python evaluate.py --synthetic

Outputs
-------
results/evaluation/
    metrics_table.csv       ← Table 3: all agents × all case studies × all metrics
    metrics_summary.txt     ← Formatted text table for thesis
    per_day_results.csv     ← Per-day breakdown for case study analysis
    metrics_comparison.png  ← Bar charts: profit, DOE compliance, participation

Case studies (Terrence's requirement)
--------------------------------------
5 specific real 2024 VIC1 dates representing distinct NEM conditions.
Each case study is run n_runs times (stochastic participation) and
mean ± std is reported — this is what goes in Table 3.

    Case Study 1 — Summer peak (Jan 2024 heatwave)
        Tests: tight DOE constraint + afternoon RRP spike
        Expected winner: SAC-GNN (learns spatial DOE correlation)

    Case Study 2 — High volatility (Mar 2024)
        Tests: precise arbitrage timing under large intraday swings
        Expected winner: SAC-GNN (learns spike timing vs greedy)

    Case Study 3 — Negative RRP (Oct 2024)
        Tests: charge direction of arbitrage (get paid to consume)
        Expected winner: SAC-GNN (learned bidirectional dispatch)

    Case Study 4 — Winter average (Jun 2024)
        Tests: baseline stable-price performance
        Expected: all agents competitive

    Case Study 5 — Weekend low demand (Aug 2024)
        Tests: incentive price adaptation under thin participation
        Expected winner: SAC-GNN (learns ρ elasticity)

Ablation interpretation
------------------------
    SAC-GNN vs SAC-GCN  → GAT attention contribution (RQ4)
    SAC-GNN vs SAC-Flat → Graph structure contribution (RQ3)
    SAC-GNN vs Greedy   → RL vs heuristic (RQ1)
    SAC-GNN vs Oracle   → Gap from oracle participation knowledge (RQ5)
"""

import os
import json
import argparse
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

os.environ.pop("FORCE_NUMPY_AGENT", None)

from nem_env.spatial_graph import HubGraphBuilder
from nem_env.aemo_price_loader import PriceLoader
from nem_env.participation_model import ParticipationModel
from nem_env.nem_doe_env import NEMDOEEnv, EnvConfig
from baselines.gnn_rl.agent import SACGNNAgent
from baselines.gnn_rl.networks import NetworkConfig
from baselines.gnn_rl.sac_gcn import SACGCNAgent
from baselines.flat_mlp.sac_flat import SACFlatAgent
from baselines.heuristics.greedy_dispatch import GreedyDispatchBaseline
from baselines.heuristics.rule_based_pricing import RuleBasedPricingBaseline
from baselines.mpc.oracle_mpc import OracleMPCBaseline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Case study definitions
# ---------------------------------------------------------------------------

CASE_STUDIES = [
    {
        "id":          "cs1_summer_peak",
        "name":        "Summer Peak",
        "date":        "2024-01-25",
        "description": "Jan 2024 heatwave — tight DOE constraint, afternoon RRP spike",
        "tests":       "Tight DOE + spike timing",
    },
    {
        "id":          "cs2_high_volatility",
        "name":        "High Volatility",
        "date":        "2024-03-12",
        "description": "Large intraday RRP swings",
        "tests":       "Precise arbitrage timing",
    },
    {
        "id":          "cs3_negative_rrp",
        "name":        "Negative RRP",
        "date":        "2024-10-03",
        "description": "Wind surplus pushes RRP negative — charge direction test",
        "tests":       "Bidirectional dispatch",
    },
    {
        "id":          "cs4_winter_average",
        "name":        "Winter Average",
        "date":        "2024-06-15",
        "description": "Moderate stable prices — baseline day",
        "tests":       "Baseline performance",
    },
    {
        "id":          "cs5_weekend_low",
        "name":        "Weekend Low Demand",
        "date":        "2024-08-20",
        "description": "Low demand weekend — thin participation pool",
        "tests":       "Incentive price adaptation",
    },
]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate all agents on thesis case studies."
    )
    parser.add_argument(
        "--sac_gnn_checkpoint", type=str,
        default="results/sac_gnn_real_seed42/checkpoints/best.pt",
        help="SAC-GNN best checkpoint path",
    )
    parser.add_argument(
        "--sac_gcn_checkpoint", type=str,
        default="results/sac_gcn_real_seed42/checkpoints/best.pt",
        help="SAC-GCN best checkpoint path",
    )
    parser.add_argument(
        "--sac_flat_checkpoint", type=str,
        default="results/sac_flat_real_seed42/checkpoints/best.pt",
        help="SAC-Flat best checkpoint path",
    )
    parser.add_argument(
        "--n_runs", type=int, default=10,
        help="Repetitions per case study (stochastic participation)",
    )
    parser.add_argument(
        "--results_dir", type=str, default="results/evaluation",
        help="Output directory for evaluation results",
    )
    parser.add_argument(
        "--seed", type=int, default=99,
        help="Evaluation RNG seed",
    )
    parser.add_argument(
        "--synthetic", action="store_true",
        help="Use synthetic prices (offline testing — not valid for thesis)",
    )
    parser.add_argument(
        "--cache_dir", type=str, default="data/nem_cache",
    )
    parser.add_argument(
        "--graph_path", type=str, default="data/graphs/inner_melbourne.pkl",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def make_eval_env(args, seed: int) -> tuple:
    """Build NEMDOEEnv loaded with 2024 held-out evaluation prices."""
    graph, hub_configs = HubGraphBuilder.load(args.graph_path)

    loader = PriceLoader(
        region="VIC1",
        cache_dir=args.cache_dir,
        seed=seed,
    )

    if args.synthetic:
        logger.warning("Using synthetic prices — results NOT valid for thesis!")
        loader.load_synthetic(n_days=365, mean_price=100.0, std_price=250.0,
                              spike_prob=0.003, spike_magnitude=2000.0)
    else:
        parquet = f"{args.cache_dir}/VIC1_2024-01-01_2024-12-31.parquet"
        if Path(parquet).exists():
            loader.load_cache(parquet)
        else:
            logger.info("Eval parquet not found — fetching 2024 prices...")
            loader.fetch_and_cache(start="2024-01-01", end="2024-12-31")

    model = ParticipationModel(seed=seed)

    env = NEMDOEEnv(
        hub_configs=hub_configs,
        price_loader=loader,
        participation_model=model,
        env_config=EnvConfig(),
        seed=seed,
    )

    return env, graph, hub_configs


# ---------------------------------------------------------------------------
# Agent loader
# ---------------------------------------------------------------------------

def load_agents(args, env, graph, hub_configs) -> list:
    """
    Load all available agents. Skips agents whose checkpoint doesn't exist
    with a warning rather than crashing — allows partial evaluation if only
    some agents have been trained.

    Returns list of (name, agent, needs_env) tuples.
    needs_env=True means select_action(obs, env) signature.
    needs_env=False means select_action(obs, deterministic=True) signature.
    """
    obs_dim    = env.observation_space.shape[0]
    n_hubs     = len(hub_configs)
    action_dim = n_hubs + 1
    net_cfg    = NetworkConfig()
    agents     = []

    # 1. SAC-GNN (proposed)
    if Path(args.sac_gnn_checkpoint).exists():
        agent = SACGNNAgent(
            n_hubs=n_hubs, graph_data=graph,
            obs_dim=obs_dim, net_cfg=net_cfg, seed=args.seed,
        )
        agent.load(args.sac_gnn_checkpoint)
        agents.append(("SAC-GNN", agent, False))
        logger.info(f"✓ SAC-GNN loaded from {args.sac_gnn_checkpoint}")
    else:
        logger.warning(f"SAC-GNN checkpoint not found: {args.sac_gnn_checkpoint}")

    # 2. SAC-GCN ablation
    if Path(args.sac_gcn_checkpoint).exists():
        agent = SACGCNAgent(
            n_hubs=n_hubs, graph_data=graph,
            obs_dim=obs_dim, net_cfg=net_cfg, seed=args.seed,
        )
        agent.load(args.sac_gcn_checkpoint)
        agents.append(("SAC-GCN", agent, False))
        logger.info(f"✓ SAC-GCN loaded from {args.sac_gcn_checkpoint}")
    else:
        logger.warning(f"SAC-GCN checkpoint not found: {args.sac_gcn_checkpoint}")

    # 3. SAC-Flat ablation
    if Path(args.sac_flat_checkpoint).exists():
        agent = SACFlatAgent(obs_dim=obs_dim, action_dim=action_dim, seed=args.seed)
        agent.load(args.sac_flat_checkpoint)
        agents.append(("SAC-Flat", agent, False))
        logger.info(f"✓ SAC-Flat loaded from {args.sac_flat_checkpoint}")
    else:
        logger.warning(f"SAC-Flat checkpoint not found: {args.sac_flat_checkpoint}")

    # 4. Greedy dispatch heuristic (no checkpoint)
    greedy = GreedyDispatchBaseline(n_hubs=n_hubs)
    agents.append(("Greedy", greedy, True))
    logger.info("✓ GreedyDispatch ready")

    # 5. Rule-based pricing heuristic (no checkpoint)
    rule = RuleBasedPricingBaseline(n_hubs=n_hubs)
    agents.append(("RulePrice", rule, True))
    logger.info("✓ RuleBasedPricing ready")

    # 6. Oracle MPC upper bound (no checkpoint)
    hub_distances = [hc.distance_km for hc in hub_configs]
    oracle_model  = ParticipationModel(seed=args.seed)
    oracle = OracleMPCBaseline(
        n_hubs=n_hubs,
        participation_model=oracle_model,
        hub_distances=hub_distances,
    )
    agents.append(("OracleMPC", oracle, True))
    logger.info("✓ OracleMPC ready")

    return agents


# ---------------------------------------------------------------------------
# Single case study evaluation
# ---------------------------------------------------------------------------

def evaluate_case_study(
    agent_name: str,
    agent,
    needs_env: bool,
    env: NEMDOEEnv,
    case_study: dict,
    n_runs: int,
) -> dict:
    """
    Evaluate one agent on one case study across n_runs repetitions.

    Each run resets with the same fixed date but different stochastic
    participation draws — captures natural variability in EV owner responses.

    Returns dict with mean ± std across n_runs for all metrics.
    """
    profits          = []
    doe_violations   = []
    participation    = []
    doe_compliant_steps = []
    total_steps_list = []

    date = case_study["date"]

    for run in range(n_runs):
        obs, _ = env.reset(options={"date": date})
        done        = False
        ep_profit   = 0.0
        ep_doe_viol = 0.0
        ep_rho      = 0.0
        n_steps     = 0
        n_zero_doe  = 0

        while not done:
            if needs_env:
                action = agent.select_action(obs, env)
            else:
                action = agent.select_action(obs, deterministic=True)

            obs, reward, done, _, info = env.step(action)
            ep_profit   += reward
            ep_doe_viol += sum(info.get("doe_violations_kw", [0.0]))
            ep_rho      += info.get("rho_hat", 0.0)
            n_steps     += 1
            if sum(info.get("doe_violations_kw", [0.0])) == 0:
                n_zero_doe += 1

        profits.append(ep_profit)
        doe_violations.append(ep_doe_viol)
        participation.append(ep_rho / n_steps if n_steps > 0 else 0.0)
        doe_compliant_steps.append(n_zero_doe / n_steps if n_steps > 0 else 0.0)
        total_steps_list.append(n_steps)

    return {
        "agent":              agent_name,
        "case_study_id":      case_study["id"],
        "case_study_name":    case_study["name"],
        "date":               date,
        "n_runs":             n_runs,
        # Primary metric
        "mean_profit":        float(np.mean(profits)),
        "std_profit":         float(np.std(profits)),
        # DOE compliance
        "mean_doe_viol_kw":   float(np.mean(doe_violations)),
        "std_doe_viol_kw":    float(np.std(doe_violations)),
        "mean_doe_compliance": float(np.mean(doe_compliant_steps)),
        # Participation
        "mean_participation": float(np.mean(participation)),
        "std_participation":  float(np.std(participation)),
        # Raw runs for distribution plots
        "profits_per_run":    profits,
    }


# ---------------------------------------------------------------------------
# Results formatting
# ---------------------------------------------------------------------------

def print_results_table(results_df: pd.DataFrame) -> str:
    """
    Print and return formatted thesis Table 3.

    Format: agents as rows, case studies as columns, showing mean±std profit.
    """
    agents      = results_df["agent"].unique()
    case_studies = results_df["case_study_name"].unique()

    lines = []
    lines.append("\n" + "=" * 100)
    lines.append("THESIS TABLE 3 — SAC-GNN vs Baselines: Net Profit ($/day), Mean ± Std")
    lines.append("=" * 100)

    # Header
    header = f"{'Agent':<14}" + "".join(f"{cs:>18}" for cs in case_studies)
    header += f"{'Mean':>12}"
    lines.append(header)
    lines.append("-" * 100)

    for agent in agents:
        agent_df   = results_df[results_df["agent"] == agent]
        row        = f"{agent:<14}"
        all_profits = []
        for cs_name in case_studies:
            cs_row = agent_df[agent_df["case_study_name"] == cs_name]
            if len(cs_row) > 0:
                mean = cs_row["mean_profit"].values[0]
                std  = cs_row["std_profit"].values[0]
                row += f"  {mean:+8.0f}±{std:5.0f}"
                all_profits.append(mean)
            else:
                row += f"{'N/A':>18}"
        if all_profits:
            row += f"  {np.mean(all_profits):+10.0f}"
        lines.append(row)

    lines.append("=" * 100)
    lines.append("DOE Compliance Rate (% steps with zero violation)")
    lines.append("-" * 100)

    for agent in agents:
        agent_df = results_df[results_df["agent"] == agent]
        row = f"{agent:<14}"
        for cs_name in case_studies:
            cs_row = agent_df[agent_df["case_study_name"] == cs_name]
            if len(cs_row) > 0:
                compliance = cs_row["mean_doe_compliance"].values[0] * 100
                row += f"  {compliance:>16.1f}%"
            else:
                row += f"{'N/A':>18}"
        lines.append(row)

    lines.append("=" * 100)
    lines.append("Mean Participation Rate ρ")
    lines.append("-" * 100)

    for agent in agents:
        agent_df = results_df[results_df["agent"] == agent]
        row = f"{agent:<14}"
        for cs_name in case_studies:
            cs_row = agent_df[agent_df["case_study_name"] == cs_name]
            if len(cs_row) > 0:
                rho = cs_row["mean_participation"].values[0] * 100
                row += f"  {rho:>16.1f}%"
            else:
                row += f"{'N/A':>18}"
        lines.append(row)

    lines.append("=" * 100)
    table_str = "\n".join(lines)
    print(table_str)
    return table_str


def make_plots(results_df: pd.DataFrame, output_dir: Path) -> None:
    """Generate comparison bar charts for the three primary metrics."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available — skipping plots")
        return

    agents       = list(results_df["agent"].unique())
    case_studies = list(results_df["case_study_name"].unique())
    n_agents     = len(agents)
    n_cs         = len(case_studies)

    colors = ["#2a78d6", "#1baf7a", "#eda100", "#e34948", "#9b59b6", "#00bcd4"]
    x      = np.arange(n_cs)
    width  = 0.8 / n_agents

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        "SAC-GNN V2G Hub — Case Study Evaluation\n"
        "Inner Melbourne 21 Hubs, VIC1 2024 Held-Out Prices",
        fontsize=12,
    )

    for col, (metric, ylabel, title) in enumerate([
        ("mean_profit",       "Net profit ($/day)", "Arbitrage Net Profit"),
        ("mean_doe_compliance","DOE compliance rate", "DOE Compliance Rate"),
        ("mean_participation", "Participation rate ρ", "EV Owner Participation"),
    ]):
        ax = axes[col]
        for j, agent in enumerate(agents):
            agent_df = results_df[results_df["agent"] == agent]
            vals, errs = [], []
            for cs_name in case_studies:
                row = agent_df[agent_df["case_study_name"] == cs_name]
                if len(row) > 0:
                    vals.append(row[metric].values[0])
                    err_col = metric.replace("mean_", "std_")
                    errs.append(row[err_col].values[0] if err_col in row.columns else 0)
                else:
                    vals.append(0)
                    errs.append(0)

            offset = (j - n_agents / 2 + 0.5) * width
            bars = ax.bar(
                x + offset, vals, width,
                yerr=errs, capsize=3,
                label=agent,
                color=colors[j % len(colors)],
                edgecolor="white", linewidth=0.5,
            )

        ax.set_title(title, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [cs.replace(" ", "\n") for cs in case_studies],
            fontsize=8,
        )
        ax.grid(axis="y", alpha=0.3)
        if col == 0:
            ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
        if col == 2:
            ax.legend(fontsize=8, loc="upper right")

    plt.tight_layout()
    out_path = output_dir / "metrics_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    logger.info(f"Plot saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args       = parse_args()
    output_dir = Path(args.results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("SAC-GNN V2G Hub — Case Study Evaluation")
    logger.info(f"Results → {output_dir}")
    logger.info(f"Case studies: {len(CASE_STUDIES)}")
    logger.info(f"Runs per case: {args.n_runs}")
    logger.info("=" * 60)

    # Build evaluation environment
    logger.info("Building evaluation environment (2024 held-out prices)...")
    env, graph, hub_configs = make_eval_env(args, seed=args.seed)

    # Load all agents
    logger.info("Loading agents...")
    agents = load_agents(args, env, graph, hub_configs)
    logger.info(f"Loaded {len(agents)} agents: {[a[0] for a in agents]}")

    if not agents:
        logger.error("No agents available — check checkpoint paths")
        return

    # Run all case studies × all agents
    all_results = []
    total = len(agents) * len(CASE_STUDIES)
    done_count = 0

    for agent_name, agent, needs_env in agents:
        logger.info(f"\n── Evaluating {agent_name} ──────────────────────────")
        if hasattr(agent, "reset"):
            agent.reset()

        for cs in CASE_STUDIES:
            done_count += 1
            logger.info(
                f"  [{done_count}/{total}] {cs['name']} ({cs['date']}) "
                f"× {args.n_runs} runs..."
            )

            result = evaluate_case_study(
                agent_name=agent_name,
                agent=agent,
                needs_env=needs_env,
                env=env,
                case_study=cs,
                n_runs=args.n_runs,
            )

            logger.info(
                f"    profit={result['mean_profit']:+8.1f} ± {result['std_profit']:5.1f} | "
                f"doe={result['mean_doe_compliance']*100:.1f}% | "
                f"ρ={result['mean_participation']*100:.1f}%"
            )
            all_results.append(result)

    # Build results DataFrame (exclude per-run list for CSV)
    csv_results = []
    for r in all_results:
        row = {k: v for k, v in r.items() if k != "profits_per_run"}
        csv_results.append(row)

    results_df = pd.DataFrame(csv_results)

    # Save CSV
    csv_path = output_dir / "metrics_table.csv"
    results_df.to_csv(csv_path, index=False)
    logger.info(f"\nMetrics saved: {csv_path}")

    # Print and save formatted table
    table_str = print_results_table(results_df)
    txt_path = output_dir / "metrics_summary.txt"
    with open(txt_path, "w") as f:
        f.write(table_str)
        f.write(f"\n\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"Agents: {[a[0] for a in agents]}\n")
        f.write(f"Runs per case study: {args.n_runs}\n")
    logger.info(f"Summary saved: {txt_path}")

    # Per-day raw results
    per_day_rows = []
    for r in all_results:
        for run_i, profit in enumerate(r["profits_per_run"]):
            per_day_rows.append({
                "agent":          r["agent"],
                "case_study_id":  r["case_study_id"],
                "date":           r["date"],
                "run":            run_i,
                "profit":         profit,
            })
    per_day_df = pd.DataFrame(per_day_rows)
    per_day_path = output_dir / "per_run_results.csv"
    per_day_df.to_csv(per_day_path, index=False)
    logger.info(f"Per-run results saved: {per_day_path}")

    # Plots
    make_plots(results_df, output_dir)

    logger.info("\n" + "=" * 60)
    logger.info("Evaluation complete.")
    logger.info(f"Results: {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
