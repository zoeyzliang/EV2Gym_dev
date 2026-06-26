"""
evaluate.py
===========
Loads trained policy checkpoints and evaluates all agents against
the four metrics from §4.3.3 of the thesis proposal.

Usage
-----
    # Evaluate SAC-GNN best checkpoint against all baselines
    python evaluate.py

    # Evaluate specific checkpoint
    python evaluate.py --checkpoint results/sac_gnn/checkpoints/best.pt

    # Evaluate on more episodes
    python evaluate.py --n_episodes 50

Outputs
-------
results/evaluation/
  metrics_table.csv        ← Table 3 equivalent: all agents × all metrics
  metrics_table.txt        ← Formatted text table for thesis
  convergence_plot.png     ← Training reward curves (RQ4)
  conformance_bar.png      ← WDR conformance rate by agent (RQ1, RQ3)
  profit_bar.png           ← Net profit by agent (RQ1, RQ2)
  participation_bar.png    ← Mean participation rate by agent (RQ2, RQ5)
"""

import os
import json
import argparse
import logging
import numpy as np
import pandas as pd
from pathlib import Path

os.environ.pop("FORCE_NUMPY_AGENT", None)

from nem_env.spatial_graph import HubGraphBuilder
from nem_env.aemo_price_loader import PriceLoader
from nem_env.participation_model import ParticipationModel
from nem_env.nem_wdr_env import NEMWDREnv, EnvConfig
from baselines.gnn_rl.agent import SACGNNAgent
from baselines.gnn_rl.networks import NetworkConfig
from baselines.gnn_rl.sac_gcn import SACGCNAgent
from baselines.flat_mlp.sac_flat import SACFlatAgent
from baselines.heuristics.greedy_dispatch import GreedyDispatchBaseline
from baselines.heuristics.rule_based_pricing import RuleBasedPricingBaseline
from baselines.mpc.oracle_mpc import OracleMPCBaseline

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate all agents")
    parser.add_argument("--checkpoint", type=str,
                        default="results/sac_gnn/checkpoints/best.pt")
    parser.add_argument("--gcn_checkpoint", type=str,
                        default="results/sac_gcn/checkpoints/best.pt")
    parser.add_argument("--flat_checkpoint", type=str,
                        default="results/sac_flat/checkpoints/best.pt")
    parser.add_argument("--n_episodes", type=int, default=30)
    parser.add_argument("--results_dir", type=str,
                        default="results/evaluation")
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument("--synthetic", action="store_true")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Environment builder (evaluation split)
# ---------------------------------------------------------------------------

def make_eval_env(seed: int, synthetic: bool = False):
    graph, hub_configs = HubGraphBuilder.load(
        "data/graphs/inner_melbourne.pkl"
    )
    loader = PriceLoader(region="VIC1", cache_dir="data/nem_cache", seed=seed)

    if synthetic:
        loader.load_synthetic(n_days=365, mean_price=100.0, std_price=250.0,
                              spike_prob=0.003, spike_magnitude=2000.0)
    else:
        parquet = "data/nem_cache/VIC1_2024-01-01_2024-12-31.parquet"
        if Path(parquet).exists():
            loader.load_cache(parquet)
        else:
            loader.fetch_and_cache(start="2024-01-01", end="2024-12-31")

    model = ParticipationModel(seed=seed)
    env = NEMWDREnv(
        hub_configs=hub_configs,
        price_loader=loader,
        participation_model=model,
        env_config=EnvConfig(),
        force_wdr=False,    # guarantee WDR in every eval episode for conformance measurement
        seed=seed,
    )
    return env, graph, hub_configs


# ---------------------------------------------------------------------------
# Single-agent evaluation
# ---------------------------------------------------------------------------

def evaluate_agent(agent, env, n_episodes: int,
                   agent_name: str) -> dict:
    """
    Evaluate one agent over n_episodes and return all four thesis metrics.

    Metrics (§4.3.3):
    1. WDR conformance rate — fraction of WDR steps within 10% of target
    2. Aggregator net profit — total reward across evaluation episodes
    3. Mean participation rate — average ρ̂ across all steps
    4. Convergence speed — loaded from training log separately
    """
    conformance_rates, net_profits, participation_rates = [], [], []
    n_wdr_episodes = 0

    is_heuristic = hasattr(agent, "reset") and not hasattr(agent, "buffer")
    is_mpc = isinstance(agent, OracleMPCBaseline)

    for ep in range(n_episodes):
        obs, _ = env.reset()
        if hasattr(agent, "reset"):
            agent.reset()

        done = False
        ep_reward = 0.0
        wdr_steps = 0
        conformant_steps = 0
        total_rho = 0.0
        total_steps = 0

        while not done:
            # Select action based on agent type
            if is_mpc:
                action = agent.select_action(obs, env)
            elif is_heuristic:
                action = agent.select_action(obs, env)
            else:
                action = agent.select_action(obs, deterministic=True)

            obs, reward, done, _, info = env.step(action)
            ep_reward += reward
            total_steps += 1
            total_rho += info.get("rho_hat", 0.0)

            if info["wdr_active"]:
                wdr_steps += 1
                e_del_mwh = info["e_del_total_kwh"] / 1000.0
                target_mwh = info["dispatch_target_mw"] * (5 / 60)
                if target_mwh > 0:
                    dev_frac = abs(e_del_mwh - target_mwh) / target_mwh
                    if dev_frac <= 0.10:
                        conformant_steps += 1

        # Compute episode metrics
        if wdr_steps > 0:
            conformance_rates.append(conformant_steps / wdr_steps)
            n_wdr_episodes += 1

        net_profits.append(ep_reward)
        participation_rates.append(
            total_rho / total_steps if total_steps > 0 else 0.0
        )

        if (ep + 1) % 10 == 0:
            logger.info(
                f"  {agent_name}: episode {ep+1}/{n_episodes} "
                f"reward={ep_reward:.1f}"
            )

    return {
        "agent": agent_name,
        "mean_conformance_rate": np.mean(conformance_rates) if conformance_rates else 0.0,
        "std_conformance_rate": np.std(conformance_rates) if conformance_rates else 0.0,
        "mean_net_profit": np.mean(net_profits),
        "std_net_profit": np.std(net_profits),
        "mean_participation_rate": np.mean(participation_rates),
        "std_participation_rate": np.std(participation_rates),
        "n_episodes_evaluated": n_episodes,
        "n_wdr_episodes": n_wdr_episodes,
        "wdr_episode_fraction": n_wdr_episodes / n_episodes,
    }


# ---------------------------------------------------------------------------
# Load training logs for convergence speed metric
# ---------------------------------------------------------------------------

def load_convergence_speed(results_dir: str, agent_name: str) -> int:
    """
    Load convergence episode from training log.
    Returns None if training log not found.
    """
    log_path = Path(results_dir) / "logs" / "training_log.csv"
    if not log_path.exists():
        return None

    df = pd.read_csv(log_path)
    if "reward" not in df.columns:
        return None

    rewards = df["reward"].values
    window = 100
    threshold = 0.90

    if len(rewards) < window:
        return None

    all_time_best = max(rewards)
    if all_time_best <= 0:
        return None

    for i in range(window, len(rewards)):
        recent = np.mean(rewards[i-window:i])
        if recent >= threshold * all_time_best:
            return int(df["episode"].values[i])

    return None


# ---------------------------------------------------------------------------
# Results table and plots
# ---------------------------------------------------------------------------

def print_results_table(results_df: pd.DataFrame):
    """Print formatted results table for thesis."""
    print("\n" + "=" * 85)
    print("EVALUATION RESULTS — NEM V2G Hub Dispatch")
    print("=" * 85)
    print(f"{'Agent':<20} {'Conformance':>14} {'Net Profit':>14} "
          f"{'Participation':>14} {'Conv. Episode':>14}")
    print("-" * 85)

    for _, row in results_df.iterrows():
        conv = f"{int(row['convergence_episode'])}" \
               if pd.notna(row.get("convergence_episode")) else "N/A"
        print(
            f"{row['agent']:<20} "
            f"{row['mean_conformance_rate']:>13.3f} "
            f"{row['mean_net_profit']:>14.1f} "
            f"{row['mean_participation_rate']:>13.3f} "
            f"{conv:>14}"
        )

    print("=" * 85)
    print("Conformance: fraction of WDR steps within 10% of dispatch target")
    print("Net Profit: mean episode reward ($)")
    print("Participation: mean empirical EV owner response rate ρ̂")
    print("Conv. Episode: episode where reward reaches 90% of asymptotic value")


def make_plots(results_df: pd.DataFrame, output_dir: Path):
    """Generate bar charts for the three primary metrics."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available — skipping plots")
        return

    agents = results_df["agent"].tolist()
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336", "#00BCD4"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel 1: Conformance rate
    vals = results_df["mean_conformance_rate"].values
    errs = results_df["std_conformance_rate"].values
    axes[0].bar(agents, vals, yerr=errs, color=colors[:len(agents)],
                capsize=5, edgecolor="white")
    axes[0].set_title("WDR Conformance Rate\n(fraction within 10% of target)")
    axes[0].set_ylabel("Conformance rate")
    axes[0].set_ylim(0, 1.05)
    axes[0].axhline(0.9, color="red", linestyle="--", alpha=0.5,
                    label="90% target")
    axes[0].legend(fontsize=8)
    axes[0].tick_params(axis="x", rotation=30)

    # Panel 2: Net profit
    vals = results_df["mean_net_profit"].values
    errs = results_df["std_net_profit"].values
    axes[1].bar(agents, vals, yerr=errs, color=colors[:len(agents)],
                capsize=5, edgecolor="white")
    axes[1].set_title("Mean Episode Net Profit ($)")
    axes[1].set_ylabel("Net profit ($)")
    axes[1].tick_params(axis="x", rotation=30)

    # Panel 3: Participation rate
    vals = results_df["mean_participation_rate"].values
    errs = results_df["std_participation_rate"].values
    axes[2].bar(agents, vals, yerr=errs, color=colors[:len(agents)],
                capsize=5, edgecolor="white")
    axes[2].set_title("Mean Participation Rate ρ̂")
    axes[2].set_ylabel("Mean participation rate")
    axes[2].set_ylim(0, 1.0)
    axes[2].tick_params(axis="x", rotation=30)

    plt.suptitle("SAC-GNN vs Baselines — Inner Melbourne VSR Zone",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / "metrics_comparison.png",
                dpi=150, bbox_inches="tight")
    logger.info(f"Plot saved: {output_dir / 'metrics_comparison.png'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    output_dir = Path(args.results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Building evaluation environment...")
    env, graph, hub_configs = make_eval_env(
        seed=args.seed, synthetic=args.synthetic
    )
    obs_dim = env.observation_space.shape[0]
    n_hubs = len(hub_configs)
    action_dim = n_hubs + 1

    # ── Build all agents ─────────────────────────────────────────────
    agents = []

    # 1. SAC-GNN (proposed)
    if Path(args.checkpoint).exists():
        net_cfg = NetworkConfig()
        sac_gnn = SACGNNAgent(
            n_hubs=n_hubs, graph_data=graph,
            obs_dim=obs_dim, net_cfg=net_cfg,
        )
        sac_gnn.load(args.checkpoint)
        agents.append(("SAC-GNN", sac_gnn))
        logger.info(f"Loaded SAC-GNN from {args.checkpoint}")
    else:
        logger.warning(f"SAC-GNN checkpoint not found: {args.checkpoint}")

    # 2. SAC-GCN ablation
    if Path(args.gcn_checkpoint).exists():
        sac_gcn = SACGCNAgent(
            n_hubs=n_hubs, graph_data=graph, obs_dim=obs_dim,
        )
        sac_gcn.load(args.gcn_checkpoint)
        agents.append(("SAC-GCN", sac_gcn))
        logger.info(f"Loaded SAC-GCN from {args.gcn_checkpoint}")
    else:
        logger.warning(f"SAC-GCN checkpoint not found: {args.gcn_checkpoint}")

    # 3. SAC-Flat ablation
    if Path(args.flat_checkpoint).exists():
        sac_flat = SACFlatAgent(obs_dim=obs_dim, action_dim=action_dim)
        agents.append(("SAC-Flat", sac_flat))
        logger.info(f"Loaded SAC-Flat from {args.flat_checkpoint}")
    else:
        logger.warning(f"SAC-Flat checkpoint not found: {args.flat_checkpoint}")

    # 4. Greedy dispatch heuristic (no checkpoint needed)
    greedy = GreedyDispatchBaseline(n_hubs=n_hubs, fixed_price=80.0)
    agents.append(("Greedy", greedy))

    # 5. Rule-based pricing heuristic
    rule_based = RuleBasedPricingBaseline(n_hubs=n_hubs, price_fraction=0.5)
    agents.append(("RulePrice", rule_based))

    # 6. Oracle MPC (upper bound)
    hub_distances = [hc.distance_km for hc in hub_configs]
    model = ParticipationModel(seed=args.seed)
    oracle = OracleMPCBaseline(
        n_hubs=n_hubs,
        participation_model=model,
        hub_distances=hub_distances,
    )
    agents.append(("OracleMPC", oracle))

    # ── Evaluate all agents ──────────────────────────────────────────
    results = []
    for name, agent in agents:
        logger.info(f"\nEvaluating {name} over {args.n_episodes} episodes...")
        metrics = evaluate_agent(agent, env, args.n_episodes, name)

        # Load convergence speed from training log
        train_log_dir = f"results/{name.lower().replace('-', '_')}"
        conv_ep = load_convergence_speed(train_log_dir, name)
        metrics["convergence_episode"] = conv_ep

        results.append(metrics)
        logger.info(f"  {name}: conformance={metrics['mean_conformance_rate']:.3f}, "
                    f"profit={metrics['mean_net_profit']:.1f}")

    # ── Save results ─────────────────────────────────────────────────
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / "metrics_table.csv", index=False)

    # Print formatted table
    print_results_table(results_df)

    # Save formatted text table
    with open(output_dir / "metrics_table.txt", "w") as f:
        f.write(results_df.to_string(index=False))

    # Generate plots
    make_plots(results_df, output_dir)

    logger.info(f"\nResults saved to {output_dir}")
    logger.info(f"  metrics_table.csv — import into thesis")
    logger.info(f"  metrics_comparison.png — Figure for thesis §5")


if __name__ == "__main__":
    main()
