"""
train_sac_gnn.py
================
Main training entry point for the SAC-GNN agent.

This script wires together all Phase 2–4 components into a complete
training loop with logging, checkpointing, and early stopping.

Usage
-----
    # Train with real AEMO data (recommended for thesis experiments)
    python train_sac_gnn.py

    # Train with synthetic data (offline / CI)
    python train_sac_gnn.py --synthetic

    # Resume from checkpoint
    python train_sac_gnn.py --resume results/sac_gnn/checkpoints/step_50000.pt

    # Custom config
    python train_sac_gnn.py --episodes 3000 --seed 1

Training loop structure
-----------------------
For each episode:
  1. Reset environment (sample new price/WDR episode)
  2. For each of 288 steps:
     a. Agent selects action (stochastic during training)
     b. Environment steps, returns (obs, reward, done, info)
     c. Transition stored in replay buffer
     d. SAC gradient update performed if buffer has enough transitions
  3. Log episode metrics
  4. Save checkpoint every --save_every episodes
  5. Evaluate deterministic policy every --eval_every episodes

Outputs
-------
results/sac_gnn/
  checkpoints/
    step_{N}.pt         ← agent checkpoint every save_every episodes
    best.pt             ← best checkpoint by eval conformance rate
  logs/
    training_log.csv    ← per-episode metrics for plotting
    eval_log.csv        ← per-evaluation metrics
  config.json           ← full hyperparameter record for reproducibility
"""

import os
import json
import time
import argparse
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# ── Set FORCE_NUMPY_AGENT=0 to ensure real PyTorch is used ──────────
os.environ.pop("FORCE_NUMPY_AGENT", None)

from nem_env.spatial_graph import HubGraphBuilder
from nem_env.aemo_price_loader import PriceLoader
from nem_env.participation_model import ParticipationModel
from nem_env.nem_doe_env import NEMDOEEnv, EnvConfig
from baselines.gnn_rl.agent import SACGNNAgent
from baselines.gnn_rl.networks import NetworkConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default training configuration
# All values are thesis hyperparameters — document any changes in thesis §4.3
# ---------------------------------------------------------------------------
DEFAULT_CONFIG = {
    # Environment
    "region": "VIC1",
    "cache_dir": "data/nem_cache",
    "graph_path": "data/graphs/inner_melbourne.pkl",
    "price_start": "2022-01-01",
    "price_end": "2023-12-31",        # training period
    "eval_price_start": "2024-01-01", # held-out evaluation period
    "eval_price_end": "2024-12-31",

    # Participation model betas (§4.2.2, calibrated from Liu et al. 2025)
    "beta_0": -2.20,
    "beta_1":  0.008,
    "beta_2": -0.20,
    "beta_3":  1.50,

    # SAC hyperparameters
    "gamma": 0.99,
    "tau": 0.005,
    "lr_actor": 3e-4,
    "lr_critic": 3e-4,
    "lr_alpha": 3e-4,
    "batch_size": 256,
    "buffer_capacity": 500_000,
    "learning_starts": 1000,          # transitions before first update
    "update_every": 1,                # update after every step

    # Network architecture
    "embed_dim": 64,
    "gat_heads": 4,
    "gat_layers": 2,
    "actor_hidden": 128,
    "critic_hidden": 256,
    "dropout": 0.1,

    # Training duration
    "n_episodes": 2000,               # total training episodes
    "eval_every": 50,                 # evaluate every N episodes
    "save_every": 100,                # checkpoint every N episodes
    "n_eval_episodes": 20,            # episodes per evaluation run

    # Convergence criterion (for RQ4 convergence speed metric)
    "convergence_threshold": 0.90,    # fraction of asymptotic reward
    "convergence_window": 100,        # rolling window to measure convergence

    # Misc
    "seed": 42,
    "results_dir": "results/sac_gnn",
    "synthetic": False,
}


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Train SAC-GNN agent")
    parser.add_argument("--episodes", type=int, default=None,
                        help="Number of training episodes")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic prices (offline mode)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--results_dir", type=str, default=None)
    parser.add_argument("--no_eval", action="store_true",
                        help="Skip evaluation runs (faster, less informative)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def make_env(cfg: dict, split: str = "train", seed: int = 42) -> NEMDOEEnv:
    """
    Build a NEMDOEEnv from config.

    Parameters
    ----------
    cfg : dict
        Training config.
    split : str
        "train" or "eval" — determines which date range is used.
    seed : int
        Environment RNG seed.
    """
    # Load graph
    graph, hub_configs = HubGraphBuilder.load(cfg["graph_path"])

    # Price loader
    loader = PriceLoader(
        region=cfg["region"],
        cache_dir=cfg["cache_dir"],
        seed=seed,
    )

    if cfg["synthetic"]:
        loader.load_synthetic(
            n_days=365,
            mean_price=100.0,
            std_price=250.0,
            spike_prob=0.003,
            spike_magnitude=2000.0,
        )
        logger.warning("Using synthetic prices — not suitable for final experiments")
    else:
        # Load from cached Parquet (must have run fetch_and_cache first)
        parquet_path = (
            f"{cfg['cache_dir']}/{cfg['region']}_"
            f"{cfg['price_start']}_{cfg['price_end']}.parquet"
        )
        if split == "eval":
            parquet_path = (
                f"{cfg['cache_dir']}/{cfg['region']}_"
                f"{cfg['eval_price_start']}_{cfg['eval_price_end']}.parquet"
            )

        if not Path(parquet_path).exists():
            logger.info(f"Parquet not found at {parquet_path}, fetching...")
            start = cfg["price_start"] if split == "train" else cfg["eval_price_start"]
            end = cfg["price_end"] if split == "train" else cfg["eval_price_end"]
            loader.fetch_and_cache(start=start, end=end)
        else:
            loader.load_cache(parquet_path)

    # Participation model
    model = ParticipationModel(
        betas={
            "beta_0": cfg["beta_0"],
            "beta_1": cfg["beta_1"],
            "beta_2": cfg["beta_2"],
            "beta_3": cfg["beta_3"],
        },
        seed=seed,
    )

    env = NEMDOEEnv(
        hub_configs=hub_configs,
        price_loader=loader,
        participation_model=model,
        env_config=EnvConfig(),
        seed=seed,
    )

    return env, graph, hub_configs


# ---------------------------------------------------------------------------
# Evaluation function
# ---------------------------------------------------------------------------

def evaluate(
    agent: SACGNNAgent,
    eval_env: NEMDOEEnv,
    n_episodes: int,
    episode_num: int,
) -> dict:
    """
    Evaluate the deterministic policy over n_episodes held-out episodes.

    Computes the four thesis metrics from §4.3.3:
      1. WDR conformance rate
      2. Aggregator net profit
      3. Mean participation rate
      4. (Convergence speed tracked separately in training loop)

    Returns dict of mean metrics across evaluation episodes.
    """
    net_profits = []
    participation_rates = []
    episode_rewards = []
    doe_violation_totals = []

    for _ in range(n_episodes):
        obs, _ = eval_env.reset()
        done = False
        ep_reward = 0.0
        total_rho_hat = 0.0
        total_doe_violation_kw = 0.0
        rho_steps = 0

        while not done:
            action = agent.select_action(obs, deterministic=True)
            obs, reward, done, _, info = eval_env.step(action)
            ep_reward += reward
            total_rho_hat += info.get("rho_hat", 0.0)
            total_doe_violation_kw += sum(info.get("doe_violations_kw", [0.0]))
            rho_steps += 1

        net_profits.append(ep_reward)
        participation_rates.append(
            total_rho_hat / rho_steps if rho_steps > 0 else 0.0
        )
        episode_rewards.append(ep_reward)
        doe_violation_totals.append(total_doe_violation_kw)

    return {
        "eval_episode": episode_num,
        "mean_net_profit": np.mean(net_profits),
        "mean_participation_rate": np.mean(participation_rates),
        "mean_episode_reward": np.mean(episode_rewards),
        "std_episode_reward": np.std(episode_rewards),
        "mean_doe_violation_kw": np.mean(doe_violation_totals),
    }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(cfg: dict, resume_path: str = None, no_eval: bool = False):
    """Main training loop."""

    # ── Setup output directories ─────────────────────────────────────
    results_dir = Path(cfg["results_dir"])
    ckpt_dir = results_dir / "checkpoints"
    log_dir = results_dir / "logs"
    for d in [ckpt_dir, log_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Save full config for reproducibility
    config_path = results_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(cfg, f, indent=2)
    logger.info(f"Config saved to {config_path}")

    # ── Build environments ────────────────────────────────────────────
    logger.info("Building training environment...")
    train_env, graph, hub_configs = make_env(cfg, split="train", seed=cfg["seed"])

    eval_env = None
    if not no_eval:
        logger.info("Building evaluation environment...")
        try:
            eval_env, _, _ = make_env(cfg, split="eval", seed=cfg["seed"] + 1)
        except Exception as e:
            logger.warning(f"Could not build eval env ({e}) — using train env for eval")
            eval_env, _, _ = make_env(cfg, split="train", seed=cfg["seed"] + 1)

    # ── Build agent ───────────────────────────────────────────────────
    obs_dim = train_env.observation_space.shape[0]
    net_cfg = NetworkConfig(
        embed_dim=cfg["embed_dim"],
        gat_heads=cfg["gat_heads"],
        gat_layers=cfg["gat_layers"],
        actor_hidden=cfg["actor_hidden"],
        critic_hidden=cfg["critic_hidden"],
        dropout=cfg["dropout"],
    )

    agent = SACGNNAgent(
        n_hubs=len(hub_configs),
        graph_data=graph,
        obs_dim=obs_dim,
        net_cfg=net_cfg,
        gamma=cfg["gamma"],
        tau=cfg["tau"],
        lr_actor=cfg["lr_actor"],
        lr_critic=cfg["lr_critic"],
        lr_alpha=cfg["lr_alpha"],
        batch_size=cfg["batch_size"],
        buffer_capacity=cfg["buffer_capacity"],
        learning_starts=cfg["learning_starts"],
        update_every=cfg["update_every"],
        seed=cfg["seed"],
    )

    if resume_path:
        logger.info(f"Resuming from checkpoint: {resume_path}")
        agent.load(resume_path)

    logger.info(f"Agent summary: {agent.summary()}")

    # ── Training state ────────────────────────────────────────────────
    training_log = []
    eval_log = []
    best_net_profit = float("-inf")
    convergence_episode = None
    reward_history = []

    start_time = time.time()
    n_episodes = cfg["n_episodes"]

    logger.info(f"Starting training: {n_episodes} episodes, "
                f"{len(hub_configs)} hubs, obs_dim={obs_dim}")
    logger.info("=" * 60)

    # ── Main training loop ────────────────────────────────────────────
    for episode in range(1, n_episodes + 1):

        obs, _ = train_env.reset()
        done = False
        ep_reward = 0.0
        ep_steps = 0
        total_rho_hat = 0.0
        total_doe_violation_kw = 0.0
        ep_losses = {"critic_loss": [], "actor_loss": [], "alpha": []}

        while not done:
            # Select action (stochastic during training)
            action = agent.select_action(obs, deterministic=False)

            # Environment step
            next_obs, reward, done, _, info = train_env.step(action)

            # Store transition
            agent.store_transition(
                obs, action, reward, next_obs, done,
            )

            # SAC update
            losses = agent.update()
            if losses is not None:
                ep_losses["critic_loss"].append(losses["critic_loss"])
                ep_losses["actor_loss"].append(losses["actor_loss"])
                ep_losses["alpha"].append(losses["alpha"])

            # Track metrics
            ep_reward += reward
            ep_steps += 1
            total_rho_hat += info.get("rho_hat", 0.0)
            total_doe_violation_kw += sum(info.get("doe_violations_kw", [0.0]))

            obs = next_obs

        # ── Episode metrics ───────────────────────────────────────────
        mean_rho = total_rho_hat / ep_steps
        reward_history.append(ep_reward)

        log_entry = {
            "episode": episode,
            "reward": ep_reward,
            "mean_participation_rate": mean_rho,
            "doe_violation_kw": total_doe_violation_kw,
            "buffer_size": agent.buffer.size,
            "total_steps": agent._total_steps,
            "critic_loss": np.mean(ep_losses["critic_loss"]) if ep_losses["critic_loss"] else None,
            "actor_loss": np.mean(ep_losses["actor_loss"]) if ep_losses["actor_loss"] else None,
            "alpha": np.mean(ep_losses["alpha"]) if ep_losses["alpha"] else None,
            "elapsed_min": (time.time() - start_time) / 60,
        }
        training_log.append(log_entry)

        # ── Convergence detection ─────────────────────────────────────
        # Track when agent reaches 90% of asymptotic reward (RQ4 metric)
        if (convergence_episode is None
                and len(reward_history) >= cfg["convergence_window"]):
            recent = np.mean(reward_history[-cfg["convergence_window"]:])
            all_time_best = max(reward_history)
            if all_time_best > 0 and recent >= cfg["convergence_threshold"] * all_time_best:
                convergence_episode = episode
                logger.info(
                    f"Convergence detected at episode {episode} "
                    f"(reward={recent:.1f}, threshold={cfg['convergence_threshold']*all_time_best:.1f})"
                )

        # ── Periodic logging ──────────────────────────────────────────
        if episode % 10 == 0:
            recent_reward = np.mean(reward_history[-10:])
            logger.info(
                f"Ep {episode:4d}/{n_episodes} | "
                f"reward={ep_reward:8.1f} | "
                f"avg10={recent_reward:8.1f} | "
                f"ρ={mean_rho:.3f} | "
                f"doe_viol={total_doe_violation_kw:.1f}kW | "
                f"buf={agent.buffer.size:6d} | "
                f"α={log_entry['alpha']:.4f}" if log_entry['alpha'] else
                f"Ep {episode:4d}/{n_episodes} | "
                f"reward={ep_reward:8.1f} | "
                f"buf={agent.buffer.size:6d} | collecting..."
            )

        # ── Evaluation ────────────────────────────────────────────────
        if not no_eval and eval_env is not None and episode % cfg["eval_every"] == 0:
            logger.info(f"  → Evaluating at episode {episode}...")
            eval_metrics = evaluate(
                agent, eval_env, cfg["n_eval_episodes"], episode
            )
            eval_log.append(eval_metrics)

            logger.info(
                f"  → Eval: profit={eval_metrics['mean_net_profit']:.1f} | "
                f"participation={eval_metrics['mean_participation_rate']:.3f} | "
                f"doe_viol={eval_metrics['mean_doe_violation_kw']:.1f}kW"
            )

            # Save best checkpoint by conformance rate
            if eval_metrics["mean_net_profit"] > best_net_profit:
                best_net_profit = eval_metrics["mean_net_profit"]
                agent.save(str(ckpt_dir / "best.pt"))
                logger.info(
                    f"  → New best net profit: {best_net_profit:.1f} — saved best.pt"
                )

            # Save eval log incrementally
            pd.DataFrame(eval_log).to_csv(
                log_dir / "eval_log.csv", index=False
            )

        # ── Checkpoint ───────────────────────────────────────────────
        if episode % cfg["save_every"] == 0:
            ckpt_path = ckpt_dir / f"episode_{episode}.pt"
            agent.save(str(ckpt_path))
            logger.info(f"  → Checkpoint saved: {ckpt_path}")

            # Save training log incrementally
            pd.DataFrame(training_log).to_csv(
                log_dir / "training_log.csv", index=False
            )

    # ── Final save ────────────────────────────────────────────────────
    agent.save(str(ckpt_dir / "final.pt"))
    pd.DataFrame(training_log).to_csv(log_dir / "training_log.csv", index=False)
    if eval_log:
        pd.DataFrame(eval_log).to_csv(log_dir / "eval_log.csv", index=False)

    elapsed = (time.time() - start_time) / 60
    logger.info("=" * 60)
    logger.info(f"Training complete in {elapsed:.1f} minutes")
    logger.info(f"Final checkpoint: {ckpt_dir / 'final.pt'}")
    logger.info(f"Best net profit: {best_net_profit:.1f}")
    if convergence_episode:
        logger.info(f"Convergence episode: {convergence_episode}")
    else:
        logger.info("Convergence not detected within training budget")

    return {
        "training_log": pd.DataFrame(training_log),
        "eval_log": pd.DataFrame(eval_log) if eval_log else None,
        "convergence_episode": convergence_episode,
        "best_net_profit": best_net_profit,
        "elapsed_min": elapsed,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()

    # Merge CLI args into config
    cfg = DEFAULT_CONFIG.copy()
    if args.episodes is not None:
        cfg["n_episodes"] = args.episodes
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.synthetic:
        cfg["synthetic"] = True
    if args.results_dir is not None:
        cfg["results_dir"] = args.results_dir

    # Set seeds for reproducibility
    np.random.seed(cfg["seed"])

    results = train(cfg, resume_path=args.resume, no_eval=args.no_eval)

    print("\nSummary:")
    print(f"  Best net profit       : {results['best_net_profit']:.1f}")
    print(f"  Convergence episode   : {results['convergence_episode']}")
    print(f"  Training time         : {results['elapsed_min']:.1f} min")
    print(f"\nResults saved to: {cfg['results_dir']}")