"""
train_sac_gnn.py
================
Main training entry point for the SAC-GNN agent.

This script wires together all Phase 2–4 components into a complete
training loop with logging, checkpointing, and early stopping.

Usage
-----
    # Train SAC-GNN (default)
    python train_sac_gnn.py

    # Train SAC-GCN ablation
    python train_sac_gnn.py --agent sac_gcn

    # Train SAC-Flat ablation
    python train_sac_gnn.py --agent sac_flat

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
    "price_end":   "2023-12-31",      # 2 years training
    "eval_price_start": "2024-01-01", # held-out: full 2024
    "eval_price_end":   "2024-12-31",

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
    "n_episodes": 1500,               # total training episodes
    "eval_every": 50,
    "n_eval_episodes": 5,             # avg over 5 price days per eval checkpoint
    "convergence_window": 10,   # episodes to check for plateau
    "convergence_threshold": 500,  # max $std of eval profit over window to declare convergence                 # evaluate every N episodes
    "save_every": 100,                # checkpoint every N episodes


    # Misc
    "seed": 42,
    "agent": "sac_gnn",                    # sac_gnn | sac_gcn | sac_flat
    "results_dir": "results/sac_gnn_real",  # separate from synthetic runs
    "synthetic": False,
}


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Train SAC-GNN agent")
    parser.add_argument("--agent", type=str, default="sac_gnn",
                        choices=["sac_gnn", "sac_gcn", "sac_flat"],
                        help="Agent architecture to train (default: sac_gnn)")
    parser.add_argument("--episodes", type=int, default=None,
                        help="Number of training episodes")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic prices (offline mode)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint .pt file to resume from")
    parser.add_argument("--start_episode", type=int, default=None,
                        help="Episode to resume from (e.g. 1601). Inferred from checkpoint if not set.")
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

# ---------------------------------------------------------------------------
# Fixed evaluation days — same 5 dates used at every checkpoint
# ---------------------------------------------------------------------------
# These are specific real 2024 VIC1 dates representing distinct NEM conditions.
# Using fixed dates (not random) makes checkpoint comparison meaningful:
# improvement in eval profit = genuine policy improvement, not lucky day sampling.
#
# Date selection rationale (master summary §4, case studies):
#   Summer peak    — Jan 2024 heatwave: tight DOE (transformer thermal stress),
#                    afternoon RRP spike; tests constraint + arbitrage timing
#   High volatility— Mar 2024: large intraday RRP swings; tests precise timing
#   Negative RRP   — Oct 2024: wind oversupply pushes RRP negative; tests
#                    whether agent learned charge direction of arbitrage
#   Winter average — Jun 2024: moderate stable prices; baseline performance day
#   Weekend low    — Aug 2024: low demand, low participation pool; tests
#                    incentive price adaptation under thin supply
#
# If a date is missing from the eval parquet (e.g. data gap), evaluate()
# falls back to a random day with a warning — training is not interrupted.
FIXED_EVAL_DAYS = [
    "2024-01-25",   # summer peak — heatwave, tight DOE, afternoon spike
    "2024-03-12",   # high volatility — large intraday RRP swings
    "2024-02-13",   # NEGATIVE RRP / STRESS TEST — 131 negative intervals,
                    # min=-$1000/MWh (2024-10-03 was missing from VIC1
                    # parquet — data gap). This day is the most extreme
                    # in the entire dataset: observed profit magnitude on
                    # this day runs 20-500x larger than the other 4 days
                    # (e.g. seed1 SAC-GNN: +$120,490 here vs $4k-8k on
                    # other days; SAC-GCN: -$539,375 here vs $5k-8k
                    # elsewhere). Index 2 in this list — see
                    # STRESS_TEST_DAY_INDEX below.
    "2024-06-15",   # winter average — moderate stable prices, baseline
    "2024-01-28",   # weekend low demand — Sunday, 106 neg RRP intervals
                    # (2024-08-20 was missing from VIC1 parquet — data gap)
]

# Index of the stress-test day within FIXED_EVAL_DAYS (0-based).
# Used to exclude it from best.pt selection and convergence detection —
# see evaluate() docstring for why pooling it in produces a misleading
# "mean_net_profit" that one extreme day can dominate or invert.
STRESS_TEST_DAY_INDEX = 2


def evaluate(
    agent: SACGNNAgent,
    eval_env: NEMDOEEnv,
    n_episodes: int,
    episode_num: int,
) -> dict:
    """
    Evaluate the deterministic policy on fixed held-out days.

    Runs the policy on each of the FIXED_EVAL_DAYS in sequence, then
    on (n_episodes - len(FIXED_EVAL_DAYS)) additional random days if
    n_episodes > 5. With n_episodes=5 this evaluates exactly the 5
    fixed days — making every checkpoint directly comparable.

    Using fixed dates eliminates the $7,000 checkpoint variance seen
    in random-day evaluation, making convergence visible and checkpoint
    comparison meaningful (§4.3.3, convergence detection).

    Per-day results are logged individually for the thesis case study
    table (Table 3), in addition to the aggregate mean.

    Stress-test day exclusion
    --------------------------
    FIXED_EVAL_DAYS[STRESS_TEST_DAY_INDEX] (2024-02-13, extreme negative
    RRP) has a profit magnitude 20-500x larger than the other 4 days.
    Pooling it into "mean_net_profit" means one day's outcome — not the
    agent's typical performance — drives best.pt selection and
    convergence detection, and can flip the ranking between checkpoints
    or agents depending on which direction that one day happened to go.

    "mean_net_profit_normal" excludes it and is the value best.pt
    selection and convergence detection should use going forward.
    "mean_net_profit" (all 5 days pooled) is retained for backward
    compatibility with existing logs/visualisers and reported alongside
    it, not in place of it.

    Returns
    -------
    dict with pooled mean, normal-day-only mean, and per-day breakdown.
    """
    net_profits = []
    participation_rates = []
    doe_violation_totals = []
    per_day_results = []

    # Evaluate on fixed days first
    eval_dates = FIXED_EVAL_DAYS[:n_episodes]
    # If n_episodes > 5, pad with random days
    n_random = max(0, n_episodes - len(FIXED_EVAL_DAYS))

    for i, date in enumerate(eval_dates):
        obs, _ = eval_env.reset(options={"date": date})
        done = False
        ep_reward = 0.0
        total_rho_hat = 0.0
        total_doe_kw = 0.0
        rho_steps = 0

        while not done:
            action = agent.select_action(obs, deterministic=True)
            obs, reward, done, _, info = eval_env.step(action)
            ep_reward += reward
            total_rho_hat += info.get("rho_hat", 0.0)
            total_doe_kw += sum(info.get("doe_violations_kw", [0.0]))
            rho_steps += 1

        mean_rho = total_rho_hat / rho_steps if rho_steps > 0 else 0.0
        net_profits.append(ep_reward)
        participation_rates.append(mean_rho)
        doe_violation_totals.append(total_doe_kw)
        per_day_results.append({
            "date": date,
            "profit": ep_reward,
            "participation": mean_rho,
            "doe_viol_kw": total_doe_kw,
            "is_stress_test": (i == STRESS_TEST_DAY_INDEX),
        })

    # Pad with random days if needed
    for _ in range(n_random):
        obs, _ = eval_env.reset()
        done = False
        ep_reward = 0.0
        total_rho_hat = 0.0
        total_doe_kw = 0.0
        rho_steps = 0

        while not done:
            action = agent.select_action(obs, deterministic=True)
            obs, reward, done, _, info = eval_env.step(action)
            ep_reward += reward
            total_rho_hat += info.get("rho_hat", 0.0)
            total_doe_kw += sum(info.get("doe_violations_kw", [0.0]))
            rho_steps += 1

        mean_rho = total_rho_hat / rho_steps if rho_steps > 0 else 0.0
        net_profits.append(ep_reward)
        participation_rates.append(mean_rho)
        doe_violation_totals.append(total_doe_kw)

    # Normal-day-only mean: excludes the stress-test day if it's within
    # the fixed-day range evaluated this call (n_episodes >= len(FIXED_EVAL_DAYS)
    # guarantees it was included; if n_episodes < 5 and truncated before
    # reaching STRESS_TEST_DAY_INDEX, there's nothing to exclude).
    if len(eval_dates) > STRESS_TEST_DAY_INDEX:
        normal_profits = [
            p for i, p in enumerate(net_profits[:len(eval_dates)])
            if i != STRESS_TEST_DAY_INDEX
        ] + net_profits[len(eval_dates):]  # keep any padded random days
    else:
        normal_profits = net_profits

    return {
        "eval_episode":            episode_num,
        "mean_net_profit":         float(np.mean(net_profits)),        # pooled (all days, incl. stress test)
        "mean_net_profit_normal":  float(np.mean(normal_profits)),      # excludes stress-test day — use for best.pt / convergence
        "std_net_profit":          float(np.std(net_profits)),
        "std_net_profit_normal":   float(np.std(normal_profits)),
        "mean_participation_rate": float(np.mean(participation_rates)),
        "mean_doe_violation_kw":   float(np.mean(doe_violation_totals)),
        "per_day":                 per_day_results,   # for thesis Table 3
    }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(cfg: dict, resume_path: str = None, no_eval: bool = False, start_episode: int = 1):
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

    agent_type = cfg.get("agent", "sac_gnn")
    n_hubs     = len(hub_configs)

    # device=None lets each agent self-detect CUDA vs CPU (agent.py picks
    # "cuda" if torch.cuda.is_available() else "cpu"). Passed explicitly
    # here (rather than relying purely on each agent's internal default)
    # so the choice is visible in one place and can be overridden via
    # cfg["device"] if ever needed (e.g. forcing CPU for a debug run).
    device = cfg.get("device", None)

    agent_kwargs = dict(
        n_hubs=n_hubs,
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
        device=device,
    )

    if agent_type == "sac_gcn":
        from baselines.gnn_rl.sac_gcn import SACGCNAgent
        agent = SACGCNAgent(**agent_kwargs)
        logger.info("Architecture: SAC-GCN (GCN encoder — fixed aggregation)")
    elif agent_type == "sac_flat":
        from baselines.flat_mlp.sac_flat import SACFlatAgent
        agent = SACFlatAgent(
            obs_dim=obs_dim,
            action_dim=n_hubs + 1,
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
            device=device,
        )
        logger.info("Architecture: SAC-Flat (MLP only — no graph encoder)")
    else:
        agent = SACGNNAgent(**agent_kwargs)
        logger.info("Architecture: SAC-GNN (GAT encoder — learned attention)")

    logger.info(f"Agent device: {agent.device}")

    if resume_path:
        logger.info(f"Resuming from checkpoint: {resume_path}")
        agent.load(resume_path)
        if start_episode <= 1 and agent._total_steps > 0:
            inferred = agent._total_steps // 288 + 1
            logger.info(f"Inferred start episode {inferred} from checkpoint total_steps={agent._total_steps}")
            start_episode = inferred

    logger.info(f"Agent summary: {agent.summary()}")

    # ── Training state ────────────────────────────────────────────────
    training_log = []
    eval_log = []
    best_net_profit = float("-inf")
    convergence_episode = None
    reward_history = []

    start_time = time.time()
    n_episodes = cfg["n_episodes"]

    ep_start = max(1, start_episode)
    ep_end   = n_episodes
    remaining = ep_end - ep_start + 1

    if ep_start > 1:
        logger.info(f"Resuming from episode {ep_start} — {remaining} episodes remaining to reach {ep_end}")
    else:
        logger.info(f"Starting training: {n_episodes} episodes, {len(hub_configs)} hubs, obs_dim={obs_dim}")
    logger.info("=" * 60)

    # ── Main training loop ────────────────────────────────────────────
    for episode in range(ep_start, ep_end + 1):

        obs, _ = train_env.reset(options={
            "episode": episode,
            "total_episodes": n_episodes,
        })
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

        # ── Convergence detection (eval-profit std based) ────────────
        # Uses mean_net_profit_normal (excludes the stress-test day at
        # FIXED_EVAL_DAYS[STRESS_TEST_DAY_INDEX]) rather than the pooled
        # mean_net_profit. The stress-test day's profit magnitude
        # (20-500x the other 4 days — see evaluate() docstring) made the
        # pooled std wildly inflated ($39k-$891k observed in seed1 runs),
        # which meant convergence could essentially never be detected
        # regardless of how stable the policy actually was on typical days.
        if convergence_episode is None and len(eval_log) >= cfg["convergence_window"]:
            recent_profits = [e["mean_net_profit_normal"] for e in eval_log[-cfg["convergence_window"]:]]
            profit_std         = float(np.std(recent_profits))
            profit_improvement = abs(recent_profits[-1] - recent_profits[0])
            profit_mean        = float(np.mean(recent_profits))
            threshold          = cfg["convergence_threshold"]

            if profit_std < threshold and profit_improvement < threshold:
                convergence_episode = episode
                logger.info(
                    f"  → CONVERGED at episode {episode} | "
                    f"normal_day_mean_profit=${profit_mean:.0f} | "
                    f"std=${profit_std:.0f} | "
                    f"improvement=${profit_improvement:.0f} "
                    f"(threshold=${threshold:.0f})"
                )
            else:
                status = "converging" if profit_std < threshold * 2 else "not yet"
                logger.info(
                    f"  → Convergence check (normal-day, excl. stress test): "
                    f"std=${profit_std:.0f} | "
                    f"improvement=${profit_improvement:.0f} | "
                    f"threshold=${threshold:.0f} ({status})"
                )

        # ── Periodic logging ──────────────────────────────────────────
        if episode % 10 == 0:
            recent_reward = np.mean(reward_history[-10:])
            critic_loss = log_entry.get('critic_loss')
            actor_loss  = log_entry.get('actor_loss')
            if log_entry.get('alpha') is not None:
                logger.info(
                    f"Ep {episode:4d}/{n_episodes} | "
                    f"reward={ep_reward:8.1f} | "
                    f"avg10={recent_reward:8.1f} | "
                    f"ρ={mean_rho:.3f} | "
                    f"doe_viol={total_doe_violation_kw:.1f}kW | "
                    f"buf={agent.buffer.size:6d} | "
                    f"α={log_entry['alpha']:.4f} | "
                    f"critic_loss={critic_loss:.4f}" if critic_loss else
                    f"Ep {episode:4d}/{n_episodes} | "
                    f"reward={ep_reward:8.1f} | "
                    f"avg10={recent_reward:8.1f} | "
                    f"ρ={mean_rho:.3f} | "
                    f"buf={agent.buffer.size:6d} | "
                    f"α={log_entry['alpha']:.4f}"
                )
            else:
                logger.info(
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
                f"  → Eval: profit(all 5)={eval_metrics['mean_net_profit']:.1f} "
                f"±{eval_metrics['std_net_profit']:.1f} | "
                f"profit(normal 4)={eval_metrics['mean_net_profit_normal']:.1f} "
                f"±{eval_metrics['std_net_profit_normal']:.1f} | "
                f"participation={eval_metrics['mean_participation_rate']:.3f} | "
                f"doe_viol={eval_metrics['mean_doe_violation_kw']:.1f}kW"
            )
            # Log per fixed day for thesis Table 3
            for d in eval_metrics.get("per_day", []):
                stress_tag = " [STRESS TEST]" if d.get("is_stress_test") else ""
                logger.info(
                    f"     [{d['date']}]{stress_tag} profit={d['profit']:8.1f} | "
                    f"ρ={d['participation']:.3f} | "
                    f"doe={d['doe_viol_kw']:.1f}kW"
                )

            # Save best checkpoint using normal-day mean (excludes the
            # stress-test day — see evaluate() docstring). Using the
            # pooled mean_net_profit here would let one extreme day's
            # outcome (20-500x the magnitude of typical days) determine
            # which checkpoint gets kept as "best", independent of
            # whether the policy is actually improving on typical days.
            if eval_metrics["mean_net_profit_normal"] > best_net_profit:
                best_net_profit = eval_metrics["mean_net_profit_normal"]
                agent.save(str(ckpt_dir / "best.pt"))
                logger.info(
                    f"  → New best normal-day profit: {best_net_profit:.1f} — saved best.pt"
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
    logger.info(f"Best normal-day net profit (excl. stress test): {best_net_profit:.1f}")
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
    cfg["agent"] = args.agent
    if args.synthetic:
        cfg["synthetic"] = True
    if args.results_dir is not None:
        cfg["results_dir"] = args.results_dir
    # Auto-set results_dir based on agent if not explicitly set
    elif args.agent != "sac_gnn":
        cfg["results_dir"] = f"results/{args.agent}_real"

    # Set seeds for reproducibility
    np.random.seed(cfg["seed"])

    start_ep = args.start_episode or 1
    results = train(cfg, resume_path=args.resume, no_eval=args.no_eval, start_episode=start_ep)

    print("\nSummary:")
    print(f"  Best net profit       : {results['best_net_profit']:.1f}")
    print(f"  Convergence episode   : {results['convergence_episode']}")
    print(f"  Training time         : {results['elapsed_min']:.1f} min")
    print(f"\nResults saved to: {cfg['results_dir']}")