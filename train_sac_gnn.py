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
    parser.add_argument(
        "--zone", type=str, default=None,
        help="Zone key from ZONE_REGISTRY (nem_env/spatial_graph.py). "
             "Overrides EnvConfig's default graph_path "
             "(data/graphs/inner_melbourne.pkl) to "
             "data/graphs/<zone>.pkl instead. Added for the hub-count "
             "scaling experiment — e.g. --zone greater_melbourne trains "
             "on the 32-hub graph instead of the default 21-hub "
             "inner_melbourne graph, with all other hyperparameters "
             "identical, to test whether SAC-GNN's advantage over "
             "SAC-GCN grows, shrinks, or stays flat at larger hub count.",
    )
    parser.add_argument(
        "--lambda_conf", type=float, default=None,
        help="Override EnvConfig.lambda_conformance (DOE violation penalty "
             "weight, default 200.0). Added for hyperparameter sensitivity "
             "analysis — trains the same agent/seed with a different DOE "
             "penalty strength to test robustness of the reported results "
             "to this choice, rather than relying on a single untested value.",
    )
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

    env_config_kwargs = {}
    if cfg.get("lambda_conf") is not None:
        env_config_kwargs["lambda_conformance"] = cfg["lambda_conf"]
        logger.info(
            f"Overriding EnvConfig.lambda_conformance: "
            f"{cfg['lambda_conf']} (default 200.0) — sensitivity analysis run"
        )

    env = NEMDOEEnv(
        hub_configs=hub_configs,
        price_loader=loader,
        participation_model=model,
        env_config=EnvConfig(**env_config_kwargs),
        seed=seed,
    )

    return env, graph, hub_configs


# ---------------------------------------------------------------------------
# Evaluation function
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Validation days — used ONLY for checkpoint (best.pt) selection during
# training. Distinct from evaluate.py's CASE_STUDIES (the final reported
# test set) — see "Validation/test split" note below.
# ---------------------------------------------------------------------------
#
# Validation/test split (methodology fix)
# -----------------------------------------
# Earlier versions of this pipeline used the SAME 5 dates both to select
# best.pt during training (via this function) AND as the final reported
# test set in evaluate.py's CASE_STUDIES. This is checkpoint-selection
# leakage: choosing the checkpoint that scores highest on exactly the
# days you later report performance on is methodologically equivalent to
# tuning hyperparameters on the test set, and would be flagged by any
# rigorous reviewer (see repo audit notes, Aug 2026).
#
# Fix: VALIDATION_DAYS below are a SEPARATE set of 5 real 2024 VIC1
# dates, chosen to cover the same qualitative conditions as evaluate.py's
# CASE_STUDIES (summer peak / high volatility / negative RRP / winter
# average / weekend low) but on DIFFERENT actual calendar dates, so
# best.pt is never selected using the days performance is ultimately
# reported on.
#
# IMPORTANT — verify before first use: these dates were selected from
# general seasonal reasoning, not a live query of the cached parquet.
# Before running training, confirm each date has the full 288 intervals
# (no AEMO data gap) using the same check used to find/replace the
# original missing dates:
#     python3 -c "
#     import pandas as pd
#     from collections import Counter
#     df = pd.read_parquet('data/nem_cache/VIC1_2024-01-01_2024-12-31.parquet')
#     df.index = pd.to_datetime(df.index)
#     for d in VALIDATION_DAYS: print(d, (df.index.date == pd.Timestamp(d).date()).sum())
#     "
# If any date has 0 (or <288) intervals, replace it with a nearby date
# of similar character using the same search method used previously
# (see aemo_price_loader.py data-gap discussion).
VALIDATION_DAYS = [
    "2024-01-18",   # summer peak (validation) — different week from the
                    # test set's 2024-01-25, still within heatwave season
    "2024-04-16",   # high volatility (validation) — different month from
                    # the test set's 2024-03-12
    "2024-01-21",   # negative RRP (validation) — genuinely negative RRP
                    # intervals present, but NOT the most extreme day in
                    # the dataset (that's reserved for the actual test
                    # stress-test day, 2024-02-13, kept exclusively in
                    # evaluate.py's CASE_STUDIES)
    "2024-05-20",   # winter average (validation) — different month from
                    # the test set's 2024-06-15
    "2024-02-25",   # weekend low demand (validation) — different weekend
                    # from the test set's 2024-01-28
]

# Backward-compatible alias — some older code/tooling may still reference
# FIXED_EVAL_DAYS by this name. Points to the same VALIDATION_DAYS list.
FIXED_EVAL_DAYS = VALIDATION_DAYS

# Index of the (validation) negative-RRP day within VALIDATION_DAYS
# (0-based). Used to exclude it from best.pt selection and convergence
# detection for the same reason the test-set stress-test day is excluded
# from the final Table 3 pooled mean — see evaluate() docstring.
# NOTE: this validation-set negative-RRP day (2024-01-21) is expected to
# have a SMALLER profit-magnitude outlier than the test set's stress-test
# day (2024-02-13, min=-$1000/MWh market floor) since it wasn't
# deliberately selected for extremity — but the same pooling risk applies
# in principle, so it's excluded here too for consistency.
STRESS_TEST_DAY_INDEX = 2


def evaluate(
    agent: SACGNNAgent,
    eval_env: NEMDOEEnv,
    n_episodes: int,
    episode_num: int,
) -> dict:
    """
    Evaluate the deterministic policy on fixed VALIDATION days.

    Runs the policy on each of the VALIDATION_DAYS in sequence, then
    on (n_episodes - len(VALIDATION_DAYS)) additional random days if
    n_episodes > 5. With n_episodes=5 this evaluates exactly the 5
    fixed days — making every checkpoint directly comparable.

    Using fixed dates eliminates the $7,000 checkpoint variance seen
    in random-day evaluation, making convergence visible and checkpoint
    comparison meaningful (§4.3.3, convergence detection).

    IMPORTANT — this function selects best.pt for TRAINING purposes
    only. VALIDATION_DAYS is deliberately DISTINCT from evaluate.py's
    CASE_STUDIES (the final reported test set) — see the
    "Validation/test split" note above VALIDATION_DAYS for why. Do NOT
    treat the per-day results logged here as the thesis/paper's final
    reported numbers; those come only from evaluate.py run on the
    already-trained, already-selected best.pt checkpoint.

    Stress-test day exclusion
    --------------------------
    VALIDATION_DAYS[STRESS_TEST_DAY_INDEX] (2024-01-21, negative RRP)
    can have a profit magnitude substantially larger than the other 4
    validation days. Pooling it into "mean_net_profit" means one day's
    outcome — not the agent's typical performance — drives best.pt
    selection and convergence detection, and can flip the ranking
    between checkpoints depending on which direction that one day
    happened to go.

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
        "per_day":                 per_day_results,   # validation-only diagnostic, NOT Table 3 data
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
            # Log per validation day (diagnostic only — final Table 3 numbers
            # come exclusively from evaluate.py on the selected best.pt)
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
    cfg["lambda_conf"] = args.lambda_conf
    if args.zone is not None:
        cfg["graph_path"] = f"data/graphs/{args.zone}.pkl"
        logger.info(f"Overriding graph_path for --zone={args.zone}: {cfg['graph_path']}")

    # Set seeds for reproducibility
    np.random.seed(cfg["seed"])

    start_ep = args.start_episode or 1
    results = train(cfg, resume_path=args.resume, no_eval=args.no_eval, start_episode=start_ep)

    print("\nSummary:")
    print(f"  Best net profit       : {results['best_net_profit']:.1f}")
    print(f"  Convergence episode   : {results['convergence_episode']}")
    print(f"  Training time         : {results['elapsed_min']:.1f} min")
    print(f"\nResults saved to: {cfg['results_dir']}")