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

Dates match FIXED_EVAL_DAYS in train_sac_gnn.py exactly, so checkpoint
selection during training and final Table 3 results are on identical
conditions. Two original dates (2024-10-03, 2024-08-20) were replaced
after discovering the VIC1 2024 parquet has a data gap from August
onward — see date comments below for the confirmed replacement dates.

    Case Study 1 — Summer peak (2024-01-25, Jan heatwave)
        Tests: tight DOE constraint + afternoon RRP spike
        Expected winner: SAC-GNN (learns spatial DOE correlation)

    Case Study 2 — High volatility (2024-03-12)
        Tests: precise arbitrage timing under large intraday swings
        Expected winner: SAC-GNN (learns spike timing vs greedy)

    Case Study 3 — Negative RRP (2024-02-13, market floor -$1000/MWh)
        Tests: charge direction of arbitrage (get paid to consume)
        Expected winner: SAC-GNN (learned bidirectional dispatch)

    Case Study 4 — Winter average (2024-06-15)
        Tests: baseline stable-price performance
        Expected: all agents competitive

    Case Study 5 — Weekend low demand (2024-01-28, mean RRP ~$0.4/MWh)
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
import time
import argparse
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from scipy import stats

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
# Case study definitions — the FINAL, HELD-OUT TEST SET
# ---------------------------------------------------------------------------
# Validation/test split (methodology note)
# -------------------------------------------
# These 5 dates are the final reported test set — the ONLY dates that
# produce the numbers in the thesis/paper's Table 3. They are never used
# to select which checkpoint to keep (best.pt) during training —
# train_sac_gnn.py uses a completely separate VALIDATION_DAYS list for
# that purpose (see train_sac_gnn.py for the corresponding dates and the
# full rationale for why this split exists). This eliminates
# checkpoint-selection leakage: the model was never optimised, directly
# or indirectly, against the specific days its final performance is
# measured on.
CASE_STUDIES = [
    {
        "id":          "cs1_summer_peak",
        "name":        "Summer Peak",
        "date":        "2024-01-25",
        "description": "Jan 2024 heatwave — tight DOE constraint, afternoon RRP spike",
        "tests":       "Tight DOE + spike timing",
        "is_stress_test": False,
    },
    {
        "id":          "cs2_high_volatility",
        "name":        "High Volatility",
        "date":        "2024-03-12",
        "description": "Large intraday RRP swings",
        "tests":       "Precise arbitrage timing",
        "is_stress_test": False,
    },
    {
        "id":          "cs3_negative_rrp",
        "name":        "Negative RRP",
        "date":        "2024-02-13",
        "description": "Extreme negative RRP — 25 intervals at market floor "
                        "(-$1,000/MWh). Original 2024-10-03 missing from VIC1 "
                        "parquet (AEMO data gap), replaced with confirmed date.",
        "tests":       "Bidirectional dispatch",
        # This day is a deliberate STRESS TEST, not a "normal" trading day —
        # it was specifically chosen for the most extreme negative RRP in
        # the entire dataset. Empirically its profit magnitude
        # (tens/hundreds of thousands, vs single-digit thousands on all
        # other days) is 20-500x larger than the other 4 case studies,
        # which means a naive pooled mean across all 5 days is dominated
        # entirely by this one day and says almost nothing about "typical"
        # performance. Excluded from the "Normal-day Mean" column in
        # print_results_table(); reported separately under "Stress Test".
        "is_stress_test": True,
    },
    {
        "id":          "cs4_winter_average",
        "name":        "Winter Average",
        "date":        "2024-06-15",
        "description": "Moderate stable prices — baseline day",
        "tests":       "Baseline performance",
        "is_stress_test": False,
    },
    {
        "id":          "cs5_weekend_low",
        "name":        "Weekend Low Demand",
        "date":        "2024-01-28",
        "description": "Sunday with mean RRP ~$0.4/MWh — near-zero average price, "
                        "thin participation pool. Original 2024-08-20 missing "
                        "from VIC1 parquet (AEMO data gap), replaced with "
                        "confirmed date.",
        "tests":       "Incentive price adaptation",
        "is_stress_test": False,
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
        "--n_runs", type=int, default=30,
        help=(
            "Repetitions per (agent, case_study) pair. Default 30 — "
            "raised from an earlier default of 10 after reviewing "
            "comparable published work (Orfanoudakis et al., "
            "Communications Engineering 2025, EV-GNN paper on the same "
            "EV2Gym simulator) which uses n=100 for its flagship "
            "large-scale result. 30 is a compute-budget compromise; "
            "pass --n_runs 100 to match that precedent exactly for a "
            "final paper submission run if compute budget allows "
            "(cost scales linearly: 6 agents x 5 case studies x n_runs "
            "episodes total, so n_runs=100 is ~3.3x the wall-clock of "
            "the default here)."
        ),
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
    base_seed: int = 1000,
    collect_actions: bool = False,
) -> dict:
    """
    Evaluate one agent on one case study across n_runs repetitions.

    Each run resets with the same fixed date but different stochastic
    participation draws — captures natural variability in EV owner responses.

    Paired evaluation via base_seed
    ---------------------------------
    Run i uses seed = base_seed + i, identical across every agent this
    function is called for. This means run i's hub-state initialisation,
    DOE noise realisation, and EV participation draws are IDENTICAL
    whether evaluating SAC-GNN, SAC-GCN, SAC-Flat, Greedy, RulePrice, or
    OracleMPC — the only thing that differs between agents is the policy
    itself, not the random conditions it's evaluated under.

    This enables a paired statistical test (paired t-test / Wilcoxon
    signed-rank) between any two agents' per-run profit arrays, which is
    considerably more statistically powerful than an unpaired test for
    the same n_runs — and, unlike the previous unseeded behaviour, makes
    results independent of the arbitrary order agents happen to be
    evaluated in (a reproducibility requirement, not just a nice-to-have).

    Returns dict with mean ± std across n_runs for all metrics, plus the
    raw per-run profit array (needed for paired significance testing in
    print_results_table()).

    Explainability data (collect_actions=True)
    ---------------------------------------------
    When enabled, also records per-step (normalised_action, doe_tightness)
    pairs for every hub, every step, across all n_runs — used by
    make_action_distribution_plot() to reproduce the action-diversity
    analysis in Orfanoudakis et al. 2025 (Communications Engineering),
    Fig. 3b: does the agent's dispatch policy vary meaningfully with
    DOE constraint tightness, or does it collapse to a small set of
    fixed actions regardless of state (a sign of a degenerate policy)?
    Off by default since it roughly doubles memory use per run and is
    only needed for the qualitative explainability figure, not the
    quantitative Table 3 metrics.
    """
    profits          = []
    doe_violations   = []
    participation    = []
    doe_compliant_steps = []
    total_steps_list = []
    inference_times_ms = []   # per-step wall-clock time for select_action()
    action_records = [] if collect_actions else None

    date = case_study["date"]

    for run in range(n_runs):
        # base_seed + run: identical seed for this run index across every
        # agent evaluated on this case study — see docstring above.
        obs, _ = env.reset(seed=base_seed + run, options={"date": date})
        done        = False
        ep_profit   = 0.0
        ep_doe_viol = 0.0
        ep_rho      = 0.0
        n_steps     = 0
        n_zero_doe  = 0

        while not done:
            t0 = time.perf_counter()
            if needs_env:
                action = agent.select_action(obs, env)
            else:
                action = agent.select_action(obs, deterministic=True)
            inference_times_ms.append((time.perf_counter() - t0) * 1000.0)

            obs, reward, done, _, info = env.step(action)
            ep_profit   += reward
            ep_doe_viol += sum(info.get("doe_violations_kw", [0.0]))
            ep_rho      += info.get("rho_hat", 0.0)
            n_steps     += 1
            if sum(info.get("doe_violations_kw", [0.0])) == 0:
                n_zero_doe += 1

            if collect_actions:
                # Per-hub: normalised dispatch fraction in [-1, 1]
                # (dispatch_kw / equipment_cap_kw) and DOE tightness
                # in [0, 1] (1 = fully constrained, 0 = full headroom).
                # node feature layout: [0]=doe_import_norm,
                # [1]=doe_export_norm, [5]=equipment_cap_kw (see
                # nem_doe_env.py _build_observation).
                node_feats = env.obs_to_node_features(obs)
                equipment_caps = node_feats[:, 5]
                doe_import_norm = node_feats[:, 0]
                doe_export_norm = node_feats[:, 1]
                # Tightness: how little DOE headroom remains relative to
                # equipment cap, averaged over both directions.
                doe_headroom_frac = np.clip(
                    (doe_import_norm + doe_export_norm) / 2.0, 0.0, 1.0
                )
                doe_tightness = 1.0 - doe_headroom_frac  # (H,)

                dispatch_kw = np.asarray(action[:env.n_hubs], dtype=np.float64)
                safe_caps = np.where(equipment_caps > 1e-6, equipment_caps, 1.0)
                norm_action = np.clip(dispatch_kw / safe_caps, -1.0, 1.0)

                for h in range(env.n_hubs):
                    action_records.append((float(norm_action[h]), float(doe_tightness[h])))

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
        "is_stress_test":     case_study.get("is_stress_test", False),
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
        # Inference latency — mean/p95 wall-clock ms per select_action()
        # call, over ALL steps across all n_runs episodes for this
        # (agent, case_study) pair. Reported because for a real-time
        # dispatch system this is directly relevant to deployability —
        # a GAT forward pass, a GCN forward pass, an MLP forward pass,
        # and MPC's per-step grid search all have very different cost
        # profiles that a profit/compliance comparison alone doesn't
        # capture (cf. Orfanoudakis et al. 2025, Communications
        # Engineering, who report MPC taking ~5 min/step vs <1s for RL
        # as a headline result for real-time applicability).
        "mean_inference_ms": float(np.mean(inference_times_ms)) if inference_times_ms else None,
        "p95_inference_ms":  float(np.percentile(inference_times_ms, 95)) if inference_times_ms else None,
        # Raw runs for distribution plots
        "profits_per_run":    profits,
        "action_records":     action_records,   # None unless collect_actions=True
    }


# ---------------------------------------------------------------------------
# Statistical significance testing
# ---------------------------------------------------------------------------

def compute_significance_tests(
    all_results: list,
    primary_agent: str = "SAC-GNN",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Paired statistical significance test between primary_agent and every
    other agent, per normal-day case study (stress-test day excluded —
    consistent with the mean_profit reporting elsewhere in this module).

    Why paired, and why this is now valid
    ----------------------------------------
    evaluate_case_study() now seeds env.reset() with
    seed=base_seed+run_index, IDENTICAL across every agent for a given
    (case_study, run) pair — see that function's docstring. This means
    profits_per_run[i] for two different agents on the same case study
    were generated under IDENTICAL hub-state, DOE, and participation
    conditions; only the policy differs. This is exactly the condition
    a paired test requires, and gives substantially more statistical
    power than an unpaired test for the same n_runs.

    Wilcoxon signed-rank (primary) vs paired t-test (reported alongside)
    -----------------------------------------------------------------------
    Wilcoxon is used as the primary test since it doesn't assume
    normally-distributed profit differences — a reasonable concern given
    the skewed, occasionally extreme profit distributions observed in
    this domain (see the stress-test-day discussion elsewhere in this
    file). The paired t-test is reported alongside for readers who
    prefer it / for comparison, but Wilcoxon's p-value is what should be
    quoted as the primary significance claim in the paper.

    Returns
    -------
    pd.DataFrame with columns:
        case_study, agent_a, agent_b, n_runs,
        mean_diff (agent_a - agent_b),
        wilcoxon_stat, wilcoxon_p, wilcoxon_significant,
        ttest_stat, ttest_p, ttest_significant
    One row per (case_study, comparison_agent) pair, comparing
    primary_agent against every other agent. Stress-test case studies
    are excluded (see is_stress_test flag on each result).
    """
    rows = []

    # Index results by (agent, case_study_name) -> profits_per_run array
    by_key = {}
    for r in all_results:
        by_key[(r["agent"], r["case_study_name"])] = r

    agents = sorted(set(r["agent"] for r in all_results))
    case_studies = sorted(set(
        r["case_study_name"] for r in all_results
        if not r.get("is_stress_test", False)
    ))

    if primary_agent not in agents:
        logger.warning(
            f"compute_significance_tests: primary_agent='{primary_agent}' "
            f"not found in results (available: {agents}) — skipping."
        )
        return pd.DataFrame()

    for cs_name in case_studies:
        primary_key = (primary_agent, cs_name)
        if primary_key not in by_key:
            continue
        primary_profits = np.array(by_key[primary_key]["profits_per_run"])

        for agent in agents:
            if agent == primary_agent:
                continue
            key = (agent, cs_name)
            if key not in by_key:
                continue
            other_profits = np.array(by_key[key]["profits_per_run"])

            if len(primary_profits) != len(other_profits):
                logger.warning(
                    f"Skipping significance test for {primary_agent} vs "
                    f"{agent} on {cs_name}: mismatched n_runs "
                    f"({len(primary_profits)} vs {len(other_profits)})"
                )
                continue

            diff = primary_profits - other_profits
            mean_diff = float(np.mean(diff))

            # Wilcoxon signed-rank requires at least one non-zero difference
            if np.all(diff == 0):
                wilcoxon_stat, wilcoxon_p = np.nan, 1.0
            else:
                try:
                    wilcoxon_stat, wilcoxon_p = stats.wilcoxon(diff)
                except ValueError:
                    # e.g. all differences identical in magnitude/sign in
                    # a way scipy can't rank — fall back to reporting NaN
                    wilcoxon_stat, wilcoxon_p = np.nan, np.nan

            ttest_stat, ttest_p = stats.ttest_rel(primary_profits, other_profits)

            rows.append({
                "case_study":            cs_name,
                "agent_a":               primary_agent,
                "agent_b":               agent,
                "n_runs":                len(diff),
                "mean_diff":             mean_diff,
                "mean_a":                float(np.mean(primary_profits)),
                "mean_b":                float(np.mean(other_profits)),
                "wilcoxon_stat":         float(wilcoxon_stat) if not np.isnan(wilcoxon_stat) else None,
                "wilcoxon_p":            float(wilcoxon_p),
                "wilcoxon_significant":  bool(wilcoxon_p < alpha),
                "ttest_stat":            float(ttest_stat),
                "ttest_p":               float(ttest_p),
                "ttest_significant":     bool(ttest_p < alpha),
            })

    return pd.DataFrame(rows)


def print_significance_summary(sig_df: pd.DataFrame, alpha: float = 0.05) -> str:
    """
    Formatted summary of compute_significance_tests() output — one line
    per (case_study, comparison agent), Wilcoxon p-value as primary,
    marked significant/not at the given alpha.
    """
    if sig_df.empty:
        return "\n(No significance tests computed — see log for why.)\n"

    lines = []
    lines.append("\n" + "=" * 100)
    lines.append(f"STATISTICAL SIGNIFICANCE — paired Wilcoxon signed-rank test (α={alpha})")
    lines.append("Paired via identical per-run seeds across agents (see evaluate_case_study docstring)")
    lines.append("=" * 100)
    lines.append(
        f"{'Case study':<20}{'Comparison':<28}{'Mean diff':>12}"
        f"{'Wilcoxon p':>14}{'Sig.':>8}{'Paired-t p':>14}{'Sig.':>8}"
    )
    lines.append("-" * 100)

    for _, row in sig_df.iterrows():
        comparison = f"{row['agent_a']} vs {row['agent_b']}"
        wp = row['wilcoxon_p']
        tp = row['ttest_p']
        wsig = "*" if row['wilcoxon_significant'] else ""
        tsig = "*" if row['ttest_significant'] else ""
        lines.append(
            f"{row['case_study']:<20}{comparison:<28}{row['mean_diff']:>+12.1f}"
            f"{wp:>14.4f}{wsig:>8}{tp:>14.4f}{tsig:>8}"
        )

    lines.append("-" * 100)
    lines.append("* = statistically significant at α=0.05")
    lines.append("=" * 100)

    table_str = "\n".join(lines)
    print(table_str)
    return table_str


# ---------------------------------------------------------------------------
# Results formatting
# ---------------------------------------------------------------------------

def print_results_table(results_df: pd.DataFrame) -> str:
    """
    Print and return formatted thesis Table 3.

    Format: agents as rows, case studies as columns, showing mean±std profit.

    Normal-day Mean vs Stress Test
    -------------------------------
    The "Normal-day Mean" column averages ONLY the case studies flagged
    is_stress_test=False (currently 4 of 5: summer peak, high volatility,
    winter average, weekend low). The negative-RRP case study
    (2024-02-13) is deliberately the most extreme day in the dataset —
    its profit magnitude runs 20-500x larger than the other 4 days
    (e.g. observed: SAC-GNN seed1 $120,490 vs $4,000-8,000 on other
    days; SAC-GCN seed1 -$539,375 vs $5,000-8,000 on other days).

    Pooling it into a single mean with the other 4 days means that one
    day's outcome — not the agent's typical performance — determines
    the headline number, and can flip the ranking between agents
    depending on which direction that one extreme day happened to go.
    It is reported in its own "Stress Test" section instead, so a
    reader can see typical-day performance and worst-case behaviour
    as two distinct, honestly-labelled numbers rather than one
    misleading blend of both.
    """
    agents      = results_df["agent"].unique()
    case_studies = results_df["case_study_name"].unique()

    # Determine which case studies are "normal" vs "stress test" from the data
    cs_stress_flag = {}
    for cs_name in case_studies:
        rows = results_df[results_df["case_study_name"] == cs_name]
        cs_stress_flag[cs_name] = bool(rows["is_stress_test"].iloc[0]) if "is_stress_test" in rows.columns and len(rows) > 0 else False

    normal_cs = [cs for cs in case_studies if not cs_stress_flag[cs]]
    stress_cs = [cs for cs in case_studies if cs_stress_flag[cs]]

    lines = []
    lines.append("\n" + "=" * 100)
    lines.append("THESIS TABLE 3 — SAC-GNN vs Baselines: Net Profit ($/day), Mean ± Std")
    lines.append("=" * 100)

    # Header — normal case studies only, plus Normal-day Mean column
    header = f"{'Agent':<14}" + "".join(f"{cs:>18}" for cs in normal_cs)
    header += f"{'Normal-day Mean':>18}"
    lines.append(header)
    lines.append("-" * 100)

    for agent in agents:
        agent_df   = results_df[results_df["agent"] == agent]
        row        = f"{agent:<14}"
        normal_profits = []
        for cs_name in normal_cs:
            cs_row = agent_df[agent_df["case_study_name"] == cs_name]
            if len(cs_row) > 0:
                mean = cs_row["mean_profit"].values[0]
                std  = cs_row["std_profit"].values[0]
                row += f"  {mean:+8.0f}±{std:5.0f}"
                normal_profits.append(mean)
            else:
                row += f"{'N/A':>18}"
        if normal_profits:
            row += f"  {np.mean(normal_profits):+16.0f}"
        lines.append(row)

    # Separate section for stress-test case studies — never pooled into
    # the headline mean above
    if stress_cs:
        lines.append("=" * 100)
        lines.append("STRESS TEST — extreme-condition day(s), reported separately")
        lines.append("(deliberately excluded from Normal-day Mean above — see")
        lines.append(" CASE_STUDIES[...]['is_stress_test'] docstring for rationale)")
        lines.append("-" * 100)
        header2 = f"{'Agent':<14}" + "".join(f"{cs:>22}" for cs in stress_cs)
        lines.append(header2)
        for agent in agents:
            agent_df = results_df[results_df["agent"] == agent]
            row = f"{agent:<14}"
            for cs_name in stress_cs:
                cs_row = agent_df[agent_df["case_study_name"] == cs_name]
                if len(cs_row) > 0:
                    mean = cs_row["mean_profit"].values[0]
                    std  = cs_row["std_profit"].values[0]
                    row += f"  {mean:+10.0f}±{std:8.0f}"
                else:
                    row += f"{'N/A':>22}"
            lines.append(row)

    lines.append("=" * 100)
    lines.append("DOE Compliance Rate (% steps with zero violation) — all case studies")
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
    lines.append("Mean Participation Rate ρ — all case studies")
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
    lines.append("Inference Latency — mean / p95 wall-clock ms per select_action() call")
    lines.append("(averaged across all case studies and runs; relevant to real-time")
    lines.append(" deployability — see e.g. Orfanoudakis et al. 2025, Comms Eng, who")
    lines.append(" report MPC at ~5 min/step vs <1s for RL as a headline result)")
    lines.append("-" * 100)
    lines.append(f"{'Agent':<14}{'Mean (ms)':>14}{'p95 (ms)':>14}")
    for agent in agents:
        agent_df = results_df[results_df["agent"] == agent]
        mean_lat_col = agent_df["mean_inference_ms"].dropna()
        p95_lat_col  = agent_df["p95_inference_ms"].dropna()
        if len(mean_lat_col) > 0:
            mean_lat = mean_lat_col.mean()
            p95_lat  = p95_lat_col.mean()
            lines.append(f"{agent:<14}{mean_lat:>14.3f}{p95_lat:>14.3f}")
        else:
            lines.append(f"{agent:<14}{'N/A':>14}{'N/A':>14}")

    lines.append("=" * 100)
    table_str = "\n".join(lines)
    print(table_str)
    return table_str


def make_plots(results_df: pd.DataFrame, output_dir: Path) -> None:
    """
    Generate comparison bar charts for the three primary metrics.

    Two separate figures are produced rather than one:
      1. metrics_comparison.png — normal-day case studies only (profit
         panel uses a shared, readable y-axis scale)
      2. stress_test_comparison.png — the negative-RRP stress-test day,
         plotted on its own scale since its magnitude (tens/hundreds of
         thousands of $) would otherwise compress the normal-day bars
         to near-invisible slivers on a shared axis.

    This mirrors the print_results_table() split — see that function's
    docstring for the full rationale on why the stress-test day is
    never pooled with the other 4.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available — skipping plots")
        return

    agents       = list(results_df["agent"].unique())
    all_case_studies = list(results_df["case_study_name"].unique())
    n_agents     = len(agents)

    # Split case studies by stress-test flag
    cs_stress_flag = {}
    for cs_name in all_case_studies:
        rows = results_df[results_df["case_study_name"] == cs_name]
        cs_stress_flag[cs_name] = bool(rows["is_stress_test"].iloc[0]) if "is_stress_test" in rows.columns and len(rows) > 0 else False

    normal_case_studies = [cs for cs in all_case_studies if not cs_stress_flag[cs]]
    stress_case_studies = [cs for cs in all_case_studies if cs_stress_flag[cs]]

    colors = ["#2a78d6", "#1baf7a", "#eda100", "#e34948", "#9b59b6", "#00bcd4"]

    def _plot_metric_panels(case_studies, title_suffix, filename):
        n_cs = len(case_studies)
        if n_cs == 0:
            return
        x = np.arange(n_cs)
        width = 0.8 / n_agents

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(
            f"SAC-GNN V2G Hub — Case Study Evaluation{title_suffix}\n"
            "Inner Melbourne 21 Hubs, VIC1 2024 Held-Out Prices",
            fontsize=12,
        )

        for col, (metric, ylabel, panel_title) in enumerate([
            ("mean_profit",        "Net profit ($/day)",  "Arbitrage Net Profit"),
            ("mean_doe_compliance","DOE compliance rate",  "DOE Compliance Rate"),
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
                ax.bar(
                    x + offset, vals, width,
                    yerr=errs, capsize=3,
                    label=agent,
                    color=colors[j % len(colors)],
                    edgecolor="white", linewidth=0.5,
                )

            ax.set_title(panel_title, fontsize=11)
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
        out_path = output_dir / filename
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Plot saved: {out_path}")

    # Normal-day comparison — shared readable scale across 4 typical days
    _plot_metric_panels(
        normal_case_studies,
        " (Normal-day Case Studies)",
        "metrics_comparison.png",
    )

    # Stress-test day — plotted separately since its $ magnitude is
    # 20-500x the normal days and would otherwise flatten them to
    # invisible slivers on a shared axis (see docstring above)
    _plot_metric_panels(
        stress_case_studies,
        " (Stress Test — Extreme Negative RRP)",
        "stress_test_comparison.png",
    )


def make_action_distribution_plot(
    action_records_by_agent: dict,
    output_dir: Path,
    moderate_range: tuple = (-0.6, 0.6),
) -> None:
    """
    Action-diversity explainability figure — reproduces the analysis in
    Orfanoudakis et al. 2025 (Communications Engineering), Fig. 3b: does
    the agent's dispatch policy vary meaningfully with the state (here,
    DOE constraint tightness), or does it collapse toward a small set
    of fixed actions regardless of context (a sign of an under-expressive
    or degenerate policy)?

    For each agent, produces 3 sub-panels — one per DOE-tightness
    tertile (loose / medium / tight headroom) — each showing the
    normalised-dispatch-action ([-1,1], dispatch_kw / equipment_cap_kw)
    probability density, with P(moderate_range) annotated (the fraction
    of actions falling in a "moderate, non-extreme" range — analogous
    to the original paper's P(0.2<=x<=0.8) metric on their [0,1] action
    space, translated to our signed [-1,1] range).

    A policy that ignores DOE tightness would show near-identical
    density shapes across all 3 tertiles. A policy that has learned to
    respond to constraint tightness should show visibly different
    shapes — e.g. more mass near 0 (conservative dispatch) under tight
    DOE headroom, more spread under loose headroom.

    Parameters
    ----------
    action_records_by_agent : dict
        {agent_name: [(norm_action, doe_tightness), ...], ...} — from
        evaluate_case_study(collect_actions=True)'s "action_records".
        Typically pooled across all case studies and runs for each agent
        before calling this function.
    output_dir : Path
    moderate_range : tuple
        (low, high) bounds for the P(moderate_range) annotation.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available — skipping action distribution plot")
        return

    agents = [a for a, recs in action_records_by_agent.items() if recs]
    if not agents:
        logger.warning("No action_records available — skipping action distribution plot")
        return

    tertile_labels = ["Loose DOE headroom\n(0-33% tight)",
                       "Medium DOE headroom\n(34-66% tight)",
                       "Tight DOE headroom\n(67-100% tight)"]
    colors = ["#2a78d6", "#1baf7a", "#eda100", "#e34948", "#9b59b6", "#00bcd4"]

    n_agents = len(agents)
    fig, axes = plt.subplots(n_agents, 3, figsize=(15, 4 * n_agents), squeeze=False)
    fig.suptitle(
        "Action-Diversity Explainability — Normalised Dispatch vs DOE Tightness\n"
        "(cf. Orfanoudakis et al. 2025, Communications Engineering, Fig. 3b)",
        fontsize=12,
    )

    lo, hi = moderate_range

    for row, agent in enumerate(agents):
        recs = np.array(action_records_by_agent[agent])   # (N, 2): [action, tightness]
        actions = recs[:, 0]
        tightness = recs[:, 1]

        # Tertile split by DOE tightness
        edges = np.quantile(tightness, [0.0, 1/3, 2/3, 1.0])
        for col in range(3):
            ax = axes[row][col]
            mask = (tightness >= edges[col]) & (tightness <= edges[col + 1])
            vals = actions[mask]

            if len(vals) > 5:
                ax.hist(
                    vals, bins=30, range=(-1, 1), density=True,
                    color=colors[row % len(colors)], alpha=0.7, edgecolor="white",
                )
                p_moderate = float(np.mean((vals >= lo) & (vals <= hi)))
                ax.text(
                    0.05, 0.92, f"P({lo:.1f}≤x≤{hi:.1f})={p_moderate:.2f}",
                    transform=ax.transAxes, fontsize=9,
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                )
            else:
                ax.text(0.5, 0.5, "insufficient data", ha="center", va="center",
                         transform=ax.transAxes, fontsize=9, color="gray")

            ax.set_xlim(-1, 1)
            ax.axvline(0, color="black", linewidth=0.5, linestyle="--")
            if row == 0:
                ax.set_title(tertile_labels[col], fontsize=10)
            if col == 0:
                ax.set_ylabel(f"{agent}\nDensity", fontsize=10)
            if row == n_agents - 1:
                ax.set_xlabel("Normalised dispatch action\n(-1=full charge, +1=full discharge)", fontsize=9)

    plt.tight_layout()
    out_path = output_dir / "action_distribution_explainability.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
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

    # Collect per-step (action, doe_tightness) pairs only for the trained
    # RL agents — the ones whose policy is a black box worth explaining.
    # The heuristics (Greedy, RulePrice) are rule-based by construction
    # (their action-vs-tightness relationship is already known from the
    # code), and OracleMPC is an analytical optimiser — collecting for
    # all 6 would work but adds memory/compute for agents where the
    # explainability question isn't really in doubt.
    EXPLAINABILITY_AGENTS = {"SAC-GNN", "SAC-GCN", "SAC-Flat"}
    action_records_by_agent = {a: [] for a in EXPLAINABILITY_AGENTS}

    for agent_name, agent, needs_env in agents:
        logger.info(f"\n── Evaluating {agent_name} ──────────────────────────")
        if hasattr(agent, "reset"):
            agent.reset()

        collect_actions = agent_name in EXPLAINABILITY_AGENTS

        for cs_idx, cs in enumerate(CASE_STUDIES):
            done_count += 1
            logger.info(
                f"  [{done_count}/{total}] {cs['name']} ({cs['date']}) "
                f"× {args.n_runs} runs..."
            )

            # base_seed is identical across every agent for a given case
            # study (so run i is a paired comparison — same hub-state/
            # DOE/participation draw for every agent), but differs
            # between case studies to avoid seed-pattern reuse.
            # See evaluate_case_study() docstring for full rationale.
            base_seed = args.seed * 1000 + cs_idx * 100

            result = evaluate_case_study(
                agent_name=agent_name,
                agent=agent,
                needs_env=needs_env,
                env=env,
                case_study=cs,
                n_runs=args.n_runs,
                base_seed=base_seed,
                collect_actions=collect_actions,
            )

            if collect_actions and result.get("action_records"):
                action_records_by_agent[agent_name].extend(result["action_records"])

            logger.info(
                f"    profit={result['mean_profit']:+8.1f} ± {result['std_profit']:5.1f} | "
                f"doe={result['mean_doe_compliance']*100:.1f}% | "
                f"ρ={result['mean_participation']*100:.1f}%"
            )
            all_results.append(result)

    # Build results DataFrame (exclude per-run list for CSV)
    csv_results = []
    for r in all_results:
        row = {k: v for k, v in r.items() if k not in ("profits_per_run", "action_records")}
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

    # Statistical significance testing — paired Wilcoxon signed-rank test
    # between SAC-GNN (the primary proposed agent) and every other agent,
    # per normal-day case study. Valid as a PAIRED test because
    # evaluate_case_study() seeds every agent identically per run index
    # (see that function's docstring).
    sig_df = compute_significance_tests(
        all_results, primary_agent="SAC-GNN", alpha=0.05
    )
    if not sig_df.empty:
        sig_csv_path = output_dir / "significance_tests.csv"
        sig_df.to_csv(sig_csv_path, index=False)
        logger.info(f"Significance tests saved: {sig_csv_path}")

        sig_str = print_significance_summary(sig_df, alpha=0.05)
        sig_txt_path = output_dir / "significance_summary.txt"
        with open(sig_txt_path, "w") as f:
            f.write(sig_str)
        logger.info(f"Significance summary saved: {sig_txt_path}")

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

    # Action-diversity explainability figure (cf. Orfanoudakis et al.
    # 2025, Fig. 3b) — pooled across all case studies and runs, per
    # trained RL agent.
    make_action_distribution_plot(action_records_by_agent, output_dir)

    logger.info("\n" + "=" * 60)
    logger.info("Evaluation complete.")
    logger.info(f"Results: {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
