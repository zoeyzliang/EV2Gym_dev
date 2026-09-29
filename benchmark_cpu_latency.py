#!/usr/bin/env python
"""
CPU inference latency benchmark for SAC-GNN, SAC-GCN, SAC-Flat.

Reuses evaluate.py's own environment construction, graph loading, and
agent loading (load_agents), so the CPU numbers this produces are directly
comparable to the GPU numbers already in Table~tab:main (1.313 / 1.060 /
0.290 ms mean). Timing methodology (perf_counter around select_action,
mean + p95) mirrors evaluate.py lines ~458-463 exactly.

Run with CUDA hidden so each agent's device selection (agent.py's
torch.device(...) logic) falls back to CPU:

    CUDA_VISIBLE_DEVICES="" python benchmark_cpu_latency.py \
        --sac_gnn_checkpoint  results/sac_gnn_21hub_seed42_20260904/checkpoints/best.pt \
        --sac_gcn_checkpoint  results/sac_gcn_21hub_seed42_20260904/checkpoints/best.pt \
        --sac_flat_checkpoint results/sac_flat_21hub_seed42_20260904/checkpoints/best.pt \
        --graph_path data/graphs/inner_melbourne.pkl \
        --seed 42 --n_warmup 50 --n_trials 500

This has NOT been executed against real checkpoints (no GPU/data available
in this environment) -- run it on M3 and check the output before citing it.
"""
import argparse
import time
import sys
import numpy as np
import torch

# CPU-inference workloads this small (single-graph, single-timestep
# forward pass) can be dominated by PyTorch's default intra-op thread-pool
# coordination overhead rather than actual compute, producing inflated and
# highly variable latency on CPU. Pinning to a single thread is a standard
# fix for this specific pattern; confirmed necessary here after an initial
# run showed p95 ~500ms with high variance even on a dedicated compute
# node (i.e. not explained by shared-node contention alone).
torch.set_num_threads(1)

# Reuse evaluate.py's own environment factory directly, rather than
# reconstructing NEMDOEEnv/PriceLoader/ParticipationModel construction by
# hand -- guessing at this pipeline from partial views has been wrong
# twice already in this project (the --zone/--graph_path mismatch, and an
# earlier incorrect guess at this exact NEMDOEEnv call). Importing the
# real function means this script tracks evaluate.py automatically.
from evaluate import make_eval_env, load_agents



def benchmark_agent(name, agent, needs_env, env, obs, n_warmup, n_trials):
    with torch.no_grad():
        for _ in range(n_warmup):
            if needs_env:
                agent.select_action(obs, env)
            else:
                agent.select_action(obs, deterministic=True)

        times_ms = []
        for _ in range(n_trials):
            t0 = time.perf_counter()
            if needs_env:
                agent.select_action(obs, env)
            else:
                agent.select_action(obs, deterministic=True)
            times_ms.append((time.perf_counter() - t0) * 1000.0)

    times_ms = np.array(times_ms)
    return {
        "mean_ms": float(times_ms.mean()),
        "p95_ms": float(np.percentile(times_ms, 95)),
        "std_ms": float(times_ms.std()),
        "n_trials": n_trials,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sac_gnn_checkpoint", type=str, default=None)
    parser.add_argument("--sac_gcn_checkpoint", type=str, default=None)
    parser.add_argument("--sac_flat_checkpoint", type=str, default=None)
    parser.add_argument("--graph_path", type=str, default="data/graphs/inner_melbourne.pkl")
    parser.add_argument("--cache_dir", type=str, default="data/nem_cache")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_warmup", type=int, default=50)
    parser.add_argument("--n_trials", type=int, default=500)
    args = parser.parse_args()

    assert not torch.cuda.is_available(), (
        "CUDA is visible to this process -- run with "
        "CUDA_VISIBLE_DEVICES=\"\" so agents fall back to CPU."
    )
    args.synthetic = False  # make_eval_env() checks args.synthetic

    env, graph, hub_configs = make_eval_env(args, args.seed)

    # load_agents() constructs each agent without an explicit device
    # argument, so it uses each class's own default device selection;
    # with CUDA hidden (checked above) that resolves to CPU for all three.
    agents = load_agents(args, env, graph, hub_configs)
    if not agents:
        raise SystemExit("No checkpoints found -- pass at least one of "
                          "--sac_gnn_checkpoint / --sac_gcn_checkpoint / "
                          "--sac_flat_checkpoint, and check the paths exist.")

    obs, _ = env.reset(seed=args.seed)

    print(f"{'Agent':<10}{'mean_ms':>10}{'p95_ms':>10}{'std_ms':>10}{'n':>8}")
    for name, agent, needs_env in agents:
        r = benchmark_agent(name, agent, needs_env, env, obs, args.n_warmup, args.n_trials)
        print(f"{name:<10}{r['mean_ms']:>10.3f}{r['p95_ms']:>10.3f}{r['std_ms']:>10.3f}{r['n_trials']:>8}")


if __name__ == "__main__":
    main()
