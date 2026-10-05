"""
benchmark_train_speed.py
========================
Measure SAC training throughput (ms per environment step, including one
gradient update per step) for one agent on the current device.

Used to size GPU jobs: run it alone, then as several concurrent processes on
the same GPU (slurm/benchmark_train_speed.sh) to see whether packing multiple
training runs into one GPU job increases total throughput.

Usage:
    python benchmark_train_speed.py --agent sac_gnn --steps 300 [--zone greater_melbourne]
"""

import argparse
import time

import numpy as np
import torch

from train_sac_gnn import make_env, DEFAULT_CONFIG
from baselines.gnn_rl.agent import SACGNNAgent
from baselines.gnn_rl.sac_gcn import SACGCNAgent
from baselines.flat_mlp.sac_flat import SACFlatAgent
from baselines.gnn_rl.networks import NetworkConfig


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--agent", default="sac_gnn", choices=["sac_gnn", "sac_gcn", "sac_flat"])
    p.add_argument("--steps", type=int, default=300, help="timed steps (each with an update)")
    p.add_argument("--zone", default=None)
    p.add_argument("--threads", type=int, default=2)
    args = p.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    cfg = dict(DEFAULT_CONFIG)
    if args.zone:
        cfg["graph_path"] = f"data/graphs/{args.zone}.pkl"
    warmup = cfg["batch_size"] + 10
    env, graph, hubs = make_env(cfg, "train", 0)
    H = len(hubs)
    common = dict(gamma=cfg["gamma"], tau=cfg["tau"], lr_actor=cfg["lr_actor"],
                  lr_critic=cfg["lr_critic"], lr_alpha=cfg["lr_alpha"],
                  batch_size=cfg["batch_size"], buffer_capacity=50_000,
                  learning_starts=warmup, update_every=1, seed=0)
    if args.agent == "sac_flat":
        agent = SACFlatAgent(obs_dim=H * 9, action_dim=H + 1, **common)
    else:
        cls = SACGCNAgent if args.agent == "sac_gcn" else SACGNNAgent
        agent = cls(n_hubs=H, graph_data=graph, obs_dim=H * 9, net_cfg=NetworkConfig(), **common)

    obs, _ = env.reset(options={"episode": 1, "total_episodes": 1500})

    def run(n):
        nonlocal obs
        for _ in range(n):
            a = agent.select_action(obs)
            nobs, r, done, _, _ = env.step(a)
            agent.store_transition(obs, a, r, nobs, done)
            agent.update()
            obs = nobs if not done else env.reset(options={"episode": 1, "total_episodes": 1500})[0]

    run(warmup + 20)                                   # fill buffer + warm up kernels
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    run(args.steps)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / args.steps * 1000
    hours = ms * 288 * 1500 / 1000 / 3600
    print(f"RESULT agent={args.agent} hubs={H} device={agent.device} "
          f"ms_per_step={ms:.1f} est_hours_1500ep={hours:.1f}", flush=True)


if __name__ == "__main__":
    main()
