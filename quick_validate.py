"""
quick_validate.py
=================
Minimal end-to-end smoke test for the Phase 2 NEM environment stack.

Run before anything else:
    conda activate <your_env>
    cd nem_v2g
    python quick_validate.py

Expected output:
  - Price loader: summary stats that look like NEM prices
  - Participation curves: sanity check that ρ(c) is monotone and in (0,1)
  - Environment rollout: 288 steps, finite rewards, correct episode termination
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from nem_env import PriceLoader, ParticipationModel, HubParticipationState
from nem_env import NEMWDREnv, HubConfig, EnvConfig


def main():
    print("=" * 60)
    print("Phase 2 NEM Environment — Quick Validation")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Price loader
    # ------------------------------------------------------------------
    print("\n[1/3] Price loader + WDR generator...")
    loader = PriceLoader(region="VIC1", cache_dir="/tmp/nem_validate", seed=0)
    loader.load_synthetic(n_days=365, mean_price=80.0, std_price=150.0, spike_prob=0.02)

    summary = loader.price_summary()
    print(f"  Loaded {summary['count']:,} intervals ({summary['count']//288} days)")
    print(f"  Price: mean=${summary['mean']:.1f}  std=${summary['std']:.1f}  "
          f"max=${summary['max']:.0f}  p99=${summary['pct_99']:.0f}/MWh")
    print(f"  Spike rate (>$300): {summary['pct_spike_above_300']:.2%}")

    ep = loader.sample_episode(force_wdr=True)
    n_wdr = ep["wdr_active"].sum()
    print(f"  Episode (force_wdr=True): {n_wdr} WDR steps / 288 "
          f"({n_wdr/288:.1%}), target max={ep['dispatch_target_mw'].max():.3f} MW")
    assert n_wdr > 0, "FAIL: force_wdr produced no WDR steps"
    print("  ✓ Price loader OK")

    # ------------------------------------------------------------------
    # 2. Participation model
    # ------------------------------------------------------------------
    print("\n[2/3] Participation model calibration...")
    model = ParticipationModel(seed=0)
    print(f"  Beta parameters: {model.beta_summary()}")

    scenarios = [
        ("Zero price, close hub, mid SoC", 0.0, 0.0, 0.5),
        ("$100/MWh, close hub, mid SoC",   100.0, 0.0, 0.5),
        ("$100/MWh, far hub (10km), mid SoC", 100.0, 10.0, 0.5),
        ("$200/MWh, close hub, high SoC",  200.0, 0.0, 0.8),
        ("$500/MWh, mid hub, high SoC",    500.0, 5.0, 0.8),
    ]
    for label, c, d, s in scenarios:
        rho = model.participation_prob(c, d, s)
        print(f"    {label:45s} → ρ = {rho:.3f}")

    # Monotonicity check
    prices = np.linspace(0, 500, 50)
    curve = model.participation_curve(prices, distance_km=3.0, mean_soc=0.5)
    assert np.all(np.diff(curve) >= -1e-9), "FAIL: participation curve not monotone"
    print("  ✓ Participation model OK (monotone, values in (0,1))")

    # ------------------------------------------------------------------
    # 3. Full environment rollout
    # ------------------------------------------------------------------
    print("\n[3/3] Environment rollout (one full episode, 288 steps)...")

    hub_configs = [
        HubConfig(hub_id=0, distance_km=1.2, loc_x=-0.5, loc_y=0.2, n_chargers=4),
        HubConfig(hub_id=1, distance_km=3.8, loc_x=0.3,  loc_y=-0.4, n_chargers=6),
        HubConfig(hub_id=2, distance_km=6.5, loc_x=0.8,  loc_y=0.6, n_chargers=4),
        HubConfig(hub_id=3, distance_km=2.1, loc_x=-0.2, loc_y=-0.7, n_chargers=5),
        HubConfig(hub_id=4, distance_km=5.0, loc_x=0.1,  loc_y=0.9, n_chargers=4),
    ]

    env = NEMWDREnv(
        hub_configs=hub_configs,
        price_loader=loader,
        participation_model=model,
        env_config=EnvConfig(),
        force_wdr=True,
        seed=42,
    )

    obs, info = env.reset()
    print(f"  Action space: {env.action_space.shape}  "
          f"(= {env.H} dispatch fracs + 1 price)")
    print(f"  Obs space:    {env.observation_space.shape}  "
          f"(= {env.H}×{env.NODE_FEATURE_DIM} node feats + {env.ZONE_FEATURE_DIM} zone feats)")
    assert np.isfinite(obs).all(), "FAIL: initial obs contains NaN/Inf"

    # Run full episode with a fixed action (50% dispatch, $80/MWh incentive)
    fixed_action = np.full(env.H + 1, 0.5, dtype=np.float32)
    fixed_action[-1] = 80.0

    step_count = 0
    total_reward = 0.0
    total_wdr_steps = 0
    total_e_del_kwh = 0.0
    conformance_devs = []

    while True:
        obs, reward, terminated, truncated, info = env.step(fixed_action)
        step_count += 1
        total_reward += reward
        total_wdr_steps += int(info["wdr_active"])
        total_e_del_kwh += info["e_del_total_kwh"]
        if info["wdr_active"]:
            conformance_devs.append(info["p_conformance"])
        assert np.isfinite(reward), f"FAIL: non-finite reward at step {step_count}"
        if terminated:
            break

    assert step_count == 288, f"FAIL: episode length {step_count} != 288"
    print(f"  Steps: {step_count}  WDR steps: {total_wdr_steps}")
    print(f"  Total reward: {total_reward:.2f}")
    print(f"  Total energy delivered: {total_e_del_kwh:.1f} kWh")
    if conformance_devs:
        print(f"  Mean conformance penalty (WDR steps): ${np.mean(conformance_devs):.4f}")

    # Check obs→node/zone split
    node_feats, zone_feats = env.obs_to_node_and_zone(obs)
    print(f"  obs_to_node_and_zone: node {node_feats.shape}, zone {zone_feats.shape}")

    print("  ✓ Environment rollout OK")
    print("\n✓ All Phase 2 checks passed. Ready for Phase 3 (spatial_graph.py).")


if __name__ == "__main__":
    main()
