"""
verify_data.py
==============
Visual and statistical checks for:
1. AEMO price loader — synthetic prices look like real NEM prices
2. Participation model — β parameters produce sensible elasticity curves

Run with: python verify_data.py
Produces plots saved to data/verification/
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from nem_env.aemo_price_loader import PriceLoader
from nem_env.participation_model import ParticipationModel

Path("data/verification").mkdir(parents=True, exist_ok=True)

# ── 1. Price loader checks ───────────────────────────────────────────

print("=" * 50)
print("1. PRICE LOADER")
print("=" * 50)

# loader = PriceLoader(seed=0)
# loader.load_synthetic(
    # n_days=365,
    # mean_price=100.0,
    # std_price=250.0,
    # spike_prob=0.003,
    # spike_magnitude=2000.0,)

loader = PriceLoader(region='VIC1', cache_dir='data/nem_cache')
loader.load_cache('data/nem_cache/VIC1_2022-01-01_2024-12-31.parquet')

summary = loader.price_summary()

print(f"  Intervals loaded : {summary['count']:,}")
print(f"  Mean price       : ${summary['mean']:.2f}/MWh")
print(f"  Std price        : ${summary['std']:.2f}/MWh")
print(f"  Min price        : ${summary['min']:.2f}/MWh")
print(f"  Max price        : ${summary['max']:.2f}/MWh")
print(f"  99th percentile  : ${summary['pct_99']:.2f}/MWh")
print(f"  Spike rate >$300 : {summary['pct_spike_above_300']:.2%}")

# Real NEM VIC1 benchmarks (2022-2024 averages):
print("\n  Real NEM VIC1 benchmarks:")
print("  Mean ~$80-120/MWh, Std ~$200-400/MWh, Spike rate ~2-5%")

# Sample 5 episodes and check WDR events
print("\n  Episode WDR checks (force_wdr=True, 5 samples):")
for i in range(5):
    ep = loader.sample_episode(force_wdr=True)
    n_wdr = ep["wdr_active"].sum()
    max_target = ep["dispatch_target_mw"].max()
    print(f"    Episode {i+1}: {n_wdr} WDR steps "
          f"({n_wdr/288:.1%}), max target={max_target:.3f} MW")

# Plot price distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

ep = loader.sample_episode(force_wdr=True)
axes[0].plot(ep.index if hasattr(ep.index, 'hour') else range(288),
             ep["spot_price"], linewidth=0.8, color="steelblue")
axes[0].axhspan(0, 300, alpha=0.1, color="green", label="Normal (<$300)")
axes[0].fill_between(range(288),
                     [300]*288, ep["spot_price"],
                     where=ep["spot_price"] > 300,
                     alpha=0.3, color="red", label="Spike (>$300)")
wdr_steps = ep["wdr_active"]
axes[0].fill_between(range(288), 0, ep["spot_price"].max(),
                     where=wdr_steps, alpha=0.2,
                     color="orange", label="WDR active")
axes[0].set_title("Sample episode: spot price + WDR windows")
axes[0].set_xlabel("5-minute interval")
axes[0].set_ylabel("Spot price ($/MWh)")
axes[0].legend(fontsize=8)

prices = loader._price_df["spot_price"]
axes[1].hist(prices.clip(-100, 500), bins=80,
             color="steelblue", edgecolor="white", linewidth=0.3)
axes[1].set_title("Price distribution (clipped at $500 for visibility)")
axes[1].set_xlabel("Spot price ($/MWh)")
axes[1].set_ylabel("Count")
axes[1].axvline(prices.mean(), color="red",
                linestyle="--", label=f"Mean=${prices.mean():.0f}")
axes[1].legend()

plt.tight_layout()
plt.savefig("data/verification/price_distribution.png", dpi=150)
print("\n  Plot saved: data/verification/price_distribution.png")

# ── 2. Participation model checks ───────────────────────────────────

print("\n" + "=" * 50)
print("2. PARTICIPATION MODEL")
print("=" * 50)

model = ParticipationModel(seed=0)
print(f"\n  Beta parameters: {model.beta_summary()}")

# Key scenario checks
scenarios = [
    ("Zero price, zero distance, mid SoC",   0.0,   0.0,  0.5),
    ("$50/MWh,  zero distance, mid SoC",    50.0,   0.0,  0.5),
    ("$100/MWh, zero distance, mid SoC",   100.0,   0.0,  0.5),
    ("$200/MWh, zero distance, mid SoC",   200.0,   0.0,  0.5),
    ("$100/MWh, 5km distance,  mid SoC",   100.0,   5.0,  0.5),
    ("$100/MWh, 10km distance, mid SoC",   100.0,  10.0,  0.5),
    ("$100/MWh, zero distance, low SoC",   100.0,   0.0,  0.2),
    ("$100/MWh, zero distance, high SoC",  100.0,   0.0,  0.8),
]
print("\n  Participation probability table:")
print(f"  {'Scenario':<45} ρ")
print("  " + "-" * 52)
for label, c, d, s in scenarios:
    rho = model.participation_prob(c, d, s)
    bar = "█" * int(rho * 20)
    print(f"  {label:<45} {rho:.3f} {bar}")

# Target check: at β₁=0.008, $100/MWh should give ρ≈0.40
rho_100 = model.participation_prob(100.0, 0.0, 0.5)
print(f"\n  ρ at $100/MWh (target ≈ 0.35–0.45): {rho_100:.3f} "
      f"{'✓ OK' if 0.30 < rho_100 < 0.55 else '✗ CHECK BETAS'}")

# Plot participation elasticity curves
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
prices = np.linspace(0, 500, 200)

# Panel 1: price elasticity at different distances
for d in [0, 3, 6, 10]:
    curve = model.participation_curve(prices, distance_km=d, mean_soc=0.5)
    axes[0].plot(prices, curve, label=f"d={d}km", linewidth=2)
axes[0].set_title("ρ vs incentive price\n(mean_soc=0.5)")
axes[0].set_xlabel("Incentive price c_t ($/MWh)")
axes[0].set_ylabel("Participation probability ρ")
axes[0].legend()
axes[0].axhline(0.5, color="gray", linestyle="--", alpha=0.5)
axes[0].set_ylim(0, 1)

# Panel 2: price elasticity at different SoC levels
for s in [0.1, 0.3, 0.5, 0.7, 0.9]:
    curve = model.participation_curve(prices, distance_km=3.0, mean_soc=s)
    axes[1].plot(prices, curve, label=f"SoC={s}", linewidth=2)
axes[1].set_title("ρ vs incentive price\n(distance=3km)")
axes[1].set_xlabel("Incentive price c_t ($/MWh)")
axes[1].set_ylabel("Participation probability ρ")
axes[1].legend(fontsize=8)
axes[1].axhline(0.5, color="gray", linestyle="--", alpha=0.5)
axes[1].set_ylim(0, 1)

# Panel 3: Binomial sample distribution at $100/MWh
n_enrolled = 20
c_t = 100.0
rng = np.random.default_rng(0)
from nem_env.participation_model import HubParticipationState
hub_states = [HubParticipationState(
    hub_id=0, n_enrolled=n_enrolled, distance_km=3.0, mean_soc=0.5
)]
samples = [model.sample_responses(c_t, hub_states)[0] for _ in range(1000)]
axes[2].hist(samples, bins=range(0, n_enrolled+2),
             color="steelblue", edgecolor="white",
             density=True, align="left")
axes[2].set_title(f"n_respond distribution\n"
                  f"(n_enrolled={n_enrolled}, c=$100, d=3km, SoC=0.5)")
axes[2].set_xlabel("Number of responding owners")
axes[2].set_ylabel("Probability")
axes[2].axvline(np.mean(samples), color="red",
                linestyle="--", label=f"mean={np.mean(samples):.1f}")
axes[2].legend()

plt.tight_layout()
plt.savefig("data/verification/participation_curves.png", dpi=150)
print("\n  Plot saved: data/verification/participation_curves.png")

print("\nDone. Open data/verification/ to inspect the plots.")