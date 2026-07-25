"""
aemo_price_loader.py
====================
NEM spot price loader for VIC1 historical dispatch prices.

Pulls historical 5-minute dispatch interval prices from NEMOSIS,
caches to Parquet, and serves 288-step (24-hour) episode DataFrames
for RL training.

Design decisions
----------------
- Episode length: 288 steps × 5 min = 24 hours, aligned to NEM trading day.
- The agent is a price-taker: it observes RRP each interval and sets its own
  signed dispatch targets. No WDR/dispatch target columns are generated —
  this loader returns spot_price only (master summary §2, Option A).
- Real AEMO VIC1 data only (2022–2024); no synthetic price generation for
  training (master summary §12 conventions).

Usage
-----
    loader = PriceLoader(region="VIC1", cache_dir="data/nem_cache")
    loader.fetch_and_cache(start="2022-01-01", end="2024-12-31")
    episode = loader.sample_episode()
    # episode: pd.DataFrame with columns [spot_price]
    # indexed by 5-min interval timestamps, length 288
"""

import os
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class PriceLoader:
    """
    Loads and caches historical NEM 5-minute spot prices from NEMOSIS,
    and assembles complete episode DataFrames for RL training.

    Parameters
    ----------
    region : str
        NEM region identifier. Default "VIC1".
    cache_dir : str or Path
        Directory for Parquet cache files.
    seed : int, optional
        RNG seed for reproducibility.

    Notes on NEMOSIS
    ----------------
    NEMOSIS (Gorman et al., 2018) fetches AEMO market data from MMS tables.
    The relevant table is DISPATCHPRICE, column RRP (regional reference price,
    $/MWh) for SETTLEMENTDATE at 5-minute resolution.

    Install: conda install -c conda-forge nemosis
    (or: pip install nemosis --break-system-packages if conda unavailable)

    The market price cap for 2025–26 is $20,300/MWh (AEMC, Feb 2025).
    Prices can go negative (market floor is -$1,000/MWh).
    """

    STEPS_PER_DAY = 288
    STEP_MINUTES = 5
    MARKET_PRICE_CAP = 20_300.0   # $/MWh, 2025–26 financial year
    MARKET_PRICE_FLOOR = -1_000.0

    def __init__(
        self,
        region: str = "VIC1",
        cache_dir: str = "data/nem_cache",
        seed: Optional[int] = None,
    ):
        self.region = region
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._rng = np.random.default_rng(seed)

        # Loaded price data; populated by fetch_and_cache() or load_cache()
        self._price_df: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Data acquisition
    # ------------------------------------------------------------------

    def fetch_and_cache(
        self,
        start: str,
        end: str,
        force_refresh: bool = False,
    ) -> None:
        """
        Download 5-minute dispatch prices from NEMOSIS and cache to Parquet.

        Parameters
        ----------
        start : str
            Start date, "YYYY-MM-DD".
        end : str
            End date inclusive, "YYYY-MM-DD".
        force_refresh : bool
            Re-download even if cache exists.
        """
        cache_path = self.cache_dir / f"{self.region}_{start}_{end}.parquet"

        if cache_path.exists() and not force_refresh:
            logger.info(f"Loading prices from cache: {cache_path}")
            self._price_df = pd.read_parquet(cache_path)
            return

        logger.info(f"Fetching NEM prices from NEMOSIS: {self.region} {start}→{end}")

        try:
            import nemosis
        except ImportError:
            raise ImportError(
                "NEMOSIS not installed. Run: conda install -c conda-forge nemosis\n"
                "or: pip install nemosis --break-system-packages"
            )

        # NEMOSIS expects datetime strings in the format it recognises.
        # DISPATCHPRICE table, RRP column = regional reference price $/MWh.
        # SETTLEMENTDATE is the end of the 5-minute dispatch interval.
        raw = nemosis.dynamic_data_compiler(
            # start_time=start + " 00:00:00",
            # end_time=end + " 23:55:00",
            start_time=start.replace("-", "/") + " 00:00:00",
            end_time=end.replace("-", "/") + " 23:55:00",
            table_name="DISPATCHPRICE",
            raw_data_location=str(self.cache_dir / "raw"),
            filter_cols=["REGIONID"],
            filter_values=([self.region],),
            select_columns=["SETTLEMENTDATE", "REGIONID", "RRP"],
        )

        # Clean and index
        df = raw[["SETTLEMENTDATE", "RRP"]].copy()
        df.columns = ["timestamp", "spot_price"]
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp").sort_index()

        # Clip to market bounds (defence against data artefacts)
        df["spot_price"] = df["spot_price"].clip(
            self.MARKET_PRICE_FLOOR, self.MARKET_PRICE_CAP
        )

        # Resample to strict 5-minute grid, forward-fill short gaps
        df = df.resample("5min").last().ffill()

        self._price_df = df
        df.to_parquet(cache_path)
        logger.info(f"Cached {len(df)} intervals to {cache_path}")

    def load_cache(self, cache_path: str) -> None:
        """Load a previously cached Parquet file directly."""
        self._price_df = pd.read_parquet(cache_path)
        logger.info(f"Loaded {len(self._price_df)} intervals from {cache_path}")

    def load_synthetic(
        self,
        n_days: int = 365,
        mean_price: float = 80.0,
        std_price: float = 150.0,
        spike_prob: float = 0.02,
        spike_magnitude: float = 5000.0,
    ) -> None:
        """
        Generate a synthetic price series for unit testing without NEMOSIS.

        Models NEM price as a log-normal base with occasional price spikes,
        capturing the heavy-tailed distribution characteristic of the NEM.
        For testing only — use real NEMOSIS data for training.

        Parameters
        ----------
        n_days : int
            Number of days to generate.
        mean_price : float
            Mean base price $/MWh (approximate).
        std_price : float
            Std of base price $/MWh.
        spike_prob : float
            Per-interval probability of a price spike.
        spike_magnitude : float
            Mean spike magnitude $/MWh above base.
        """
        n_steps = n_days * self.STEPS_PER_DAY
        timestamps = pd.date_range(
            start="2022-01-01", periods=n_steps, freq="5min"
        )

        # Base price: log-normal to ensure positivity (mostly)
        base = self._rng.lognormal(
            mean=np.log(max(mean_price, 1.0)),
            sigma=std_price / max(mean_price, 1.0),
            size=n_steps,
        )

        # Spikes: Bernoulli indicator × exponential magnitude
        spikes = (
            self._rng.random(size=n_steps) < spike_prob
        ) * self._rng.exponential(spike_magnitude, n_steps)

        prices = np.clip(base + spikes, self.MARKET_PRICE_FLOOR, self.MARKET_PRICE_CAP)

        self._price_df = pd.DataFrame(
            {"spot_price": prices}, index=timestamps
        )
        logger.info(f"Generated {n_days} days of synthetic NEM prices")

    # ------------------------------------------------------------------
    # Episode sampling
    # ------------------------------------------------------------------

    def build_curriculum_index(self) -> None:
        """
        Pre-classify all loaded days into volatility tiers for curriculum sampling.

        Tiers are defined by the daily standard deviation of RRP ($/MWh):
          Tier 1 — Calm:     std < $100/MWh  — safe early training days
          Tier 2 — Normal:   $100 ≤ std < $500
          Tier 3 — Volatile: $500 ≤ std < $2000
          Tier 4 — Extreme:  std ≥ $2000     — spikes + negative RRP events

        Called automatically on first sample_episode() call if not already built.
        Stored in self._tier_days: dict[int, list[date]].
        """
        if self._price_df is None:
            return

        df = self._price_df
        date_counts = df.groupby(df.index.date).size()
        full_days = date_counts[date_counts >= self.STEPS_PER_DAY].index

        tier_days = {1: [], 2: [], 3: [], 4: []}
        for d in full_days:
            mask = df.index.date == d
            std = float(df.loc[mask, "spot_price"].std())
            if std < 100:
                tier_days[1].append(d)
            elif std < 500:
                tier_days[2].append(d)
            elif std < 2000:
                tier_days[3].append(d)
            else:
                tier_days[4].append(d)

        self._tier_days = tier_days
        counts = {t: len(v) for t, v in tier_days.items()}
        logger.info(
            f"Curriculum index built: "
            f"Tier1(calm)={counts[1]} "
            f"Tier2(normal)={counts[2]} "
            f"Tier3(volatile)={counts[3]} "
            f"Tier4(extreme)={counts[4]} days"
        )

    def _curriculum_tier(self, episode: int, total_episodes: int) -> int:
        """
        Return the curriculum tier for the current training episode.

        Tier schedule (aligned to master summary training strategy §3):
          0–20%  of training → Tier 1 only  (calm days)
          20–50% of training → Tier 1+2     (normal days)
          50–80% of training → Tier 1+2+3   (volatile days)
          80–100% of training → all tiers    (full NEM distribution)

        This progressive exposure prevents the catastrophic early losses
        (−$10M at episode 10) that destabilised the synthetic-price runs.
        """
        progress = episode / max(total_episodes, 1)
        if progress < 0.20:
            return 1
        elif progress < 0.50:
            return 2
        elif progress < 0.80:
            return 3
        else:
            return 4

    def sample_episode(
        self,
        date: Optional[str] = None,
        force_wdr: bool = False,   # retained for API compatibility; unused
        episode: Optional[int] = None,
        total_episodes: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Sample a single 288-step (24-hour) price episode for RL training.

        Parameters
        ----------
        date : str, optional
            "YYYY-MM-DD" — if set, samples this specific day (used for
            fixed eval days). Ignores curriculum.
        force_wdr : bool
            Unused — retained for API compatibility.
        episode : int, optional
            Current training episode number. Used for curriculum sampling.
            If None, samples uniformly at random (no curriculum).
        total_episodes : int, optional
            Total training episodes. Required when episode is set.

        Returns
        -------
        pd.DataFrame
            288 rows × [spot_price], indexed by 5-minute timestamps.

        Curriculum behaviour
        --------------------
        When episode and total_episodes are both provided, days are sampled
        from the curriculum tier appropriate for the current training progress.
        Early episodes see only calm days (std < $100/MWh); later episodes
        progressively include volatile and extreme spike days.

        This compresses DOE constraint learning from ~800 to ~300 episodes
        by avoiding catastrophic early losses from random spike days.
        """
        if self._price_df is None:
            raise RuntimeError(
                "No price data loaded. Call fetch_and_cache() or load_synthetic() first."
            )

        # Build curriculum index on first use
        if not hasattr(self, '_tier_days') or self._tier_days is None:
            self.build_curriculum_index()

        prices_day = self._sample_price_day(
            date=date,
            episode=episode,
            total_episodes=total_episodes,
        )

        episode_df = prices_day.reset_index(drop=False)
        episode_df.columns = ["timestamp", "spot_price"]
        episode_df = episode_df.set_index("timestamp")

        return episode_df

    def _sample_price_day(
        self,
        date: Optional[str] = None,
        episode: Optional[int] = None,
        total_episodes: Optional[int] = None,
    ) -> pd.Series:
        """
        Extract or sample a 288-step price series from loaded data.

        If date is specified: return that exact date (for fixed eval days).
        If episode/total_episodes provided: curriculum sampling by tier.
        Otherwise: uniform random day.
        """
        df = self._price_df

        # Fixed date requested (eval mode)
        if date is not None:
            target = pd.Timestamp(date)
            mask = df.index.date == target.date()
            day_prices = df.loc[mask, "spot_price"]

            if len(day_prices) < self.STEPS_PER_DAY:
                logger.warning(
                    f"Date {date} has only {len(day_prices)} intervals "
                    f"(expected {self.STEPS_PER_DAY}). Falling back to random day."
                )
                return self._random_day(df)

            return day_prices.iloc[: self.STEPS_PER_DAY]

        # Curriculum sampling (training mode with episode info)
        if (episode is not None
                and total_episodes is not None
                and hasattr(self, '_tier_days')
                and self._tier_days is not None):
            max_tier = self._curriculum_tier(episode, total_episodes)
            # Pool = all days up to and including current max tier
            pool = []
            for t in range(1, max_tier + 1):
                pool.extend(self._tier_days.get(t, []))

            if pool:
                chosen_date = self._rng.choice(pool)
                mask = df.index.date == chosen_date
                return df.loc[mask, "spot_price"].iloc[: self.STEPS_PER_DAY]

        # Fallback: uniform random day
        return self._random_day(df)

    def _random_day(self, df: pd.DataFrame) -> pd.Series:
        """Sample a complete 288-step day at random from df."""
        # Identify all dates that have a full 288-step day
        date_counts = df.groupby(df.index.date).size()
        full_days = date_counts[date_counts >= self.STEPS_PER_DAY].index

        if len(full_days) == 0:
            raise RuntimeError("No complete days found in loaded price data.")

        chosen_date = self._rng.choice(full_days)
        mask = df.index.date == chosen_date
        return df.loc[mask, "spot_price"].iloc[: self.STEPS_PER_DAY]

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def price_summary(self) -> dict:
        """Basic statistics on loaded prices, for sanity-checking."""
        if self._price_df is None:
            return {}
        p = self._price_df["spot_price"]
        return {
            "count": len(p),
            "mean": float(p.mean()),
            "std": float(p.std()),
            "min": float(p.min()),
            "max": float(p.max()),
            "pct_99": float(p.quantile(0.99)),
            "pct_spike_above_300": float((p > 300).mean()),
        }
