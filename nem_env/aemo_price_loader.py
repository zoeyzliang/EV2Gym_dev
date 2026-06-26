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
    zone_peak_mw : float
        VSR zone maximum aggregate dispatch capacity. Passed to WDR generator.
    wdr_params : dict, optional
        Override WDR_DEFAULTS for event generator.
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

    def sample_episode(
        self,
        date: Optional[str] = None,
        force_wdr: bool = False,   # retained for API compatibility; unused
    ) -> pd.DataFrame:
        """
        Sample a single 288-step (24-hour) price episode for RL training.

        Parameters
        ----------
        date : str, optional
            "YYYY-MM-DD" of the target day. If None, samples a random day.
        force_wdr : bool
            Unused — retained for API compatibility. WDR logic has been
            removed. The agent operates as a price-taker with no AEMO
            dispatch target (master summary §2, Option A).

        Returns
        -------
        pd.DataFrame
            288 rows, index = 5-minute timestamps, columns:
              - spot_price (float): NEM spot price $/MWh
        """
        if self._price_df is None:
            raise RuntimeError(
                "No price data loaded. Call fetch_and_cache() or load_synthetic() first."
            )

        prices_day = self._sample_price_day(date)

        # Return only spot_price — no WDR/dispatch target columns.
        # The agent is a price-taker: it observes RRP and sets its own
        # dispatch position. AEMO does not issue dispatch targets to the
        # aggregator in this model (Option A, master summary §2).
        episode = prices_day.reset_index(drop=False)
        episode.columns = ["timestamp", "spot_price"]
        episode = episode.set_index("timestamp")

        return episode

    def _sample_price_day(self, date: Optional[str]) -> pd.Series:
        """
        Extract or randomly sample a 288-step price series from loaded data.

        If the requested date is missing from cache (e.g. public holiday with
        no data), falls back to a random day with a warning.
        """
        df = self._price_df

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