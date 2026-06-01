"""
aemo_price_loader.py
====================
NEM spot price loader and synthetic WDR dispatch target generator.

Pulls historical 5-minute dispatch interval prices for VIC1 from NEMOSIS,
caches to Parquet, and generates synthetic WDR activation events for each
training episode.

Design decisions
----------------
- Episode length: 288 steps × 5 min = 24 hours, aligned to NEM trading day.
- WDR events are sparse in reality (~2–5 activations/year in VIC). To keep
  training signal non-degenerate, `sample_episode()` accepts a
  `force_wdr=True` flag that guarantees at least one WDR window per episode.
  This implements the curriculum described in §4.4.
- Outside WDR windows, `wdr_active=False` and `dispatch_target_mw=0`. The
  env still runs but the conformance penalty term is zeroed (see nem_wdr_env).
- WDR event generator: Poisson inter-arrival calibrated to historical AEMO
  frequency; event duration log-normal fitted to AEMO WDRM review data
  (median ~30 min, max ~4 hr).

Usage
-----
    loader = PriceLoader(region="VIC1", cache_dir="data/nem_cache")
    loader.fetch_and_cache(start="2022-01-01", end="2024-12-31")
    episode = loader.sample_episode(force_wdr=True)
    # episode: pd.DataFrame with columns
    #   [spot_price, wdr_active, dispatch_target_mw]
    # indexed by 5-min interval timestamps, length 288
"""

import os
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# WDR event generation parameters
# These are calibrated to AEMO WDRM review data (AEMC EPR0099, Oct 2025).
# Treat as configuration — sweep in sensitivity analysis (§4.3.4).
# ---------------------------------------------------------------------------
WDR_DEFAULTS = {
    # Poisson rate: expected activations per year in VIC under WDRM
    # AEMC review recorded ~3 activations/year 2021–2025 in VIC
    "annual_activation_rate": 3.0,

    # Log-normal parameters for event duration (minutes)
    # Fitted to AEMO dispatch events: median ~30 min, 95th pct ~120 min
    "duration_lognormal_mu": 3.4,   # ln(30) ≈ 3.4
    "duration_lognormal_sigma": 0.8,

    # Dispatch target as fraction of VSR zone peak capacity
    # Agent must deliver this fraction of its declared maximum capacity
    "target_fraction_mean": 0.6,
    "target_fraction_std": 0.15,

    # Earliest / latest hour an activation can start (avoid overnight)
    "activation_start_hour_min": 14,   # 2 PM
    "activation_start_hour_max": 21,   # 9 PM (peak demand window)
}


class WDREventGenerator:
    """
    Generates synthetic WDR activation windows for a single 24-hour episode.

    The generator models the sequence of events as:
      1. Number of activations in a day ~ Poisson(λ_day)
         where λ_day = annual_rate / 365
      2. Start time of each activation ~ Uniform over peak window
      3. Duration ~ LogNormal(mu, sigma), clipped to fit within episode
      4. Dispatch target ~ Uniform(0, zone_peak_mw) clipped by N(mean, std)

    Parameters
    ----------
    zone_peak_mw : float
        Maximum aggregate discharge capacity of the VSR zone (MW).
        Used to scale dispatch targets. Should match your hub config.
    rng : np.random.Generator, optional
        Reproducible RNG. If None, a fresh default_rng() is used.
    params : dict, optional
        Override any WDR_DEFAULTS key.
    """

    STEPS_PER_DAY = 288          # 24 hr × 12 intervals/hr
    STEP_MINUTES = 5

    def __init__(
        self,
        zone_peak_mw: float = 2.0,
        rng: Optional[np.random.Generator] = None,
        params: Optional[dict] = None,
    ):
        self.zone_peak_mw = zone_peak_mw
        self.rng = rng if rng is not None else np.random.default_rng()
        self.p = {**WDR_DEFAULTS, **(params or {})}

    def generate(self) -> pd.DataFrame:
        """
        Generate a 288-row DataFrame with WDR event labels for one episode.

        Returns
        -------
        pd.DataFrame
            Columns: wdr_active (bool), dispatch_target_mw (float).
            Index: integer 0..287 (step index within episode).
        """
        wdr_active = np.zeros(self.STEPS_PER_DAY, dtype=bool)
        dispatch_target_mw = np.zeros(self.STEPS_PER_DAY, dtype=float)

        # Step 1: draw number of activations today
        lambda_day = self.p["annual_activation_rate"] / 365.0
        n_events = self.rng.poisson(lambda_day)

        for _ in range(n_events):
            event_start_step, duration_steps, target_mw = self._draw_event()
            event_end_step = min(event_start_step + duration_steps, self.STEPS_PER_DAY)
            wdr_active[event_start_step:event_end_step] = True
            # Constant dispatch target within each event window
            dispatch_target_mw[event_start_step:event_end_step] = target_mw

        return pd.DataFrame({
            "wdr_active": wdr_active,
            "dispatch_target_mw": dispatch_target_mw,
        })

    def generate_forced(self) -> pd.DataFrame:
        """
        Guarantee exactly one WDR activation window (curriculum mode).

        Used when `force_wdr=True` in PriceLoader.sample_episode(), ensuring
        the training signal contains at least one conformance-penalty episode
        per training iteration. Without this, the agent would rarely encounter
        WDR intervals during early training and receive no learning signal from
        the conformance term.
        """
        wdr_active = np.zeros(self.STEPS_PER_DAY, dtype=bool)
        dispatch_target_mw = np.zeros(self.STEPS_PER_DAY, dtype=float)

        event_start_step, duration_steps, target_mw = self._draw_event()
        event_end_step = min(event_start_step + duration_steps, self.STEPS_PER_DAY)
        wdr_active[event_start_step:event_end_step] = True
        dispatch_target_mw[event_start_step:event_end_step] = target_mw

        return pd.DataFrame({
            "wdr_active": wdr_active,
            "dispatch_target_mw": dispatch_target_mw,
        })

    def _draw_event(self):
        """Sample start step, duration steps, and target MW for one event."""
        # Start time: uniform over peak demand window (step index)
        start_hour_min = self.p["activation_start_hour_min"]
        start_hour_max = self.p["activation_start_hour_max"]
        start_step_min = int(start_hour_min * 60 / self.STEP_MINUTES)
        start_step_max = int(start_hour_max * 60 / self.STEP_MINUTES)
        event_start_step = int(self.rng.integers(start_step_min, start_step_max))

        # Duration: log-normal in minutes → steps
        duration_min = self.rng.lognormal(
            self.p["duration_lognormal_mu"],
            self.p["duration_lognormal_sigma"],
        )
        # Clip: min 1 step (5 min), max fills remaining day
        duration_min = np.clip(duration_min, self.STEP_MINUTES, 240)
        duration_steps = max(1, int(duration_min / self.STEP_MINUTES))

        # Dispatch target: fraction of zone peak capacity
        fraction = self.rng.normal(
            self.p["target_fraction_mean"],
            self.p["target_fraction_std"],
        )
        fraction = np.clip(fraction, 0.1, 1.0)
        target_mw = fraction * self.zone_peak_mw

        return event_start_step, duration_steps, target_mw


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
        zone_peak_mw: float = 2.0,
        wdr_params: Optional[dict] = None,
        seed: Optional[int] = None,
    ):
        self.region = region
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.zone_peak_mw = zone_peak_mw
        self._rng = np.random.default_rng(seed)
        self._wdr_generator = WDREventGenerator(
            zone_peak_mw=zone_peak_mw,
            rng=self._rng,
            params=wdr_params,
        )

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
        force_wdr: bool = True,
    ) -> pd.DataFrame:
        """
        Sample a single 288-step (24-hour) episode for RL training.

        Parameters
        ----------
        date : str, optional
            "YYYY-MM-DD" of the target day. If None, samples a random day
            from the loaded price data.
        force_wdr : bool
            If True (default), guarantee at least one WDR activation window
            in the episode (curriculum training mode).
            If False, WDR events are sampled from the Poisson process and may
            be absent. Use False for evaluation to assess real-world sparsity.

        Returns
        -------
        pd.DataFrame
            288 rows, index = 5-minute timestamps, columns:
              - spot_price (float): NEM spot price $/MWh
              - wdr_active (bool): whether a WDR activation is in progress
              - dispatch_target_mw (float): AEMO dispatch target this step (MW)
                0.0 when wdr_active is False
        """
        if self._price_df is None:
            raise RuntimeError(
                "No price data loaded. Call fetch_and_cache() or load_synthetic() first."
            )

        prices_day = self._sample_price_day(date)
        wdr_events = (
            self._wdr_generator.generate_forced()
            if force_wdr
            else self._wdr_generator.generate()
        )

        # Align indices: prices_day has datetime index, wdr_events has 0..287
        episode = prices_day.reset_index(drop=False)
        episode.columns = ["timestamp", "spot_price"]
        episode["wdr_active"] = wdr_events["wdr_active"].values
        episode["dispatch_target_mw"] = wdr_events["dispatch_target_mw"].values
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
