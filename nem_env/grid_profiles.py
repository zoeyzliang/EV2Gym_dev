"""
grid_profiles.py
================
Regional demand and rooftop-PV profiles from AEMO, aligned to the 5-minute
dispatch grid used for prices.

Used by the feeder environment (nem_env/feeder.py) to shape each feeder bus's
background load and PV on the *same dates* as the RRP series, so that price
and network stress are jointly realistic (e.g. midday rooftop PV coincides
with low/negative RRP and with tight export DOEs).

Sources (via NEMOSIS):
  DISPATCHREGIONSUM.TOTALDEMAND  — 5-min regional operational demand (MW)
  ROOFTOP_PV_ACTUAL.POWER        — 30-min regional rooftop PV estimate (MW),
                                   interpolated to 5 min

Only the *shape* of each series is used: demand is normalised by its mean
over the cached period and PV by its peak, so the feeder's own nominal load
and PV capacity set the absolute scale.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class GridProfiles:
    """
    Cached 5-min demand/PV shapes for one NEM region.

    Parameters
    ----------
    region : str
        NEM region id, e.g. "VIC1".
    cache_dir : str
        Directory holding NEMOSIS raw files and the processed parquet.
    """

    STEPS_PER_DAY = 288

    def __init__(self, region: str = "VIC1", cache_dir: str = "data/nem_cache"):
        self.region = region
        self.cache_dir = Path(cache_dir)
        self._df: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    def cache_path(self, start: str, end: str) -> Path:
        return self.cache_dir / f"{self.region}_grid_profiles_{start}_{end}.parquet"

    def fetch_and_cache(self, start: str, end: str, force_refresh: bool = False) -> None:
        """
        Download demand and rooftop PV for [start, end] (YYYY-MM-DD, inclusive)
        and cache a 5-min frame with columns demand_mw, pv_mw.
        """
        path = self.cache_path(start, end)
        if path.exists() and not force_refresh:
            self.load_cache(str(path))
            return

        import nemosis

        raw_dir = str(self.cache_dir / "raw")
        t0 = start.replace("-", "/") + " 00:00:00"
        t1 = end.replace("-", "/") + " 23:55:00"

        dem = nemosis.dynamic_data_compiler(
            t0, t1, "DISPATCHREGIONSUM", raw_dir,
            filter_cols=["REGIONID"], filter_values=([self.region],),
            select_columns=["SETTLEMENTDATE", "REGIONID", "TOTALDEMAND", "INTERVENTION"],
        )
        # Keep the non-intervention run when both are published
        if "INTERVENTION" in dem.columns:
            dem = dem[dem["INTERVENTION"].astype(float) == 0]
        dem = (dem.assign(timestamp=pd.to_datetime(dem["SETTLEMENTDATE"]),
                          demand_mw=dem["TOTALDEMAND"].astype(float))
                  .groupby("timestamp")["demand_mw"].last())

        pv = nemosis.dynamic_data_compiler(
            t0, t1, "ROOFTOP_PV_ACTUAL", raw_dir,
            filter_cols=["REGIONID"], filter_values=([self.region],),
        )
        pv = pv.assign(timestamp=pd.to_datetime(pv["INTERVAL_DATETIME"]),
                       power=pv["POWER"].astype(float))
        # Several estimate TYPEs per interval: prefer MEASUREMENT, else the
        # latest of whatever is available.
        pv["pref"] = (pv["TYPE"] == "MEASUREMENT").astype(int)
        if "LASTCHANGED" in pv.columns:
            pv["lc"] = pd.to_datetime(pv["LASTCHANGED"])
            pv = pv.sort_values(["timestamp", "pref", "lc"])
        else:
            pv = pv.sort_values(["timestamp", "pref"])
        pv = pv.groupby("timestamp")["power"].last()

        idx = pd.date_range(dem.index.min(), dem.index.max(), freq="5min")
        df = pd.DataFrame(index=idx)
        df.index.name = "timestamp"
        df["demand_mw"] = dem.reindex(idx).interpolate(limit=12).ffill().bfill()
        # PV is a 30-min interval-ending average: interpolate onto 5-min grid
        df["pv_mw"] = (pv.reindex(pv.index.union(idx)).interpolate(method="time")
                         .reindex(idx).fillna(0.0).clip(lower=0.0))

        self._df = df
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path)
        logger.info(f"Cached {len(df)} grid-profile intervals to {path}")

    def load_cache(self, path: str) -> None:
        self._df = pd.read_parquet(path)
        logger.info(f"Loaded {len(self._df)} grid-profile intervals from {path}")

    # ------------------------------------------------------------------
    @property
    def df(self) -> pd.DataFrame:
        if self._df is None:
            raise RuntimeError("GridProfiles not loaded: call fetch_and_cache() or load_cache().")
        return self._df

    def day(self, index: pd.DatetimeIndex) -> tuple:
        """
        Normalised (demand_shape, pv_shape) arrays aligned to the given 5-min
        timestamps (e.g. an episode's price index).

        demand_shape : demand / 99th-percentile demand over the cached period
                       (so ≤ 1 except on the most extreme peaks)
        pv_shape     : PV / peak PV over the cached period (0..1)
        """
        df = self.df
        sub = df.reindex(index, method="nearest", tolerance=pd.Timedelta("10min"))
        if sub.isna().any().any():
            missing = int(sub.isna().any(axis=1).sum())
            raise KeyError(f"{missing} timestamps not covered by grid profiles "
                           f"({index[0]} .. {index[-1]})")
        demand = sub["demand_mw"].to_numpy() / df["demand_mw"].quantile(0.99)
        # Rooftop PV capacity grew strongly over 2022–2024, so normalising by
        # the overall peak would give training years (2022–23) systematically
        # less PV than the test year. Normalise by each calendar year's 99.5th
        # percentile instead, so the feeder's PV penetration means the same
        # thing in every year.
        yearly_ref = df["pv_mw"].groupby(df.index.year).quantile(0.995)
        pv = sub["pv_mw"].to_numpy() / yearly_ref.reindex(sub.index.year).to_numpy()
        return demand, np.clip(pv, 0.0, None)
