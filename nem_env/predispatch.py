"""
predispatch.py
==============
AEMO 30-minute predispatch price forecasts for one region, as they were
published (every predispatch run, not only the last one), for the
forecast-based MPC baseline (spec §5).

Source: NEMweb MMSDM archive, PREDISP_ALL_DATA/PUBLIC_DVD_PREDISPATCHPRICE
(NEMOSIS does not cover predispatch). Each run, published every 30 minutes,
forecasts the regional RRP for every 30-minute period to the end of the next
trading day. Rows with INTERVENTION = 1 are dropped.

Cached columns:
  published  run publication time (LASTCHANGED, market time)
  period     end of the 30-minute period forecast (DATETIME, market time)
  rrp        forecast RRP ($/MWh)

Usage:
  from nem_env.predispatch import Predispatch
  pd_ = Predispatch.fetch_and_cache("VIC1", 2024, "data/nem_cache")
  fc = pd_.forecast_at(pd.Timestamp("2024-03-05 17:02"))   # Series indexed by period
"""

import io
import logging
import re
import zipfile
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

HOST = "https://nemweb.com.au"
DIR = ("/Data_Archive/Wholesale_Electricity/MMSDM/{y}/MMSDM_{y}_{m:02d}/"
       "MMSDM_Historical_Data_SQLLoader/PREDISP_ALL_DATA/")


def _month_files(y: int, m: int) -> list:
    """Predispatch price files for one month, read from the directory listing.
    AEMO renamed them in mid-2024 (PUBLIC_DVD_PREDISPATCHPRICE_... ->
    PUBLIC_ARCHIVE#PREDISPATCHPRICE#ALL#FILEnn#...), and a month may be split
    into several FILEnn parts."""
    with urlopen(HOST + DIR.format(y=y, m=m), timeout=120) as r:
        html = r.read().decode("utf-8", "replace")
    hrefs = re.findall(r'HREF="([^"]+)"', html, flags=re.I)
    files = sorted({h for h in hrefs
                    if re.search(r"PREDISPATCHPRICE(_|%23|#)", h, re.I) and "SENSITIVIT" not in h.upper()})
    if not files:
        raise RuntimeError(f"no predispatch price file listed for {y}-{m:02d}")
    return [h if h.startswith("http") else HOST + h if h.startswith("/") else HOST + DIR.format(y=y, m=m) + h
            for h in files]


class Predispatch:
    def __init__(self, df: pd.DataFrame):
        self.df = df.sort_values(["published", "period"]).reset_index(drop=True)
        self._runs = np.sort(self.df["published"].unique())

    @staticmethod
    def cache_path(region: str, year: int, cache_dir: str) -> Path:
        return Path(cache_dir) / f"{region}_predispatch_{year}.parquet"

    @classmethod
    def fetch_and_cache(cls, region: str, year: int, cache_dir: str, force_refresh: bool = False):
        path = cls.cache_path(region, year, cache_dir)
        if path.exists() and not force_refresh:
            return cls(pd.read_parquet(path))
        frames = []
        # A month's file holds the runs published in that month; the first
        # runs of January forecast periods of the previous year's last day,
        # and the January file of the next year holds 31 Dec's late runs.
        for y, m in [(year, k) for k in range(1, 13)] + [(year + 1, 1)]:
            for url in _month_files(y, m):
                logger.info(f"Downloading {url}")
                with urlopen(url, timeout=600) as r:
                    zf = zipfile.ZipFile(io.BytesIO(r.read()))
                with zf.open(zf.namelist()[0]) as f:
                    raw = pd.read_csv(f, skiprows=1, low_memory=False)
                raw = raw[(raw.iloc[:, 0] == "D") & (raw["REGIONID"] == region)]
                raw = raw[raw["INTERVENTION"].astype(int) == 0]
                frames.append(pd.DataFrame({
                    "published": pd.to_datetime(raw["LASTCHANGED"], format="%Y/%m/%d %H:%M:%S"),
                    "period": pd.to_datetime(raw["DATETIME"], format="%Y/%m/%d %H:%M:%S"),
                    "rrp": raw["RRP"].astype(float),
                }))
        df = pd.concat(frames, ignore_index=True).drop_duplicates(["published", "period"])
        df = df[(df["period"] > pd.Timestamp(f"{year}-01-01")) &
                (df["period"] <= pd.Timestamp(f"{year + 1}-01-01 04:00"))]
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        logger.info(f"Cached {len(df)} rows, {df['published'].nunique()} runs -> {path}")
        return cls(df)

    @classmethod
    def load_cache(cls, path: str):
        return cls(pd.read_parquet(path))

    def forecast_at(self, t: pd.Timestamp) -> pd.Series:
        """RRP forecast of the latest run published at or before t, by period end."""
        i = np.searchsorted(self._runs, np.datetime64(t), side="right") - 1
        if i < 0:
            raise ValueError(f"no predispatch run published before {t}")
        run = self._runs[i]
        g = self.df[self.df["published"] == run]
        return g.set_index("period")["rrp"]
