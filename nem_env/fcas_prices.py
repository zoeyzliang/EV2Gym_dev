"""
fcas_prices.py
==============
5-minute FCAS enablement prices for one region and year (AEMO DISPATCHPRICE,
NEMweb MMSDM archive), for the FCAS availability bound (spec §6, S4).
NEMOSIS's DISPATCHPRICE column list predates the 1-second services, so the
monthly files are read directly. Rows with INTERVENTION = 1 are dropped.

Cached columns (index: SETTLEMENTDATE, interval end, market time):
  RAISE1SECRRP, RAISE6SECRRP, RAISE60SECRRP, RAISE5MINRRP, RAISEREGRRP,
  LOWER1SECRRP, LOWER6SECRRP, LOWER60SECRRP, LOWER5MINRRP, LOWERREGRRP  ($/MW/h)
(1-second columns are NaN where a file predates them.)
"""

import io
import logging
import re
import zipfile
from pathlib import Path
from urllib.request import urlopen

import pandas as pd

logger = logging.getLogger(__name__)

HOST = "https://nemweb.com.au"
DIR = "/Data_Archive/Wholesale_Electricity/MMSDM/{y}/MMSDM_{y}_{m:02d}/MMSDM_Historical_Data_SQLLoader/DATA/"
SERVICES = [f"{d}{s}RRP" for d in ("RAISE", "LOWER") for s in ("1SEC", "6SEC", "60SEC", "5MIN", "REG")]


def _month_files(y: int, m: int) -> list:
    with urlopen(HOST + DIR.format(y=y, m=m), timeout=120) as r:
        html = r.read().decode("utf-8", "replace")
    hrefs = re.findall(r'HREF="([^"]+)"', html, flags=re.I)
    # DISPATCHPRICE only (not PREDISPATCHPRICE...): name ends ..._DISPATCHPRICE_<date> or ...#DISPATCHPRICE#FILEnn#
    files = sorted({h for h in hrefs if re.search(r"(_|%23|#)DISPATCHPRICE(_2|%23|#)", h, re.I)})
    if not files:
        raise RuntimeError(f"no DISPATCHPRICE file listed for {y}-{m:02d}")
    return [HOST + h if h.startswith("/") else h for h in files]


def fetch_fcas(region: str, year: int, cache_dir: str, force_refresh: bool = False) -> pd.DataFrame:
    path = Path(cache_dir) / f"{region}_fcas_{year}.parquet"
    if path.exists() and not force_refresh:
        return pd.read_parquet(path)
    frames = []
    for y, m in [(year, k) for k in range(1, 13)] + [(year + 1, 1)]:
        for url in _month_files(y, m):
            logger.info(f"Downloading {url}")
            with urlopen(url, timeout=600) as r:
                zf = zipfile.ZipFile(io.BytesIO(r.read()))
            with zf.open(zf.namelist()[0]) as f:
                raw = pd.read_csv(f, skiprows=1, low_memory=False)
            raw = raw[(raw.iloc[:, 0] == "D") & (raw["REGIONID"] == region)]
            raw = raw[raw["INTERVENTION"].astype(int) == 0]
            df = pd.DataFrame({"SETTLEMENTDATE": pd.to_datetime(raw["SETTLEMENTDATE"], format="%Y/%m/%d %H:%M:%S")})
            for c in SERVICES:
                df[c] = raw[c].astype(float).to_numpy() if c in raw else float("nan")
            frames.append(df)
    df = pd.concat(frames, ignore_index=True).drop_duplicates("SETTLEMENTDATE")
    df = df[(df["SETTLEMENTDATE"] > pd.Timestamp(f"{year}-01-01")) &
            (df["SETTLEMENTDATE"] <= pd.Timestamp(f"{year + 1}-01-01"))].set_index("SETTLEMENTDATE").sort_index()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    logger.info(f"Cached {len(df)} intervals -> {path}")
    return df
