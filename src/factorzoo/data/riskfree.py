"""Kenneth French Data Library loader: risk-free rate and benchmark factors.

Used only for validation: the replica's own from-scratch market/size/value
portfolios should correlate above ~0.9 with these before any new factor
result is trusted.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import httpx
import pandas as pd

from factorzoo.config import CACHE_DIR

FF5_DAILY_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
)
FF5_MONTHLY_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Research_Data_5_Factors_2x3_CSV.zip"
)
MOMENTUM_DAILY_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_daily_CSV.zip"
)

FF5_COLUMNS = {"Mkt-RF": "mkt_rf", "SMB": "smb", "HML": "hml", "RMW": "rmw", "CMA": "cma", "RF": "rf"}


def fetch_ff5_daily(cache_dir: Path | None = None, force: bool = False) -> pd.DataFrame:
    """Daily Fama-French five factors + RF, as decimal returns (the source
    file reports percent, e.g. -0.67 meaning -0.67%; this divides by 100
    so downstream code never has to remember which convention a column is
    in)."""
    cache_dir = cache_dir or CACHE_DIR
    cache_path = cache_dir / "ff5_daily.csv"
    raw_zip = _download_or_cached(FF5_DAILY_URL, cache_dir / "ff5_daily.zip", force)
    df = _parse_ff_zip(raw_zip, inner_name_hint="5_Factors_2x3_daily")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False)
    return df


def _download_or_cached(url: str, cache_path: Path, force: bool) -> bytes:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists() and not force:
        return cache_path.read_bytes()
    with httpx.Client(timeout=60.0) as client:
        resp = client.get(url)
        resp.raise_for_status()
        content = resp.content
    cache_path.write_bytes(content)
    return content


def _parse_ff_zip(raw_zip: bytes, inner_name_hint: str) -> pd.DataFrame:
    """Pure function: Ken French's CSVs all share a format -- a few header
    comment lines, a header row with a blank first column name, YYYYMMDD
    (or YYYYMM for monthly) dates, percent-unit values, and a trailing
    copyright line with no valid date. Rows that don't parse as a date are
    dropped rather than assumed to be data.
    """
    with zipfile.ZipFile(io.BytesIO(raw_zip)) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith((".csv", ".CSV".lower()))]
        if not names:
            names = zf.namelist()
        inner = next((n for n in names if inner_name_hint.lower() in n.lower()), names[0])
        text = zf.read(inner).decode("utf-8", errors="replace")

    lines = text.splitlines()
    header_idx = next(i for i, line in enumerate(lines) if line.strip().startswith(","))
    body = "\n".join(lines[header_idx:])
    df = pd.read_csv(io.StringIO(body))
    df = df.rename(columns={df.columns[0]: "date_raw"})
    df["date_raw"] = df["date_raw"].astype(str).str.strip()
    df = df[df["date_raw"].str.match(r"^\d{6,8}$")].copy()

    date_len = df["date_raw"].str.len().mode()[0]
    fmt = "%Y%m%d" if date_len == 8 else "%Y%m"
    df["date"] = pd.to_datetime(df["date_raw"], format=fmt)
    df = df.rename(columns=FF5_COLUMNS)
    value_cols = [c for c in FF5_COLUMNS.values() if c in df.columns]
    for c in value_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce") / 100.0
    return df[["date", *value_cols]].reset_index(drop=True)
