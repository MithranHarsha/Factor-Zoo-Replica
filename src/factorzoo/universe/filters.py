"""Point-in-time universe construction.

The point-in-time fix: SEC's company_tickers.json lists only *current*
EDGAR filers, so starting from it and trying to "add back" delisted names
afterward has no real mechanism for knowing which names to add -- that is
survivorship bias baked into the design, not an oversight that backfilling
can patch.

Instead this module treats a maintained historical index-membership history
as the *primary* source of which tickers were investable in which month: it
explicitly logs additions AND removals (delisting, acquisition, bankruptcy,
index demotion) with dates. The pilot source is the community-maintained
S&P 500 historical constituents dataset (fja05680/sp500 on GitHub), which
gives ticker, start_date, end_date (end_date blank = still a member).
`company_tickers.json` (edgar.py) is used only downstream, as a secondary
CIK lookup for whichever tickers this history already names.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pandas as pd

from factorzoo.config import CACHE_DIR

SP500_HISTORY_URL = (
    "https://raw.githubusercontent.com/fja05680/sp500/master/sp500_ticker_start_end.csv"
)
SP500_HISTORY_SOURCE_NOTE = (
    "fja05680/sp500 (community-maintained S&P 500 historical constituents, "
    "originally sourced from 'Trading Evolved' by Andreas Clenow plus ongoing "
    "Wikipedia-tracked changes). Free-data proxy universe, not CRSP-equivalent."
)


def fetch_sp500_history(cache_path: Path | None = None, force: bool = False) -> pd.DataFrame:
    """Download (or load cached) ticker/start_date/end_date membership
    history. Network I/O lives here; parsing/validation lives in
    `parse_sp500_history` below so it can be unit tested offline.
    """
    cache_path = cache_path or (CACHE_DIR / "sp500_ticker_start_end.csv")
    if cache_path.exists() and not force:
        raw = cache_path.read_text()
    else:
        with httpx.Client(timeout=30.0) as client:
            resp = client.get(SP500_HISTORY_URL)
            resp.raise_for_status()
            raw = resp.text
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(raw)
    return parse_sp500_history(raw)


def parse_sp500_history(raw_csv: str) -> pd.DataFrame:
    """Pure function. Returns DataFrame[ticker, start_date, end_date]
    (end_date is NaT for names still in the index as of the source's last
    update)."""
    import io

    df = pd.read_csv(io.StringIO(raw_csv))
    df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")
    df["end_date"] = pd.to_datetime(df["end_date"], errors="coerce")
    df["ticker"] = df["ticker"].str.upper().str.strip()
    df = df.dropna(subset=["ticker", "start_date"]).reset_index(drop=True)
    return df[["ticker", "start_date", "end_date"]]


def point_in_time_members(history: pd.DataFrame, as_of: pd.Timestamp) -> list[str]:
    """Which tickers were index members on `as_of`. A ticker with a blank
    end_date is treated as a member through the present; this is exactly
    the mechanism that keeps a name like AABA (delisted 2017) out of a
    2026-dated universe while still correctly including it for any
    as-of date between 1999-12-08 and 2017-06-19.
    """
    as_of = pd.Timestamp(as_of)
    mask = (history["start_date"] <= as_of) & (history["end_date"].isna() | (history["end_date"] >= as_of))
    return sorted(history.loc[mask, "ticker"].unique().tolist())


def monthly_universe_panel(
    history: pd.DataFrame, start: str = "1996-01-31", end: str | None = None, freq: str = "ME"
) -> pd.DataFrame:
    """Long panel [formation_date, ticker] for every month end in range.
    This is what gets written to the point-in-time store: for any given
    formation date, which tickers were actually investable then, not which
    tickers happen to be in the S&P 500 today.
    """
    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    month_ends = pd.date_range(start=start, end=end, freq=freq)
    rows: list[dict] = []
    for d in month_ends:
        for ticker in point_in_time_members(history, d):
            rows.append({"formation_date": d, "ticker": ticker})
    return pd.DataFrame(rows)


def summarize_coverage(history: pd.DataFrame) -> dict:
    """Section 4's transparency requirement made concrete: report size and
    scope honestly rather than letting a silently-narrow universe pass
    unnoticed."""
    n_unique_tickers = history["ticker"].nunique()
    still_active = history["end_date"].isna().sum()
    span_start = history["start_date"].min()
    span_end = history["end_date"].max()
    return {
        "n_ticker_spells": len(history),
        "n_unique_tickers": n_unique_tickers,
        "n_currently_active_spells": int(still_active),
        "history_start": span_start,
        "history_end": span_end,
        "source": SP500_HISTORY_SOURCE_NOTE,
    }
