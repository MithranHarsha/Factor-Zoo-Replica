"""Decile (or quintile-fallback) sorting with a large-cap breakpoint
proxy: true NYSE-only breakpoints need an exchange-listing flag this
project's free data doesn't cleanly provide, so breakpoints are computed
from the largest 25% of names by COUNT, not 60% of cumulative dollar
market cap (the two are not interchangeable -- count-based is simpler to
compute from this data but skews the cutoff toward smaller names than a
true NYSE dollar-cap breakpoint would), then applied to the full eligible
universe.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

LARGE_CAP_BREAKPOINT_PCT = 0.25  # fraction of names, by count, not cumulative cap
MIN_NAMES_FOR_DECILES = 300
N_DECILES = 10
N_QUINTILES_FALLBACK = 5


def assign_quantile_buckets(
    df: pd.DataFrame,
    factor_col: str,
    market_cap_col: str = "market_cap",
    large_cap_pct: float = LARGE_CAP_BREAKPOINT_PCT,
    min_names_for_deciles: int = MIN_NAMES_FOR_DECILES,
) -> pd.Series:
    """Returns a Series of bucket numbers (1 = lowest factor value, N =
    highest), aligned to `df`'s index. Rows with a missing factor value or
    missing market cap are left as <NA> (not silently dropped from the
    index, so callers can still see which rows were ineligible).
    """
    valid_mask = df[factor_col].notna() & df[market_cap_col].notna()
    valid = df.loc[valid_mask]
    out = pd.Series(pd.NA, index=df.index, dtype="Int64")
    if valid.empty:
        return out

    n_quantiles = N_DECILES if len(valid) >= min_names_for_deciles else N_QUINTILES_FALLBACK

    breakpoint_n = max(1, int(np.ceil(len(valid) * large_cap_pct)))
    breakpoint_set = valid.nlargest(breakpoint_n, market_cap_col)

    quantile_points = np.linspace(0, 1, n_quantiles + 1)
    edges = np.quantile(breakpoint_set[factor_col].to_numpy(), quantile_points)
    edges = np.unique(edges)  # collapse duplicate edges (common with a small breakpoint set)
    if len(edges) < 2:
        # Every breakpoint-set observation has the same value (degenerate
        # case, e.g. a tiny pilot universe): everyone falls in one bucket
        # rather than raising.
        out.loc[valid.index] = 1
        return out

    edges[0] = -np.inf
    edges[-1] = np.inf
    bucket = pd.cut(valid[factor_col], bins=edges, labels=False, include_lowest=True, duplicates="drop")
    out.loc[valid.index] = (bucket + 1).astype("Int64")
    return out


def n_buckets_used(df: pd.DataFrame, factor_col: str, market_cap_col: str = "market_cap") -> int:
    """How many quantile buckets a given cross-section actually used
    (10, 5, or fewer in a degenerate case) -- useful for reporting/tests
    without re-deriving the sort logic's internal thresholds."""
    n_valid = df[factor_col].notna().sum()
    return N_DECILES if n_valid >= MIN_NAMES_FOR_DECILES else N_QUINTILES_FALLBACK
