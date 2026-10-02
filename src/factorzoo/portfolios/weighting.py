"""Value- and equal-weighted portfolio weights within a bucket (build
guide Section 6: both are computed for every factor; value-weighted is
the primary series, equal-weighted is kept to compare against original
papers that report it).
"""

from __future__ import annotations

import pandas as pd


def value_weights(df: pd.DataFrame, bucket_col: str, market_cap_col: str = "market_cap") -> pd.Series:
    """Within each bucket, weight proportional to market cap, summing to
    1.0 per bucket. Rows with no bucket assignment get weight 0 (not NaN),
    so a weighted-average return computed with these weights never
    silently includes an ineligible row."""
    weights = pd.Series(0.0, index=df.index)
    if df.empty:
        # An empty-slice assignment into an empty float64 Series trips a
        # pandas dtype edge case (LossySetitemError) -- short-circuit
        # rather than let an empty cross-section (e.g. a formation date
        # with no eligible names yet) raise instead of returning nothing.
        return weights
    grouped = df.groupby(bucket_col, observed=True)[market_cap_col]
    bucket_totals = grouped.transform("sum")
    eligible = df[bucket_col].notna() & (bucket_totals > 0)
    weights.loc[eligible] = df.loc[eligible, market_cap_col] / bucket_totals.loc[eligible]
    return weights


def equal_weights(df: pd.DataFrame, bucket_col: str) -> pd.Series:
    weights = pd.Series(0.0, index=df.index)
    if df.empty:
        return weights
    grouped = df.groupby(bucket_col, observed=True)[bucket_col]
    bucket_counts = grouped.transform("count")
    eligible = df[bucket_col].notna() & (bucket_counts > 0)
    weights.loc[eligible] = 1.0 / bucket_counts.loc[eligible]
    return weights
