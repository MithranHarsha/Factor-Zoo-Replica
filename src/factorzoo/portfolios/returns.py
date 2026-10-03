"""Bucket-level portfolio returns and the long-short spread: Decile 10
minus Decile 1, sign-flipped according to the factor's direction field,
so the spread is always defined as high expected return minus low
expected return.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def bucket_returns(df: pd.DataFrame, bucket_col: str, weight_col: str, return_col: str) -> pd.Series:
    """Weighted average forward return per bucket. Weights are
    renormalized within each bucket among only the rows that actually
    have a return (a stock can have a valid formation-date weight but no
    next-period return -- it was delisted, or price data hasn't been
    pulled for it yet -- and naively applying the original weight would
    silently understate that bucket's return rather than reallocating
    among what's actually observed).
    """
    eligible = df[bucket_col].notna() & df[return_col].notna() & (df[weight_col] > 0)
    sub = df.loc[eligible].copy()
    if sub.empty:
        return pd.Series(dtype="float64")
    sub["_w_renorm"] = sub.groupby(bucket_col, observed=True)[weight_col].transform(lambda w: w / w.sum())
    sub["_contrib"] = sub["_w_renorm"] * sub[return_col]
    return sub.groupby(bucket_col, observed=True)["_contrib"].sum()


def long_short_spread(bucket_ret: pd.Series, direction: int, n_quantiles: int) -> float:
    """direction=+1: long the top bucket, short the bottom.
    direction=-1: long the bottom bucket, short the top (e.g. asset
    growth, where low values predict high returns). Either way the
    result is defined as "long leg minus short leg", i.e. always meant
    to be a positive-expected-return bet if the factor works.
    """
    if n_quantiles not in bucket_ret.index or 1 not in bucket_ret.index:
        return np.nan
    top = bucket_ret.loc[n_quantiles]
    bottom = bucket_ret.loc[1]
    return direction * (top - bottom)


def factor_return_series(
    panel_by_date: dict, factor_col: str, direction: int, return_col: str = "forward_return"
) -> pd.DataFrame:
    """Runs assign_quantile_buckets + both weighting schemes + the
    long-short spread across every formation date in `panel_by_date`
    (date -> cross-sectional DataFrame, already containing factor_col,
    market_cap, and return_col). Returns one row per date with the VW and
    EW long-short return -- this is the time series Section 7's
    statistical taming consumes.
    """
    from factorzoo.portfolios.sorts import assign_quantile_buckets, n_buckets_used
    from factorzoo.portfolios.weighting import equal_weights, value_weights

    rows = []
    for date, df in sorted(panel_by_date.items()):
        if df.empty or df[factor_col].notna().sum() == 0:
            continue
        buckets = assign_quantile_buckets(df, factor_col)
        n_q = n_buckets_used(df, factor_col)
        work = df.assign(_bucket=buckets)
        work["_vw"] = value_weights(work, "_bucket")
        work["_ew"] = equal_weights(work, "_bucket")

        vw_bucket_ret = bucket_returns(work, "_bucket", "_vw", return_col)
        ew_bucket_ret = bucket_returns(work, "_bucket", "_ew", return_col)

        rows.append(
            {
                "formation_date": date,
                "n_quantiles": n_q,
                "n_names": int(df[factor_col].notna().sum()),
                "vw_long_short": long_short_spread(vw_bucket_ret, direction, n_q),
                "ew_long_short": long_short_spread(ew_bucket_ret, direction, n_q),
            }
        )
    return pd.DataFrame(rows)
