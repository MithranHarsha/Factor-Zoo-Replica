"""Offline tests for decile sorting, weighting, and the long-short spread
(build guide Section 6). Synthetic cross-sections with known, hand-worked
answers -- these are the mechanics every factor's backtest depends on, so
bugs here would silently corrupt every single factor's result.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factorzoo.portfolios.returns import bucket_returns, factor_return_series, long_short_spread
from factorzoo.portfolios.sorts import assign_quantile_buckets, n_buckets_used
from factorzoo.portfolios.weighting import equal_weights, value_weights


def _make_cross_section(n=400, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    factor = rng.normal(size=n)
    # Deliberately correlate forward return with the factor so a
    # long-short spread computed on it should come out positive for a
    # direction=+1 factor -- an end-to-end sanity check, not just shape.
    forward_return = 0.01 * factor + rng.normal(scale=0.005, size=n)
    market_cap = rng.lognormal(mean=10, sigma=1.5, size=n)
    return pd.DataFrame(
        {
            "entity_id": [f"E{i}" for i in range(n)],
            "my_factor": factor,
            "market_cap": market_cap,
            "forward_return": forward_return,
        }
    )


class TestAssignQuantileBuckets:
    def test_large_universe_uses_deciles(self):
        df = _make_cross_section(n=400)
        buckets = assign_quantile_buckets(df, "my_factor")
        assert n_buckets_used(df, "my_factor") == 10
        assert buckets.dropna().max() <= 10
        assert buckets.dropna().min() >= 1

    def test_small_universe_falls_back_to_quintiles(self):
        df = _make_cross_section(n=100)
        assert n_buckets_used(df, "my_factor") == 5
        buckets = assign_quantile_buckets(df, "my_factor")
        assert buckets.dropna().max() <= 5

    def test_rows_missing_factor_value_are_left_unassigned(self):
        df = _make_cross_section(n=400)
        df.loc[0, "my_factor"] = np.nan
        buckets = assign_quantile_buckets(df, "my_factor")
        assert pd.isna(buckets.loc[0])

    def test_rows_missing_market_cap_are_left_unassigned(self):
        df = _make_cross_section(n=400)
        df.loc[0, "market_cap"] = np.nan
        buckets = assign_quantile_buckets(df, "my_factor")
        assert pd.isna(buckets.loc[0])

    def test_breakpoints_come_from_large_cap_subset_not_whole_universe(self):
        # Build a universe where the factor values among the largest 25%
        # by market cap are systematically different from the small-cap
        # majority. The breakpoints must reflect the large-cap subset,
        # not the full-universe distribution -- that's the entire point
        # of the proxy (build guide Section 6).
        n = 400
        df = pd.DataFrame(
            {
                "entity_id": [f"E{i}" for i in range(n)],
                "market_cap": [1000.0] * 100 + [1.0] * 300,  # top 100 = top 25%
                # Large caps: factor values 0..99. Small caps: all 9999
                # (would blow out the breakpoints if wrongly included).
                "my_factor": list(range(100)) + [9999] * 300,
            }
        )
        buckets = assign_quantile_buckets(df, "my_factor")
        # The large-cap names (0..99) should span the full bucket range
        # 1..10 based on their own distribution.
        large_cap_buckets = buckets.iloc[:100]
        assert large_cap_buckets.min() == 1
        assert large_cap_buckets.max() == 10
        # Every small-cap name (value 9999) is far above all large-cap
        # breakpoints, so all 300 of them land in the top bucket.
        small_cap_buckets = buckets.iloc[100:]
        assert (small_cap_buckets == 10).all()

    def test_degenerate_single_value_breakpoint_set_does_not_raise(self):
        df = pd.DataFrame(
            {"entity_id": ["E1", "E2", "E3"], "market_cap": [100.0, 100.0, 100.0], "my_factor": [5.0, 5.0, 5.0]}
        )
        buckets = assign_quantile_buckets(df, "my_factor")
        assert (buckets == 1).all()

    def test_empty_frame_returns_empty_series_not_an_error(self):
        df = pd.DataFrame({"entity_id": [], "market_cap": [], "my_factor": []})
        buckets = assign_quantile_buckets(df, "my_factor")
        assert buckets.empty


class TestWeighting:
    def test_value_weights_sum_to_one_within_bucket(self):
        df = _make_cross_section(n=400)
        buckets = assign_quantile_buckets(df, "my_factor")
        work = df.assign(_bucket=buckets)
        w = value_weights(work, "_bucket")
        totals = work.assign(_w=w).groupby("_bucket", observed=True)["_w"].sum()
        assert np.allclose(totals.to_numpy(), 1.0)

    def test_equal_weights_sum_to_one_within_bucket(self):
        df = _make_cross_section(n=400)
        buckets = assign_quantile_buckets(df, "my_factor")
        work = df.assign(_bucket=buckets)
        w = equal_weights(work, "_bucket")
        totals = work.assign(_w=w).groupby("_bucket", observed=True)["_w"].sum()
        assert np.allclose(totals.to_numpy(), 1.0)

    def test_value_weight_proportional_to_market_cap_within_bucket(self):
        df = pd.DataFrame(
            {"entity_id": ["A", "B"], "market_cap": [100.0, 300.0], "_bucket": pd.array([1, 1], dtype="Int64")}
        )
        w = value_weights(df, "_bucket")
        assert w.iloc[0] == pytest.approx(0.25)
        assert w.iloc[1] == pytest.approx(0.75)

    def test_empty_dataframe_returns_empty_weights_not_an_error(self):
        df = pd.DataFrame(columns=["entity_id", "market_cap", "_bucket"])
        assert value_weights(df, "_bucket").empty
        assert equal_weights(df, "_bucket").empty

    def test_unassigned_rows_get_zero_weight(self):
        df = pd.DataFrame(
            {"entity_id": ["A", "B"], "market_cap": [100.0, 300.0], "_bucket": pd.array([1, pd.NA], dtype="Int64")}
        )
        w = value_weights(df, "_bucket")
        assert w.iloc[1] == 0.0


class TestBucketReturnsAndSpread:
    def test_bucket_return_is_weighted_average(self):
        df = pd.DataFrame(
            {
                "_bucket": pd.array([1, 1], dtype="Int64"),
                "_w": [0.25, 0.75],
                "ret": [0.10, 0.20],
            }
        )
        result = bucket_returns(df, "_bucket", "_w", "ret")
        assert result.loc[1] == pytest.approx(0.25 * 0.10 + 0.75 * 0.20)

    def test_weights_renormalize_among_rows_with_a_return(self):
        # Three names with equal weight, but one has no return this
        # period (delisted/missing). The other two should fully absorb
        # the weight, not silently count the missing one as a zero.
        df = pd.DataFrame(
            {
                "_bucket": pd.array([1, 1, 1], dtype="Int64"),
                "_w": [1 / 3, 1 / 3, 1 / 3],
                "ret": [0.10, 0.20, np.nan],
            }
        )
        result = bucket_returns(df, "_bucket", "_w", "ret")
        assert result.loc[1] == pytest.approx(0.5 * 0.10 + 0.5 * 0.20)

    def test_long_short_spread_direction_positive(self):
        bucket_ret = pd.Series({1: 0.01, 10: 0.05})
        assert long_short_spread(bucket_ret, direction=1, n_quantiles=10) == pytest.approx(0.04)

    def test_long_short_spread_direction_negative_flips_sign(self):
        bucket_ret = pd.Series({1: 0.01, 10: 0.05})
        # direction=-1 (e.g. asset growth): long the LOW bucket, short
        # the HIGH bucket, so the spread should be bottom - top.
        assert long_short_spread(bucket_ret, direction=-1, n_quantiles=10) == pytest.approx(-0.04)

    def test_long_short_spread_missing_extreme_bucket_returns_nan(self):
        bucket_ret = pd.Series({1: 0.01, 5: 0.02})  # no bucket 10
        assert np.isnan(long_short_spread(bucket_ret, direction=1, n_quantiles=10))


class TestFactorReturnSeries:
    def test_positive_direction_factor_on_correlated_data_yields_positive_mean_spread(self):
        # End-to-end sanity check: the synthetic cross-section was built
        # so the factor positively predicts forward returns; a
        # direction=+1 long-short spread, averaged over many independent
        # draws, should come out positive.
        panel_by_date = {pd.Timestamp("2020-01-31") + pd.DateOffset(months=i): _make_cross_section(seed=i) for i in range(24)}
        result = factor_return_series(panel_by_date, factor_col="my_factor", direction=1)
        assert result["vw_long_short"].mean() > 0
        assert result["ew_long_short"].mean() > 0

    def test_negated_direction_flips_the_sign_of_the_mean_spread(self):
        panel_by_date = {pd.Timestamp("2020-01-31") + pd.DateOffset(months=i): _make_cross_section(seed=i) for i in range(24)}
        positive = factor_return_series(panel_by_date, factor_col="my_factor", direction=1)
        negative = factor_return_series(panel_by_date, factor_col="my_factor", direction=-1)
        assert negative["vw_long_short"].mean() == pytest.approx(-positive["vw_long_short"].mean())

    def test_output_has_one_row_per_date_with_data(self):
        panel_by_date = {
            pd.Timestamp("2020-01-31"): _make_cross_section(seed=1),
            pd.Timestamp("2020-02-29"): pd.DataFrame(columns=["entity_id", "my_factor", "market_cap", "forward_return"]),
        }
        result = factor_return_series(panel_by_date, factor_col="my_factor", direction=1)
        assert len(result) == 1  # the empty month is skipped, not a row of NaN
