"""Offline tests for Newey-West t-statistics and basic performance stats
(build guide Section 7)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factorzoo.eval.performance import newey_west_tstat, sharpe_ratio, summary_stats


class TestNeweyWestTstat:
    def test_strong_positive_mean_low_noise_gives_large_positive_tstat(self):
        rng = np.random.default_rng(0)
        returns = pd.Series(rng.normal(0.02, 0.001, 200))
        assert newey_west_tstat(returns) > 10

    def test_zero_mean_noise_gives_small_tstat_in_magnitude(self):
        rng = np.random.default_rng(1)
        returns = pd.Series(rng.normal(0.0, 0.02, 500))
        assert abs(newey_west_tstat(returns)) < 3  # not a hard guarantee, but should usually hold

    def test_matches_textbook_tstat_closely_when_returns_are_iid(self):
        # With no real autocorrelation, the HAC-adjusted t-stat should be
        # close to the ordinary t-stat (mean / (std/sqrt(n))).
        rng = np.random.default_rng(2)
        returns = pd.Series(rng.normal(0.01, 0.05, 300))
        nw = newey_west_tstat(returns, max_lags=1)
        textbook = returns.mean() / (returns.std(ddof=1) / np.sqrt(len(returns)))
        assert nw == pytest.approx(textbook, rel=0.2)

    def test_nan_values_are_dropped_not_propagated(self):
        returns = pd.Series([0.01, np.nan, 0.012, 0.009, np.nan, 0.011] * 20)
        result = newey_west_tstat(returns)
        assert not np.isnan(result)

    def test_too_few_observations_returns_nan(self):
        assert np.isnan(newey_west_tstat(pd.Series([0.01, 0.02])))

    def test_autocorrelated_series_gives_smaller_tstat_than_naive_textbook_formula(self):
        # Positive autocorrelation inflates the naive standard error's
        # understatement of true uncertainty; NW correction should pull
        # the t-stat down relative to the (overstated) textbook one.
        rng = np.random.default_rng(4)
        n = 300
        eps = rng.normal(0, 0.01, n)
        returns = np.zeros(n)
        returns[0] = 0.01
        for i in range(1, n):
            returns[i] = 0.01 + 0.7 * (returns[i - 1] - 0.01) + eps[i]
        returns = pd.Series(returns)
        nw = newey_west_tstat(returns)
        textbook = returns.mean() / (returns.std(ddof=1) / np.sqrt(len(returns)))
        assert abs(nw) < abs(textbook)


class TestSharpeRatio:
    def test_known_value(self):
        returns_noisy = pd.Series(np.array([0.01, 0.02, 0.0, 0.015, 0.005, 0.01, 0.02, 0.0, 0.015, 0.005, 0.01, 0.02]))
        expected = returns_noisy.mean() / returns_noisy.std(ddof=1) * np.sqrt(12)
        assert sharpe_ratio(returns_noisy) == pytest.approx(expected)

    def test_zero_volatility_returns_nan(self):
        returns = pd.Series([0.01] * 12)
        assert np.isnan(sharpe_ratio(returns))


class TestSummaryStats:
    def test_returns_all_expected_keys(self):
        rng = np.random.default_rng(6)
        returns = pd.Series(rng.normal(0.01, 0.02, 60))
        result = summary_stats(returns)
        assert set(result.keys()) == {"n_obs", "mean", "std", "sharpe", "newey_west_tstat"}
        assert result["n_obs"] == 60
