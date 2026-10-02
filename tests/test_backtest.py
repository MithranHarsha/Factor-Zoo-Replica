"""Offline tests for the shuffle test and walk-forward split helper
(build guide Section 8)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factorzoo.backtest.leakage_tests import run_shuffle_test, shuffle_factor_within_date
from factorzoo.backtest.walkforward import expanding_windows, n_windows


def _informative_panel(n_dates=36, n_names=200, seed=0) -> dict:
    """A panel where the factor genuinely predicts forward returns --
    real_tstat should be large, and shuffling should destroy that."""
    rng = np.random.default_rng(seed)
    panel = {}
    for i in range(n_dates):
        date = pd.Timestamp("2020-01-31") + pd.DateOffset(months=i)
        factor = rng.normal(size=n_names)
        forward_return = 0.02 * factor + rng.normal(scale=0.01, size=n_names)
        market_cap = rng.lognormal(mean=10, sigma=1.5, size=n_names)
        panel[date] = pd.DataFrame(
            {
                "entity_id": [f"E{j}" for j in range(n_names)],
                "my_factor": factor,
                "market_cap": market_cap,
                "forward_return": forward_return,
            }
        )
    return panel


@pytest.mark.leakage
class TestShuffleFactorWithinDate:
    def test_shuffled_values_are_a_permutation_of_the_original_within_each_date(self):
        panel = _informative_panel(n_dates=3, n_names=50)
        shuffled = shuffle_factor_within_date(panel, "my_factor", seed=1)
        for date in panel:
            original_sorted = np.sort(panel[date]["my_factor"].to_numpy())
            shuffled_sorted = np.sort(shuffled[date]["my_factor"].to_numpy())
            np.testing.assert_allclose(original_sorted, shuffled_sorted)

    def test_other_columns_are_unchanged(self):
        panel = _informative_panel(n_dates=2, n_names=20)
        shuffled = shuffle_factor_within_date(panel, "my_factor", seed=1)
        for date in panel:
            pd.testing.assert_series_equal(panel[date]["market_cap"], shuffled[date]["market_cap"])

    def test_different_seeds_give_different_permutations(self):
        panel = _informative_panel(n_dates=1, n_names=100)
        a = shuffle_factor_within_date(panel, "my_factor", seed=1)
        b = shuffle_factor_within_date(panel, "my_factor", seed=2)
        date = next(iter(panel))
        assert not np.allclose(a[date]["my_factor"].to_numpy(), b[date]["my_factor"].to_numpy())


@pytest.mark.leakage
class TestRunShuffleTest:
    def test_informative_factor_passes(self):
        panel = _informative_panel()
        result = run_shuffle_test(panel, "my_factor", direction=1, n_shuffles=15, seed=0)
        assert result["passed"] is True
        assert abs(result["real_tstat"]) > abs(result["mean_shuffled_abs_tstat"])

    def test_pure_noise_factor_does_not_reliably_pass(self):
        # A factor with NO real relationship to forward returns: the real
        # t-stat should be comparable in magnitude to the shuffled ones,
        # not dramatically larger.
        rng = np.random.default_rng(9)
        panel = {}
        for i in range(36):
            date = pd.Timestamp("2020-01-31") + pd.DateOffset(months=i)
            n = 200
            panel[date] = pd.DataFrame(
                {
                    "entity_id": [f"E{j}" for j in range(n)],
                    "my_factor": rng.normal(size=n),
                    "market_cap": rng.lognormal(mean=10, sigma=1.5, size=n),
                    "forward_return": rng.normal(scale=0.01, size=n),  # independent of the factor
                }
            )
        result = run_shuffle_test(panel, "my_factor", direction=1, n_shuffles=15, seed=0, collapse_threshold=2.0)
        # Not asserting False outright (noise is noisy -- an occasional
        # false pass is expected by chance), but the real and shuffled
        # t-stats should be the same order of magnitude.
        assert abs(result["real_tstat"]) < 5 * max(result["mean_shuffled_abs_tstat"], 0.5)

    def test_result_has_expected_keys(self):
        panel = _informative_panel(n_dates=12)
        result = run_shuffle_test(panel, "my_factor", direction=1, n_shuffles=5)
        assert set(result.keys()) == {"factor", "real_tstat", "mean_shuffled_abs_tstat", "n_shuffles", "passed"}


class TestExpandingWindows:
    def test_yields_expected_number_of_windows(self):
        dates = [pd.Timestamp("2020-01-01") + pd.DateOffset(months=i) for i in range(24)]
        windows = list(expanding_windows(dates, min_train_periods=12, step=1))
        assert len(windows) == n_windows(len(dates), min_train_periods=12, step=1)
        assert len(windows) == 12

    def test_train_set_grows_each_window(self):
        dates = [pd.Timestamp("2020-01-01") + pd.DateOffset(months=i) for i in range(15)]
        windows = list(expanding_windows(dates, min_train_periods=10, step=1))
        train_sizes = [len(train) for train, _ in windows]
        assert train_sizes == sorted(train_sizes)
        assert train_sizes == list(range(10, 15))

    def test_test_set_never_overlaps_train_set(self):
        dates = [pd.Timestamp("2020-01-01") + pd.DateOffset(months=i) for i in range(20)]
        for train, test in expanding_windows(dates, min_train_periods=10, step=2):
            assert set(train).isdisjoint(set(test))
            assert max(train) < min(test)

    def test_unsorted_dates_raise(self):
        dates = [pd.Timestamp("2020-03-01"), pd.Timestamp("2020-01-01")]
        with pytest.raises(ValueError, match="sorted ascending"):
            list(expanding_windows(dates, min_train_periods=1))

    def test_step_larger_than_one_batches_test_dates(self):
        dates = [pd.Timestamp("2020-01-01") + pd.DateOffset(months=i) for i in range(16)]
        windows = list(expanding_windows(dates, min_train_periods=10, step=3))
        assert len(windows[0][1]) == 3
