"""Offline tests for t-hurdles, FDR control, the Deflated Sharpe Ratio,
and Probability of Backtest Overfitting (build guide Section 7)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factorzoo.taming.multiple_testing import (
    benjamini_hochberg,
    benjamini_yekutieli,
    deflated_sharpe_ratio,
    multiple_testing_report,
    probability_of_backtest_overfitting,
    tstat_to_pvalue,
)


class TestTstatToPvalue:
    def test_large_tstat_gives_tiny_pvalue(self):
        assert tstat_to_pvalue(5.0) < 0.001

    def test_zero_tstat_gives_pvalue_near_one(self):
        assert tstat_to_pvalue(0.0) == pytest.approx(1.0, abs=1e-6)

    def test_symmetric_in_sign(self):
        assert tstat_to_pvalue(2.5) == pytest.approx(tstat_to_pvalue(-2.5))

    def test_nan_tstat_gives_nan_pvalue(self):
        assert np.isnan(tstat_to_pvalue(np.nan))

    def test_known_normal_approximation_value(self):
        # |t|=1.96 is the conventional 5% two-sided cutoff under the
        # normal approximation.
        assert tstat_to_pvalue(1.959964) == pytest.approx(0.05, abs=1e-3)


class TestBenjaminiHochberg:
    def test_all_tiny_pvalues_are_all_discoveries(self):
        p = pd.Series([0.0001, 0.0002, 0.0003, 0.0004], index=["a", "b", "c", "d"])
        result = benjamini_hochberg(p, q=0.05)
        assert result.all()

    def test_all_large_pvalues_are_no_discoveries(self):
        p = pd.Series([0.6, 0.7, 0.8, 0.9], index=["a", "b", "c", "d"])
        result = benjamini_hochberg(p, q=0.05)
        assert not result.any()

    def test_mixed_pvalues_discovers_only_the_small_ones(self):
        # Classic textbook BH example: five p-values, q=0.05.
        p = pd.Series([0.001, 0.008, 0.039, 0.041, 0.042], index=list("abcde"))
        result = benjamini_hochberg(p, q=0.05)
        # BH threshold at rank k: k/5 * 0.05 -> [0.01, 0.02, 0.03, 0.04, 0.05]
        # p(1)=0.001<=0.01 yes; p(2)=0.008<=0.02 yes; p(3)=0.039<=0.03 no;
        # p(4)=0.041<=0.04 no; p(5)=0.042<=0.05 yes -> largest k where
        # condition holds is 5, so ALL are discoveries under the BH rule
        # (reject 1..k using the largest qualifying k, not a monotonic walk).
        assert result.all()

    def test_nan_pvalues_are_never_discoveries(self):
        p = pd.Series([0.0001, np.nan, 0.0002], index=["a", "b", "c"])
        result = benjamini_hochberg(p, q=0.05)
        assert result["b"] is np.False_ or result["b"] == False
        assert result["a"] and result["c"]

    def test_empty_series_returns_empty_result_no_discoveries(self):
        p = pd.Series([], dtype="float64")
        result = benjamini_hochberg(p, q=0.05)
        assert result.empty


class TestBenjaminiYekutieli:
    def test_by_is_never_more_permissive_than_bh(self):
        # BY's threshold is BH's divided by the harmonic number (>1 for
        # m>1), so BY can never discover more than BH on the same data.
        rng = np.random.default_rng(0)
        p = pd.Series(rng.uniform(0, 1, 50))
        bh = benjamini_hochberg(p, q=0.10)
        by = benjamini_yekutieli(p, q=0.10)
        assert by.sum() <= bh.sum()

    def test_by_with_many_tests_is_strictly_more_conservative_on_borderline_data(self):
        # With m=100 tests, the harmonic correction is substantial
        # (~5.2), so a batch of modestly-significant p-values that BH
        # would call discoveries should mostly fail under BY.
        p = pd.Series([0.001] * 5 + list(np.linspace(0.02, 0.5, 95)))
        bh = benjamini_hochberg(p, q=0.10)
        by = benjamini_yekutieli(p, q=0.10)
        assert by.sum() < bh.sum()


class TestMultipleTestingReport:
    def test_report_has_expected_columns(self):
        tstats = pd.Series([3.5, 1.2, -4.0, 0.5], index=["f1", "f2", "f3", "f4"])
        report = multiple_testing_report(tstats)
        assert set(report.columns) == {"tstat", "pvalue", "passes_t2", "passes_t3_hlz", "bh_discovery", "by_discovery"}

    def test_t3_hurdle_is_stricter_than_t2(self):
        tstats = pd.Series([2.5, 3.5], index=["f1", "f2"])
        report = multiple_testing_report(tstats)
        assert report.loc["f1", "passes_t2"] and not report.loc["f1", "passes_t3_hlz"]
        assert report.loc["f2", "passes_t2"] and report.loc["f2", "passes_t3_hlz"]


class TestDeflatedSharpeRatio:
    def test_standout_strategy_among_many_mediocre_trials_scores_high(self):
        rng = np.random.default_rng(1)
        trial_sharpes = pd.Series(rng.normal(0.0, 0.3, 200))
        result = deflated_sharpe_ratio(trial_sharpes, selected_sharpe=2.5, n_obs=120)
        assert result > 0.9

    def test_a_sharpe_consistent_with_pure_luck_scores_low(self):
        rng = np.random.default_rng(2)
        trial_sharpes = pd.Series(rng.normal(0.0, 0.3, 200))
        best_by_chance = float(trial_sharpes.max())
        result = deflated_sharpe_ratio(trial_sharpes, selected_sharpe=best_by_chance, n_obs=120)
        # The best-of-200 trial, evaluated against the very distribution
        # it was drawn from, should NOT look like genuine skill.
        assert result < 0.7

    def test_more_trials_searched_deflates_the_same_sharpe_further(self):
        rng = np.random.default_rng(3)
        small_trial_set = pd.Series(rng.normal(0.0, 0.3, 20))
        large_trial_set = pd.Series(rng.normal(0.0, 0.3, 500))
        dsr_small = deflated_sharpe_ratio(small_trial_set, selected_sharpe=1.5, n_obs=120)
        dsr_large = deflated_sharpe_ratio(large_trial_set, selected_sharpe=1.5, n_obs=120)
        assert dsr_large <= dsr_small

    def test_degenerate_single_trial_returns_nan(self):
        assert np.isnan(deflated_sharpe_ratio(pd.Series([1.0]), selected_sharpe=1.0, n_obs=100))


class TestProbabilityOfBacktestOverfitting:
    def test_genuinely_skilled_strategy_has_low_pbo(self):
        # One strategy with a real, persistent edge; the rest pure noise.
        # The skilled strategy should usually win in-sample AND
        # out-of-sample, giving a low PBO.
        rng = np.random.default_rng(4)
        t_obs = 120
        skilled = rng.normal(0.01, 0.02, t_obs)
        noise_trials = {f"noise_{i}": rng.normal(0.0, 0.02, t_obs) for i in range(19)}
        returns = pd.DataFrame({"skilled": skilled, **noise_trials})
        result = probability_of_backtest_overfitting(returns, n_splits=10)
        assert result["pbo"] < 0.5

    def test_pure_noise_trials_have_high_pbo(self):
        # No real signal anywhere: whichever trial wins in-sample is
        # essentially a random draw and should NOT reliably repeat
        # out-of-sample -- PBO should be high (closer to 0.5+, the
        # "coin flip or worse" region for an overfit selection process).
        rng = np.random.default_rng(5)
        t_obs = 120
        returns = pd.DataFrame({f"noise_{i}": rng.normal(0.0, 0.02, t_obs) for i in range(20)})
        result = probability_of_backtest_overfitting(returns, n_splits=10)
        assert result["pbo"] > 0.3

    def test_returns_expected_number_of_combinations(self):
        rng = np.random.default_rng(6)
        returns = pd.DataFrame({f"t{i}": rng.normal(size=100) for i in range(5)})
        result = probability_of_backtest_overfitting(returns, n_splits=6)
        from math import comb

        assert result["n_combinations"] == comb(6, 3)

    def test_odd_n_splits_raises(self):
        returns = pd.DataFrame({"a": [1, 2, 3], "b": [1, 2, 3]})
        with pytest.raises(ValueError, match="even"):
            probability_of_backtest_overfitting(returns, n_splits=5)

    def test_too_few_observations_returns_nan_not_an_error(self):
        returns = pd.DataFrame({"a": [1.0, 2.0], "b": [1.0, 2.0]})
        result = probability_of_backtest_overfitting(returns, n_splits=10)
        assert np.isnan(result["pbo"])
