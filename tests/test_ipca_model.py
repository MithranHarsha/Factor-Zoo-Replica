"""Offline tests for the IPCA wrapper. Fits are
on small synthetic panels -- IPCA's ALS estimator is iterative and not
free, so these stay small (tens of entities, tens of periods) rather than
pilot-scale, which is enough to confirm the wrapper's data plumbing and
result shapes are correct.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from factorzoo.taming.ipca_model import (
    fit_ipca,
    fit_ipca_for_k_range,
    pca_benchmark_r2,
    standardize_characteristics,
)


def _synthetic_panel(n_entities=25, n_time=20, n_chars=3, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    beta_true = rng.normal(size=n_chars)
    factor_true = rng.normal(0.01, 0.05, n_time)
    for e in range(n_entities):
        chars = rng.normal(size=n_chars)  # time-invariant characteristics, kept simple
        for t in range(n_time):
            ret = float(chars @ beta_true) * factor_true[t] + rng.normal(0, 0.005)
            rows.append({"entity": e, "time": t, "c0": chars[0], "c1": chars[1], "c2": chars[2], "ret": ret})
    df = pd.DataFrame(rows).set_index(["entity", "time"])
    return df[["c0", "c1", "c2"]], df["ret"]


class TestStandardizeCharacteristics:
    def test_output_bounded_between_negative_half_and_half(self):
        panel = pd.DataFrame(
            {
                "time": [1, 1, 1, 2, 2, 2],
                "c0": [10.0, 20.0, 5.0, 100.0, 50.0, 75.0],
            }
        )
        result = standardize_characteristics(panel, time_col="time")
        assert result["c0"].min() >= -0.5
        assert result["c0"].max() <= 0.5

    def test_preserves_within_period_ranking(self):
        panel = pd.DataFrame({"time": [1, 1, 1], "c0": [5.0, 1.0, 9.0]})
        result = standardize_characteristics(panel, time_col="time")
        # Largest raw value should map to the largest standardized value.
        assert result["c0"].iloc[2] > result["c0"].iloc[0] > result["c0"].iloc[1]

    def test_single_name_in_a_period_does_not_raise(self):
        panel = pd.DataFrame({"time": [1, 2, 2], "c0": [5.0, 1.0, 9.0]})
        result = standardize_characteristics(panel, time_col="time")
        assert result["c0"].iloc[0] == 0.0  # degenerate single-name period


class TestFitIpcaForKRange:
    def test_returns_one_row_per_k(self):
        chars, returns = _synthetic_panel()
        result = fit_ipca_for_k_range(chars, returns, k_range=range(1, 3))
        assert list(result["n_factors"]) == [1, 2]

    def test_r_squared_values_are_bounded_reasonably(self):
        chars, returns = _synthetic_panel()
        result = fit_ipca_for_k_range(chars, returns, k_range=range(1, 2))
        # Total R2 for a model fit on its own training data should be
        # high and not above 1 (allowing tiny numerical slack).
        assert 0.0 <= result["total_r2"].iloc[0] <= 1.0001

    def test_predictive_r2_is_not_larger_than_total_r2_by_much(self):
        # Predictive R2 (using the time-averaged factor) is a harder
        # standard than total R2 (using the actual realized factor); it
        # should generally be lower, not dramatically higher.
        chars, returns = _synthetic_panel(n_time=40, seed=1)
        result = fit_ipca_for_k_range(chars, returns, k_range=range(1, 2))
        row = result.iloc[0]
        assert row["predictive_r2"] <= row["total_r2"] + 0.05


class TestFitIpca:
    def test_gamma_indexed_by_characteristic_names(self):
        chars, returns = _synthetic_panel()
        result = fit_ipca(chars, returns, n_factors=2)
        assert list(result["gamma"].index) == ["c0", "c1", "c2"]
        assert result["gamma"].shape == (3, 2)

    def test_factors_indexed_by_time(self):
        chars, returns = _synthetic_panel(n_time=15)
        result = fit_ipca(chars, returns, n_factors=1)
        assert len(result["factors"]) == 15
        assert list(result["factors"].columns) == ["ipca_factor_1"]

    def test_result_has_both_r_squared_values(self):
        chars, returns = _synthetic_panel()
        result = fit_ipca(chars, returns, n_factors=1)
        assert "total_r2" in result
        assert "predictive_r2" in result


class TestPcaBenchmarkR2:
    def test_single_dominant_factor_gives_high_explained_variance(self):
        rng = np.random.default_rng(2)
        n_time, n_entities = 60, 20
        common_factor = rng.normal(0, 0.05, n_time)
        returns = pd.DataFrame(
            {f"e{i}": common_factor * rng.uniform(0.8, 1.2) + rng.normal(0, 0.002, n_time) for i in range(n_entities)}
        )
        r2 = pca_benchmark_r2(returns, n_factors=1)
        assert r2 > 0.8

    def test_too_few_entities_returns_nan(self):
        returns = pd.DataFrame({"e1": [0.01, 0.02, 0.03]})
        assert np.isnan(pca_benchmark_r2(returns, n_factors=3))
