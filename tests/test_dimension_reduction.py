"""Offline tests for correlation clustering and the LASSO spanning test."""

from __future__ import annotations

import numpy as np
import pandas as pd

from factorzoo.taming.dimension_reduction import (
    correlation_clusters,
    lasso_spanning_test,
    select_cluster_representatives,
)


def _factor_returns_with_near_duplicates(n=120, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base_momentum = rng.normal(0.01, 0.05, n)
    base_value = rng.normal(0.005, 0.04, n)
    return pd.DataFrame(
        {
            "momentum_12_1": base_momentum,
            "momentum_6_1": base_momentum + rng.normal(0, 0.005, n),  # near-duplicate of momentum_12_1
            "short_term_reversal": -base_momentum + rng.normal(0, 0.005, n),  # near-NEGATIVE duplicate
            "book_to_market": base_value,
            "earnings_to_price": base_value + rng.normal(0, 0.005, n),  # near-duplicate of book_to_market
            "size": rng.normal(0.0, 0.03, n),  # genuinely independent
        }
    )


class TestCorrelationClusters:
    def test_near_duplicates_land_in_the_same_cluster(self):
        df = _factor_returns_with_near_duplicates()
        clusters = correlation_clusters(df, distance_threshold=0.3)
        assert clusters["momentum_12_1"] == clusters["momentum_6_1"]

    def test_negatively_correlated_near_duplicate_also_clusters(self):
        # short_term_reversal is built as -momentum + small noise: an
        # abs-correlation-based distance should still cluster it with
        # the momentum factors, since |corr| is high even though corr is
        # strongly negative.
        df = _factor_returns_with_near_duplicates()
        clusters = correlation_clusters(df, distance_threshold=0.3)
        assert clusters["short_term_reversal"] == clusters["momentum_12_1"]

    def test_independent_factor_lands_in_its_own_cluster(self):
        df = _factor_returns_with_near_duplicates()
        clusters = correlation_clusters(df, distance_threshold=0.3)
        momentum_cluster = clusters["momentum_12_1"]
        value_cluster = clusters["book_to_market"]
        assert clusters["size"] != momentum_cluster
        assert clusters["size"] != value_cluster

    def test_momentum_cluster_and_value_cluster_are_different(self):
        df = _factor_returns_with_near_duplicates()
        clusters = correlation_clusters(df, distance_threshold=0.3)
        assert clusters["momentum_12_1"] != clusters["book_to_market"]

    def test_sparse_columns_with_no_pairwise_overlap_do_not_raise(self):
        # Confirmed live on a real cross-section: two factors can each
        # individually have plenty of non-null values but almost no
        # overlap with EACH OTHER, giving a NaN pairwise correlation even
        # though neither column is degenerate alone. squareform/linkage
        # require every distance to be finite, so an unfilled NaN used to
        # crash with an opaque scipy error instead of just treating
        # "can't tell" as "uncorrelated."
        n = 40
        df = pd.DataFrame(
            {
                "dense_a": np.linspace(0, 1, n),
                "dense_b": np.linspace(1, 0, n),
                "sparse_early": [float(i) if i < 15 else np.nan for i in range(n)],
                "sparse_late": [float(i) if i >= 25 else np.nan for i in range(n)],
            }
        )
        clusters = correlation_clusters(df, distance_threshold=0.3)  # must not raise
        assert set(clusters.index) == set(df.columns)


class TestSelectClusterRepresentatives:
    def test_picks_the_highest_tstat_factor_per_cluster(self):
        clusters = pd.Series({"a": 1, "b": 1, "c": 2})
        tstats = pd.Series({"a": 2.0, "b": 5.0, "c": 1.0})
        reps = select_cluster_representatives(clusters, tstats)
        assert set(reps) == {"b", "c"}

    def test_handles_negative_tstats_by_magnitude(self):
        clusters = pd.Series({"a": 1, "b": 1})
        tstats = pd.Series({"a": -6.0, "b": 2.0})
        reps = select_cluster_representatives(clusters, tstats)
        assert reps == ["a"]

    def test_one_representative_per_singleton_cluster(self):
        clusters = pd.Series({"a": 1, "b": 2, "c": 3})
        tstats = pd.Series({"a": 1.0, "b": 1.0, "c": 1.0})
        reps = select_cluster_representatives(clusters, tstats)
        assert set(reps) == {"a", "b", "c"}


class TestLassoSpanningTest:
    def test_candidate_fully_explained_by_spanning_factors_has_small_alpha(self):
        rng = np.random.default_rng(1)
        n = 200
        spanning = pd.DataFrame({"mkt": rng.normal(0.01, 0.04, n), "smb": rng.normal(0.0, 0.02, n)})
        candidate = 0.8 * spanning["mkt"] + 0.3 * spanning["smb"] + rng.normal(0, 0.001, n)
        result = lasso_spanning_test(candidate, spanning)
        assert abs(result["alpha_tstat"]) < 2.0
        assert result["adds_value"] is False

    def test_candidate_with_genuine_independent_alpha_is_detected(self):
        rng = np.random.default_rng(2)
        n = 200
        spanning = pd.DataFrame({"mkt": rng.normal(0.01, 0.04, n), "smb": rng.normal(0.0, 0.02, n)})
        # Candidate has a real, persistent mean return NOT explained by
        # the spanning factors (near-zero loading on both).
        candidate = pd.Series(0.015 + rng.normal(0, 0.005, n))
        result = lasso_spanning_test(candidate, spanning)
        assert abs(result["alpha_tstat"]) > 2.0
        assert result["adds_value"] is True

    def test_irrelevant_spanning_factors_tend_to_be_dropped_by_lasso(self):
        rng = np.random.default_rng(3)
        n = 300
        spanning = pd.DataFrame(
            {
                "mkt": rng.normal(0.01, 0.04, n),
                "irrelevant_1": rng.normal(0, 0.03, n),
                "irrelevant_2": rng.normal(0, 0.03, n),
            }
        )
        candidate = 0.9 * spanning["mkt"] + rng.normal(0, 0.001, n)
        result = lasso_spanning_test(candidate, spanning)
        assert "mkt" in result["selected_factors"]

    def test_empty_spanning_set_returns_nan_gracefully(self):
        candidate = pd.Series(np.random.default_rng(4).normal(size=50))
        result = lasso_spanning_test(candidate, pd.DataFrame(index=candidate.index))
        assert np.isnan(result["alpha_tstat"])
        assert result["adds_value"] is False

    def test_too_few_observations_returns_nan_gracefully(self):
        candidate = pd.Series([0.01, 0.02, 0.03])
        spanning = pd.DataFrame({"mkt": [0.01, 0.02, 0.03]})
        result = lasso_spanning_test(candidate, spanning)
        assert np.isnan(result["alpha_tstat"])
