"""Instrumented Principal Component Analysis (Kelly, Pruitt & Su,
2019/2020): fit with 1 to 6 latent factors,
using firm characteristics as instruments, and compare total vs.
predictive R-squared to choose K. Thin wrapper around the `ipca` package
(github.com/bkelly-lab/ipca, `pip install ipca`), which does the actual
alternating-least-squares estimation; this module's job is getting data
into and summaries out of that package's expected shape.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from ipca import InstrumentedPCA


def standardize_characteristics(panel: pd.DataFrame, time_col: str = "time") -> pd.DataFrame:
    """Cross-sectional rank-standardization to [-0.5, 0.5] within each
    time period (Kelly-Pruitt-Su's convention): robust to outliers and
    puts every characteristic on a comparable scale regardless of its
    raw units, which matters because IPCA's factor loadings are a linear
    combination of the characteristics.
    """

    def _rank_standardize(s: pd.Series) -> pd.Series:
        ranks = s.rank(method="average")
        n = ranks.notna().sum()
        if n <= 1:
            return pd.Series(0.0, index=s.index)
        return (ranks - 1) / (n - 1) - 0.5

    char_cols = [c for c in panel.columns if c != time_col]
    return panel.groupby(time_col)[char_cols].transform(_rank_standardize)


def fit_ipca_for_k_range(
    characteristics: pd.DataFrame, returns: pd.Series, k_range: range = range(1, 7), intercept: bool = False
) -> pd.DataFrame:
    """`characteristics` and `returns` must share a (entity, time)
    MultiIndex (the `ipca` package's expected shape). Fits one model per
    K in `k_range` and reports both total R-squared (in-sample, using the
    actual realized factor each period) and predictive R-squared
    (out-of-sample-style, using the time-series average factor instead) --
    the exact total-vs-predictive comparison Section 7 asks for to choose
    the number of latent factors.
    """
    rows = []
    for k in k_range:
        model = InstrumentedPCA(n_factors=k, intercept=intercept)
        model = model.fit(X=characteristics, y=returns)
        total_r2 = model.score(X=characteristics, y=returns, mean_factor=False)
        predictive_r2 = model.score(X=characteristics, y=returns, mean_factor=True)
        rows.append(
            {
                "n_factors": k,
                "total_r2": float(total_r2),
                "predictive_r2": float(predictive_r2),
            }
        )
    return pd.DataFrame(rows)


def fit_ipca(characteristics: pd.DataFrame, returns: pd.Series, n_factors: int, intercept: bool = False) -> dict:
    """Fits a single IPCA model and returns its loadings (Gamma, one row
    per characteristic) and estimated factor realizations (Factors, one
    row per time period) as labeled DataFrames, not the package's bare
    numpy arrays -- so results can be joined back to characteristic/date
    names downstream (the dashboard, a written report) without the
    caller having to remember column order.
    """
    model = InstrumentedPCA(n_factors=n_factors, intercept=intercept)
    model = model.fit(X=characteristics, y=returns)
    gamma, factors = model.get_factors()

    char_names = list(characteristics.columns)
    time_values = sorted(characteristics.index.get_level_values(-1).unique())
    factor_names = [f"ipca_factor_{i + 1}" for i in range(n_factors)]

    gamma_df = pd.DataFrame(gamma, index=char_names, columns=factor_names)
    factors_df = pd.DataFrame(factors.T, index=time_values, columns=factor_names)

    return {
        "model": model,
        "gamma": gamma_df,
        "factors": factors_df,
        "total_r2": float(model.score(X=characteristics, y=returns, mean_factor=False)),
        "predictive_r2": float(model.score(X=characteristics, y=returns, mean_factor=True)),
    }


def pca_benchmark_r2(returns_wide: pd.DataFrame, n_factors: int) -> float:
    """Naive PCA-on-returns baseline, to compare against the IPCA fit
    above. `returns_wide` is a
    time x entity matrix (unlike IPCA's long panel); entities/periods with
    any missing data are dropped, since plain PCA has no missing-data
    handling of its own (that gap is exactly the kind of thing IPCA, built
    for unbalanced panels, is meant to handle better).
    """
    from sklearn.decomposition import PCA

    clean = returns_wide.dropna(axis=1, how="any").dropna(axis=0, how="any")
    if clean.shape[1] < n_factors or clean.shape[0] < n_factors:
        return np.nan
    pca = PCA(n_components=n_factors)
    pca.fit(clean.to_numpy())
    return float(np.sum(pca.explained_variance_ratio_))
