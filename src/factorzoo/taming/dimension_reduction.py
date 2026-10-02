"""Step 2 of statistical taming (build guide Section 7): correlation
clustering to find near-duplicate factors, and a LASSO/elastic-net
spanning test (Feng, Giglio & Xiu's two-pass idea, simplified) to check
whether a candidate factor adds explanatory power beyond a spanning set
of existing ones.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.linear_model import LassoCV


def correlation_clusters(factor_returns: pd.DataFrame, distance_threshold: float = 0.3) -> pd.Series:
    """Hierarchical (average-linkage) clustering on a 1 - |correlation|
    distance -- |correlation| rather than signed correlation, so two
    factors that are near-perfect NEGATIVE images of each other (a common
    way the same underlying signal shows up twice with an inconsistent
    sign convention) cluster together too, not just positively-correlated
    near-duplicates.
    """
    corr = factor_returns.corr()
    distance = 1 - corr.abs()
    # .to_numpy(copy=True), not .values: pandas' copy-on-write can hand
    # back a read-only view, and np.fill_diagonal mutates in place --
    # confirmed live by this exact call raising "underlying array is
    # read-only" against the pandas version this project pins.
    distance_arr = distance.to_numpy(copy=True)
    np.fill_diagonal(distance_arr, 0.0)
    # Floating-point noise can make the distance matrix only
    # approximately symmetric; squareform is strict about exact symmetry.
    distance_arr = (distance_arr + distance_arr.T) / 2
    condensed = squareform(distance_arr, checks=False)
    z = linkage(condensed, method="average")
    labels = fcluster(z, t=distance_threshold, criterion="distance")
    return pd.Series(labels, index=factor_returns.columns, name="cluster")


def select_cluster_representatives(cluster_labels: pd.Series, abs_tstats: pd.Series) -> list[str]:
    """One representative per cluster: the factor with the largest
    |t-statistic| in that cluster (build guide Section 7: "keep one
    representative factor per cluster")."""
    aligned = abs_tstats.reindex(cluster_labels.index).abs()
    df = pd.DataFrame({"cluster": cluster_labels, "abs_t": aligned})
    return df.groupby("cluster")["abs_t"].apply(lambda s: s.idxmax()).tolist()


def lasso_spanning_test(
    candidate: pd.Series, spanning_factors: pd.DataFrame, cv: int = 5, random_state: int = 0
) -> dict:
    """Feng-Giglio-Xiu-style spanning test, simplified to a practical
    two-step procedure: (1) LASSO selects which spanning factors have a
    nonzero relationship with the candidate's returns (standardized first,
    so the penalty is comparable across factors of different scale); (2)
    an un-penalized OLS refit on just the selected factors ("post-LASSO")
    gives a proper HAC-robust standard error on the intercept, which is
    the candidate's "alpha" -- the part of its return the selected
    spanning factors don't explain. A candidate with a significant alpha
    adds information beyond what's already in the spanning set; one
    without is likely redundant.
    """
    merged = pd.concat([candidate.rename("_candidate"), spanning_factors], axis=1).dropna()
    if len(merged) < 10 or spanning_factors.shape[1] == 0:
        return {"alpha": np.nan, "alpha_tstat": np.nan, "selected_factors": [], "adds_value": False, "n_obs": len(merged)}

    y = merged["_candidate"].to_numpy()
    x_df = merged.drop(columns=["_candidate"])
    x = x_df.to_numpy()
    x_std = (x - x.mean(axis=0)) / np.clip(x.std(axis=0, ddof=1), 1e-12, None)

    n_folds = max(2, min(cv, len(merged) // 2))
    lasso = LassoCV(cv=n_folds, random_state=random_state).fit(x_std, y)
    selected_mask = np.abs(lasso.coef_) > 1e-10
    selected_cols = x_df.columns[selected_mask].tolist()

    design = sm.add_constant(x_df[selected_cols].to_numpy()) if selected_cols else np.ones((len(y), 1))
    max_lags = max(1, int(np.floor(4 * (len(y) / 100) ** (2 / 9))))
    ols = sm.OLS(y, design).fit(cov_type="HAC", cov_kwds={"maxlags": max_lags})

    return {
        "alpha": float(ols.params[0]),
        "alpha_tstat": float(ols.tvalues[0]),
        "selected_factors": selected_cols,
        "adds_value": bool(abs(ols.tvalues[0]) > 2.0),
        "n_obs": len(merged),
    }
