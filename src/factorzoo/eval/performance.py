"""Return/risk performance statistics, principally the Newey-West
adjusted t-statistic on a value-weighted long-short return, which
corrects for autocorrelation and is standard in this literature.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm


def newey_west_tstat(returns: pd.Series, max_lags: int | None = None) -> float:
    """t-statistic on the mean of `returns`, HAC (Newey-West) corrected
    for autocorrelation -- regress returns on a constant and read the
    constant's t-stat under a Newey-West covariance estimator, which is
    algebraically the standard way to get a HAC-corrected mean t-stat.

    `max_lags` defaults to the common rule of thumb floor(4*(n/100)^(2/9))
    (Newey & West 1994) when not given explicitly.
    """
    clean = returns.dropna()
    n = len(clean)
    if n < 3:
        return np.nan
    if max_lags is None:
        max_lags = max(1, int(np.floor(4 * (n / 100) ** (2 / 9))))
    x = np.ones(n)
    model = sm.OLS(clean.to_numpy(), x).fit(cov_type="HAC", cov_kwds={"maxlags": max_lags})
    return float(model.tvalues[0])


def sharpe_ratio(returns: pd.Series, periods_per_year: int = 12) -> float:
    clean = returns.dropna()
    if len(clean) < 2:
        return np.nan
    std = clean.std(ddof=1)
    # A "constant" series of floats (e.g. [0.01] * 12) is almost never
    # EXACTLY 0.0 std due to floating-point representation -- `== 0` lets
    # a near-zero denominator through and returns a huge-but-finite
    # number instead of the NaN a zero-volatility series should produce.
    if np.isclose(std, 0.0, atol=1e-12):
        return np.nan
    return float(clean.mean() / std * np.sqrt(periods_per_year))


def summary_stats(returns: pd.Series, periods_per_year: int = 12) -> dict:
    clean = returns.dropna()
    return {
        "n_obs": len(clean),
        "mean": float(clean.mean()) if len(clean) else np.nan,
        "std": float(clean.std(ddof=1)) if len(clean) > 1 else np.nan,
        "sharpe": sharpe_ratio(clean, periods_per_year),
        "newey_west_tstat": newey_west_tstat(clean),
    }
