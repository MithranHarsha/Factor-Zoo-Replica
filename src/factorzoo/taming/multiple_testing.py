"""Step 1 of statistical taming: t-stat hurdles,
false-discovery-rate control, the Deflated Sharpe Ratio, and the
Probability of Backtest Overfitting -- the tools that answer "how much of
the zoo's apparent performance is multiple-testing luck."
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.stats import t as student_t

EULER_GAMMA = 0.5772156649015329


def tstat_to_pvalue(tstat: float, df: int | None = None) -> float:
    """Two-sided p-value for a t-statistic. Uses a t-distribution with
    `df` degrees of freedom when given (appropriate for a small sample),
    otherwise the standard normal (the large-sample approximation most
    Newey-West t-stats in this literature are interpreted under)."""
    if pd.isna(tstat):
        return np.nan
    if df is not None and df > 0:
        return float(2 * student_t.sf(abs(tstat), df))
    return float(2 * norm.sf(abs(tstat)))


def benjamini_hochberg(pvalues: pd.Series, q: float = 0.05) -> pd.Series:
    """Classic BH (1995) FDR control. Returns a boolean Series aligned to
    `pvalues`' index: True = reject the null (a "discovery") at FDR level
    q. NaN p-values are never discoveries."""
    return _bh_family(pvalues, q, correction=1.0)


def benjamini_yekutieli(pvalues: pd.Series, q: float = 0.05) -> pd.Series:
    """BY (2001) FDR control: the same procedure as BH but with the
    threshold divided by c(m) = sum_{i=1}^{m} 1/i (the m-th harmonic
    number), which stays valid under arbitrary dependence between tests
    -- the reason to use it alongside BH, since this
    project's factors are correlated with each other by construction."""
    valid_m = pvalues.notna().sum()
    harmonic = float(np.sum(1.0 / np.arange(1, max(valid_m, 1) + 1)))
    return _bh_family(pvalues, q, correction=harmonic)


def _bh_family(pvalues: pd.Series, q: float, correction: float) -> pd.Series:
    clean = pvalues.dropna().sort_values()
    m = len(clean)
    result = pd.Series(False, index=pvalues.index)
    if m == 0:
        return result
    ranks = np.arange(1, m + 1)
    thresholds = ranks / (m * correction) * q
    below = clean.to_numpy() <= thresholds
    if not below.any():
        return result
    k = int(np.max(np.where(below)[0])) + 1
    result.loc[clean.index[:k]] = True
    return result


def multiple_testing_report(factor_tstats: pd.Series, df: int | None = None, fdr_q: float = 0.10) -> pd.DataFrame:
    """One row per factor: its t-stat, implied p-value, whether it clears
    the conventional (|t|>2) and Harvey-Liu-Zhu-elevated (|t|>3) hurdles,
    and whether it's a BH/BY discovery at `fdr_q`. This table IS Section
    7's "report both the simple t>3 count and the FDR-controlled count
    side by side."
    """
    pvalues = factor_tstats.apply(lambda tv: tstat_to_pvalue(tv, df=df))
    bh = benjamini_hochberg(pvalues, q=fdr_q)
    by = benjamini_yekutieli(pvalues, q=fdr_q)
    return pd.DataFrame(
        {
            "tstat": factor_tstats,
            "pvalue": pvalues,
            "passes_t2": factor_tstats.abs() > 2.0,
            "passes_t3_hlz": factor_tstats.abs() > 3.0,
            "bh_discovery": bh,
            "by_discovery": by,
        }
    )


# --- Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014) ------------------


def deflated_sharpe_ratio(
    trial_sharpe_ratios: pd.Series, selected_sharpe: float, n_obs: int, skew: float = 0.0, kurtosis: float = 3.0
) -> float:
    """Probability that the SELECTED strategy's true Sharpe ratio exceeds
    zero, after deflating for (a) having picked the best of
    `len(trial_sharpe_ratios)` independent trials and (b) non-normal
    returns. `kurtosis` is the ordinary (not excess) kurtosis; a normal
    distribution has kurtosis=3, which is this function's default.

    `trial_sharpe_ratios` should be the Sharpe ratios of the WHOLE zoo
    (or however many independent trials were searched over) -- it is what
    estimates the expected maximum Sharpe ratio under a null of no real
    skill, per Bailey & Lopez de Prado's closed-form approximation using
    the cross-sectional variance of trial Sharpe ratios.
    """
    n_trials = len(trial_sharpe_ratios)
    if n_trials < 2 or n_obs < 2:
        return np.nan
    var_sr = float(np.var(trial_sharpe_ratios, ddof=1))
    if var_sr <= 0:
        return np.nan
    sr_max_expected = np.sqrt(var_sr) * (
        (1 - EULER_GAMMA) * norm.ppf(1 - 1.0 / n_trials) + EULER_GAMMA * norm.ppf(1 - 1.0 / (n_trials * np.e))
    )
    sigma_sr_hat = np.sqrt(
        max(1 - skew * selected_sharpe + ((kurtosis - 1) / 4) * selected_sharpe**2, 1e-12) / (n_obs - 1)
    )
    z = (selected_sharpe - sr_max_expected) / sigma_sr_hat
    return float(norm.cdf(z))


# --- Probability of Backtest Overfitting, via CSCV --------------------------
# (Bailey, Borwein, Lopez de Prado & Zhu, 2017)


def probability_of_backtest_overfitting(returns_matrix: pd.DataFrame, n_splits: int = 10) -> dict:
    """Combinatorially Symmetric Cross-Validation: split the T return
    observations (rows of `returns_matrix`, one column per trial/strategy)
    into `n_splits` equal blocks, and for every way of combining half of
    them into a training set and the other half into a test set, pick the
    best-in-training strategy and see how it ranks out-of-sample. PBO is
    the fraction of combinations where the in-sample winner finishes in
    the bottom half out-of-sample -- direct evidence of overfitting to
    the backtest, not genuine skill.

    `n_splits` must be even; C(n_splits, n_splits/2) combinations are
    evaluated, so this grows fast (C(10,5)=252, C(16,8)=12,870) -- kept at
    a default of 10 deliberately, not the paper's sometimes-larger choices,
    to keep this tractable for an interactive pilot run.
    """
    if n_splits % 2 != 0:
        raise ValueError("n_splits must be even (CSCV splits into two equal-sized halves)")
    clean = returns_matrix.dropna(axis=1, how="all").dropna(axis=0, how="any")
    t_obs, n_trials = clean.shape
    if t_obs < n_splits or n_trials < 2:
        return {"pbo": np.nan, "n_combinations": 0, "logits": []}

    block_edges = np.array_split(np.arange(t_obs), n_splits)
    block_indices = list(range(n_splits))
    logits: list[float] = []

    for train_blocks in itertools.combinations(block_indices, n_splits // 2):
        test_blocks = [b for b in block_indices if b not in train_blocks]
        train_idx = np.concatenate([block_edges[b] for b in train_blocks])
        test_idx = np.concatenate([block_edges[b] for b in test_blocks])

        train_perf = clean.iloc[train_idx].mean()  # mean return as the in-sample performance metric
        test_perf = clean.iloc[test_idx].mean()

        best_in_sample = train_perf.idxmax()
        # relative rank of that SAME strategy, out-of-sample: 1 = worst, n_trials = best
        oos_rank = test_perf.rank(method="average")[best_in_sample]
        omega = oos_rank / (n_trials + 1)
        omega = min(max(omega, 1e-6), 1 - 1e-6)  # keep the logit finite
        logits.append(float(np.log(omega / (1 - omega))))

    pbo = float(np.mean(np.array(logits) <= 0))
    return {"pbo": pbo, "n_combinations": len(logits), "logits": logits}
