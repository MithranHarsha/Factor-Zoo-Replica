"""The shuffle test (build guide Section 8, leakage test #4): randomly
reshuffle one factor's cross-sectional values within each formation date
(destroys real predictive content but preserves the panel's shape) and
confirm its long-short t-statistic collapses toward zero. A basic sanity
check that the whole pipeline isn't manufacturing significance from a bug
-- the kind of bug that produces a real-looking t-stat regardless of
whether the factor values are actually informative (a stale join, a sign
flip baked into the data path, a leakage channel elsewhere in the
pipeline) would NOT disappear under shuffling, which is exactly what this
test is for.

Leakage tests #1-3 from Section 8 (every factor value's available_date
<= formation date; an amendment never overwrites a pre-amendment vintage;
universe size/coverage reported per year) are exercised directly by
pit_store.get_facts_as_of and universe.filters.summarize_coverage and
their own tests (test_pit_store.py, test_universe.py) -- this module is
specifically the shuffle test, #4.
"""

from __future__ import annotations

import numpy as np

from factorzoo.eval.performance import newey_west_tstat
from factorzoo.portfolios.returns import factor_return_series


def shuffle_factor_within_date(panel_by_date: dict, factor_col: str, seed: int | None = None) -> dict:
    """Returns a NEW panel_by_date dict where `factor_col`'s values are
    randomly permuted across names WITHIN each date -- the cross-section
    on any one date still has the same set of factor values, just
    reassigned to different companies, so market-cap/return
    distributions and cross-sectional moments are unchanged, only the
    pairing between a company and its factor value is destroyed.
    """
    rng = np.random.default_rng(seed)
    shuffled = {}
    for date, df in panel_by_date.items():
        new_df = df.copy()
        values = new_df[factor_col].to_numpy().copy()
        rng.shuffle(values)
        new_df[factor_col] = values
        shuffled[date] = new_df
    return shuffled


def run_shuffle_test(
    panel_by_date: dict,
    factor_col: str,
    direction: int,
    return_col: str = "forward_return",
    n_shuffles: int = 10,
    seed: int = 0,
    collapse_threshold: float = 2.0,
) -> dict:
    """Computes the real factor's Newey-West t-stat, then the same
    statistic averaged over `n_shuffles` independent within-date
    shufflings. `passed` is True when the real |t-stat| exceeds the
    shuffled average by at least `collapse_threshold`x (a looser bar than
    literally "near zero", since a small panel's shuffled t-stat is noisy
    rather than exactly 0) -- this is a sanity check on the PIPELINE, not
    a claim that the factor itself is significant (that's Section 7's
    job).
    """
    real_series = factor_return_series(panel_by_date, factor_col, direction, return_col)
    real_tstat = newey_west_tstat(real_series["vw_long_short"])

    shuffled_tstats = []
    for i in range(n_shuffles):
        shuffled_panel = shuffle_factor_within_date(panel_by_date, factor_col, seed=seed + i)
        shuffled_series = factor_return_series(shuffled_panel, factor_col, direction, return_col)
        shuffled_tstats.append(newey_west_tstat(shuffled_series["vw_long_short"]))

    shuffled_tstats = [t for t in shuffled_tstats if not np.isnan(t)]
    mean_shuffled_abs_tstat = float(np.mean(np.abs(shuffled_tstats))) if shuffled_tstats else np.nan

    passed = (
        not np.isnan(real_tstat)
        and not np.isnan(mean_shuffled_abs_tstat)
        and abs(real_tstat) >= collapse_threshold * max(mean_shuffled_abs_tstat, 1e-6)
    )
    return {
        "factor": factor_col,
        "real_tstat": real_tstat,
        "mean_shuffled_abs_tstat": mean_shuffled_abs_tstat,
        "n_shuffles": len(shuffled_tstats),
        "passed": bool(passed),
    }
