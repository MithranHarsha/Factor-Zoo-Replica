"""Expanding-window walk-forward splits: fit any
model parameter on data through year Y, evaluate on year Y+1 only, and
roll forward. Falls out of the point-in-time store directly (any as-of
query only ever sees what was available by that date), so this module is
just the date bookkeeping on top of that -- time-based splits only, never
a random shuffle across the panel.
"""

from __future__ import annotations

from collections.abc import Iterator

import pandas as pd


def expanding_windows(
    dates: list[pd.Timestamp], min_train_periods: int, step: int = 1
) -> Iterator[tuple[list[pd.Timestamp], list[pd.Timestamp]]]:
    """Yields (train_dates, test_dates) pairs: train is every date up to
    and including some point, test is the next `step` dates after it.
    `dates` must already be sorted ascending -- this function trusts the
    caller rather than re-sorting (and silently masking a caller bug that
    assumed sorted input).
    """
    if dates != sorted(dates):
        raise ValueError("expanding_windows requires `dates` sorted ascending")
    n = len(dates)
    cursor = min_train_periods
    while cursor < n:
        train = dates[:cursor]
        test = dates[cursor : min(cursor + step, n)]
        yield train, test
        cursor += step


def n_windows(n_dates: int, min_train_periods: int, step: int = 1) -> int:
    if n_dates <= min_train_periods:
        return 0
    return -(-(n_dates - min_train_periods) // step)  # ceil division
