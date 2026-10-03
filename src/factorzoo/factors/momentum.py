"""Momentum factors (factors #1-6). All operate on
the monthly price panel (factors/panel.py's `build_monthly_price_panel`),
grouped and rolled per entity_id_hint -- a momentum column computed
without grouping by entity rolls returns across different companies at
the panel's seams, so every rolling window here is explicitly
per-entity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from factorzoo.factors.registry import FactorSpec, register


def _sorted_groups(p: pd.DataFrame):
    return p.sort_values(["entity_id_hint", "month"]).groupby("entity_id_hint", group_keys=False)


def _cumulative_return_window(p: pd.DataFrame, window: int, skip: int) -> pd.Series:
    """Cumulative return over `window` months ending `skip` months before
    the formation month (e.g. window=11, skip=1 => months t-12..t-2,
    "12-1 momentum"). Always computed within one entity's own time series.
    """
    p_sorted = p.sort_values(["entity_id_hint", "month"])

    def _roll(s: pd.Series) -> pd.Series:
        return (1 + s).rolling(window).apply(np.prod, raw=True).shift(skip) - 1

    return p_sorted.groupby("entity_id_hint")["monthly_return"].transform(_roll).reindex(p.index)


momentum_12_1 = register(
    FactorSpec(
        name="momentum_12_1",
        category="momentum",
        description="Cumulative return from month t-12 to t-2, skipping the most recent month.",
        source="Jegadeesh & Titman (1993)",
        lag_days=0,
        direction=+1,
        exclude_financials=False,
        panel="monthly",
        compute=lambda p: _cumulative_return_window(p, window=11, skip=1),
    )
)

momentum_6_1 = register(
    FactorSpec(
        name="momentum_6_1",
        category="momentum",
        description="Cumulative return from month t-6 to t-2.",
        source="Jegadeesh & Titman (1993)",
        lag_days=0,
        direction=+1,
        exclude_financials=False,
        panel="monthly",
        compute=lambda p: _cumulative_return_window(p, window=5, skip=1),
    )
)

week_52_high = register(
    FactorSpec(
        name="week_52_high",
        category="momentum",
        description="Month-end price divided by its trailing 12-month high.",
        source="George & Hwang (2004)",
        lag_days=0,
        direction=+1,
        exclude_financials=False,
        panel="monthly",
        compute=lambda p: p["close_month_end"]
        / _sorted_groups(p)["close_month_end"].transform(lambda s: s.rolling(12, min_periods=1).max()).reindex(
            p.index
        ),
    )
)

long_term_reversal = register(
    FactorSpec(
        name="long_term_reversal",
        category="momentum",
        description="Cumulative return from month t-60 to t-13.",
        source="De Bondt & Thaler (1985)",
        lag_days=0,
        direction=-1,
        exclude_financials=False,
        panel="monthly",
        compute=lambda p: _cumulative_return_window(p, window=47, skip=13),
    )
)

short_term_reversal = register(
    FactorSpec(
        name="short_term_reversal",
        category="momentum",
        description="Prior one-month return.",
        source="Jegadeesh (1990)",
        lag_days=0,
        direction=-1,
        exclude_financials=False,
        panel="monthly",
        compute=lambda p: p["monthly_return"],
    )
)


def _industry_momentum(p: pd.DataFrame) -> pd.Series:
    """Industry-average 12-1 momentum. Needs a `sic` column joined onto
    the monthly panel (see `attach_industry` below) -- the one factor in
    this registry whose input isn't the bare panel from
    `build_monthly_price_panel`. Degrades to all-NaN, not a crash, if
    `sic` isn't present, since FactorSpec.compute's contract elsewhere in
    this project assumes it can be called on the bare monthly panel.
    """
    if "sic" not in p.columns:
        return pd.Series(np.nan, index=p.index)
    own_momentum = _cumulative_return_window(p, window=11, skip=1)
    tmp = p.assign(_own_momentum=own_momentum)
    industry_avg = tmp.groupby(["sic", "month"])["_own_momentum"].transform("mean")
    return industry_avg


industry_momentum = register(
    FactorSpec(
        name="industry_momentum",
        category="momentum",
        description="Industry-average (SIC code) 12-1 momentum. Requires the monthly panel to "
        "have a `sic` column pre-joined via momentum.attach_industry(con, panel) -- the only "
        "factor in this registry with that extra requirement, since SIC lives in EDGAR's "
        "submissions endpoint, not in price data).",
        source="Moskowitz & Grinblatt (1999)",
        lag_days=0,
        direction=+1,
        exclude_financials=False,
        panel="monthly",
        compute=_industry_momentum,
    )
)


def attach_industry(con, monthly_panel: pd.DataFrame) -> pd.DataFrame:
    """Joins `sic` onto the monthly price panel via entity_crosswalk
    (ticker -> sic), for industry_momentum's use. Separate from
    factors/panel.py's build_monthly_price_panel because every other
    momentum/trading-frictions factor has no use for it.
    """
    crosswalk = con.execute("SELECT ticker, sic FROM entity_crosswalk").fetchdf()
    return monthly_panel.merge(crosswalk, left_on="entity_id_hint", right_on="ticker", how="left").drop(
        columns=["ticker"]
    )
