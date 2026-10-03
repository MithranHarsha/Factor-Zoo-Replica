"""Trading-frictions / risk factors (factors
#36-43). Size uses the annual panel (market cap already lives there via
factors/panel.py's attach_market_cap); the rest are pure price-based
signals on the monthly panel. A few (idiosyncratic volatility, beta,
coskewness) use a simplified rolling approximation rather than a full
daily multi-factor regression -- documented per factor, not silently
substituted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from factorzoo.factors.panel import get_monthly_market_return
from factorzoo.factors.registry import FactorSpec, register

ROLLING_WINDOW_MONTHS = 36  # practical substitute for a 60-month window when history is shorter


def _sorted(p: pd.DataFrame) -> pd.DataFrame:
    return p.sort_values(["entity_id_hint", "month"])


size = register(
    FactorSpec(
        name="size",
        category="trading_frictions",
        description="Log of market cap.",
        source="Banz (1981)",
        lag_days=0,
        direction=-1,
        exclude_financials=False,
        panel="annual",
        compute=lambda p: np.log(p["market_cap"]),
    )
)


def _idiosyncratic_volatility(p: pd.DataFrame) -> pd.Series:
    """Simplified proxy: std. dev. of (stock monthly return - market
    monthly return) over a trailing window, rather than the literal
    "daily FF3-residual std dev within the prior month" definition (which
    needs daily FF3 factor data regressed per company-month -- out of
    scope for this phase; see market_beta's note for the same trade-off).
    """
    if "mkt_return" not in p.columns:
        return pd.Series(np.nan, index=p.index)
    excess = p["monthly_return"] - p["mkt_return"]
    return (
        p.assign(_excess=excess)
        .sort_values(["entity_id_hint", "month"])
        .groupby("entity_id_hint")["_excess"]
        .transform(lambda s: s.rolling(ROLLING_WINDOW_MONTHS, min_periods=6).std())
        .reindex(p.index)
    )


idiosyncratic_volatility = register(
    FactorSpec(
        name="idiosyncratic_volatility",
        category="trading_frictions",
        description="Std. dev. of (monthly return - market return) over a trailing 36-month "
        "window -- a simplified proxy for the literal daily Fama-French-3 residual std. dev. "
        "within the prior month, which needs daily factor regressions out of this phase's scope. "
        "Requires `mkt_return` joined onto the panel (see attach_market_return below).",
        source="Ang, Hodrick, Xing & Zhang (2006)",
        lag_days=0,
        direction=-1,
        exclude_financials=False,
        panel="monthly",
        compute=_idiosyncratic_volatility,
    )
)

amihud_illiquidity = register(
    FactorSpec(
        name="amihud_illiquidity",
        category="trading_frictions",
        description="Average of |daily return| / dollar volume within the month, approximated at "
        "monthly resolution as |monthly return| / (month-end close x average daily volume) since "
        "this panel does not retain every daily observation at the factor-computation stage.",
        source="Amihud (2002)",
        lag_days=0,
        direction=+1,
        exclude_financials=False,
        panel="monthly",
        compute=lambda p: p["monthly_return"].abs() / (p["close_month_end"] * p["volume_avg"]).replace(0, np.nan),
    )
)

share_turnover = register(
    FactorSpec(
        name="share_turnover",
        category="trading_frictions",
        description="Average daily dollar volume divided by month-end price (a volume-in-shares "
        "proxy; true turnover needs shares outstanding, joined in by the caller if available).",
        source="Datar, Naik & Radcliffe (1998)",
        lag_days=0,
        direction=-1,
        exclude_financials=False,
        panel="monthly",
        compute=lambda p: p["volume_avg"],
    )
)

max_daily_return = register(
    FactorSpec(
        name="max_daily_return",
        category="trading_frictions",
        description="Maximum daily return within the prior month.",
        source="Bali, Cakici & Whitelaw (2011)",
        lag_days=0,
        direction=-1,
        exclude_financials=False,
        panel="monthly",
        compute=lambda p: p["max_daily_return"],
    )
)


def _market_beta(p: pd.DataFrame) -> pd.Series:
    """Rolling 36-month beta via the covariance/variance identity
    (equivalent to a univariate OLS slope), not the more conventional
    60-month window -- shortened window
    since 60 months of monthly history isn't available this early in the
    project, and the cov/var shortcut avoids a per-group statsmodels loop.
    """
    if "mkt_return" not in p.columns:
        return pd.Series(np.nan, index=p.index)

    def _beta(group: pd.DataFrame) -> pd.Series:
        cov = group["monthly_return"].rolling(ROLLING_WINDOW_MONTHS, min_periods=6).cov(group["mkt_return"])
        var = group["mkt_return"].rolling(ROLLING_WINDOW_MONTHS, min_periods=6).var()
        return cov / var

    return _sorted(p).groupby("entity_id_hint", group_keys=False).apply(_beta).reindex(p.index)


market_beta = register(
    FactorSpec(
        name="market_beta",
        category="trading_frictions",
        description="Rolling 36-month CAPM beta vs. the Fama-French market return (cov/var "
        "shortcut, equivalent to an OLS slope). Shortened from the more conventional 60-month "
        "window since this early in the project there isn't 60 months of panel history yet; "
        "revisit once more history has accumulated. Requires `mkt_return` joined onto the panel.",
        source="Frazzini & Pedersen (2014)",
        lag_days=0,
        direction=-1,
        exclude_financials=False,
        panel="monthly",
        compute=_market_beta,
    )
)

total_volatility = register(
    FactorSpec(
        name="total_volatility",
        category="trading_frictions",
        description="Std. dev. of daily returns within the prior month (computed in the monthly "
        "panel itself, factors/panel.py).",
        source="Ang, Hodrick, Xing & Zhang (2006)",
        lag_days=0,
        direction=-1,
        exclude_financials=False,
        panel="monthly",
        compute=lambda p: p["volatility_daily_std"],
    )
)


def _coskewness(p: pd.DataFrame) -> pd.Series:
    """Harvey-Siddique-style coskewness, simplified to demeaned raw
    returns rather than regression residuals (a standard practical
    simplification -- the sign and ranking are materially similar for
    this purpose): mean(e_i * e_m^2) / (std(e_i) * var(e_m)), rolling
    36 months.
    """
    if "mkt_return" not in p.columns:
        return pd.Series(np.nan, index=p.index)

    def _roll(group: pd.DataFrame) -> pd.Series:
        r = group["monthly_return"]
        m = group["mkt_return"]
        e_i = r - r.rolling(ROLLING_WINDOW_MONTHS, min_periods=6).mean()
        e_m = m - m.rolling(ROLLING_WINDOW_MONTHS, min_periods=6).mean()
        numerator = (e_i * e_m**2).rolling(ROLLING_WINDOW_MONTHS, min_periods=6).mean()
        denom = e_i.rolling(ROLLING_WINDOW_MONTHS, min_periods=6).std() * (
            e_m.rolling(ROLLING_WINDOW_MONTHS, min_periods=6).var()
        )
        return numerator / denom

    return _sorted(p).groupby("entity_id_hint", group_keys=False).apply(_roll).reindex(p.index)


coskewness = register(
    FactorSpec(
        name="coskewness",
        category="trading_frictions",
        description="Rolling 36-month coskewness with the market return, simplified to demeaned "
        "raw returns rather than full regression residuals. Requires `mkt_return` joined onto the "
        "panel.",
        source="Harvey & Siddique (2000)",
        lag_days=0,
        direction=-1,
        exclude_financials=False,
        panel="monthly",
        compute=_coskewness,
    )
)


def attach_market_return(con, monthly_panel: pd.DataFrame) -> pd.DataFrame:
    """Joins the monthly market return (factors/panel.py's
    get_monthly_market_return) onto the price panel, for the three
    factors above that need it (idiosyncratic_volatility, market_beta,
    coskewness)."""
    market = get_monthly_market_return(con)
    return monthly_panel.merge(market, on="month", how="left")
