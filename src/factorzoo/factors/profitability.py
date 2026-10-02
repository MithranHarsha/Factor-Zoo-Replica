"""Profitability factors (build guide Section 5, factors #13-18). All
operate on the annual snapshot (factors/panel.py); none need market_cap
except implicitly through exclusion rules, so these still work even before
price data is available -- unlike the value factors.
"""

from __future__ import annotations

import pandas as pd

from factorzoo.factors.registry import FactorSpec, register

gross_profitability = register(
    FactorSpec(
        name="gross_profitability",
        category="profitability",
        description="(Revenues - COGS) divided by total assets.",
        source="Novy-Marx (2013)",
        lag_days=91,
        direction=+1,
        exclude_financials=True,
        panel="annual",
        compute=lambda p: (p["revenues"] - p["cost_of_goods_sold"]) / p["assets"],
    )
)

operating_profitability = register(
    FactorSpec(
        name="operating_profitability",
        category="profitability",
        description="(Revenues - COGS - SG&A - interest expense) divided by book equity.",
        source="Fama & French (2015)",
        lag_days=91,
        direction=+1,
        exclude_financials=True,
        panel="annual",
        compute=lambda p: (
            p["revenues"] - p["cost_of_goods_sold"] - p["sga_expense"].fillna(0) - p["interest_expense"].fillna(0)
        )
        / p["stockholders_equity"],
    )
)

return_on_equity = register(
    FactorSpec(
        name="return_on_equity",
        category="profitability",
        description="Net income divided by lagged (prior fiscal year) book equity.",
        source="Haugen & Baker (1996)",
        lag_days=91,
        direction=+1,
        exclude_financials=True,
        panel="annual",
        compute=lambda p: p["net_income"] / p["stockholders_equity_prior"],
    )
)

return_on_assets = register(
    FactorSpec(
        name="return_on_assets",
        category="profitability",
        description="Net income divided by lagged (prior fiscal year) assets.",
        source="Balakrishnan, Bartov & Faurel (2010)",
        lag_days=91,
        direction=+1,
        exclude_financials=True,
        panel="annual",
        compute=lambda p: p["net_income"] / p["assets_prior"],
    )
)


def _piotroski_f_score(p):
    """Nine binary tests, summed. Known simplification: the snapshot
    (factors/panel.py) carries only the current and prior fiscal year for
    each tag, not a third year back, so the two tests that strictly need
    Assets(t-2) -- ROA improvement and asset-turnover improvement --
    substitute Assets(t-1) for Assets(t-2) in the prior-period
    denominator. This is a standard practical approximation when a
    two-vintage panel is what's available, not Piotroski's original
    three-year construction; results will differ slightly from a
    from-scratch three-year implementation. Missing inputs make that one
    test score 0 rather than raising, since not every filer reports every
    line item (e.g. smaller companies often omit a separate
    long-term-debt-current breakout).
    """

    def safe_gt(a, b):
        # pd.notna(), not a.notna()/b.notna(): two of the calls below
        # compare a Series against the plain scalar 0 (safe_gt(roa, 0)),
        # and a bare int has no .notna() method -- pd.notna() handles a
        # scalar or a Series uniformly.
        return ((a > b) & pd.notna(a) & pd.notna(b)).astype(int)

    roa = p["net_income"] / p["assets_prior"]
    roa_prior = p["net_income_prior"] / p.get("assets_prior_prior", p["assets_prior"])  # best-effort
    leverage = (p[["lt_debt_noncurrent", "lt_debt_current"]].fillna(0).sum(axis=1)) / p["assets"]
    leverage_prior = (p[["lt_debt_noncurrent_prior", "lt_debt_current_prior"]].fillna(0).sum(axis=1)) / p[
        "assets_prior"
    ]
    current_ratio = p["assets_current"] / p["liabilities_current"]
    current_ratio_prior = p["assets_current_prior"] / p["liabilities_current_prior"]
    gross_margin = (p["revenues"] - p["cost_of_goods_sold"]) / p["revenues"]
    gross_margin_prior = (p["revenues_prior"] - p["cost_of_goods_sold_prior"]) / p["revenues_prior"]
    asset_turnover = p["revenues"] / p["assets_prior"]
    asset_turnover_prior = p["revenues_prior"] / p.get("assets_prior_prior", p["assets_prior"])

    score = (
        safe_gt(roa, 0)
        + safe_gt(p["cfo"], 0)
        + safe_gt(roa, roa_prior)
        + safe_gt(p["cfo"], p["net_income"])  # accrual quality: CFO > net income
        + safe_gt(leverage_prior, leverage)  # leverage DECREASED
        + safe_gt(current_ratio, current_ratio_prior)
        + (p["shares_outstanding_gaap"].fillna(p["shares_outstanding_dei"]) <= p["shares_outstanding_gaap_prior"].fillna(p["shares_outstanding_dei_prior"])).astype(int)
        + safe_gt(gross_margin, gross_margin_prior)
        + safe_gt(asset_turnover, asset_turnover_prior)
    )
    return score


piotroski_f_score = register(
    FactorSpec(
        name="piotroski_f_score",
        category="profitability",
        description="Nine-point binary accounting quality score (profitability, leverage/liquidity, "
        "operating-efficiency tests).",
        source="Piotroski (2000)",
        lag_days=91,
        direction=+1,
        exclude_financials=True,
        panel="annual",
        compute=_piotroski_f_score,
    )
)

gross_margin_stability = register(
    FactorSpec(
        name="gross_margin_stability",
        category="profitability",
        description="Negative of the standard deviation of gross margin across the fiscal years "
        "present in the snapshot (current + prior; a fuller trailing window needs multiple "
        "snapshot calls accumulated over time, left to the caller).",
        source="Novy-Marx (2011)",
        lag_days=91,
        direction=+1,  # already sign-flipped in the computation: higher = more stable
        exclude_financials=True,
        panel="annual",
        compute=lambda p: -(
            (
                (p["revenues"] - p["cost_of_goods_sold"]) / p["revenues"]
                - (p["revenues_prior"] - p["cost_of_goods_sold_prior"]) / p["revenues_prior"]
            ).abs()
            / 2
        ),
    )
)
