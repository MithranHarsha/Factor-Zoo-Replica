"""Accruals / earnings-quality factors (factors
#24-27). All negative-weight: high accruals predict LOWER returns."""

from __future__ import annotations

from factorzoo.factors.registry import FactorSpec, register


def _avg_assets(p):
    return (p["assets"] + p["assets_prior"]) / 2


total_accruals = register(
    FactorSpec(
        name="total_accruals",
        category="accruals",
        description="(d(CurrentAssets) - d(Cash) - d(CurrentLiabilities) + d(ShortTermDebt) - "
        "Depreciation) / average total assets.",
        source="Sloan (1996)",
        lag_days=91,
        direction=-1,
        exclude_financials=True,
        panel="annual",
        compute=lambda p: (
            (p["assets_current"] - p["assets_current_prior"])
            - (p["cash"].fillna(0) - p["cash_prior"].fillna(0))
            - (p["liabilities_current"] - p["liabilities_current_prior"])
            + (p["st_borrowings"].fillna(0) - p["st_borrowings_prior"].fillna(0))
            - p["depreciation"].fillna(0)
        )
        / _avg_assets(p),
    )
)

change_in_net_operating_assets = register(
    FactorSpec(
        name="change_in_net_operating_assets",
        category="accruals",
        description="Year-over-year change in net operating assets, divided by lagged assets.",
        source="Hirshleifer, Hou, Teoh & Zhang (2004)",
        lag_days=91,
        direction=-1,
        exclude_financials=True,
        panel="annual",
        compute=lambda p: (
            (
                (p["assets"] - p["cash"].fillna(0))
                - (
                    p["assets"]
                    - p["stockholders_equity"]
                    - p[["lt_debt_noncurrent", "lt_debt_current", "st_borrowings"]].fillna(0).sum(axis=1)
                )
            )
            - (
                (p["assets_prior"] - p["cash_prior"].fillna(0))
                - (
                    p["assets_prior"]
                    - p["stockholders_equity_prior"]
                    - p[["lt_debt_noncurrent_prior", "lt_debt_current_prior", "st_borrowings_prior"]]
                    .fillna(0)
                    .sum(axis=1)
                )
            )
        )
        / p["assets_prior"],
    )
)

discretionary_accruals = register(
    FactorSpec(
        name="discretionary_accruals",
        category="accruals",
        description="Approximation of the Jones-model residual: total accruals in excess of the "
        "change in revenue scaled by lagged assets (a simplified proxy -- the full Jones model "
        "needs a cross-sectional regression per industry-year, deferred to the taming stage's "
        "LASSO/IPCA machinery rather than duplicated here).",
        source="Jones (1991); Xie (2001)",
        lag_days=91,
        direction=-1,
        exclude_financials=True,
        panel="annual",
        compute=lambda p: (
            (
                (p["assets_current"] - p["assets_current_prior"])
                - (p["cash"].fillna(0) - p["cash_prior"].fillna(0))
                - (p["liabilities_current"] - p["liabilities_current_prior"])
                + (p["st_borrowings"].fillna(0) - p["st_borrowings_prior"].fillna(0))
                - p["depreciation"].fillna(0)
            )
            / _avg_assets(p)
            - (p["revenues"] - p["revenues_prior"]) / p["assets_prior"]
        ),
    )
)

percent_operating_accruals = register(
    FactorSpec(
        name="percent_operating_accruals",
        category="accruals",
        description="Total accruals divided by the absolute value of net income.",
        source="Hafzalla, Lundholm & Van Winkle (2011)",
        lag_days=91,
        direction=-1,
        exclude_financials=True,
        panel="annual",
        compute=lambda p: (
            (p["assets_current"] - p["assets_current_prior"])
            - (p["cash"].fillna(0) - p["cash_prior"].fillna(0))
            - (p["liabilities_current"] - p["liabilities_current_prior"])
            + (p["st_borrowings"].fillna(0) - p["st_borrowings_prior"].fillna(0))
            - p["depreciation"].fillna(0)
        )
        / p["net_income"].abs(),
    )
)
