"""Intangibles / R&D factors (build guide Section 5, factors #32-35)."""

from __future__ import annotations

from factorzoo.factors.registry import FactorSpec, register

rd_to_market = register(
    FactorSpec(
        name="rd_to_market",
        category="intangibles",
        description="R&D expense divided by market cap.",
        source="Chan, Lakonishok & Sougiannis (2001)",
        lag_days=91,
        direction=+1,
        exclude_financials=False,
        panel="annual",
        compute=lambda p: p["rd_expense"].fillna(0) / p["market_cap"],
    )
)

rd_to_sales = register(
    FactorSpec(
        name="rd_to_sales",
        category="intangibles",
        description="R&D expense divided by revenues.",
        source="Chambers, Jennings & Thompson (2002)",
        lag_days=91,
        direction=+1,
        exclude_financials=False,
        panel="annual",
        compute=lambda p: p["rd_expense"].fillna(0) / p["revenues"],
    )
)

advertising_to_market = register(
    FactorSpec(
        name="advertising_to_market",
        category="intangibles",
        description="Proxy: 30% of SG&A (advertising is not broken out as its own standardized "
        "XBRL tag -- it is usually folded into SellingGeneralAndAdministrativeExpense -- so this "
        "is a documented proxy, not a direct observation; see build guide Section 4). Divided by "
        "market cap.",
        source="Chan, Lakonishok & Sougiannis (2001)",
        lag_days=91,
        direction=+1,
        exclude_financials=False,
        panel="annual",
        compute=lambda p: 0.30 * p["sga_expense"].fillna(0) / p["market_cap"],
    )
)

organizational_capital = register(
    FactorSpec(
        name="organizational_capital",
        category="intangibles",
        description="Capitalized SG&A (perpetual-inventory style: current SG&A plus 80% of the "
        "prior period's capitalized stock, a simplified one-period version of Eisfeldt-"
        "Papanikolaou's recursive formula) divided by assets.",
        source="Eisfeldt & Papanikolaou (2013)",
        lag_days=91,
        direction=+1,
        exclude_financials=True,
        panel="annual",
        compute=lambda p: (p["sga_expense"].fillna(0) + 0.80 * p["sga_expense_prior"].fillna(0)) / p["assets"],
    )
)
