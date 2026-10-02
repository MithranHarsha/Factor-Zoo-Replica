"""Value factors (build guide Section 5, factors #7-12). All operate on the
annual snapshot with market_cap attached (factors/panel.py) -- every one
of these needs both a fundamentals figure and a market value.
"""

from __future__ import annotations

from factorzoo.factors.registry import FactorSpec, register

book_to_market = register(
    FactorSpec(
        name="book_to_market",
        category="value",
        description="Book equity (point-in-time) divided by market cap at formation.",
        source="Fama & French (1992)",
        lag_days=91,
        direction=+1,
        exclude_financials=False,
        panel="annual",
        compute=lambda p: p["stockholders_equity"] / p["market_cap"],
    )
)

earnings_to_price = register(
    FactorSpec(
        name="earnings_to_price",
        category="value",
        description="Net income divided by market cap.",
        source="Basu (1983)",
        lag_days=91,
        direction=+1,
        exclude_financials=False,
        panel="annual",
        compute=lambda p: p["net_income"] / p["market_cap"],
    )
)

cash_flow_to_price = register(
    FactorSpec(
        name="cash_flow_to_price",
        category="value",
        description="Operating cash flow divided by market cap.",
        source="Lakonishok, Shleifer & Vishny (1994)",
        lag_days=91,
        direction=+1,
        exclude_financials=False,
        panel="annual",
        compute=lambda p: p["cfo"] / p["market_cap"],
    )
)

sales_to_price = register(
    FactorSpec(
        name="sales_to_price",
        category="value",
        description="Revenues divided by market cap.",
        source="Barbee, Mukherji & Raines (1996)",
        lag_days=91,
        direction=+1,
        exclude_financials=False,
        panel="annual",
        compute=lambda p: p["revenues"] / p["market_cap"],
    )
)

dividend_yield = register(
    FactorSpec(
        name="dividend_yield",
        category="value",
        description="Trailing annual dividends paid divided by market cap (XBRL PaymentsOfDividends, "
        "not a price vendor's dividend field -- see build guide Section 4).",
        source="Litzenberger & Ramaswamy (1979)",
        lag_days=91,
        direction=+1,
        exclude_financials=False,
        panel="annual",
        # PaymentsOfDividends is reported as a positive cash outflow magnitude.
        compute=lambda p: p["dividends_paid"].abs() / p["market_cap"],
    )
)

ebitda_to_ev = register(
    FactorSpec(
        name="ebitda_to_ev",
        category="value",
        description="(Operating income + D&A) divided by (market cap + total debt - cash).",
        source="Loughran & Wellman (2011)",
        lag_days=91,
        direction=+1,
        exclude_financials=False,
        panel="annual",
        compute=lambda p: (p["operating_income"] + p["depreciation"].fillna(0))
        / (
            p["market_cap"]
            + p[["lt_debt_noncurrent", "lt_debt_current", "st_borrowings"]].fillna(0).sum(axis=1)
            - p["cash"].fillna(0)
        ),
    )
)
