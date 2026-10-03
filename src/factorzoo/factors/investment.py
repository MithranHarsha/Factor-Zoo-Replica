"""Investment factors (factors #19-23). All are
negative-weight: firms that invested/grew the most have historically
earned LOWER subsequent returns."""

from __future__ import annotations

from factorzoo.factors.registry import FactorSpec, register

asset_growth = register(
    FactorSpec(
        name="asset_growth",
        category="investment",
        description="Assets(t) / Assets(t-1) - 1; higher growth predicts LOWER returns.",
        source="Cooper, Gulen & Schill (2008)",
        lag_days=91,
        direction=-1,
        exclude_financials=True,
        panel="annual",
        compute=lambda p: p["assets"] / p["assets_prior"] - 1,
    )
)

investment_to_assets = register(
    FactorSpec(
        name="investment_to_assets",
        category="investment",
        description="(Change in PP&E + change in inventory) divided by lagged assets.",
        source="Lyandres, Sun & Zhang (2008)",
        lag_days=91,
        direction=-1,
        exclude_financials=True,
        panel="annual",
        compute=lambda p: (
            (p["ppe_net"] - p["ppe_net_prior"]).fillna(0) + (p["inventory"] - p["inventory_prior"]).fillna(0)
        )
        / p["assets_prior"],
    )
)

net_operating_assets = register(
    FactorSpec(
        name="net_operating_assets",
        category="investment",
        description="(Operating assets - operating liabilities) divided by lagged assets, where "
        "operating assets = assets - cash and operating liabilities = assets - equity - debt.",
        source="Hirshleifer, Hou, Teoh & Zhang (2004)",
        lag_days=91,
        direction=-1,
        exclude_financials=True,
        panel="annual",
        compute=lambda p: (
            (p["assets"] - p["cash"].fillna(0))
            - (
                p["assets"]
                - p["stockholders_equity"]
                - p[["lt_debt_noncurrent", "lt_debt_current", "st_borrowings"]].fillna(0).sum(axis=1)
            )
        )
        / p["assets_prior"],
    )
)

capex_growth = register(
    FactorSpec(
        name="capex_growth",
        category="investment",
        description="Capex(t) / Capex(t-1) - 1.",
        source="Titman, Wei & Xie (2004)",
        lag_days=91,
        direction=-1,
        exclude_financials=True,
        panel="annual",
        compute=lambda p: p["capex"].abs() / p["capex_prior"].abs() - 1,
    )
)

investment_to_capital = register(
    FactorSpec(
        name="investment_to_capital",
        category="investment",
        description="Capex divided by lagged PP&E.",
        source="Xing (2008)",
        lag_days=91,
        direction=-1,
        exclude_financials=True,
        panel="annual",
        compute=lambda p: p["capex"].abs() / p["ppe_net_prior"],
    )
)
