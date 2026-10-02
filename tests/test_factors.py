"""Offline tests for the 48-factor starter library (build guide Section
5). Two layers: a registry-wide smoke test (every factor computes without
error on a synthetic panel shaped the way it expects), and hand-computed
spot checks for a representative factor per category, including a direct
regression test for the groupby-across-companies bug the Phase 1 critique
flagged in momentum_12_1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import factorzoo.factors as f
from factorzoo.factors.panel import EXPECTED_ANNUAL_COLUMNS

ANNUAL_COLUMNS = list(EXPECTED_ANNUAL_COLUMNS)


def _synthetic_annual_panel() -> pd.DataFrame:
    """Three companies, each with current + prior fiscal year values for
    every column the registry's annual factors might touch. Values are
    chosen to be realistic-ish and to avoid accidental zeros in
    denominators, so a factor returning NaN signals a real bug, not a
    divide-by-zero fixture artifact.
    """
    rng = np.random.default_rng(42)
    rows = []
    for i, entity_id in enumerate(["E1", "E2", "E3"]):
        base = 1000.0 * (i + 1)
        row = {"entity_id": entity_id}
        for col in ANNUAL_COLUMNS:
            if col in ("entity_id", "market_cap", "price_at_formation"):
                continue
            is_prior = col.endswith("_prior")
            scale = base * rng.uniform(0.05, 0.5)
            row[col] = scale * (0.9 if is_prior else 1.0)
        # Keep a few economically sensible relationships so ratios aren't nonsense.
        row["assets"] = base
        row["assets_prior"] = base * 0.9
        row["liabilities"] = base * 0.4
        row["liabilities_prior"] = base * 0.38
        row["assets_current"] = base * 0.3
        row["assets_current_prior"] = base * 0.28
        row["liabilities_current"] = base * 0.15
        row["liabilities_current_prior"] = base * 0.14
        row["stockholders_equity"] = row["assets"] - row["liabilities"]
        row["stockholders_equity_prior"] = row["assets_prior"] - row["liabilities_prior"]
        row["revenues"] = base * 1.2
        row["revenues_prior"] = base * 1.1
        row["cost_of_goods_sold"] = base * 0.7
        row["cost_of_goods_sold_prior"] = base * 0.65
        row["net_income"] = base * 0.08
        row["net_income_prior"] = base * 0.07
        row["cfo"] = base * 0.1
        row["shares_outstanding_dei"] = 10_000_000.0 * (i + 1)
        row["shares_outstanding_dei_prior"] = 9_800_000.0 * (i + 1)
        row["shares_outstanding_gaap"] = row["shares_outstanding_dei"]
        row["shares_outstanding_gaap_prior"] = row["shares_outstanding_dei_prior"]
        row["market_cap"] = row["shares_outstanding_dei"] * (5.0 + i)
        row["price_at_formation"] = 5.0 + i
        rows.append(row)
    return pd.DataFrame(rows)


def _synthetic_monthly_panel(n_months: int = 72) -> pd.DataFrame:
    # 72 months, not 48: long_term_reversal alone needs a 47-month rolling
    # window shifted 13 months (60 months minimum) before it can produce
    # its first non-NaN value -- a shorter fixture would make every
    # factor's smoke test pass EXCEPT this one for a reason that has
    # nothing to do with its formula being correct, just fixture length.
    rng = np.random.default_rng(7)
    months = pd.date_range("2020-01-31", periods=n_months, freq="ME")
    rows = []
    for entity, drift in [("AAA", 0.01), ("BBB", -0.002), ("CCC", 0.004)]:
        price = 50.0
        for m in months:
            ret = rng.normal(drift, 0.05)
            price *= 1 + ret
            rows.append(
                {
                    "entity_id_hint": entity,
                    "month": m,
                    "monthly_return": ret,
                    "close_month_end": price,
                    "volume_avg": rng.uniform(1e5, 1e6),
                    "volatility_daily_std": rng.uniform(0.01, 0.03),
                    "max_daily_return": rng.uniform(0.01, 0.08),
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def annual_panel():
    return _synthetic_annual_panel()


@pytest.fixture(scope="module")
def monthly_panel():
    panel = _synthetic_monthly_panel()
    # Fake a market return series and sic codes so the factors that need
    # them (industry_momentum, idiosyncratic_volatility, market_beta,
    # coskewness) can be smoke-tested too, not just skipped.
    months = panel["month"].unique()
    market = pd.DataFrame({"month": months, "mkt_return": np.random.default_rng(1).normal(0.006, 0.04, len(months))})
    panel = panel.merge(market, on="month", how="left")
    sic_map = {"AAA": "7372", "BBB": "7372", "CCC": "2834"}
    panel["sic"] = panel["entity_id_hint"].map(sic_map)
    return panel


class TestRegistryShape:
    def test_exactly_48_factors_registered(self):
        assert len(f.FACTOR_REGISTRY) == 48

    def test_every_factor_has_a_valid_direction(self):
        for spec in f.FACTOR_REGISTRY.values():
            assert spec.direction in (1, -1), f"{spec.name} has invalid direction {spec.direction}"

    def test_every_factor_has_a_valid_panel_type(self):
        for spec in f.FACTOR_REGISTRY.values():
            assert spec.panel in ("annual", "monthly"), f"{spec.name} has invalid panel '{spec.panel}'"

    def test_category_counts_match_build_guide_table(self):
        expected = {
            "momentum": 6, "value": 6, "profitability": 6, "investment": 5,
            "accruals": 4, "financing": 4, "intangibles": 4, "trading_frictions": 8, "distress": 5,
        }
        from collections import Counter

        actual = Counter(spec.category for spec in f.FACTOR_REGISTRY.values())
        assert dict(actual) == expected

    def test_registering_a_duplicate_name_raises(self):
        from factorzoo.factors.registry import FactorSpec, register

        with pytest.raises(ValueError, match="already registered"):
            register(
                FactorSpec(
                    name="book_to_market", category="value", description="dup", source="x",
                    lag_days=0, direction=1, exclude_financials=False, panel="annual",
                    compute=lambda p: p["assets"],
                )
            )


class TestEveryFactorComputesWithoutError:
    """The registry-wide smoke test: every one of the 48 factors, run
    against a panel shaped the way its own spec declares, must return a
    numeric (or NaN) Series of the same length as the input -- not raise."""

    def test_all_annual_factors_compute(self, annual_panel):
        failures = []
        for spec in f.FACTOR_REGISTRY.values():
            if spec.panel != "annual":
                continue
            try:
                result = spec.compute(annual_panel)
            except Exception as exc:  # noqa: BLE001 -- intentionally broad: this IS the check
                failures.append(f"{spec.name}: {exc!r}")
                continue
            if len(result) != len(annual_panel):
                failures.append(f"{spec.name}: returned length {len(result)}, expected {len(annual_panel)}")
        assert not failures, "Annual factors that failed to compute:\n" + "\n".join(failures)

    def test_all_monthly_factors_compute(self, monthly_panel):
        # industry_momentum and the three market-return-dependent factors
        # need columns already attached in the monthly_panel fixture above
        # (sic, mkt_return) -- the fixture does this directly rather than
        # via attach_industry/attach_market_return (which need a live
        # DuckDB connection), since this is a pure-function smoke test.
        failures = []
        for spec in f.FACTOR_REGISTRY.values():
            if spec.panel != "monthly":
                continue
            try:
                result = spec.compute(monthly_panel)
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{spec.name}: {exc!r}")
                continue
            if len(result) != len(monthly_panel):
                failures.append(f"{spec.name}: returned length {len(result)}, expected {len(monthly_panel)}")
            elif result.notna().sum() == 0:
                failures.append(f"{spec.name}: produced all-NaN output on a well-formed synthetic panel")
        assert not failures, "Monthly factors that failed to compute:\n" + "\n".join(failures)


class TestHandComputedSpotChecks:
    """One directly-verified formula per category, plus the specific
    regression test for the cross-company momentum bug the critique found."""

    def test_book_to_market(self, annual_panel):
        result = f.value.book_to_market.compute(annual_panel)
        row = annual_panel.iloc[0]
        expected = row["stockholders_equity"] / row["market_cap"]
        assert result.iloc[0] == pytest.approx(expected)

    def test_gross_profitability(self, annual_panel):
        result = f.profitability.gross_profitability.compute(annual_panel)
        row = annual_panel.iloc[0]
        expected = (row["revenues"] - row["cost_of_goods_sold"]) / row["assets"]
        assert result.iloc[0] == pytest.approx(expected)

    def test_asset_growth_sign_and_value(self, annual_panel):
        result = f.investment.asset_growth.compute(annual_panel)
        row = annual_panel.iloc[0]
        expected = row["assets"] / row["assets_prior"] - 1
        assert result.iloc[0] == pytest.approx(expected)
        assert f.investment.asset_growth.direction == -1  # high growth -> long the LOW decile

    def test_size_is_log_market_cap(self, annual_panel):
        result = f.trading_frictions.size.compute(annual_panel)
        row = annual_panel.iloc[0]
        assert result.iloc[0] == pytest.approx(np.log(row["market_cap"]))

    def test_momentum_12_1_does_not_roll_across_companies(self):
        # The exact bug class flagged in the Phase 1 critique: a momentum
        # column computed without grouping by entity rolls returns across
        # different companies at the panel's seams. Two companies, 13
        # months each, company B's returns are all zero; if momentum_12_1
        # ever used company A's late-window returns to compute company
        # B's early-window momentum, B's values would be nonzero.
        months = pd.date_range("2020-01-31", periods=13, freq="ME")
        rows = []
        for m in months:
            rows.append({"entity_id_hint": "A", "month": m, "monthly_return": 0.05})
        for m in months:
            rows.append({"entity_id_hint": "B", "month": m, "monthly_return": 0.0})
        panel = pd.DataFrame(rows)

        result = f.momentum.momentum_12_1.compute(panel)
        panel = panel.assign(_result=result)
        b_values = panel[panel["entity_id_hint"] == "B"]["_result"].dropna()
        assert (b_values == 0).all(), "momentum_12_1 leaked company A's returns into company B"

        a_values = panel[panel["entity_id_hint"] == "A"]["_result"].dropna()
        expected_a = (1.05**11) - 1
        assert a_values.iloc[-1] == pytest.approx(expected_a)

    def test_short_term_reversal_is_just_last_months_return(self, monthly_panel):
        result = f.momentum.short_term_reversal.compute(monthly_panel)
        pd.testing.assert_series_equal(result, monthly_panel["monthly_return"], check_names=False)

    def test_total_accruals_uses_average_assets_in_denominator(self, annual_panel):
        result = f.accruals.total_accruals.compute(annual_panel)
        row = annual_panel.iloc[0]
        numerator = (
            (row["assets_current"] - row["assets_current_prior"])
            - (row["cash"] - row["cash_prior"])
            - (row["liabilities_current"] - row["liabilities_current_prior"])
            + (row["st_borrowings"] - row["st_borrowings_prior"])
            - row["depreciation"]
        )
        expected = numerator / ((row["assets"] + row["assets_prior"]) / 2)
        assert result.iloc[0] == pytest.approx(expected)

    def test_industry_momentum_groups_only_within_same_sic(self, monthly_panel):
        panel_with_sic = monthly_panel  # fixture already attaches sic
        result = f.momentum.industry_momentum.compute(panel_with_sic)
        # AAA and BBB share SIC 7372; CCC is alone in 2834, so CCC's
        # industry-momentum value must equal CCC's own momentum exactly.
        own = pd.Series(
            f.momentum._cumulative_return_window(panel_with_sic, window=11, skip=1), index=panel_with_sic.index
        )
        tagged = panel_with_sic.assign(_industry=result, _own=own)
        ccc = tagged[tagged["entity_id_hint"] == "CCC"].dropna(subset=["_own"])
        pd.testing.assert_series_equal(ccc["_industry"], ccc["_own"], check_names=False)

    def test_industry_momentum_returns_nan_without_sic_column(self):
        panel = _synthetic_monthly_panel(n_months=13)
        result = f.momentum.industry_momentum.compute(panel)  # no sic column attached
        assert result.isna().all()
