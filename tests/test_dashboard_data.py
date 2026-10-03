"""Offline tests for the dashboard's data-loading helpers, against a
small synthetic store built with the same schema
as production (not the real project database, so these stay deterministic
and don't depend on what's been pulled into data/factorzoo.duckdb so far).
"""

from __future__ import annotations

import pandas as pd
import pytest

from factorzoo.dashboard.data import (
    compute_zoo_overview,
    factor_detail,
    load_data_health,
    load_recent_manifest,
)
from factorzoo.data import pit_store


@pytest.fixture
def con(tmp_path):
    connection = pit_store.init_db(tmp_path / "dashboard_test.duckdb")
    yield connection
    connection.close()


def _seed_minimal_fundamentals(con, n_companies=3):
    rows = []
    for i in range(n_companies):
        cik = 1000 + i
        entity_id = f"CIK{cik:010d}"
        base = 1000.0 * (i + 1)
        for tag, val in [
            ("Assets", base), ("Revenues", base * 1.2), ("CostOfGoodsAndServicesSold", base * 0.7),
            ("NetIncomeLoss", base * 0.08), ("StockholdersEquity", base * 0.6),
        ]:
            rows.append(
                {
                    "cik": cik, "entity_id": entity_id, "taxonomy": "us-gaap", "tag": tag, "unit": "USD",
                    "val": val, "period_start": pd.Timestamp("2022-01-01") if tag not in ("Assets", "StockholdersEquity") else pd.NaT,
                    "period_end": pd.Timestamp("2022-12-31"), "fy": 2022, "fp": "FY", "form": "10-K",
                    "filed": pd.Timestamp("2023-02-15"), "accn": f"A{i}", "frame": None,
                    "available_date": pd.Timestamp("2023-02-15"),
                }
            )
    pit_store.write_xbrl_facts(con, pd.DataFrame(rows))


class TestLoadDataHealth:
    def test_empty_store_reports_zero_everywhere(self, con):
        health = load_data_health(con)
        assert all(v == 0 for v in health.values())

    def test_reflects_actual_row_counts(self, con):
        _seed_minimal_fundamentals(con, n_companies=2)
        health = load_data_health(con)
        assert health["xbrl_facts"] == 10  # 5 tags x 2 companies


class TestLoadRecentManifest:
    def test_empty_store_returns_empty_frame(self, con):
        result = load_recent_manifest(con)
        assert result.empty

    def test_returns_most_recent_first(self, con):
        pit_store.record_manifest(con, "step-1", {"n": 1})
        pit_store.record_manifest(con, "step-2", {"n": 2})
        result = load_recent_manifest(con)
        assert result.iloc[0]["component"] == "step-2"


class TestComputeZooOverview:
    def test_monthly_factors_reported_as_needing_price_data(self, con):
        overview = compute_zoo_overview(con)
        momentum_row = overview[overview["factor"] == "momentum_12_1"].iloc[0]
        assert momentum_row["status"] == "needs price data"

    def test_annual_factors_with_no_fundamentals_report_that_clearly(self, con):
        overview = compute_zoo_overview(con)
        gp_row = overview[overview["factor"] == "gross_profitability"].iloc[0]
        assert gp_row["status"] == "no fundamentals pulled yet"
        assert gp_row["n_valid"] == 0

    def test_annual_factors_compute_once_fundamentals_exist(self, con):
        _seed_minimal_fundamentals(con, n_companies=3)
        overview = compute_zoo_overview(con)
        gp_row = overview[overview["factor"] == "gross_profitability"].iloc[0]
        assert gp_row["status"] == "ok"
        assert gp_row["n_valid"] == 3

    def test_covers_all_48_registered_factors(self, con):
        overview = compute_zoo_overview(con)
        assert len(overview) == 48


class TestFactorDetail:
    def test_unknown_factor_reports_error(self, con):
        result = factor_detail(con, "not_a_real_factor")
        assert "error" in result

    def test_monthly_factor_returns_a_note_not_values(self, con):
        result = factor_detail(con, "momentum_12_1")
        assert result["values"].empty
        assert "price data" in result["note"]

    def test_annual_factor_with_data_returns_sorted_values(self, con):
        _seed_minimal_fundamentals(con, n_companies=3)
        result = factor_detail(con, "gross_profitability")
        assert len(result["values"]) == 3
        assert result["note"] is None
        # gross_profitability direction is +1 (long high values), so the
        # detail table should be sorted with the highest value first.
        assert result["values"]["value"].is_monotonic_decreasing
