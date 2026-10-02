"""Offline tests for the annual point-in-time snapshot and monthly price
panel -- the two data structures every factor formula computes against.
These are synthetic fixtures by design (see test_network_smoke.py for the
live-data checks), so they run in CI and pin down exact expected values.
"""

from __future__ import annotations

import pandas as pd
import pytest

from factorzoo.data import pit_store
from factorzoo.factors.panel import (
    attach_market_cap,
    build_monthly_price_panel,
    build_point_in_time_snapshot,
    resolve_shares_outstanding,
)


@pytest.fixture
def con(tmp_path):
    connection = pit_store.init_db(tmp_path / "panel_test.duckdb")
    yield connection
    connection.close()


def _fact(entity_id, tag, val, period_start, period_end, filed, accn, available_date=None, form="10-K"):
    return {
        "cik": 1,
        "entity_id": entity_id,
        "taxonomy": "us-gaap",
        "tag": tag,
        "unit": "USD",
        "val": val,
        "period_start": pd.Timestamp(period_start) if period_start else pd.NaT,
        "period_end": pd.Timestamp(period_end),
        "fy": pd.Timestamp(period_end).year,
        "fp": "FY",
        "form": form,
        "filed": pd.Timestamp(filed),
        "accn": accn,
        "frame": None,
        "available_date": pd.Timestamp(available_date or filed),
    }


class TestBuildPointInTimeSnapshot:
    def test_instant_tag_picks_latest_known_value(self, con):
        # Assets at two fiscal year-ends; as-of date after both.
        facts = pd.DataFrame(
            [
                _fact("E1", "Assets", 1000.0, None, "2022-12-31", "2023-02-15", "A1"),
                _fact("E1", "Assets", 1200.0, None, "2023-12-31", "2024-02-15", "A2"),
            ]
        )
        pit_store.write_xbrl_facts(con, facts)
        snap = build_point_in_time_snapshot(con, "2025-01-01")
        row = snap[snap["entity_id"] == "E1"].iloc[0]
        assert row["assets"] == 1200.0
        assert row["assets_prior"] == 1000.0

    def test_future_fact_not_yet_available_is_excluded(self, con):
        # The more recent Assets fact is NOT yet available as of the query
        # date -- the snapshot must not leak it in.
        facts = pd.DataFrame(
            [
                _fact("E1", "Assets", 1000.0, None, "2022-12-31", "2023-02-15", "A1"),
                _fact("E1", "Assets", 1200.0, None, "2023-12-31", "2024-02-15", "A2"),
            ]
        )
        pit_store.write_xbrl_facts(con, facts)
        snap = build_point_in_time_snapshot(con, "2023-06-01")  # before A2 is available
        row = snap[snap["entity_id"] == "E1"].iloc[0]
        assert row["assets"] == 1000.0
        assert pd.isna(row.get("assets_prior"))  # only one year known yet

    def test_amendment_available_later_supersedes_original_once_available(self, con):
        original = _fact("E1", "Assets", 1000.0, None, "2023-12-31", "2024-02-15", "A1")
        amended = _fact(
            "E1", "Assets", 950.0, None, "2023-12-31", "2024-08-01", "A2",
            available_date="2024-08-01", form="10-K/A",
        )
        pit_store.write_xbrl_facts(con, pd.DataFrame([original]))
        pit_store.write_xbrl_facts(con, pd.DataFrame([amended]))

        before_amendment = build_point_in_time_snapshot(con, "2024-03-01")
        assert before_amendment[before_amendment["entity_id"] == "E1"].iloc[0]["assets"] == 1000.0

        after_amendment = build_point_in_time_snapshot(con, "2024-09-01")
        assert after_amendment[after_amendment["entity_id"] == "E1"].iloc[0]["assets"] == 950.0

    def test_duration_tag_requires_annual_span_not_quarterly(self, con):
        # A quarterly Revenues fact (period_start..period_end ~90 days)
        # must NOT be picked up even though its period_end matches a
        # fiscal year-end -- only the genuinely annual-span fact should be.
        quarterly = _fact("E1", "Revenues", 300.0, "2023-10-01", "2023-12-31", "2024-02-01", "Q1", form="10-Q")
        annual = _fact("E1", "Revenues", 1200.0, "2023-01-01", "2023-12-31", "2024-02-15", "A1", form="10-K")
        pit_store.write_xbrl_facts(con, pd.DataFrame([quarterly, annual]))
        snap = build_point_in_time_snapshot(con, "2025-01-01")
        assert snap[snap["entity_id"] == "E1"].iloc[0]["revenues"] == 1200.0

    def test_revenue_tag_fallback_prefers_newer_tag(self, con):
        old_tag = _fact("E1", "Revenues", 500.0, "2023-01-01", "2023-12-31", "2024-02-15", "A1")
        new_tag = _fact(
            "E1", "RevenueFromContractWithCustomerExcludingAssessedTax", 500.0,
            "2023-01-01", "2023-12-31", "2024-02-15", "A1b",
        )
        pit_store.write_xbrl_facts(con, pd.DataFrame([old_tag, new_tag]))
        snap = build_point_in_time_snapshot(con, "2025-01-01")
        # Both map to the "revenues" column; must collapse to one value,
        # not raise or silently duplicate the row.
        assert len(snap[snap["entity_id"] == "E1"]) == 1
        assert snap[snap["entity_id"] == "E1"].iloc[0]["revenues"] == 500.0

    def test_entity_filter_restricts_to_requested_entities(self, con):
        facts = pd.DataFrame(
            [
                _fact("E1", "Assets", 1000.0, None, "2023-12-31", "2024-02-15", "A1"),
                _fact("E2", "Assets", 2000.0, None, "2023-12-31", "2024-02-15", "B1"),
            ]
        )
        pit_store.write_xbrl_facts(con, facts)
        snap = build_point_in_time_snapshot(con, "2025-01-01", entity_ids=["E1"])
        assert set(snap["entity_id"]) == {"E1"}

    def test_empty_store_returns_frame_with_entity_id_column_only(self, con):
        snap = build_point_in_time_snapshot(con, "2025-01-01")
        assert list(snap.columns) == ["entity_id"]
        assert snap.empty


class TestResolveSharesOutstanding:
    def test_prefers_dei_tag_when_both_present(self):
        df = pd.DataFrame({"shares_outstanding_dei": [100.0], "shares_outstanding_gaap": [90.0]})
        result = resolve_shares_outstanding(df)
        assert result.iloc[0] == 100.0

    def test_falls_back_to_gaap_tag_when_dei_missing(self):
        df = pd.DataFrame({"shares_outstanding_dei": [None], "shares_outstanding_gaap": [90.0]})
        result = resolve_shares_outstanding(df)
        assert result.iloc[0] == 90.0

    def test_handles_neither_column_present(self):
        df = pd.DataFrame({"assets": [1000.0]})
        result = resolve_shares_outstanding(df)
        assert result.isna().all()


class TestAttachMarketCap:
    def _setup_entity(self, con, entity_id="CIK0000000001", ticker="AAA", shares=100.0):
        pit_store.write_crosswalk(
            con,
            pd.DataFrame(
                [{"ticker": ticker, "cik": 1, "entity_id": entity_id, "title": "Test Co", "sic": "7372", "sic_description": "Software"}]
            ),
        )
        facts = pd.DataFrame(
            [_fact(entity_id, "CommonStockSharesOutstanding", shares, None, "2023-12-31", "2024-02-15", "A1")]
        )
        pit_store.write_xbrl_facts(con, facts)

    def test_market_cap_is_shares_times_latest_price_on_or_before_as_of(self, con):
        self._setup_entity(con, shares=100.0)
        pit_store.write_prices(
            con,
            pd.DataFrame(
                [
                    {"entity_id_hint": "AAA", "date": pd.Timestamp("2024-03-01"), "open": 9, "high": 11,
                     "low": 9, "close": 10.0, "volume": 100},
                    {"entity_id_hint": "AAA", "date": pd.Timestamp("2024-06-01"), "open": 11, "high": 13,
                     "low": 11, "close": 12.0, "volume": 100},
                ]
            ),
            source="test",
        )
        snapshot = build_point_in_time_snapshot(con, "2025-01-01")
        result = attach_market_cap(con, snapshot, as_of="2024-04-15")  # between the two price dates
        row = result[result["entity_id"] == "CIK0000000001"].iloc[0]
        assert row["price_at_formation"] == 10.0  # the March price, not June (which is after as_of)
        assert row["market_cap"] == pytest.approx(1000.0)

    def test_price_after_as_of_is_not_used(self, con):
        self._setup_entity(con, shares=100.0)
        pit_store.write_prices(
            con,
            pd.DataFrame(
                [{"entity_id_hint": "AAA", "date": pd.Timestamp("2024-06-01"), "open": 11, "high": 13,
                  "low": 11, "close": 12.0, "volume": 100}]
            ),
            source="test",
        )
        snapshot = build_point_in_time_snapshot(con, "2025-01-01")
        result = attach_market_cap(con, snapshot, as_of="2024-01-01")  # before the only price
        row = result[result["entity_id"] == "CIK0000000001"].iloc[0]
        assert pd.isna(row["price_at_formation"])

    def test_empty_snapshot_returns_market_cap_column(self, con):
        empty = pd.DataFrame(columns=["entity_id"])
        result = attach_market_cap(con, empty, as_of="2024-01-01")
        assert "market_cap" in result.columns


class TestBuildMonthlyPricePanel:
    def _write_prices(self, con, rows):
        df = pd.DataFrame(rows)
        pit_store.write_prices(con, df, source="test")

    def test_monthly_return_compounds_daily_returns(self, con):
        # +1% then +1% on two trading days in the same month -> monthly
        # return should be the compounded (1.01*1.01 - 1), not the sum.
        self._write_prices(
            con,
            [
                {"entity_id_hint": "AAA", "date": pd.Timestamp("2024-01-02"), "open": 100, "high": 101,
                 "low": 99, "close": 100.0, "volume": 1000},
                {"entity_id_hint": "AAA", "date": pd.Timestamp("2024-01-03"), "open": 100, "high": 102,
                 "low": 100, "close": 101.0, "volume": 1100},
                {"entity_id_hint": "AAA", "date": pd.Timestamp("2024-01-04"), "open": 101, "high": 103,
                 "low": 101, "close": 102.01, "volume": 1200},
            ],
        )
        panel = build_monthly_price_panel(con)
        row = panel[(panel["entity_id_hint"] == "AAA")].iloc[0]
        assert row["monthly_return"] == pytest.approx(0.0201, abs=1e-6)

    def test_max_daily_return_captures_largest_single_day_move(self, con):
        self._write_prices(
            con,
            [
                {"entity_id_hint": "AAA", "date": pd.Timestamp("2024-01-02"), "open": 100, "high": 101,
                 "low": 99, "close": 100.0, "volume": 1000},
                {"entity_id_hint": "AAA", "date": pd.Timestamp("2024-01-03"), "open": 100, "high": 110,
                 "low": 100, "close": 110.0, "volume": 1100},  # +10% day
                {"entity_id_hint": "AAA", "date": pd.Timestamp("2024-01-04"), "open": 110, "high": 111,
                 "low": 109, "close": 109.0, "volume": 1200},
            ],
        )
        panel = build_monthly_price_panel(con)
        row = panel[(panel["entity_id_hint"] == "AAA")].iloc[0]
        assert row["max_daily_return"] == pytest.approx(0.10, abs=1e-6)

    def test_two_companies_kept_separate(self, con):
        self._write_prices(
            con,
            [
                {"entity_id_hint": "AAA", "date": pd.Timestamp("2024-01-02"), "open": 100, "high": 101,
                 "low": 99, "close": 100.0, "volume": 1000},
                {"entity_id_hint": "AAA", "date": pd.Timestamp("2024-01-03"), "open": 100, "high": 102,
                 "low": 100, "close": 101.0, "volume": 1100},
                {"entity_id_hint": "BBB", "date": pd.Timestamp("2024-01-02"), "open": 50, "high": 51,
                 "low": 49, "close": 50.0, "volume": 500},
                {"entity_id_hint": "BBB", "date": pd.Timestamp("2024-01-03"), "open": 50, "high": 49,
                 "low": 47, "close": 48.0, "volume": 600},
            ],
        )
        panel = build_monthly_price_panel(con)
        aaa = panel[panel["entity_id_hint"] == "AAA"].iloc[0]
        bbb = panel[panel["entity_id_hint"] == "BBB"].iloc[0]
        assert aaa["monthly_return"] > 0
        assert bbb["monthly_return"] < 0

    def test_source_filter_restricts_to_one_vendor(self, con):
        pit_store.write_prices(
            con,
            pd.DataFrame(
                [{"entity_id_hint": "AAA", "date": pd.Timestamp("2024-01-02"), "open": 100, "high": 101,
                  "low": 99, "close": 100.0, "volume": 1000}]
            ),
            source="yahoo",
        )
        pit_store.write_prices(
            con,
            pd.DataFrame(
                [{"entity_id_hint": "AAA", "date": pd.Timestamp("2024-01-02"), "open": 100, "high": 101,
                  "low": 99, "close": 100.0, "volume": 1000}]
            ),
            source="tiingo",
        )
        panel = build_monthly_price_panel(con, source="yahoo")
        assert len(panel) == 1
