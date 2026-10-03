"""Offline tests for point-in-time universe construction -- the
survivorship-bias fix. A synthetic membership
history stands in for the real S&P 500 dataset so these never touch the
network.
"""

from __future__ import annotations

import pandas as pd
import pytest

from factorzoo.universe.filters import (
    monthly_universe_panel,
    parse_sp500_history,
    point_in_time_members,
    summarize_coverage,
)

SYNTHETIC_HISTORY_CSV = """ticker,start_date,end_date
AAA,1996-01-02,
BBB,1996-01-02,2010-06-15
CCC,2015-03-23,2024-09-23
CCC,1996-01-02,1997-01-15
DDD,2020-01-01,
"""
# AAA: a long-lived member, never removed.
# BBB: delisted/removed in 2010 -- must appear for pre-2010 as-of dates and
#      disappear for post-2010 ones.
# CCC: two separate membership spells (left and rejoined, like AAL in the
#      real dataset) -- must be present in BOTH windows and absent in the
#      gap between them.
# DDD: a recent addition, absent before 2020.


@pytest.fixture
def history() -> pd.DataFrame:
    return parse_sp500_history(SYNTHETIC_HISTORY_CSV)


class TestParseSp500History:
    def test_parses_expected_columns(self, history):
        assert list(history.columns) == ["ticker", "start_date", "end_date"]

    def test_blank_end_date_becomes_nat(self, history):
        aaa = history[history["ticker"] == "AAA"].iloc[0]
        assert pd.isna(aaa["end_date"])

    def test_ticker_with_two_spells_has_two_rows(self, history):
        assert len(history[history["ticker"] == "CCC"]) == 2


class TestPointInTimeMembers:
    def test_long_lived_member_present_at_any_date(self, history):
        for as_of in ["1996-06-01", "2010-01-01", "2026-01-01"]:
            assert "AAA" in point_in_time_members(history, as_of)

    def test_removed_ticker_present_before_removal(self, history):
        members = point_in_time_members(history, "2005-01-01")
        assert "BBB" in members

    def test_removed_ticker_absent_after_removal(self, history):
        # This is the exact bug the critique flagged: a universe built
        # from "current filers only" would never exclude BBB correctly
        # for a historical as-of date, because it would never have
        # included BBB in the first place. Here BBB is correctly dropped
        # only after its real removal date.
        members = point_in_time_members(history, "2020-01-01")
        assert "BBB" not in members

    def test_removed_ticker_present_exactly_on_removal_date(self, history):
        members = point_in_time_members(history, "2010-06-15")
        assert "BBB" in members

    def test_recent_addition_absent_before_its_start_date(self, history):
        members = point_in_time_members(history, "2015-01-01")
        assert "DDD" not in members

    def test_recent_addition_present_after_its_start_date(self, history):
        members = point_in_time_members(history, "2021-01-01")
        assert "DDD" in members

    def test_ticker_absent_in_gap_between_two_membership_spells(self, history):
        # CCC: member 1996-01-02 to 1997-01-15, then absent, then member
        # again 2015-03-23 to 2024-09-23.
        assert "CCC" in point_in_time_members(history, "1996-06-01")
        assert "CCC" not in point_in_time_members(history, "2005-01-01")
        assert "CCC" in point_in_time_members(history, "2020-01-01")
        assert "CCC" not in point_in_time_members(history, "2025-01-01")

    @pytest.mark.leakage
    def test_no_future_leakage_into_a_past_as_of_date(self, history):
        # The universe as of 1996 cannot possibly include DDD, which
        # doesn't exist until 2020. A naive "current constituents" source
        # would get this wrong by construction.
        members = point_in_time_members(history, "1996-06-01")
        assert "DDD" not in members


class TestMonthlyUniversePanel:
    def test_panel_has_expected_columns(self, history):
        panel = monthly_universe_panel(history, start="1996-01-31", end="1996-03-31")
        assert set(panel.columns) == {"formation_date", "ticker"}

    def test_panel_row_count_matches_pointwise_membership(self, history):
        panel = monthly_universe_panel(history, start="2010-01-31", end="2010-12-31")
        for d in panel["formation_date"].unique():
            expected = set(point_in_time_members(history, d))
            actual = set(panel.loc[panel["formation_date"] == d, "ticker"])
            assert actual == expected

    def test_bbb_disappears_from_panel_after_its_removal_month(self, history):
        panel = monthly_universe_panel(history, start="2010-01-31", end="2010-12-31")
        before = panel[(panel["formation_date"] == pd.Timestamp("2010-05-31"))]["ticker"].tolist()
        after = panel[(panel["formation_date"] == pd.Timestamp("2010-07-31"))]["ticker"].tolist()
        assert "BBB" in before
        assert "BBB" not in after


class TestSummarizeCoverage:
    def test_reports_unique_ticker_count(self, history):
        summary = summarize_coverage(history)
        assert summary["n_unique_tickers"] == 4  # AAA, BBB, CCC, DDD
        assert summary["n_ticker_spells"] == 5  # CCC counted twice

    def test_reports_source_note(self, history):
        summary = summarize_coverage(history)
        assert "fja05680/sp500" in summary["source"]
