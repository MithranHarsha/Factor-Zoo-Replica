"""Offline tests for the DuckDB point-in-time store: schema creation, and
the leakage-blocking property of `get_facts_as_of` -- this is Phase 1's
gate ("PIT store & universe pass schema tests") made literal.
"""

from __future__ import annotations

import pandas as pd
import pytest

from factorzoo.data import pit_store

EXPECTED_TABLES = {
    "xbrl_facts",
    "universe_membership",
    "entity_crosswalk",
    "prices_daily",
    "ff5_daily",
    "pull_manifest",
}


@pytest.fixture
def con(tmp_path):
    db_path = tmp_path / "test.duckdb"
    connection = pit_store.init_db(db_path)
    yield connection
    connection.close()


def _facts_df(**overrides) -> pd.DataFrame:
    base = {
        "cik": 320193,
        "entity_id": "CIK0000320193",
        "taxonomy": "us-gaap",
        "tag": "Assets",
        "unit": "USD",
        "val": 1000.0,
        "period_start": pd.NaT,
        "period_end": pd.Timestamp("2023-12-31"),
        "fy": 2023,
        "fp": "FY",
        "form": "10-K",
        "filed": pd.Timestamp("2024-02-15"),
        "accn": "0001-ORIGINAL",
        "frame": None,
        "available_date": pd.Timestamp("2024-02-15"),
    }
    base.update(overrides)
    return pd.DataFrame([base])


class TestSchema:
    def test_all_expected_tables_created(self, con):
        tables = set(con.execute("SHOW TABLES").fetchdf()["name"])
        assert EXPECTED_TABLES <= tables

    def test_init_db_is_idempotent(self, tmp_path):
        db_path = tmp_path / "idempotent.duckdb"
        con1 = pit_store.init_db(db_path)
        con1.close()
        con2 = pit_store.init_db(db_path)  # should not raise on re-create
        tables = set(con2.execute("SHOW TABLES").fetchdf()["name"])
        assert EXPECTED_TABLES <= tables
        con2.close()


class TestWriteXbrlFacts:
    def test_insert_returns_row_count(self, con):
        n = pit_store.write_xbrl_facts(con, _facts_df())
        assert n == 1
        assert con.execute("SELECT COUNT(*) FROM xbrl_facts").fetchone()[0] == 1

    def test_duplicate_natural_key_is_not_double_inserted(self, con):
        pit_store.write_xbrl_facts(con, _facts_df())
        pit_store.write_xbrl_facts(con, _facts_df())  # same cik/tag/unit/accn/period_end
        assert con.execute("SELECT COUNT(*) FROM xbrl_facts").fetchone()[0] == 1

    def test_amendment_with_different_accn_is_a_separate_row(self, con):
        original = _facts_df(accn="0001-ORIGINAL", val=1000.0)
        amended = _facts_df(accn="0002-AMENDED", val=950.0, available_date=pd.Timestamp("2024-08-01"))
        pit_store.write_xbrl_facts(con, original)
        pit_store.write_xbrl_facts(con, amended)
        rows = con.execute("SELECT accn, val FROM xbrl_facts ORDER BY accn").fetchdf()
        assert len(rows) == 2
        assert set(rows["accn"]) == {"0001-ORIGINAL", "0002-AMENDED"}


class TestGetFactsAsOf:
    """The leakage-blocking query itself."""

    def test_fact_excluded_before_its_available_date(self, con):
        pit_store.write_xbrl_facts(con, _facts_df(available_date=pd.Timestamp("2024-02-15")))
        before = pit_store.get_facts_as_of(con, "2024-01-01")
        assert before.empty

    def test_fact_included_on_its_available_date(self, con):
        pit_store.write_xbrl_facts(con, _facts_df(available_date=pd.Timestamp("2024-02-15")))
        on_date = pit_store.get_facts_as_of(con, "2024-02-15")
        assert len(on_date) == 1

    def test_fact_included_after_its_available_date(self, con):
        pit_store.write_xbrl_facts(con, _facts_df(available_date=pd.Timestamp("2024-02-15")))
        after = pit_store.get_facts_as_of(con, "2024-06-01")
        assert len(after) == 1

    def test_no_returned_row_has_available_date_past_the_query_date(self, con):
        # The general-case leakage assertion: for ANY as-of date, every
        # row returned must satisfy available_date <= as_of.
        pit_store.write_xbrl_facts(con, _facts_df(accn="A", available_date=pd.Timestamp("2024-02-15")))
        pit_store.write_xbrl_facts(con, _facts_df(accn="B", available_date=pd.Timestamp("2024-08-01"), val=950.0))
        as_of = pd.Timestamp("2024-05-01")
        result = pit_store.get_facts_as_of(con, as_of)
        assert (result["available_date"] <= as_of).all()
        assert set(result["accn"]) == {"A"}  # B is not yet available

    def test_tag_filter_restricts_results(self, con):
        pit_store.write_xbrl_facts(con, _facts_df(tag="Assets"))
        pit_store.write_xbrl_facts(con, _facts_df(tag="Liabilities", accn="0005-OTHER-TAG"))
        result = pit_store.get_facts_as_of(con, "2030-01-01", tags=["Assets"])
        assert set(result["tag"]) == {"Assets"}


class TestUniverseQueries:
    def test_write_and_read_universe(self, con):
        panel = pd.DataFrame(
            {
                "formation_date": [pd.Timestamp("2020-01-31")] * 2,
                "ticker": ["AAA", "BBB"],
            }
        )
        pit_store.write_universe(con, panel)
        members = pit_store.get_universe_as_of(con, "2020-02-15")
        assert set(members) == {"AAA", "BBB"}

    def test_get_universe_as_of_uses_nearest_prior_formation_date(self, con):
        panel = pd.DataFrame(
            {
                "formation_date": [pd.Timestamp("2020-01-31"), pd.Timestamp("2020-02-29")],
                "ticker": ["AAA", "BBB"],
            }
        )
        pit_store.write_universe(con, panel)
        # As-of a date between the two formation dates: must use Jan, not Feb.
        members = pit_store.get_universe_as_of(con, "2020-02-10")
        assert members == ["AAA"]

    def test_get_universe_as_of_before_any_data_returns_empty(self, con):
        assert pit_store.get_universe_as_of(con, "1900-01-01") == []


class TestManifest:
    def test_record_manifest_writes_a_row(self, con):
        pit_store.record_manifest(con, "test-component", {"k": "v"})
        rows = con.execute("SELECT component, detail FROM pull_manifest").fetchdf()
        assert len(rows) == 1
        assert rows.iloc[0]["component"] == "test-component"
        assert "k" in rows.iloc[0]["detail"]
