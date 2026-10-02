"""Offline tests for the point-in-time rule and XBRL fact extraction.

These are the tests behind Phase 1's gate ("PIT store & universe pass
schema tests") for the data-correctness half of that gate: no network
call, pure functions, so they run in CI every time.
"""

from __future__ import annotations

import pandas as pd

from factorzoo.data.edgar import compute_available_date, extract_xbrl_facts


class TestComputeAvailableDate:
    def test_annual_fact_filed_before_floor_uses_floor(self):
        # period_end 2023-12-31, filed only 32 days later -- well inside
        # the 91-day annual floor, so available_date must be the floor,
        # not the (implausibly early) filed date.
        result = compute_available_date("10-K", pd.Timestamp("2023-12-31"), pd.Timestamp("2024-02-01"))
        assert result == pd.Timestamp("2023-12-31") + pd.Timedelta(days=91)

    def test_annual_fact_filed_after_floor_uses_filed(self):
        # A late 10-K: filed well past the floor, so the real filing date
        # governs availability, not the floor.
        result = compute_available_date("10-K", pd.Timestamp("2023-12-31"), pd.Timestamp("2024-06-01"))
        assert result == pd.Timestamp("2024-06-01")

    def test_10k_amendment_still_uses_annual_floor(self):
        result = compute_available_date("10-K/A", pd.Timestamp("2023-12-31"), pd.Timestamp("2024-02-01"))
        assert result == pd.Timestamp("2023-12-31") + pd.Timedelta(days=91)

    def test_quarterly_fact_filed_before_floor_uses_floor(self):
        result = compute_available_date("10-Q", pd.Timestamp("2023-09-30"), pd.Timestamp("2023-10-20"))
        assert result == pd.Timestamp("2023-09-30") + pd.Timedelta(days=45)

    def test_quarterly_fact_filed_after_floor_uses_filed(self):
        result = compute_available_date("10-Q", pd.Timestamp("2023-09-30"), pd.Timestamp("2023-12-01"))
        assert result == pd.Timestamp("2023-12-01")

    def test_missing_filed_date_returns_nat(self):
        assert pd.isna(compute_available_date("10-K", pd.Timestamp("2023-12-31"), None))

    def test_missing_period_end_returns_nat(self):
        assert pd.isna(compute_available_date("10-K", None, pd.Timestamp("2024-02-01")))

    def test_annual_hurdle_is_strictly_later_than_quarterly_for_same_dates(self):
        # Sanity check on the two floors relative to each other.
        period_end = pd.Timestamp("2023-12-31")
        filed = pd.Timestamp("2024-01-05")  # earlier than both floors
        annual = compute_available_date("10-K", period_end, filed)
        quarterly = compute_available_date("10-Q", period_end, filed)
        assert annual > quarterly


SYNTHETIC_FACTS_JSON = {
    "cik": 320193,
    "entityName": "TEST CORP",
    "facts": {
        "us-gaap": {
            "Assets": {
                "units": {
                    "USD": [
                        # Original 10-K, filed on time.
                        {
                            "val": 1000.0,
                            "end": "2023-12-31",
                            "fy": 2023,
                            "fp": "FY",
                            "form": "10-K",
                            "filed": "2024-02-15",
                            "accn": "0001-ORIGINAL",
                        },
                        # Later restatement of the SAME period via 10-K/A --
                        # must NOT silently replace the row above.
                        {
                            "val": 950.0,
                            "end": "2023-12-31",
                            "fy": 2023,
                            "fp": "FY",
                            "form": "10-K/A",
                            "filed": "2024-08-01",
                            "accn": "0002-AMENDED",
                        },
                        # A different period entirely.
                        {
                            "val": 900.0,
                            "end": "2022-12-31",
                            "fy": 2022,
                            "fp": "FY",
                            "form": "10-K",
                            "filed": "2023-02-10",
                            "accn": "0003-PRIOR-YEAR",
                        },
                    ]
                }
            },
            "SomeOtherTagNotRequested": {
                "units": {"USD": [{"val": 1, "end": "2023-12-31", "filed": "2024-02-15", "accn": "x"}]}
            },
        },
        "dei": {
            "EntityCommonStockSharesOutstanding": {
                "units": {
                    "shares": [
                        {
                            "val": 15000000000,
                            "end": "2023-12-31",
                            "fy": 2023,
                            "fp": "FY",
                            "form": "10-K",
                            "filed": "2024-02-15",
                            "accn": "0001-ORIGINAL",
                        }
                    ]
                }
            }
        },
    },
}


class TestExtractXbrlFacts:
    def test_only_requested_tags_are_extracted(self):
        df = extract_xbrl_facts(SYNTHETIC_FACTS_JSON, tags=("Assets",))
        assert set(df["tag"].unique()) <= {"Assets", "EntityCommonStockSharesOutstanding"}
        assert "SomeOtherTagNotRequested" not in df["tag"].unique()

    def test_original_and_amended_filings_both_retained(self):
        # The core correctness requirement from the build guide: an
        # amendment is a NEW vintage row, never an in-place overwrite.
        df = extract_xbrl_facts(SYNTHETIC_FACTS_JSON, tags=("Assets",))
        same_period = df[(df["tag"] == "Assets") & (df["period_end"] == pd.Timestamp("2023-12-31"))]
        assert len(same_period) == 2
        assert set(same_period["accn"]) == {"0001-ORIGINAL", "0002-AMENDED"}
        assert set(same_period["val"]) == {1000.0, 950.0}

    def test_amendment_has_later_available_date_than_original(self):
        df = extract_xbrl_facts(SYNTHETIC_FACTS_JSON, tags=("Assets",))
        original = df[df["accn"] == "0001-ORIGINAL"].iloc[0]
        amended = df[df["accn"] == "0002-AMENDED"].iloc[0]
        assert amended["available_date"] > original["available_date"]

    def test_entity_id_derived_from_cik(self):
        df = extract_xbrl_facts(SYNTHETIC_FACTS_JSON, tags=("Assets",))
        assert (df["entity_id"] == "CIK0000320193").all()

    def test_every_row_has_available_date_populated(self):
        df = extract_xbrl_facts(SYNTHETIC_FACTS_JSON, tags=("Assets",))
        assert df["available_date"].notna().all()

    def test_empty_facts_json_returns_empty_frame(self):
        df = extract_xbrl_facts({"cik": 1, "facts": {}}, tags=("Assets",))
        assert df.empty
