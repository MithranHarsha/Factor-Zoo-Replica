"""Offline test for the Ken French CSV parser, using a synthetic zip that
mirrors the real file's layout (a few comment lines, a header row with a
blank first column name, YYYYMMDD dates, percent-unit values, a trailing
copyright line)."""

from __future__ import annotations

import io
import zipfile

import pandas as pd
import pytest

from factorzoo.data.riskfree import _parse_ff_zip

SAMPLE_INNER_CSV = """This file was created by using the 202608 CRSP database.
The Tbill return is the simple daily rate.


,Mkt-RF,SMB,HML,RMW,CMA,RF
19630701,   -0.67,    0.00,   -0.33,   -0.01,    0.16,    0.01
19630702,    0.79,   -0.26,    0.26,   -0.07,   -0.20,    0.01

Copyright 2026 Eugene F. Fama and Kenneth R. French
"""


def _make_zip(inner_name: str, inner_text: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(inner_name, inner_text)
    return buf.getvalue()


class TestParseFfZip:
    def test_parses_expected_row_count(self):
        raw = _make_zip("F-F_Research_Data_5_Factors_2x3_daily.CSV", SAMPLE_INNER_CSV)
        df = _parse_ff_zip(raw, inner_name_hint="5_Factors_2x3_daily")
        assert len(df) == 2

    def test_copyright_footer_line_is_dropped_not_parsed_as_data(self):
        raw = _make_zip("F-F_Research_Data_5_Factors_2x3_daily.CSV", SAMPLE_INNER_CSV)
        df = _parse_ff_zip(raw, inner_name_hint="5_Factors_2x3_daily")
        assert df["date"].notna().all()
        assert len(df) == 2  # not 3 -- the copyright line must not survive

    def test_dates_parsed_correctly(self):
        raw = _make_zip("F-F_Research_Data_5_Factors_2x3_daily.CSV", SAMPLE_INNER_CSV)
        df = _parse_ff_zip(raw, inner_name_hint="5_Factors_2x3_daily")
        assert df.iloc[0]["date"] == pd.Timestamp("1963-07-01")

    def test_percent_values_converted_to_decimal(self):
        raw = _make_zip("F-F_Research_Data_5_Factors_2x3_daily.CSV", SAMPLE_INNER_CSV)
        df = _parse_ff_zip(raw, inner_name_hint="5_Factors_2x3_daily")
        # Source shows -0.67 meaning -0.67%; parsed value should be the
        # decimal fraction -0.0067, not -0.67.
        assert df.iloc[0]["mkt_rf"] == pytest.approx(-0.0067)

    def test_expected_columns_present(self):
        raw = _make_zip("F-F_Research_Data_5_Factors_2x3_daily.CSV", SAMPLE_INNER_CSV)
        df = _parse_ff_zip(raw, inner_name_hint="5_Factors_2x3_daily")
        assert set(df.columns) == {"date", "mkt_rf", "smb", "hml", "rmw", "cma", "rf"}
