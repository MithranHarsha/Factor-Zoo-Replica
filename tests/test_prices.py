"""Offline tests for price parsing and the Tiingo free-tier budget tracker."""

from __future__ import annotations

import httpx
import pytest

from factorzoo.data.prices import (
    TiingoBudget,
    _is_rate_limit_error,
    parse_stooq_csv,
    parse_tiingo_json,
    parse_yahoo_chart_json,
)

STOOQ_SAMPLE = """Date,Open,High,Low,Close,Volume
2024-01-02,185.0,186.0,184.5,185.5,1000000
2024-01-03,185.5,187.0,185.0,186.8,900000
"""


class TestParseStooqCsv:
    def test_parses_expected_rows(self):
        df = parse_stooq_csv(STOOQ_SAMPLE, "aapl")
        assert len(df) == 2
        assert list(df.columns) == ["entity_id_hint", "date", "open", "high", "low", "close", "volume"]

    def test_entity_id_hint_is_uppercased(self):
        df = parse_stooq_csv(STOOQ_SAMPLE, "aapl")
        assert (df["entity_id_hint"] == "AAPL").all()

    def test_no_data_response_returns_empty_frame_not_an_error(self):
        # Stooq returns the literal string "No data" (HTTP 200) for an
        # unknown ticker -- must not be parsed as if it were a CSV body.
        df = parse_stooq_csv("No data", "ZZZZNOTREAL")
        assert df.empty

    def test_empty_string_returns_empty_frame(self):
        df = parse_stooq_csv("", "ZZZZNOTREAL")
        assert df.empty


TIINGO_SAMPLE = [
    {
        "date": "2024-01-02T00:00:00.000Z",
        "adjOpen": 185.0,
        "adjHigh": 186.0,
        "adjLow": 184.5,
        "adjClose": 185.5,
        "adjVolume": 1000000,
    }
]


class TestParseTiingoJson:
    def test_parses_expected_rows(self):
        df = parse_tiingo_json(TIINGO_SAMPLE, "aapl")
        assert len(df) == 1
        assert df.iloc[0]["close"] == 185.5

    def test_empty_payload_returns_empty_frame(self):
        df = parse_tiingo_json([], "ZZZZNOTREAL")
        assert df.empty


YAHOO_SAMPLE = {
    "chart": {
        "result": [
            {
                "meta": {"symbol": "AAPL"},
                "timestamp": [1704207600, 1704294000],  # 2024-01-02, 2024-01-03 (approx, UTC)
                "indicators": {
                    "quote": [
                        {
                            "open": [185.0, 185.5],
                            "high": [186.0, 187.0],
                            "low": [184.5, 185.0],
                            "close": [185.5, 186.8],
                            "volume": [1000000, 900000],
                        }
                    ],
                    "adjclose": [{"adjclose": [185.3, 186.6]}],
                },
            }
        ],
        "error": None,
    }
}

YAHOO_SAMPLE_UNKNOWN_TICKER = {"chart": {"result": None, "error": {"code": "Not Found", "description": "No data found"}}}


class TestParseYahooChartJson:
    def test_parses_expected_row_count(self):
        df = parse_yahoo_chart_json(YAHOO_SAMPLE, "aapl")
        assert len(df) == 2

    def test_uses_adjusted_close_when_present(self):
        df = parse_yahoo_chart_json(YAHOO_SAMPLE, "aapl")
        assert df.iloc[0]["close"] == 185.3  # adjclose, not raw close (185.5)

    def test_entity_id_hint_is_uppercased(self):
        df = parse_yahoo_chart_json(YAHOO_SAMPLE, "aapl")
        assert (df["entity_id_hint"] == "AAPL").all()

    def test_unknown_ticker_returns_empty_frame_not_an_error(self):
        df = parse_yahoo_chart_json(YAHOO_SAMPLE_UNKNOWN_TICKER, "ZZZZNOTREAL")
        assert df.empty

    def test_empty_payload_returns_empty_frame(self):
        df = parse_yahoo_chart_json({}, "ZZZZNOTREAL")
        assert df.empty

    def test_columns_match_other_price_sources(self):
        # Same shape as parse_stooq_csv / parse_tiingo_json so downstream
        # code (pit_store.write_prices) never has to special-case a source.
        df = parse_yahoo_chart_json(YAHOO_SAMPLE, "aapl")
        assert list(df.columns) == ["entity_id_hint", "date", "open", "high", "low", "close", "volume"]


class TestIsRateLimitError:
    def test_429_is_a_rate_limit_error(self):
        req = httpx.Request("GET", "https://example.com")
        resp = httpx.Response(429, request=req)
        exc = httpx.HTTPStatusError("429", request=req, response=resp)
        assert _is_rate_limit_error(exc) is True

    def test_404_is_not_a_rate_limit_error(self):
        req = httpx.Request("GET", "https://example.com")
        resp = httpx.Response(404, request=req)
        exc = httpx.HTTPStatusError("404", request=req, response=resp)
        assert _is_rate_limit_error(exc) is False

    def test_non_http_exception_is_not_a_rate_limit_error(self):
        assert _is_rate_limit_error(ValueError("not an http error")) is False


class TestTiingoBudget:
    def test_fresh_budget_has_full_cap_remaining(self, tmp_path):
        budget = TiingoBudget(cap=5, ledger_path=tmp_path / "ledger.json")
        assert budget.remaining() == 5

    def test_registering_a_symbol_decrements_remaining(self, tmp_path):
        budget = TiingoBudget(cap=5, ledger_path=tmp_path / "ledger.json")
        budget.register("AAPL")
        assert budget.remaining() == 4

    def test_registering_same_symbol_twice_does_not_double_count(self, tmp_path):
        budget = TiingoBudget(cap=5, ledger_path=tmp_path / "ledger.json")
        budget.register("AAPL")
        budget.register("AAPL")
        assert budget.remaining() == 4

    def test_exceeding_cap_raises(self, tmp_path):
        budget = TiingoBudget(cap=2, ledger_path=tmp_path / "ledger.json")
        budget.register("AAA")
        budget.register("BBB")
        with pytest.raises(RuntimeError, match="free-tier symbol cap"):
            budget.register("CCC")
