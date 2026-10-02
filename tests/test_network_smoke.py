"""Live smoke tests against real, free data sources. Not run by default
(see pyproject.toml's pytest marker config) -- run explicitly with
`uv run pytest -m network` when you want to confirm the actual endpoints
still behave as documented, as opposed to the offline unit tests, which
check logic against fixtures and run in CI every time.
"""

from __future__ import annotations

import os

import pytest

from factorzoo.data import edgar, prices, riskfree
from factorzoo.universe import filters as universe_filters

pytestmark = pytest.mark.network

SEC_USER_AGENT = os.environ.get("SEC_USER_AGENT", "factorzoo-test test@example.com")


def test_sec_company_tickers_live(tmp_path):
    client = edgar.EdgarClient(edgar.Settings(sec_user_agent=SEC_USER_AGENT, tiingo_api_key=None))
    df = client.fetch_company_tickers(cache_path=tmp_path / "ct.json")
    assert len(df) > 5000  # thousands of EDGAR filers
    assert "ticker" in df.columns
    client.close()


def test_sec_company_facts_live_for_apple():
    client = edgar.EdgarClient(edgar.Settings(sec_user_agent=SEC_USER_AGENT, tiingo_api_key=None))
    facts = client.fetch_company_facts(320193)  # Apple's CIK
    assert facts is not None
    df = edgar.extract_xbrl_facts(facts)
    assert not df.empty
    assert "Assets" in df["tag"].unique()
    client.close()


def test_yahoo_chart_live_for_aapl():
    df = prices.fetch_yahoo_chart_daily("AAPL", range_="5d")
    assert not df.empty
    assert df["close"].min() > 0


@pytest.mark.xfail(
    reason=(
        "Live finding as of this build: Stooq's bulk CSV endpoint returns a "
        "JavaScript bot-check page for automated clients, confirmed even with a "
        "browser User-Agent and cookies replayed. Kept as an xfail, not deleted, "
        "so a future pass (if Stooq changes this) shows up as an unexpected pass "
        "rather than silence. See data/prices.py's module docstring."
    ),
    strict=False,
)
def test_stooq_live_for_aapl():
    df = prices.fetch_stooq_daily("AAPL")
    assert not df.empty
    assert df["close"].min() > 0


def test_sp500_history_live():
    history = universe_filters.fetch_sp500_history()
    assert len(history) > 500
    members = universe_filters.point_in_time_members(history, "2024-01-01")
    assert "AAPL" in members


def test_ken_french_ff5_live():
    df = riskfree.fetch_ff5_daily()
    assert not df.empty
    assert df["date"].min() < df["date"].max()
