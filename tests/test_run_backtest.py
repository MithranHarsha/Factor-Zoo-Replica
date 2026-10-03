"""Offline smoke tests for the backtest orchestration module. The
statistical mechanics (decile sorts, Newey-West t-stats, BH/BY, DSR,
clustering, LASSO, IPCA) are each already tested against synthetic
fixtures with known answers in their own modules (test_portfolios.py,
test_performance.py, test_multiple_testing.py, test_dimension_reduction.py,
test_ipca_model.py) -- this file only checks that real-data wiring
(assembling point-in-time cross-sections, sequencing the calls, persisting
results) doesn't crash and produces plausibly-shaped output on a small
synthetic store.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factorzoo.backtest.run_backtest import (
    BacktestResult,
    build_cross_sections,
    persist_backtest_result,
    run_full_backtest,
)
from factorzoo.data import pit_store


@pytest.fixture
def con(tmp_path):
    connection = pit_store.init_db(tmp_path / "backtest_test.duckdb")
    yield connection
    connection.close()


N_ENTITIES = 14
N_MONTHS = 14  # +1 extra month of prices for the last formation date's forward return


def _seed_synthetic_store(con, seed: int = 0) -> list[pd.Timestamp]:
    rng = np.random.default_rng(seed)
    tickers = [f"T{i:02d}" for i in range(N_ENTITIES)]
    entity_ids = [f"CIK{i:010d}" for i in range(N_ENTITIES)]

    crosswalk = pd.DataFrame(
        {
            "ticker": tickers,
            "cik": range(N_ENTITIES),
            "entity_id": entity_ids,
            "title": [f"Test Co {i}" for i in range(N_ENTITIES)],
            "sic": ["7372"] * N_ENTITIES,  # software, non-financial
            "sic_description": ["Services-Prepackaged Software"] * N_ENTITIES,
        }
    )
    pit_store.write_crosswalk(con, crosswalk)

    formation_dates = [pd.Timestamp("2022-01-31") + pd.DateOffset(months=i) for i in range(N_MONTHS)]
    universe = pd.DataFrame(
        [{"formation_date": d, "ticker": t} for d in formation_dates for t in tickers]
    )
    pit_store.write_universe(con, universe)

    # One Assets/Revenues/shares fact per entity per fiscal year, filed
    # well before the backtest window starts, plus a real-looking spread
    # of values so book-to-market/size sorts aren't degenerate.
    facts_rows = []
    for i, eid in enumerate(entity_ids):
        for fy, period_end in [(2020, "2020-12-31"), (2021, "2021-12-31")]:
            filed = pd.Timestamp(period_end) + pd.Timedelta(days=45)
            base = 1_000_000.0 * (1 + i)
            for tag, val in [
                ("Assets", base * 2),
                ("Liabilities", base * 0.8),
                ("StockholdersEquity", base * 1.2),
                ("CommonStockSharesOutstanding", 1_000_000.0 + i * 10_000),
            ]:
                facts_rows.append(
                    {
                        "cik": i, "entity_id": eid, "taxonomy": "us-gaap", "tag": tag, "unit": "USD" if tag != "CommonStockSharesOutstanding" else "shares",
                        "val": val, "period_start": pd.NaT, "period_end": pd.Timestamp(period_end),
                        "fy": fy, "fp": "FY", "form": "10-K", "filed": filed, "accn": f"A{i}-{fy}",
                        "frame": None, "available_date": filed,
                    }
                )
    pit_store.write_xbrl_facts(con, pd.DataFrame(facts_rows))

    # Daily prices: one more month than the formation-date range needs, so
    # the LAST formation date still has a real forward return to score.
    price_rows = []
    price_dates = pd.date_range("2022-01-01", periods=(N_MONTHS + 1) * 22, freq="B")
    for i, ticker in enumerate(tickers):
        price = 50.0 + i * 5
        for d in price_dates:
            price *= 1 + rng.normal(0.004, 0.03)
            price_rows.append({"entity_id_hint": ticker, "date": d, "open": price, "high": price * 1.01, "low": price * 0.99, "close": price, "volume": 100_000})
    pit_store.write_prices(con, pd.DataFrame(price_rows), source="test")

    return formation_dates


class TestBuildCrossSections:
    def test_returns_one_frame_per_scoreable_date(self, con):
        dates = _seed_synthetic_store(con)
        panel_by_date = build_cross_sections(con, dates, min_companies=5)
        assert len(panel_by_date) > 0
        for d, df in panel_by_date.items():
            assert d in dates
            assert "forward_return" in df.columns
            assert "market_cap" in df.columns
            assert df["forward_return"].notna().sum() >= 5

    def test_too_few_companies_drops_the_date(self, con):
        dates = _seed_synthetic_store(con)
        panel_by_date = build_cross_sections(con, dates, min_companies=1000)
        assert panel_by_date == {}


class TestRunFullBacktest:
    def test_returns_well_formed_result_without_crashing(self, con):
        dates = _seed_synthetic_store(con)
        result = run_full_backtest(con, dates, min_companies=5)

        assert isinstance(result, BacktestResult)
        assert len(result.dates) > 0
        assert result.factor_vw_returns.shape[1] > 0
        assert set(result.taming_report.columns) >= {
            "tstat", "pvalue", "passes_t2", "passes_t3_hlz", "bh_discovery", "by_discovery",
            "one_per_cluster", "lasso_survives",
        }
        # Every factor scored in the taming report must actually have a
        # return series -- no orphan rows from a mismatched index.
        assert set(result.taming_report.index) <= set(result.factor_vw_returns.columns)

    def test_return_series_is_labeled_by_the_month_the_return_was_earned(self, con):
        # build_cross_sections keys each cross-section by its FORMATION
        # date, but the return in it is the FOLLOWING month's realized
        # return. Confirmed live: leaving the return series labeled by
        # formation date instead of earned-month made the replica's
        # market return correlate at r=-0.09 against Ken French's
        # same-month series; relabeling by earned month (this test's
        # subject) brought that to r=0.92 on the real data, so the dates
        # on every return series this function produces must be shifted
        # one month forward from the formation dates passed in.
        dates = _seed_synthetic_store(con)
        result = run_full_backtest(con, dates, min_companies=5)
        earned_months = {d.to_period("M") for d in result.factor_vw_returns.index}
        formation_months = {d.to_period("M") for d in dates}
        shifted_formation_months = {m + 1 for m in formation_months}
        # The earned-return index should NOT just be a relabeled copy of
        # the formation dates -- every label in it must be some formation
        # date's month, shifted one month later.
        assert earned_months != formation_months
        assert earned_months <= shifted_formation_months
        assert min(earned_months) == min(formation_months) + 1

    def test_raises_when_no_date_has_enough_companies(self, con):
        dates = _seed_synthetic_store(con)
        with pytest.raises(RuntimeError, match="nothing to backtest"):
            run_full_backtest(con, dates, min_companies=1000)


class TestPersistBacktestResult:
    def test_writes_queryable_tables(self, con):
        dates = _seed_synthetic_store(con)
        result = run_full_backtest(con, dates, min_companies=5)
        persist_backtest_result(con, result)

        n_returns = con.execute("SELECT COUNT(*) FROM factor_returns_monthly").fetchone()[0]
        n_taming = con.execute("SELECT COUNT(*) FROM taming_report").fetchone()[0]
        assert n_returns > 0
        assert n_taming == len(result.taming_report)

        manifest = con.execute(
            "SELECT detail FROM pull_manifest WHERE component = 'run-backtest' ORDER BY pulled_at DESC LIMIT 1"
        ).fetchone()
        assert manifest is not None

    def test_rerunning_replaces_rather_than_accumulates(self, con):
        dates = _seed_synthetic_store(con)
        result = run_full_backtest(con, dates, min_companies=5)
        persist_backtest_result(con, result)
        n_first = con.execute("SELECT COUNT(*) FROM taming_report").fetchone()[0]

        persist_backtest_result(con, result)  # same result, written again
        n_second = con.execute("SELECT COUNT(*) FROM taming_report").fetchone()[0]

        assert n_first == n_second  # replaced, not doubled
