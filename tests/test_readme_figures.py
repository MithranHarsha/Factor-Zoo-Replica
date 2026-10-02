"""Offline tests for the README proof-of-work figures. Synthetic fixtures
by design (same pattern as test_factor_panel.py) -- these check that each
figure function is wired up correctly (returns a real figure, writes real
files, skips honestly when its input is missing), not that the specific
numbers are meaningful on fake data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from factorzoo.data import pit_store
from factorzoo.viz import readme_figures as rf


@pytest.fixture
def con(tmp_path, monkeypatch):
    connection = pit_store.init_db(tmp_path / "viz_test.duckdb")
    monkeypatch.setattr(rf, "IMG_DIR", tmp_path / "img")
    monkeypatch.setattr(rf, "INTERACTIVE_DIR", tmp_path / "interactive")
    yield connection
    connection.close()


def _fact(entity_id, tag, val, period_end, filed, accn):
    return {
        "cik": 1, "entity_id": entity_id, "taxonomy": "us-gaap", "tag": tag, "unit": "USD",
        "val": val, "period_start": pd.NaT, "period_end": pd.Timestamp(period_end),
        "fy": pd.Timestamp(period_end).year, "fp": "FY", "form": "10-K",
        "filed": pd.Timestamp(filed), "accn": accn, "frame": None,
        "available_date": pd.Timestamp(filed),
    }


def _seed_minimal_facts(con):
    """Just enough for core_tables_present() and _real_sample_years() --
    doesn't need to support factor computation."""
    facts = pd.DataFrame(
        [
            _fact("E1", "Assets", 1_000_000.0, "2022-12-31", "2023-02-15", "A1"),
            _fact("E1", "Assets", 1_100_000.0, "2023-12-31", "2024-02-20", "A2"),
        ]
    )
    pit_store.write_xbrl_facts(con, facts)
    pit_store.write_universe(con, pd.DataFrame([{"formation_date": pd.Timestamp("2023-01-31"), "ticker": "E1"}]))


class TestCoreTablesPresent:
    def test_false_on_empty_store(self, con):
        ok, reason = rf.core_tables_present(con)
        assert ok is False
        assert "universe_membership" in reason

    def test_true_once_universe_and_facts_exist(self, con):
        _seed_minimal_facts(con)
        ok, reason = rf.core_tables_present(con)
        assert ok is True
        assert reason == ""


class TestHurdleAndDsrSurfaces:
    def test_hurdle_surface_generated(self, con):
        _seed_minimal_facts(con)
        result = rf.hurdle_surface(con)
        assert result.status == "generated"
        assert result.output_png is not None
        assert result.output_png.exists()
        assert result.output_png.stat().st_size > 0

    def test_dsr_surface_generated_and_labeled_illustration(self, con):
        _seed_minimal_facts(con)
        result = rf.dsr_surface(con)
        assert result.status == "generated"
        assert result.output_png.exists()
        assert "illustration" in (result.note or "").lower()

    def test_dsr_fig_helper_returns_nonempty_figure(self, con):
        _seed_minimal_facts(con)
        fig = rf._dsr_fig(con)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0


class TestClusterHeatmap:
    def test_generated_from_synthetic_correlated_factors(self, con, monkeypatch):
        _seed_minimal_facts(con)
        rng = np.random.default_rng(0)
        base_a = rng.normal(size=40)
        base_b = rng.normal(size=40)
        synthetic = pd.DataFrame(
            {
                "factor_a": base_a,
                "factor_a_echo": base_a + rng.normal(scale=0.05, size=40),  # near-duplicate -> should cluster
                "factor_b": base_b,
                "factor_c": rng.normal(size=40),  # independent -> singleton
            }
        )
        monkeypatch.setattr(rf, "_real_annual_factor_values", lambda _con: synthetic)

        result = rf.cluster_heatmap(con)

        assert result.status == "generated"
        assert result.output_png.exists()
        assert result.output_html.exists()

    def test_skips_factors_below_min_valid(self, con):
        _seed_minimal_facts(con)
        values = rf._real_annual_factor_values(con)
        assert list(values.columns) == []  # no price/share data -> nothing clears the 30-valid floor


class TestUniverseByYear:
    def test_generated_with_real_coverage_overlap(self, con):
        panel = pd.DataFrame(
            [
                {"formation_date": pd.Timestamp("2020-01-31"), "ticker": "AAA"},
                {"formation_date": pd.Timestamp("2020-06-30"), "ticker": "AAA"},
                {"formation_date": pd.Timestamp("2020-06-30"), "ticker": "BBB"},
                {"formation_date": pd.Timestamp("2021-03-31"), "ticker": "AAA"},
                {"formation_date": pd.Timestamp("2021-03-31"), "ticker": "BBB"},
                {"formation_date": pd.Timestamp("2021-03-31"), "ticker": "CCC"},
            ]
        )
        pit_store.write_universe(con, panel)
        pit_store.write_crosswalk(
            con, pd.DataFrame([{"ticker": "AAA", "cik": 1, "entity_id": "CIK1", "title": "A", "sic": None, "sic_description": None}])
        )

        result = rf.universe_by_year(con)

        assert result.status == "generated"
        assert result.output_png.exists()


class TestPriceGatedSkips:
    @pytest.mark.parametrize(
        "fn", [rf.ff_validation, rf.cumulative_ls_returns, rf.taming_funnel, rf.ipca_r2]
    )
    def test_skips_with_reason_when_no_price_data(self, con, fn):
        result = fn(con)
        assert result.status == "skipped"
        assert result.missing_input
        assert result.produced_by


class TestMain:
    def test_exits_nonzero_when_core_tables_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(rf, "IMG_DIR", tmp_path / "img")
        monkeypatch.setattr(rf, "INTERACTIVE_DIR", tmp_path / "interactive")
        code = rf.main(db_path=tmp_path / "empty.duckdb")
        assert code == 1

    def test_exits_zero_and_reports_mixed_status_with_real_core_data(self, tmp_path, monkeypatch):
        monkeypatch.setattr(rf, "IMG_DIR", tmp_path / "img")
        monkeypatch.setattr(rf, "INTERACTIVE_DIR", tmp_path / "interactive")
        db_path = tmp_path / "seeded.duckdb"
        con = pit_store.init_db(db_path)
        _seed_minimal_facts(con)
        con.close()

        code = rf.main(db_path=db_path)

        assert code == 0
