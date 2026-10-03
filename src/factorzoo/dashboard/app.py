"""Streamlit dashboard. Five pages: Zoo Overview,
Factor Detail, Correlation Map, Taming Report, Data Health. Reads only
from the point-in-time store (never recomputes a full pipeline run), so
it stays fast and can be redeployed independently of a data refresh --
all actual computation lives in dashboard/data.py, imported here, so this
file is presentation only.

Run with: uv run streamlit run src/factorzoo/dashboard/app.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

import plotly.express as px
import streamlit as st

from factorzoo.config import DB_PATH, ensure_data_dirs
from factorzoo.dashboard.data import (
    compute_zoo_overview,
    factor_detail,
    load_data_health,
    load_recent_manifest,
)
from factorzoo.data import pit_store

st.set_page_config(page_title="Factor Zoo Replica", layout="wide")

# A small (35MB, 80-company), real-data snapshot committed at this path
# specifically for a fresh deploy (e.g. Streamlit Community Cloud) that
# has no persistent volume and so starts with no point-in-time store at
# all. On a machine that's actually run the CLI pull commands, DB_PATH
# already exists and this is a no-op -- local development is unaffected.
DEMO_SNAPSHOT_PATH = Path(__file__).resolve().parents[3] / "demo" / "factorzoo_demo.duckdb"


def _bootstrap_from_demo_snapshot_if_needed() -> None:
    if DB_PATH.exists():
        return
    if DEMO_SNAPSHOT_PATH.exists():
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(DEMO_SNAPSHOT_PATH, DB_PATH)


@st.cache_resource
def get_connection():
    ensure_data_dirs()
    _bootstrap_from_demo_snapshot_if_needed()
    return pit_store.init_db()


def page_data_health(con) -> None:
    st.header("Data Health")
    st.caption(
        "What's actually in the point-in-time store right now -- the honest version of "
        "'is this safe to trust', not a claim that everything is fully loaded."
    )
    health = load_data_health(con)
    cols = st.columns(len(health))
    for col, (label, value) in zip(cols, health.items()):
        col.metric(label.replace("_", " "), f"{value:,}")

    if health["prices_daily"] == 0:
        st.warning(
            "No price data loaded yet. Price-based factors (momentum, trading-frictions, value-category "
            "ratios needing market cap) will show as 'needs price data' on the Zoo Overview page until "
            "`factorzoo pull-prices` succeeds -- see the live finding on free price-source "
            "rate limiting (data/prices.py's module docstring) for why that may take a few tries."
        )

    st.subheader("Recent pulls")
    manifest = load_recent_manifest(con)
    if manifest.empty:
        st.info("No pipeline runs recorded yet. Run the Phase 1 CLI commands first.")
    else:
        st.dataframe(manifest, use_container_width=True)


def page_zoo_overview(con) -> None:
    st.header("Zoo Overview")
    st.caption(
        "All 48 starter-library factors. Fundamentals-only (annual-panel) factors are computed live "
        "against the current point-in-time snapshot; price-based factors show their status honestly "
        "rather than being hidden."
    )
    overview = compute_zoo_overview(con)

    category_filter = st.multiselect(
        "Category", sorted(overview["category"].unique()), default=sorted(overview["category"].unique())
    )
    status_filter = st.multiselect("Status", sorted(overview["status"].unique()), default=sorted(overview["status"].unique()))
    filtered = overview[overview["category"].isin(category_filter) & overview["status"].isin(status_filter)]

    st.dataframe(filtered, use_container_width=True, hide_index=True)
    st.caption(f"{len(filtered)} of {len(overview)} factors shown.")


def page_factor_detail(con) -> None:
    st.header("Factor Detail")
    import factorzoo.factors as f

    factor_name = st.selectbox("Factor", sorted(f.FACTOR_REGISTRY.keys()))
    result = factor_detail(con, factor_name)

    if "error" in result:
        st.error(result["error"])
        return

    spec = result["spec"]
    st.markdown(f"**Category:** {spec.category} &nbsp;&nbsp; **Direction:** {spec.direction:+d} &nbsp;&nbsp; **Source:** {spec.source}")
    st.caption(spec.description)

    if result["note"]:
        st.info(result["note"])
        return

    values = result["values"]
    if values.empty:
        st.info("No values computed for the current universe/as-of date.")
        return

    st.plotly_chart(px.bar(values, x="entity_id", y="value", title=f"{factor_name} by company"), use_container_width=True)
    st.dataframe(values, use_container_width=True, hide_index=True)


def page_correlation_map(con) -> None:
    st.header("Correlation Map")
    st.caption(
        "Factor-by-factor correlation of long-short returns, with hierarchical clusters outlined "
        "-- the visual version of 'which factors are near-duplicates'."
    )
    st.info(
        "This page reads from a factor-return time series table that Phase 3's backtest run "
        "populates (one column per factor, one row per formation date). None has been computed into "
        "the store yet in this build -- see the live finding on price-data access "
        "(data/prices.py's module docstring). "
        "The underlying mechanics (factorzoo.taming.dimension_reduction.correlation_clusters) are "
        "implemented and unit-tested; this page will render as soon as a backtest run's long-short "
        "return series are written back to the store."
    )


def page_taming_report(con) -> None:
    st.header("Taming Report")
    st.caption(
        "How many factors survive each stage of the multiple-testing and dimension-reduction "
        "pipeline."
    )
    st.info(
        "Populated once a full backtest run has produced a long-short return series for every "
        "factor (needs price data, which is not loaded yet in this build -- see Data Health). The "
        "taming machinery itself (t-hurdles, Benjamini-Hochberg/Yekutieli FDR, the Deflated Sharpe "
        "Ratio, Probability of Backtest Overfitting, correlation clustering, the LASSO spanning "
        "test, and IPCA) is implemented and unit-tested in factorzoo.taming -- this page renders "
        "its output, it doesn't compute anything itself."
    )


PAGES = {
    "Data Health": page_data_health,
    "Zoo Overview": page_zoo_overview,
    "Factor Detail": page_factor_detail,
    "Correlation Map": page_correlation_map,
    "Taming Report": page_taming_report,
}


def main() -> None:
    st.sidebar.title("Factor Zoo Replica")
    st.sidebar.caption("Free-data, point-in-time-correct factor zoo replication")
    page_name = st.sidebar.radio("Page", list(PAGES.keys()))
    con = get_connection()
    PAGES[page_name](con)


if __name__ == "__main__":
    main()
