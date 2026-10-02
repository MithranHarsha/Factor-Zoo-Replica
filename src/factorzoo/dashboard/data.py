"""Data-loading and computation helpers for the dashboard (build guide
Section 9), kept separate from app.py's Streamlit calls so the logic can
be unit tested directly -- app.py should contain no computation of its
own, only `st.*` presentation calls against what this module returns.
"""

from __future__ import annotations

import duckdb
import pandas as pd

import factorzoo.factors as f
from factorzoo.factors.panel import get_annual_factor_panel
from factorzoo.factors.registry import summarize_values

DATA_HEALTH_QUERIES: dict[str, str] = {
    "xbrl_facts": "SELECT COUNT(*) FROM xbrl_facts",
    "universe_membership_rows": "SELECT COUNT(*) FROM universe_membership",
    "universe_distinct_tickers": "SELECT COUNT(DISTINCT ticker) FROM universe_membership",
    "entity_crosswalk": "SELECT COUNT(*) FROM entity_crosswalk",
    "prices_daily": "SELECT COUNT(*) FROM prices_daily",
    "ff5_daily": "SELECT COUNT(*) FROM ff5_daily",
}


def load_data_health(con: duckdb.DuckDBPyConnection) -> dict[str, int]:
    return {label: con.execute(q).fetchone()[0] for label, q in DATA_HEALTH_QUERIES.items()}


def load_recent_manifest(con: duckdb.DuckDBPyConnection, limit: int = 20) -> pd.DataFrame:
    return con.execute(
        "SELECT pulled_at, component, detail FROM pull_manifest ORDER BY pulled_at DESC LIMIT ?", [limit]
    ).fetchdf()


def compute_zoo_overview(con: duckdb.DuckDBPyConnection, as_of: str | pd.Timestamp | None = None) -> pd.DataFrame:
    """One row per registered factor. Annual (fundamentals-only) factors
    are actually computed against the current point-in-time snapshot;
    monthly (price-based) factors are reported as "needs price data"
    rather than silently omitted, so the zoo's full 48-factor shape is
    always visible even before a price pull has succeeded.
    """
    as_of_ts = pd.Timestamp(as_of) if as_of else pd.Timestamp.today()
    panel = get_annual_factor_panel(con, as_of_ts)
    has_data = not panel.empty and panel["entity_id"].notna().any()

    rows = []
    for spec in f.FACTOR_REGISTRY.values():
        if spec.panel != "annual":
            rows.append(
                {
                    "factor": spec.name, "category": spec.category, "direction": spec.direction,
                    "n_valid": None, "median": None, "status": "needs price data",
                }
            )
            continue
        if not has_data:
            rows.append(
                {
                    "factor": spec.name, "category": spec.category, "direction": spec.direction,
                    "n_valid": 0, "median": None, "status": "no fundamentals pulled yet",
                }
            )
            continue
        try:
            values = spec.compute(panel)
            summary = summarize_values(values)
            rows.append(
                {
                    "factor": spec.name, "category": spec.category, "direction": spec.direction,
                    "n_valid": summary["n_valid"], "median": summary["median"],
                    "status": "ok" if summary["n_valid"] else "no data for this universe",
                }
            )
        except Exception as exc:  # noqa: BLE001 -- surfaced in the table, not a crashed page
            rows.append(
                {
                    "factor": spec.name, "category": spec.category, "direction": spec.direction,
                    "n_valid": 0, "median": None, "status": f"error: {exc}",
                }
            )
    return pd.DataFrame(rows).sort_values(["category", "factor"]).reset_index(drop=True)


def factor_detail(con: duckdb.DuckDBPyConnection, factor_name: str, as_of: str | pd.Timestamp | None = None) -> dict:
    """Per-company values for one factor, plus its spec metadata --
    what the Factor Detail page needs for a single selected factor."""
    spec = f.FACTOR_REGISTRY.get(factor_name)
    if spec is None:
        return {"error": f"Unknown factor '{factor_name}'"}
    if spec.panel != "annual":
        return {"spec": spec, "values": pd.DataFrame(), "note": "Price-based factor: needs price data."}

    as_of_ts = pd.Timestamp(as_of) if as_of else pd.Timestamp.today()
    panel = get_annual_factor_panel(con, as_of_ts)
    if panel.empty:
        return {"spec": spec, "values": pd.DataFrame(), "note": "No fundamentals in the store yet."}

    values = spec.compute(panel)
    out = pd.DataFrame({"entity_id": panel["entity_id"], "value": values}).dropna().sort_values(
        "value", ascending=(spec.direction == -1)
    )
    return {"spec": spec, "values": out, "note": None}
