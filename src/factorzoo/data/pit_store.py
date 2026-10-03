"""The point-in-time data store: DuckDB + Parquet-backed, filed-date indexed.

The point-in-time rule made concrete in a schema: every XBRL fact
carries its own `available_date`, every filing is kept as its own immutable
vintage row keyed by `accn` (a 10-K/A never overwrites the original), and
any query for "what did we know as of date D" is a WHERE clause on
`available_date`, not a join on period_end. The leakage test is
exactly `get_facts_as_of` used with an assertion.
"""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import pandas as pd

from factorzoo.config import DB_PATH

SCHEMA_STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS xbrl_facts (
        cik BIGINT,
        entity_id VARCHAR,
        taxonomy VARCHAR,
        tag VARCHAR,
        unit VARCHAR,
        val DOUBLE,
        period_start TIMESTAMP,
        period_end TIMESTAMP,
        fy INTEGER,
        fp VARCHAR,
        form VARCHAR,
        filed TIMESTAMP,
        accn VARCHAR,
        frame VARCHAR,
        available_date TIMESTAMP,
        pulled_at TIMESTAMP,
        PRIMARY KEY (cik, tag, unit, accn, period_end)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS universe_membership (
        formation_date TIMESTAMP,
        ticker VARCHAR,
        PRIMARY KEY (formation_date, ticker)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS entity_crosswalk (
        ticker VARCHAR,
        cik BIGINT,
        entity_id VARCHAR,
        title VARCHAR,
        sic VARCHAR,
        sic_description VARCHAR,
        as_of TIMESTAMP,
        PRIMARY KEY (ticker)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS prices_daily (
        entity_id_hint VARCHAR,
        date TIMESTAMP,
        open DOUBLE,
        high DOUBLE,
        low DOUBLE,
        close DOUBLE,
        volume DOUBLE,
        source VARCHAR,
        PRIMARY KEY (entity_id_hint, date, source)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS ff5_daily (
        date TIMESTAMP PRIMARY KEY,
        mkt_rf DOUBLE,
        smb DOUBLE,
        hml DOUBLE,
        rmw DOUBLE,
        cma DOUBLE,
        rf DOUBLE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS pull_manifest (
        pulled_at TIMESTAMP,
        component VARCHAR,
        detail VARCHAR,
        git_commit VARCHAR
    )
    """,
    # The tables below hold DERIVED backtest output, not raw pulled data --
    # unlike xbrl_facts/prices_daily (append-only, vintage-preserving),
    # these are fully REPLACED on every `run-backtest` (see
    # backtest/run_backtest.py::persist_backtest_result), since they
    # represent "the current run's result for this date range/universe",
    # not an immutable historical record.
    """
    CREATE TABLE IF NOT EXISTS factor_returns_monthly (
        formation_date TIMESTAMP,
        factor VARCHAR,
        vw_return DOUBLE,
        ew_return DOUBLE,
        PRIMARY KEY (formation_date, factor)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS taming_report (
        factor VARCHAR PRIMARY KEY,
        tstat DOUBLE,
        pvalue DOUBLE,
        passes_t2 BOOLEAN,
        passes_t3_hlz BOOLEAN,
        bh_discovery BOOLEAN,
        by_discovery BOOLEAN,
        one_per_cluster BOOLEAN,
        lasso_survives BOOLEAN,
        annualized_sharpe DOUBLE,
        dsr DOUBLE,
        cluster_label BIGINT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS ff3_replica (
        formation_date TIMESTAMP PRIMARY KEY,
        mkt DOUBLE,
        smb DOUBLE,
        hml DOUBLE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS french_validation (
        factor VARCHAR PRIMARY KEY,
        n_months INTEGER,
        correlation DOUBLE,
        clears_0_9_gate BOOLEAN
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS ipca_r2 (
        n_factors INTEGER PRIMARY KEY,
        total_r2 DOUBLE,
        predictive_r2 DOUBLE
    )
    """,
)


def init_db(db_path: Path | None = None) -> duckdb.DuckDBPyConnection:
    db_path = db_path or DB_PATH
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    for stmt in SCHEMA_STATEMENTS:
        con.execute(stmt)
    return con


def _git_commit() -> str | None:
    """Best-effort: a manifest row without a commit hash (no git repo, git
    not installed, etc.) is still useful, so failures here are swallowed
    deliberately -- this is reproducibility metadata, not the pull itself.
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def record_manifest(con: duckdb.DuckDBPyConnection, component: str, detail: dict) -> None:
    """Every pull writes a manifest row: what ran, when, against what code
    -- the reproducibility requirement every other table's provenance
    depends on."""
    con.execute(
        "INSERT INTO pull_manifest VALUES (?, ?, ?, ?)",
        [datetime.now(UTC), component, json.dumps(detail, default=str), _git_commit()],
    )


def write_xbrl_facts(con: duckdb.DuckDBPyConnection, df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    df = df.copy()
    df["pulled_at"] = pd.Timestamp.now(tz="UTC").tz_localize(None)
    cols = [
        "cik", "entity_id", "taxonomy", "tag", "unit", "val", "period_start", "period_end",
        "fy", "fp", "form", "filed", "accn", "frame", "available_date", "pulled_at",
    ]
    df = df[cols].dropna(subset=["cik", "tag", "unit", "accn", "period_end"])
    con.register("_stage", df)
    con.execute(
        f"""
        INSERT INTO xbrl_facts
        SELECT {', '.join(cols)} FROM _stage
        ON CONFLICT (cik, tag, unit, accn, period_end) DO NOTHING
        """
    )
    con.unregister("_stage")
    return len(df)


def write_universe(con: duckdb.DuckDBPyConnection, df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    con.register("_stage", df[["formation_date", "ticker"]])
    con.execute(
        "INSERT INTO universe_membership SELECT * FROM _stage "
        "ON CONFLICT (formation_date, ticker) DO NOTHING"
    )
    con.unregister("_stage")
    return len(df)


def write_crosswalk(con: duckdb.DuckDBPyConnection, df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    df = df.copy()
    df["as_of"] = pd.Timestamp.now(tz="UTC").tz_localize(None)
    cols = ["ticker", "cik", "entity_id", "title", "sic", "sic_description", "as_of"]
    for c in cols:
        if c not in df.columns:
            df[c] = None
    con.register("_stage", df[cols])
    con.execute(
        """
        INSERT INTO entity_crosswalk SELECT * FROM _stage
        ON CONFLICT (ticker) DO UPDATE SET
            cik = excluded.cik, entity_id = excluded.entity_id, title = excluded.title,
            sic = excluded.sic, sic_description = excluded.sic_description, as_of = excluded.as_of
        """
    )
    con.unregister("_stage")
    return len(df)


def write_prices(con: duckdb.DuckDBPyConnection, df: pd.DataFrame, source: str) -> int:
    if df.empty:
        return 0
    df = df.copy()
    df["source"] = source
    cols = ["entity_id_hint", "date", "open", "high", "low", "close", "volume", "source"]
    con.register("_stage", df[cols])
    con.execute(
        "INSERT INTO prices_daily SELECT * FROM _stage "
        "ON CONFLICT (entity_id_hint, date, source) DO NOTHING"
    )
    con.unregister("_stage")
    return len(df)


def write_ff5(con: duckdb.DuckDBPyConnection, df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    con.register("_stage", df)
    con.execute("INSERT INTO ff5_daily SELECT * FROM _stage ON CONFLICT (date) DO NOTHING")
    con.unregister("_stage")
    return len(df)


def _replace_table(con: duckdb.DuckDBPyConnection, table: str, df: pd.DataFrame, columns: list[str]) -> int:
    """Full replace, not upsert -- these tables hold one backtest run's
    derived output, not accumulated history (see the schema comment
    above factor_returns_monthly)."""
    con.execute(f"DELETE FROM {table}")
    if df.empty:
        return 0
    con.register("_stage", df[columns])
    con.execute(f"INSERT INTO {table} SELECT * FROM _stage")
    con.unregister("_stage")
    return len(df)


def write_factor_returns(con: duckdb.DuckDBPyConnection, vw: pd.DataFrame, ew: pd.DataFrame) -> int:
    vw_long = vw.reset_index(names="formation_date").melt(id_vars="formation_date", var_name="factor", value_name="vw_return")
    ew_long = ew.reset_index(names="formation_date").melt(id_vars="formation_date", var_name="factor", value_name="ew_return")
    merged = vw_long.merge(ew_long, on=["formation_date", "factor"], how="outer").dropna(subset=["vw_return", "ew_return"], how="all")
    return _replace_table(con, "factor_returns_monthly", merged, ["formation_date", "factor", "vw_return", "ew_return"])


def write_taming_report(con: duckdb.DuckDBPyConnection, taming: pd.DataFrame) -> int:
    df = taming.reset_index(names="factor")
    cols = [
        "factor", "tstat", "pvalue", "passes_t2", "passes_t3_hlz", "bh_discovery", "by_discovery",
        "one_per_cluster", "lasso_survives", "annualized_sharpe", "dsr", "cluster_label",
    ]
    for c in cols:
        if c not in df.columns:
            df[c] = pd.NA
    return _replace_table(con, "taming_report", df, cols)


def write_ff3_replica(con: duckdb.DuckDBPyConnection, df: pd.DataFrame) -> int:
    """Expects a flat frame (not date-indexed) with a 'date' column --
    i.e. ff3_series' own output column name, or BacktestResult.ff3_replica
    .reset_index() -- confirmed live: an earlier version of this function
    tried to handle an already-reset-index frame as if it might still
    need resetting, which re-ran reset_index on a plain RangeIndex and
    wrote row numbers (0, 1, 2, ...) into formation_date instead of the
    real dates.
    """
    data = df.rename(columns={"date": "formation_date"}) if "date" in df.columns else df
    return _replace_table(con, "ff3_replica", data, ["formation_date", "mkt", "smb", "hml"])


def write_french_validation(con: duckdb.DuckDBPyConnection, df: pd.DataFrame) -> int:
    data = df.rename(columns={"clears_0.9_gate": "clears_0_9_gate"})
    return _replace_table(con, "french_validation", data, ["factor", "n_months", "correlation", "clears_0_9_gate"])


def write_ipca_r2(con: duckdb.DuckDBPyConnection, df: pd.DataFrame) -> int:
    return _replace_table(con, "ipca_r2", df, ["n_factors", "total_r2", "predictive_r2"])


# --- point-in-time queries ---------------------------------------------------


def get_facts_as_of(
    con: duckdb.DuckDBPyConnection, as_of: str | pd.Timestamp, tags: list[str] | None = None
) -> pd.DataFrame:
    """The query that makes point-in-time correctness real rather than
    aspirational: every fact returned has available_date <= as_of. Section
    8's leakage test is this function plus an assertion that no returned
    row's available_date exceeds the requested date.
    """
    q = "SELECT * FROM xbrl_facts WHERE available_date <= ?"
    params: list = [pd.Timestamp(as_of)]
    if tags:
        placeholders = ", ".join(["?"] * len(tags))
        q += f" AND tag IN ({placeholders})"
        params.extend(tags)
    return con.execute(q, params).fetchdf()


def get_universe_as_of(con: duckdb.DuckDBPyConnection, as_of: str | pd.Timestamp) -> list[str]:
    """Nearest formation_date at or before `as_of` (universe is formed
    monthly)."""
    row = con.execute(
        "SELECT MAX(formation_date) FROM universe_membership WHERE formation_date <= ?",
        [pd.Timestamp(as_of)],
    ).fetchone()
    nearest = row[0] if row else None
    if nearest is None:
        return []
    out = con.execute(
        "SELECT ticker FROM universe_membership WHERE formation_date = ? ORDER BY ticker", [nearest]
    ).fetchdf()
    return out["ticker"].tolist()
