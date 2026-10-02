"""Command-line entry point for Phase 1: init the store, build the
point-in-time universe, pull EDGAR fundamentals and prices for a pilot, pull
the Ken French benchmark series, and report data health.

Usage (after `uv sync` and copying .env.example to .env):
    uv run factorzoo init-db
    uv run factorzoo build-universe
    uv run factorzoo pull-riskfree
    uv run factorzoo pull-edgar --limit 25
    uv run factorzoo pull-prices --limit 25
    uv run factorzoo status
"""

from __future__ import annotations

import logging

import pandas as pd
import typer

from factorzoo.config import ConfigError, ensure_data_dirs, load_settings
from factorzoo.data import edgar, pit_store, prices, riskfree
from factorzoo.universe import filters as universe_filters

app = typer.Typer(add_completion=False, help="factorzoo: free-data, point-in-time factor zoo replica")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("factorzoo.cli")


@app.command("init-db")
def init_db_cmd() -> None:
    """Create (or verify) the DuckDB schema."""
    ensure_data_dirs()
    con = pit_store.init_db()
    tables = con.execute("SHOW TABLES").fetchdf()
    typer.echo(f"Schema ready at {pit_store.DB_PATH}. Tables: {', '.join(tables['name'])}")
    con.close()


@app.command("build-universe")
def build_universe_cmd(
    start: str = typer.Option("1996-01-31", help="First month-end to materialize"),
    force: bool = typer.Option(False, help="Re-download the source history even if cached"),
) -> None:
    """Build the point-in-time universe panel from historical S&P 500
    membership and write it to the store. This is the survivorship-bias
    fix from build guide Section 4: additions AND removals are both in the
    source data, not reconstructed after the fact.
    """
    ensure_data_dirs()
    history = universe_filters.fetch_sp500_history(force=force)
    coverage = universe_filters.summarize_coverage(history)
    typer.echo("Universe source coverage:")
    for k, v in coverage.items():
        typer.echo(f"  {k}: {v}")

    panel = universe_filters.monthly_universe_panel(history, start=start)
    con = pit_store.init_db()
    n = pit_store.write_universe(con, panel)
    pit_store.record_manifest(con, "build-universe", {**coverage, "rows_written": n, "panel_start": start})
    con.close()
    typer.echo(f"Wrote {n} (formation_date, ticker) rows to universe_membership.")


@app.command("pull-riskfree")
def pull_riskfree_cmd(force: bool = typer.Option(False)) -> None:
    """Pull Ken French daily FF5 factors (validation benchmark, Section 4/11)."""
    ensure_data_dirs()
    df = riskfree.fetch_ff5_daily(force=force)
    con = pit_store.init_db()
    n = pit_store.write_ff5(con, df)
    pit_store.record_manifest(con, "pull-riskfree", {"rows": n, "date_min": str(df["date"].min()), "date_max": str(df["date"].max())})
    con.close()
    typer.echo(f"Wrote {n} daily FF5 rows ({df['date'].min().date()} to {df['date'].max().date()}).")


@app.command("pull-edgar")
def pull_edgar_cmd(
    limit: int = typer.Option(25, help="Number of tickers to pull (pilot scope; keep modest for a smoke test)"),
    as_of: str = typer.Option(None, help="Universe as-of date (defaults to latest formation date)"),
) -> None:
    """Pull SEC EDGAR company facts + submissions (SIC) for a pilot slice
    of the current universe. Full-scale (~500-2,500 companies) pulls take
    a while under the rate limiter by design -- see Section 4 on SEC's
    fair-use guidance -- run this in the background for a larger `limit`.
    """
    ensure_data_dirs()
    try:
        settings = load_settings()
    except ConfigError as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(1)

    con = pit_store.init_db()
    as_of_ts = pd.Timestamp(as_of) if as_of else pd.Timestamp.today()
    universe = pit_store.get_universe_as_of(con, as_of_ts)
    if not universe:
        typer.secho("Universe is empty -- run `build-universe` first.", fg=typer.colors.RED)
        raise typer.Exit(1)

    tickers_df = edgar.EdgarClient(settings).fetch_company_tickers()
    ticker_to_cik = dict(zip(tickers_df["ticker"], tickers_df["cik"]))

    client = edgar.EdgarClient(settings)
    pulled, skipped, fact_rows = 0, 0, 0
    for ticker in universe[:limit]:
        cik = ticker_to_cik.get(ticker)
        if cik is None:
            skipped += 1
            log.info("No CIK found for %s (not a current EDGAR filer, or ticker changed)", ticker)
            continue
        submissions = client.fetch_submissions(cik)
        sic_info = edgar.extract_sic(submissions)
        pit_store.write_crosswalk(con, pd.DataFrame([{
            "ticker": ticker, "cik": cik, "entity_id": f"CIK{cik:010d}",
            "title": sic_info.get("entity_name"), "sic": sic_info.get("sic"),
            "sic_description": sic_info.get("sic_description"),
        }]))

        facts_json = client.fetch_company_facts(cik)
        if facts_json is None:
            skipped += 1
            continue
        facts_df = edgar.extract_xbrl_facts(facts_json)
        n = pit_store.write_xbrl_facts(con, facts_df)
        fact_rows += n
        pulled += 1
        log.info("[%d/%d] %s (CIK %s): %d fact rows", pulled, min(limit, len(universe)), ticker, cik, n)

    pit_store.record_manifest(
        con, "pull-edgar", {"limit": limit, "pulled": pulled, "skipped": skipped, "fact_rows": fact_rows}
    )
    con.close()
    client.close()
    typer.echo(f"Pulled {pulled} companies ({fact_rows} fact rows), skipped {skipped}.")


@app.command("pull-prices")
def pull_prices_cmd(
    limit: int = typer.Option(25, help="Number of tickers to pull"),
    source: str = typer.Option("yahoo", help="'yahoo' (default, free/no key) or 'stooq' (currently blocked)"),
    as_of: str = typer.Option(None),
) -> None:
    """Pull daily price history for a pilot slice of the current universe.

    Default source is Yahoo Finance's chart API (free, no key). Stooq is
    documented in the build guide as the primary source but is currently
    blocked behind a JavaScript bot-check for automated clients -- a live
    finding from this build, not a hypothetical -- see data/prices.py's
    module docstring. Pass --source stooq to retry it once that changes.
    """
    ensure_data_dirs()
    con = pit_store.init_db()
    as_of_ts = pd.Timestamp(as_of) if as_of else pd.Timestamp.today()
    universe = pit_store.get_universe_as_of(con, as_of_ts)
    if not universe:
        typer.secho("Universe is empty -- run `build-universe` first.", fg=typer.colors.RED)
        raise typer.Exit(1)

    fetch_fn = prices.fetch_stooq_daily if source == "stooq" else prices.fetch_yahoo_chart_daily
    pulled, empty, failed, total_rows = 0, 0, 0, 0
    failures: list[str] = []
    for ticker in universe[:limit]:
        try:
            df = fetch_fn(ticker)
        except Exception as exc:  # noqa: BLE001 -- one bad/rate-limited ticker must not kill the whole batch
            failed += 1
            failures.append(f"{ticker}: {exc}")
            log.warning("%s: failed (%s)", ticker, exc)
            continue
        if df.empty:
            empty += 1
            continue
        n = pit_store.write_prices(con, df, source=source)
        total_rows += n
        pulled += 1
        log.info("%s: %d daily rows", ticker, n)

    pit_store.record_manifest(
        con, "pull-prices",
        {"limit": limit, "source": source, "pulled": pulled, "empty": empty, "failed": failed, "rows": total_rows},
    )
    con.close()
    typer.echo(f"Pulled {pulled} tickers ({total_rows} rows) from {source}, {empty} returned no data, {failed} failed.")
    if failures:
        typer.secho(f"\n{len(failures)} ticker(s) failed (showing up to 5):", fg=typer.colors.YELLOW)
        for line in failures[:5]:
            typer.echo(f"  {line}")
        if failed > 0 and pulled == 0:
            typer.secho(
                "\nEvery ticker failed. If these are all rate-limit (429) errors, this is the live external "
                "rate-limiting documented in data/prices.py and the README, not a code problem -- retry later, "
                "from a different network, or configure a Tiingo key.",
                fg=typer.colors.YELLOW,
            )


@app.command("compute-factors")
def compute_factors_cmd(
    as_of: str = typer.Option(None, help="Snapshot as-of date (defaults to today)"),
    category: str = typer.Option(None, help="Restrict to one factor category, e.g. 'profitability'"),
) -> None:
    """Computes every registered annual (fundamentals-only) factor against
    the current point-in-time snapshot and prints summary statistics. The
    annual factors that don't need market_cap (profitability, investment,
    accruals, most financing/distress -- see registry_summary()) run on
    EDGAR data alone, so this works even while price data is unavailable;
    value-category and all monthly (price-based) factors will be all-NaN
    until pull-prices has real data loaded.
    """
    ensure_data_dirs()
    import factorzoo.factors as f
    from factorzoo.factors.panel import get_annual_factor_panel
    from factorzoo.factors.registry import summarize_values

    con = pit_store.init_db()
    as_of_ts = pd.Timestamp(as_of) if as_of else pd.Timestamp.today()
    panel = get_annual_factor_panel(con, as_of_ts)
    if panel.empty or "entity_id" not in panel.columns or panel["entity_id"].isna().all():
        typer.secho("No fundamentals in the point-in-time store yet -- run `pull-edgar` first.", fg=typer.colors.RED)
        raise typer.Exit(1)

    specs = [s for s in f.FACTOR_REGISTRY.values() if s.panel == "annual" and (category is None or s.category == category)]
    rows = []
    for spec in sorted(specs, key=lambda s: (s.category, s.name)):
        try:
            values = spec.compute(panel)
        except Exception as exc:  # noqa: BLE001 -- surfaced as a per-factor row, not a crash
            rows.append({"factor": spec.name, "category": spec.category, "n_valid": 0, "median": None, "error": repr(exc)})
            continue
        summary = summarize_values(values)
        rows.append(
            {
                "factor": spec.name,
                "category": spec.category,
                "n_valid": summary["n_valid"],
                "median": summary["median"],
                "error": None,
            }
        )
    report = pd.DataFrame(rows)
    typer.echo(f"Computed {len(specs)} annual factors against {panel['entity_id'].notna().sum()} companies "
               f"as of {as_of_ts.date()}:\n")
    typer.echo(report.to_string(index=False))
    con.close()


@app.command("make-figures")
def make_figures_cmd() -> None:
    """Generate the README proof-of-work figures (docs/img/*.png + GIF,
    docs/interactive/*.html) from whatever real data is currently in the
    point-in-time store. Figures whose inputs don't exist yet are skipped
    and reported, not faked -- see viz/readme_figures.py's module docstring.
    Exits nonzero only if the core tables (universe, EDGAR facts) are
    missing entirely; a price-data gap is an expected, reported skip.
    """
    from factorzoo.viz import readme_figures

    raise typer.Exit(readme_figures.main())


@app.command("status")
def status_cmd() -> None:
    """Data-health report -- the human-readable version of Phase 1's gate."""
    ensure_data_dirs()
    con = pit_store.init_db()
    counts = {
        "xbrl_facts": con.execute("SELECT COUNT(*) FROM xbrl_facts").fetchone()[0],
        "universe_membership (rows)": con.execute("SELECT COUNT(*) FROM universe_membership").fetchone()[0],
        "universe_membership (distinct tickers)": con.execute(
            "SELECT COUNT(DISTINCT ticker) FROM universe_membership"
        ).fetchone()[0],
        "entity_crosswalk": con.execute("SELECT COUNT(*) FROM entity_crosswalk").fetchone()[0],
        "prices_daily": con.execute("SELECT COUNT(*) FROM prices_daily").fetchone()[0],
        "ff5_daily": con.execute("SELECT COUNT(*) FROM ff5_daily").fetchone()[0],
    }
    typer.echo("Data health:")
    for k, v in counts.items():
        typer.echo(f"  {k}: {v}")

    manifest = con.execute(
        "SELECT pulled_at, component, detail FROM pull_manifest ORDER BY pulled_at DESC LIMIT 10"
    ).fetchdf()
    if not manifest.empty:
        typer.echo("\nRecent pulls:")
        for _, row in manifest.iterrows():
            typer.echo(f"  {row['pulled_at']}  {row['component']}  {row['detail']}")
    con.close()


if __name__ == "__main__":
    app()
