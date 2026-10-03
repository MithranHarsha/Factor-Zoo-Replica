"""README proof-of-work figures.

RULES (non-negotiable, enforced by this module's structure, not just this
comment): every figure is generated from real tables in the point-in-time
DuckDB store (data/factorzoo.duckdb) -- never mocked or hand-typed
numbers. A figure whose required input doesn't exist yet (price-derived
returns, principally) is SKIPPED, not faked, and shows up in the status
table main() prints with the phase that produces it. A figure that is
pure parametric theory (no per-point real data, e.g. the DSR/hurdle
surfaces' underlying math) is explicitly titled "illustration".

Architecture note: the brief this was built from assumed a `data/outputs/`
Parquet pipeline with a run manifest. This project's actual pipeline
writes to a DuckDB store instead (`factorzoo.data.pit_store`), which
already carries an equivalent manifest (`pull_manifest`, with a git
commit hash per pull -- see `pit_store.record_manifest`). Every function
below reads from that store directly rather than a Parquet layer that
was never built, which is an architecture adaptation, not an invented
data source: the underlying facts (what's real, what's missing) are
unchanged.

Run with: uv run factorzoo make-figures
"""

from __future__ import annotations

import subprocess
import textwrap
from dataclasses import dataclass
from pathlib import Path

import duckdb
import imageio.v2 as imageio
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats

from factorzoo.data import pit_store

REPO_ROOT = Path(__file__).resolve().parents[3]
IMG_DIR = REPO_ROOT / "docs" / "img"
INTERACTIVE_DIR = REPO_ROOT / "docs" / "interactive"

# -- shared dark theme (plotly_dark base + explicit near-black surface) -----
BG = "#0b0e14"
GRID = "#262b36"
TEXT = "#e8e8e8"
MUTED = "#9aa0a6"
FONT_FAMILY = "Arial, Helvetica, sans-serif"
BLUE_YELLOW = [[0.0, "#0d2340"], [0.5, "#2a78d6"], [1.0, "#eda100"]]
ACCENT_BLUE = "#3987e5"
ACCENT_YELLOW = "#eda100"
ACCENT_RED = "#e66767"
ACCENT_GREEN = "#1baf7a"


@dataclass
class FigureResult:
    name: str
    status: str  # "generated" or "skipped"
    output_png: Path | None = None
    output_html: Path | None = None
    missing_input: str | None = None
    produced_by: str | None = None
    note: str | None = None


def _git_commit_now() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5, check=False
        )
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def snapshot_footer_text(con: duckdb.DuckDBPyConnection) -> str:
    """'data snapshot <date> Â· commit <hash>', from the real latest
    pull_manifest row (data recency) and the current git HEAD (code
    recency) -- deliberately not the same thing: the data can be a day
    old while the code generating the figure is from right now.
    """
    row = con.execute("SELECT MAX(pulled_at) FROM pull_manifest").fetchone()
    snapshot_date = row[0].date().isoformat() if row and row[0] else "no data pulled yet"
    return f"data snapshot {snapshot_date} · commit {_git_commit_now()}"


def _dark_layout(fig: go.Figure, title: str, subtitle: str, footer: str, height: int = 820) -> go.Figure:
    # Plotly titles don't auto-wrap to the figure width -- a long subtitle
    # just runs off the right edge (seen live on the first render of this
    # figure) unless line breaks are inserted explicitly.
    wrapped_subtitle = "<br>".join(textwrap.wrap(subtitle, width=100))
    full_title = f"{title}<br><span style='font-size:13px;color={MUTED}'>{wrapped_subtitle}</span>"
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font={"family": FONT_FAMILY, "color": TEXT, "size": 13},
        title={"text": full_title, "x": 0.5, "xanchor": "center", "font": {"size": 19}},
        width=1200,
        height=height,
        margin={"l": 70, "r": 40, "t": 135, "b": 70},
        annotations=list(fig.layout.annotations or [])
        + [
            {
                "text": footer, "showarrow": False, "x": 0.5, "y": -0.08, "xref": "paper", "yref": "paper",
                "font": {"size": 10.5, "color": MUTED},
            }
        ],
    )
    return fig


def _save_figure(fig: go.Figure, name: str, scale: float = 2) -> tuple[Path, Path]:
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    INTERACTIVE_DIR.mkdir(parents=True, exist_ok=True)
    png_path = IMG_DIR / f"{name}.png"
    html_path = INTERACTIVE_DIR / f"{name}.html"
    fig.write_image(str(png_path), scale=scale)
    # The PNG needs the fixed 1200px canvas, but the interactive page should
    # fill the browser window, so drop the fixed size for the HTML only.
    html_fig = go.Figure(fig)
    html_fig.update_layout(width=None, height=None, autosize=True)
    html_fig.write_html(
        str(html_path), include_plotlyjs="cdn", full_html=True,
        default_width="100vw", default_height="100vh", config={"responsive": True},
    )
    html_path.write_text(
        html_path.read_text().replace("<head>", f"<head><style>html,body{{margin:0;overflow:hidden;background:{BG}}}</style>", 1)
    )
    size_kb = png_path.stat().st_size / 1024
    if size_kb > 500:
        print(f"  warning: {png_path.name} is {size_kb:.0f}KB, over the 500KB target")
    return png_path, html_path


def core_tables_present(con: duckdb.DuckDBPyConnection) -> tuple[bool, str]:
    """The one thing that's a hard error, not a graceful skip: no data
    pulled at all. Price-data-specific gaps (most of the figures below)
    are expected and reported, not fatal."""
    universe_n = con.execute("SELECT COUNT(*) FROM universe_membership").fetchone()[0]
    facts_n = con.execute("SELECT COUNT(*) FROM xbrl_facts").fetchone()[0]
    if universe_n == 0:
        return False, "universe_membership is empty -- run `factorzoo build-universe` first"
    if facts_n == 0:
        return False, "xbrl_facts is empty -- run `factorzoo pull-edgar` first"
    return True, ""


def returns_data_present(con: duckdb.DuckDBPyConnection) -> bool:
    return con.execute("SELECT COUNT(*) FROM prices_daily").fetchone()[0] > 0


def backtest_results_present(con: duckdb.DuckDBPyConnection) -> bool:
    """Having price data isn't enough on its own -- `factorzoo
    run-backtest` has to have actually been run to turn it into factor
    return series, the FF3 replication, and the taming report these
    figures read from."""
    return con.execute("SELECT COUNT(*) FROM taming_report").fetchone()[0] > 0


def _latest_backtest_manifest(con: duckdb.DuckDBPyConnection) -> dict:
    import json

    row = con.execute(
        "SELECT detail FROM pull_manifest WHERE component = 'run-backtest' ORDER BY pulled_at DESC LIMIT 1"
    ).fetchone()
    return json.loads(row[0]) if row else {}


def _real_sample_years(con: duckdb.DuckDBPyConnection) -> float:
    """Years of point-in-time-available EDGAR data, measured from `filed`
    (when a fact actually became knowable), not `period_end` (which can
    reach earlier via prior-year comparative figures inside a later
    filing) -- the same distinction the point-in-time rule itself turns
    on."""
    lo, hi = con.execute("SELECT MIN(filed), MAX(filed) FROM xbrl_facts").fetchone()
    return (hi - lo).days / 365.25


# --- (b) hurdle_surface: fully real (Bonferroni math + real/cited markers) --


def hurdle_surface(con: duckdb.DuckDBPyConnection) -> FigureResult:
    from factorzoo.factors.registry import FACTOR_REGISTRY

    name = "hurdle_surface"
    alpha = 0.05
    log_n = np.linspace(0, 3.5, 90)
    years = np.linspace(5, 60, 90)
    log_n_grid, years_grid = np.meshgrid(log_n, years)
    n_tests = 10.0**log_n_grid
    df = years_grid * 12 - 1
    required_t = stats.t.ppf(1 - alpha / (2 * n_tests), df)

    fig = go.Figure(
        data=[
            go.Surface(
                x=log_n, y=years, z=required_t, colorscale="Plasma",
                contours={"z": {"show": True, "usecolormap": True, "project_z": False, "width": 1}},
                colorbar={"title": {"text": "required |t|", "font": {"color": TEXT}}, "tickfont": {"color": TEXT}},
                opacity=0.95,
            )
        ]
    )

    n_real = len([s for s in FACTOR_REGISTRY.values()])
    years_real = _real_sample_years(con)
    t_real = float(stats.t.ppf(1 - alpha / (2 * n_real), years_real * 12 - 1))
    fig.add_trace(
        go.Scatter3d(
            x=[np.log10(n_real)], y=[years_real], z=[t_real + 0.5],
            mode="markers+text", text=[f"this replica<br>t≈{t_real:.2f}"], textposition="top left",
            marker={"size": 7, "color": ACCENT_BLUE, "symbol": "diamond"},
            textfont={"color": ACCENT_BLUE, "size": 12}, showlegend=False, hoverinfo="skip",
        )
    )

    n_hxz, years_hxz = 452, 55.0  # Hou, Xue & Zhang (2020) -- cited literature value, not computed here
    t_hxz = float(stats.t.ppf(1 - alpha / (2 * n_hxz), years_hxz * 12 - 1))
    fig.add_trace(
        go.Scatter3d(
            x=[np.log10(n_hxz)], y=[years_hxz], z=[t_hxz + 0.5],
            mode="markers+text", text=[f"Hou-Xue-Zhang 2020<br>t≈{t_hxz:.2f}"], textposition="top left",
            marker={"size": 7, "color": ACCENT_YELLOW, "symbol": "diamond"},
            textfont={"color": ACCENT_YELLOW, "size": 12}, showlegend=False, hoverinfo="skip",
        )
    )

    fig.update_scenes(
        xaxis={"title": "log10(number of tests)", "gridcolor": GRID, "backgroundcolor": BG, "color": TEXT},
        yaxis={"title": "sample length (years)", "gridcolor": GRID, "backgroundcolor": BG, "color": TEXT},
        zaxis={"title": "required |t|", "gridcolor": GRID, "backgroundcolor": BG, "color": TEXT},
        camera={"eye": {"x": 1.7, "y": -1.9, "z": 0.7}},
    )
    fig.update_traces(colorbar={"x": 1.0, "len": 0.6, "y": 0.3, "thickness": 16}, selector={"type": "surface"})
    _dark_layout(
        fig,
        "How many tests you ran sets the bar a real factor has to clear",
        f"Bonferroni α=5%; diamonds: this replica (real, {n_real} factors / {years_real:.0f}y EDGAR) "
        "vs. Hou-Xue-Zhang 2020 (cited, 452 / ~55y)",
        snapshot_footer_text(con),
    )
    png, html = _save_figure(fig, name)
    return FigureResult(name=name, status="generated", output_png=png, output_html=html)


# --- (a)/(c) DSR surface: illustration only -- real overlay needs returns --

DSR_ASSUMED_VAR_SR = 0.35**2  # assumed cross-sectional std of trial Sharpes = 0.35; cited, not fit to real trials


def _dsr_surface_z(sharpe_grid: np.ndarray, log_n_grid: np.ndarray, var_sr: float, n_obs: float) -> np.ndarray:
    """Mirrors taming/multiple_testing.py::deflated_sharpe_ratio's closed
    form exactly, parameterized directly by (var_sr, n_obs) instead of a
    synthetic trial-Sharpe array, so the surface is smooth and
    deterministic rather than re-sampling noise at every grid point."""
    euler_gamma = 0.5772156649015329
    n_trials = 10.0**log_n_grid
    sr_max_expected = np.sqrt(var_sr) * (
        (1 - euler_gamma) * stats.norm.ppf(1 - 1.0 / n_trials) + euler_gamma * stats.norm.ppf(1 - 1.0 / (n_trials * np.e))
    )
    sigma_sr_hat = np.sqrt(np.maximum(1 + 0.25 * sharpe_grid**2, 1e-12) / (n_obs - 1))  # skew=0, kurtosis=3 (normal)
    return stats.norm.cdf((sharpe_grid - sr_max_expected) / sigma_sr_hat)


def _dsr_fig(con: duckdb.DuckDBPyConnection) -> go.Figure:
    sharpe = np.linspace(0, 2.5, 90)
    # Starts at 0.15 (~1.4 trials), not 0: N=1 is a genuine degeneracy in
    # the formula (comparing against the max of ONE trial is undefined --
    # confirmed live, the N=1 column saturates to DSR=1 for every Sharpe,
    # which is a real edge case, not a rendering bug) and the whole point
    # of this figure is N>1.
    log_n = np.linspace(0.15, 3.5, 90)
    sharpe_grid, log_n_grid = np.meshgrid(sharpe, log_n)
    # n_obs is illustrative (36 months, a common minimum track-record
    # length in this literature), NOT tied to this replica's real ~17y
    # EDGAR span: tried that first, and at n_obs~210 the transition band
    # between "luck" and "real" becomes razor-thin (confirmed by printing
    # the actual grid values), which is mathematically correct -- more
    # data really does make the DSR test sharper -- but makes for a
    # near-binary, uninformative illustration rather than a legible one.
    n_obs = 36
    z = _dsr_surface_z(sharpe_grid, log_n_grid, DSR_ASSUMED_VAR_SR, n_obs)
    years_real = _real_sample_years(con)

    fig = go.Figure(
        data=[
            go.Surface(
                x=sharpe, y=log_n, z=z, colorscale=BLUE_YELLOW,
                contours={"z": {"show": True, "usecolormap": True, "project_z": False, "width": 1}},
                colorbar={"title": {"text": "DSR", "font": {"color": TEXT}}, "tickfont": {"color": TEXT}, "x": 1.0, "len": 0.6, "y": 0.3, "thickness": 16},
            )
        ]
    )
    fig.update_scenes(
        xaxis={"title": "observed annualized Sharpe", "gridcolor": GRID, "backgroundcolor": BG, "color": TEXT},
        yaxis={"title": "log10(number of trials)", "gridcolor": GRID, "backgroundcolor": BG, "color": TEXT},
        zaxis={"title": "Deflated Sharpe Ratio", "gridcolor": GRID, "backgroundcolor": BG, "color": TEXT, "range": [0, 1]},
        camera={"eye": {"x": 1.7, "y": -1.9, "z": 0.7}},
    )
    overlay_note = "No real per-factor overlay yet -- needs price data (Phase 3 backtest)"
    if backtest_results_present(con):
        taming = con.execute("SELECT * FROM taming_report").fetchdf().dropna(subset=["annualized_sharpe", "dsr"])
        if not taming.empty:
            n_trials = con.execute("SELECT COUNT(*) FROM taming_report").fetchone()[0]
            log_n_trials = np.log10(max(n_trials, 2))
            survived = taming["bh_discovery"].astype(bool)
            for mask, color, label in [(~survived, ACCENT_RED, "did not survive BH"), (survived, ACCENT_GREEN, "BH discovery")]:
                sub = taming[mask]
                if sub.empty:
                    continue
                fig.add_trace(
                    go.Scatter3d(
                        x=sub["annualized_sharpe"], y=[log_n_trials] * len(sub), z=sub["dsr"],
                        mode="markers", name=label, marker={"size": 4, "color": color, "opacity": 0.75},
                        hovertext=sub["factor"], hoverinfo="text",
                    )
                )
            best = taming.loc[taming["dsr"].idxmax()]
            fig.add_trace(
                go.Scatter3d(
                    x=[best["annualized_sharpe"]], y=[log_n_trials], z=[best["dsr"] + 0.03],
                    mode="markers+text", marker={"size": 7, "color": ACCENT_YELLOW, "symbol": "diamond"},
                    text=[f"{best['factor']}<br>Sharpe={best['annualized_sharpe']:.2f}, DSR={best['dsr']:.2f}"],
                    textposition="top center", textfont={"color": ACCENT_YELLOW, "size": 11}, showlegend=False,
                )
            )
            fig.update_layout(legend={"bgcolor": "rgba(0,0,0,0)", "font": {"color": TEXT}, "x": 0.02, "y": 0.9})
            overlay_note = (
                f"Overlay: {len(taming)} real factors from the actual backtest (VW long-short annualized Sharpe, "
                f"log10({n_trials}) trials tested, real DSR), green = survives BH at q=10%. Best by DSR: "
                f"{best['factor']}."
            )

    _dark_layout(
        fig,
        "ILLUSTRATION: how many strategies you tried deflates an impressive Sharpe ratio",
        f"Parametric surface (Bailey & Lopez de Prado 2014 closed form); assumed trial-Sharpe std=0.35, "
        f"n_obs={n_obs:.0f} months (illustrative -- this replica's real EDGAR span is {years_real:.0f}y, but that "
        f"makes the surface a near-vertical cliff, not a legible one). {overlay_note}",
        snapshot_footer_text(con),
    )
    return fig


def dsr_surface(con: duckdb.DuckDBPyConnection) -> FigureResult:
    fig = _dsr_fig(con)
    png, html = _save_figure(fig, "dsr_surface", scale=1.6)
    note = (
        "surface is illustration (parametric); per-factor markers are real backtest results"
        if backtest_results_present(con)
        else "illustration only -- real per-factor Sharpe overlay needs `factorzoo run-backtest`"
    )
    return FigureResult(name="dsr_surface", status="generated", output_png=png, output_html=html, note=note)


def dsr_surface_rotating_gif(con: duckdb.DuckDBPyConnection) -> FigureResult:
    name = "dsr_surface_rotating"
    fig = _dsr_fig(con)
    gif_path = IMG_DIR / f"{name}.gif"
    IMG_DIR.mkdir(parents=True, exist_ok=True)

    frames = []
    n_frames = 36
    for i in range(n_frames):
        angle = 2 * np.pi * i / n_frames
        fig.update_scenes(camera={"eye": {"x": 1.9 * np.cos(angle), "y": 1.9 * np.sin(angle), "z": 0.7}})
        png_bytes = fig.to_image(format="png", scale=1, width=800, height=600)
        frames.append(imageio.imread(png_bytes))
    imageio.mimsave(gif_path, frames, duration=1000 / 10, loop=0)  # 10 fps, looping

    size_mb = gif_path.stat().st_size / (1024 * 1024)
    if size_mb > 5:
        print(f"  warning: {gif_path.name} is {size_mb:.1f}MB, over the 5MB target")
    return FigureResult(
        name=name, status="generated", output_png=gif_path, output_html=None,
        note=f"illustration only (same caveat as dsr_surface); {size_mb:.1f}MB, {n_frames} frames @ 10fps",
    )


# --- (g) cluster_heatmap: real data (cross-sectional VALUE correlation) ----


def _real_annual_factor_values(con: duckdb.DuckDBPyConnection, min_valid: int = 30) -> pd.DataFrame:
    """Every annual factor's real computed value, one column per factor,
    for whichever factors have at least `min_valid` non-missing/non-infinite
    companies in the current snapshot. Shared by cluster_heatmap and the
    taming-funnel skip check.
    """
    import factorzoo.factors as f
    from factorzoo.factors.panel import get_annual_factor_panel

    panel = get_annual_factor_panel(con, pd.Timestamp.today())
    values: dict[str, pd.Series] = {}
    for spec in f.FACTOR_REGISTRY.values():
        if spec.panel != "annual":
            continue
        try:
            v = spec.compute(panel).replace([np.inf, -np.inf], np.nan)
        except Exception:  # noqa: BLE001, S112 -- a factor that can't compute just isn't included
            continue
        if v.notna().sum() >= min_valid:
            values[spec.name] = v
    return pd.DataFrame(values)


def cluster_heatmap(con: duckdb.DuckDBPyConnection) -> FigureResult:
    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.spatial.distance import squareform

    from factorzoo.taming.dimension_reduction import correlation_clusters

    name = "cluster_heatmap"
    factor_df = _real_annual_factor_values(con)
    n_companies = con.execute("SELECT COUNT(*) FROM entity_crosswalk").fetchone()[0]
    corr = factor_df.corr(min_periods=30).fillna(0.0)

    dist_arr = (1 - corr.abs()).to_numpy(copy=True)
    np.fill_diagonal(dist_arr, 0.0)
    dist_arr = (dist_arr + dist_arr.T) / 2
    link = linkage(squareform(dist_arr, checks=False), method="average")
    order = dendrogram(link, no_plot=True)["leaves"]
    ordered = [corr.columns[i] for i in order]
    corr_ordered = corr.loc[ordered, ordered]

    clusters = correlation_clusters(factor_df, distance_threshold=0.3)
    cluster_seq = [int(clusters[name_]) for name_ in ordered]

    fig = go.Figure(
        data=go.Heatmap(
            z=corr_ordered.to_numpy(), x=ordered, y=ordered,
            colorscale=[[0, ACCENT_RED], [0.5, "#15181f"], [1, ACCENT_BLUE]], zmid=0, zmin=-1, zmax=1,
            colorbar={"title": {"text": "correlation", "font": {"color": TEXT}}, "tickfont": {"color": TEXT}},
            hovertemplate="%{x} vs %{y}: %{z:.2f}<extra></extra>",
        )
    )

    # Outline multi-member clusters only (singletons add visual noise
    # without adding information -- there's nothing to outline).
    i = 0
    n = len(cluster_seq)
    while i < n:
        j = i
        while j + 1 < n and cluster_seq[j + 1] == cluster_seq[i]:
            j += 1
        if j > i:
            fig.add_shape(
                type="rect", x0=i - 0.5, x1=j + 0.5, y0=i - 0.5, y1=j + 0.5,
                line={"color": ACCENT_YELLOW, "width": 2.5}, fillcolor="rgba(0,0,0,0)",
            )
        i = j + 1

    fig.update_layout(
        xaxis={"tickfont": {"size": 8.5, "color": TEXT}, "tickangle": 90, "showgrid": False},
        yaxis={"tickfont": {"size": 8.5, "color": TEXT}, "showgrid": False, "autorange": "reversed"},
    )
    _dark_layout(
        fig,
        f"Cross-sectional correlation among {len(ordered)} real fundamentals-only factors",
        f"{n_companies} real companies (EDGAR point-in-time snapshot); rows/columns ordered by hierarchical "
        "clustering; yellow boxes outline multi-factor clusters (taming/dimension_reduction.py::"
        "correlation_clusters). This is cross-sectional VALUE correlation, not yet the long-short RETURN "
        "correlation Section 7 uses -- that needs price data",
        snapshot_footer_text(con),
        height=980,
    )
    png, html = _save_figure(fig, name)
    return FigureResult(name=name, status="generated", output_png=png, output_html=html)


# --- (i) universe_by_year: real data, S&P 1500 overlap adapted -------------


def universe_by_year(con: duckdb.DuckDBPyConnection) -> FigureResult:
    """Spec asked for 'universe size by year plus overlap with S&P 1500'.
    This replica's universe source is point-in-time S&P 500 membership, not
    1500 (flagged to the user before building) -- there is no real S&P 1500
    table to overlap against. Adapted to the nearest real, honest analog:
    overlap with tickers this build has actually pulled real EDGAR facts
    for, i.e. how much of the point-in-time universe the data pipeline
    currently covers, which is the same underlying question (coverage, not
    just universe size) the original ask was getting at.
    """
    name = "universe_by_year"
    df = con.execute(
        """
        SELECT formation_date, ticker
        FROM universe_membership
        WHERE formation_date = (
            SELECT MAX(formation_date) FROM universe_membership u2
            WHERE date_trunc('year', u2.formation_date) = date_trunc('year', universe_membership.formation_date)
        )
        """
    ).fetchdf()
    df["year"] = df["formation_date"].dt.year
    pulled = set(con.execute("SELECT DISTINCT ticker FROM entity_crosswalk").fetchdf()["ticker"])

    by_year = df.groupby("year")["ticker"].apply(set).reset_index(name="tickers")
    by_year["universe_size"] = by_year["tickers"].apply(len)
    by_year["edgar_covered"] = by_year["tickers"].apply(lambda s: len(s & pulled))
    latest_year = int(by_year["year"].max())
    latest_covered = int(by_year.loc[by_year["year"] == latest_year, "edgar_covered"].iloc[0])
    latest_size = int(by_year.loc[by_year["year"] == latest_year, "universe_size"].iloc[0])

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=by_year["year"], y=by_year["universe_size"], name="point-in-time universe size",
            marker_color=ACCENT_BLUE, opacity=0.85,
        )
    )
    fig.add_trace(
        go.Bar(
            x=by_year["year"], y=by_year["edgar_covered"], name="covered by real EDGAR pull",
            marker_color=ACCENT_YELLOW,
        )
    )
    fig.update_layout(
        barmode="overlay",
        xaxis={"title": "year", "gridcolor": GRID, "color": TEXT, "dtick": 5},
        yaxis={"title": "number of tickers", "gridcolor": GRID, "color": TEXT},
        legend={"x": 0.02, "y": 0.98, "bgcolor": "rgba(0,0,0,0)", "font": {"color": TEXT}},
    )
    _dark_layout(
        fig,
        "Point-in-time universe size by year, and how much of it this build actually pulled",
        f"Adapted from the brief's 'S&P 1500 overlap' (this replica's universe source is S&P 500, not 1500 -- "
        f"flagged before building): as of {latest_year}, {latest_covered}/{latest_size} tickers have real "
        "EDGAR facts pulled. Year = last point-in-time monthly snapshot in that year",
        snapshot_footer_text(con),
        height=620,
    )
    png, html = _save_figure(fig, name)
    return FigureResult(name=name, status="generated", output_png=png, output_html=html)


# --- (d)/(e)/(f)/(h): real once `factorzoo run-backtest` has been run ------

_BACKTEST_MISSING_INPUT = (
    "taming_report is empty -- prices_daily may have data, but the backtest that turns it into real "
    "factor returns hasn't been run yet"
)
_BACKTEST_PRODUCED_BY = "`factorzoo run-backtest`, after `factorzoo pull-prices` has real price data"


def ff_validation(con: duckdb.DuckDBPyConnection) -> FigureResult:
    from plotly.subplots import make_subplots

    name = "ff_validation"
    if not backtest_results_present(con):
        return FigureResult(name=name, status="skipped", missing_input=_BACKTEST_MISSING_INPUT, produced_by=_BACKTEST_PRODUCED_BY)

    replica = con.execute("SELECT * FROM ff3_replica ORDER BY formation_date").fetchdf()
    validation = con.execute("SELECT * FROM french_validation").fetchdf()
    if replica.empty or validation.empty:
        return FigureResult(
            name=name, status="skipped",
            missing_input="ff3_replica/french_validation are empty -- the backtest ran but produced no FF3 series "
            "(too few formation dates, or ff5_daily wasn't pulled)",
            produced_by=_BACKTEST_PRODUCED_BY,
        )

    ff5 = con.execute("SELECT * FROM ff5_daily").fetchdf()
    ff5["date"] = pd.to_datetime(ff5["date"])
    ff5_monthly = ff5.set_index("date")[["mkt_rf", "smb", "hml"]].resample("ME").apply(lambda s: (1 + s).prod() - 1)
    ff5_monthly.index = ff5_monthly.index.to_period("M").to_timestamp()
    ff5_monthly = ff5_monthly.rename(columns={"mkt_rf": "mkt"})

    # ff3_replica's formation_date is month-END (e.g. 2017-01-31, the
    # universe's own convention); ff5_monthly's index above is
    # month-START (Period.to_timestamp()'s default). Joining the two
    # as-is compares disjoint calendar days and silently drops every row
    # -- confirmed live: both sides individually had zero NaNs, but the
    # join produced zero overlapping rows. Normalize both to the same
    # month-start representation (run_full_backtest's own French
    # validation already does this; this figure function had not).
    replica = replica.set_index("formation_date")
    replica.index = pd.to_datetime(replica.index).to_period("M").to_timestamp()
    pairs = [("mkt", "market"), ("smb", "SMB"), ("hml", "HML")]
    fig = make_subplots(
        rows=3, cols=2, column_widths=[0.62, 0.38], horizontal_spacing=0.09, vertical_spacing=0.08,
        subplot_titles=[t for p in pairs for t in (f"{p[1]}: replica vs. Ken French", f"{p[1]} scatter")],
    )
    corr_by_factor = validation.set_index("factor")["correlation"].to_dict()
    for i, (col, label) in enumerate(pairs):
        row = i + 1
        joined = pd.DataFrame({"replica": replica[col], "french": ff5_monthly[col]}).dropna()
        fig.add_trace(
            go.Scatter(x=joined.index, y=joined["replica"], mode="lines", name=f"replica {label}",
                       line={"color": ACCENT_BLUE, "width": 1.5}, showlegend=(i == 0)),
            row=row, col=1,
        )
        fig.add_trace(
            go.Scatter(x=joined.index, y=joined["french"], mode="lines", name=f"Ken French {label}",
                       line={"color": ACCENT_YELLOW, "width": 1.5}, showlegend=(i == 0)),
            row=row, col=1,
        )
        fig.add_trace(
            go.Scatter(x=joined["french"], y=joined["replica"], mode="markers", showlegend=False,
                       marker={"color": ACCENT_BLUE, "size": 5, "opacity": 0.6}),
            row=row, col=2,
        )
        lims = [joined.to_numpy().min(), joined.to_numpy().max()]
        fig.add_trace(
            go.Scatter(x=lims, y=lims, mode="lines", line={"color": MUTED, "width": 1, "dash": "dot"}, showlegend=False),
            row=row, col=2,
        )

    fig.update_xaxes(gridcolor=GRID, color=TEXT)
    fig.update_yaxes(gridcolor=GRID, color=TEXT)
    for ann in fig.layout.annotations:
        ann.font = {"color": TEXT, "size": 12}

    corr_bits = ", ".join(f"{lab} r={corr_by_factor.get(lab2, float('nan')):.2f}" for (col, lab), lab2 in zip(pairs, ["market", "SMB", "HML"]))
    all_pass = all(validation["clears_0_9_gate"])
    headline_color = ACCENT_GREEN if all_pass else ACCENT_RED
    fig.update_layout(
        template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=BG, font={"family": FONT_FAMILY, "color": TEXT, "size": 13},
        width=1200, height=1000,
        title={
            "text": f"<span style='color:{headline_color}'>Replica vs. Ken French: {corr_bits}</span>"
            f"<br><span style='font-size:13px;color={MUTED}'>Real monthly series, "
            f"{int(validation['n_months'].max())} months overlap. Green if the 0.9 correlation gate is cleared "
            "on every factor, red otherwise.</span>",
            "x": 0.5, "xanchor": "center",
        },
        margin={"l": 70, "r": 40, "t": 120, "b": 90},
        annotations=list(fig.layout.annotations) + [
            {"text": snapshot_footer_text(con), "showarrow": False, "x": 0.5, "y": -0.045, "xref": "paper", "yref": "paper",
                 "font": {"size": 10.5, "color": MUTED}},
        ],
    )
    png, html = _save_figure(fig, name, scale=1.5)
    return FigureResult(name=name, status="generated", output_png=png, output_html=html)


def cumulative_ls_returns(con: duckdb.DuckDBPyConnection) -> FigureResult:
    name = "cumulative_ls_returns"
    if not backtest_results_present(con):
        return FigureResult(name=name, status="skipped", missing_input=_BACKTEST_MISSING_INPUT, produced_by=_BACKTEST_PRODUCED_BY)

    taming = con.execute("SELECT * FROM taming_report").fetchdf()
    returns = con.execute("SELECT * FROM factor_returns_monthly ORDER BY formation_date").fetchdf()
    ff3 = con.execute("SELECT * FROM ff3_replica ORDER BY formation_date").fetchdf()
    if returns.empty:
        return FigureResult(name=name, status="skipped", missing_input="factor_returns_monthly is empty", produced_by=_BACKTEST_PRODUCED_BY)

    survivors = taming[taming["bh_discovery"] & taming["one_per_cluster"]].sort_values("dsr", ascending=False)
    top = survivors.head(5)["factor"].tolist()
    if not top:
        top = taming.reindex(taming["tstat"].abs().sort_values(ascending=False).index).head(5)["factor"].tolist()

    wide = returns.pivot(index="formation_date", columns="factor", values="vw_return")
    fig = go.Figure()
    colors = [ACCENT_BLUE, ACCENT_YELLOW, ACCENT_GREEN, ACCENT_RED, "#b36ae2"]
    for i, factor in enumerate(top):
        cum = (1 + wide[factor].fillna(0)).cumprod()
        fig.add_trace(go.Scatter(x=cum.index, y=cum, mode="lines", name=factor, line={"color": colors[i % len(colors)], "width": 2}))

    if not ff3.empty:
        mkt_cum = (1 + ff3.set_index("formation_date")["mkt"].fillna(0)).cumprod()
        fig.add_trace(go.Scatter(x=mkt_cum.index, y=mkt_cum, mode="lines", name="market (VW, for reference)",
                                  line={"color": MUTED, "width": 2, "dash": "dot"}))

    fig.update_layout(
        xaxis={"title": "formation date", "gridcolor": GRID, "color": TEXT},
        yaxis={"title": "cumulative growth of $1 (log scale)", "type": "log", "gridcolor": GRID, "color": TEXT},
        legend={"bgcolor": "rgba(0,0,0,0)", "font": {"color": TEXT}},
    )
    _dark_layout(
        fig,
        f"Cumulative value-weighted long-short returns: the {len(top)} top surviving factors",
        "Real monthly VW long-short returns, compounded. 'Surviving' = BH-discovery AND one-per-correlation-"
        "cluster, ranked by Deflated Sharpe Ratio; falls back to |t|-ranked if nothing clears BH. Market line is "
        "the replica's own real value-weighted return, for reference.",
        snapshot_footer_text(con),
        height=700,
    )
    png, html = _save_figure(fig, name)
    return FigureResult(name=name, status="generated", output_png=png, output_html=html)


def taming_funnel(con: duckdb.DuckDBPyConnection) -> FigureResult:
    name = "taming_funnel"
    if not backtest_results_present(con):
        return FigureResult(name=name, status="skipped", missing_input=_BACKTEST_MISSING_INPUT, produced_by=_BACKTEST_PRODUCED_BY)

    taming = con.execute("SELECT * FROM taming_report").fetchdf()
    # A funnel is strictly cumulative -- each stage must be a SUBSET of
    # the one before it, never computed independently. taming_report's
    # own one_per_cluster/lasso_survives columns are computed over the
    # FULL factor set (useful for the DSR-surface overlay and other
    # figures, which want "is this factor redundant" independent of
    # whether it individually cleared a t-hurdle), so this function
    # re-intersects them stage-by-stage rather than reading the column
    # sums directly -- confirmed live: real data clears BH/BY discovery
    # for ZERO factors, and reading one_per_cluster's raw sum (30) would
    # have shown the bar going back UP after BH's bar hit zero, which
    # looks like a funnel chart bug rather than the real (and honestly
    # the whole point of this project) finding that almost nothing
    # survives honest multiple-testing correction.
    survivors = set(taming["factor"])
    stages = [("all factors tested", len(survivors))]
    for label, col in [
        ("|t| > 2", "passes_t2"), ("|t| > 3 (HLZ)", "passes_t3_hlz"),
        ("BH discovery", "bh_discovery"), ("BY discovery", "by_discovery"),
        ("one per cluster", "one_per_cluster"), ("LASSO-survives spanning test", "lasso_survives"),
    ]:
        survivors &= set(taming.loc[taming[col].astype(bool), "factor"])
        stages.append((label, len(survivors)))
    fig = go.Figure(
        go.Funnel(
            y=[s for s, _ in stages], x=[n for _, n in stages],
            textposition="inside", textinfo="value+percent initial",
            marker={"color": [BLUE_YELLOW[0][1], "#1d3a63", "#2a5590", ACCENT_BLUE, "#5aa0eb", ACCENT_YELLOW, ACCENT_GREEN]},
            connector={"line": {"color": GRID, "width": 1}},
        )
    )
    fig.update_layout(yaxis={"color": TEXT}, xaxis={"color": TEXT})
    final_n = stages[-1][1]
    _dark_layout(
        fig,
        "How many of the 48 factors survive each stage of statistical taming",
        f"Real t-statistics and Deflated Sharpe Ratios from {len(taming)} factors with computed return series "
        f"(alpha=5% for t-hurdles; q=10% for BH/BY); each stage is a strict subset of the one before it. "
        f"{final_n} of {len(taming)} survive every stage -- real multiple-testing correction is exactly this "
        "harsh on a real cross-section.",
        snapshot_footer_text(con),
        height=620,
    )
    png, html = _save_figure(fig, name)
    return FigureResult(name=name, status="generated", output_png=png, output_html=html)


def ipca_r2(con: duckdb.DuckDBPyConnection) -> FigureResult:
    name = "ipca_r2"
    if not backtest_results_present(con):
        return FigureResult(name=name, status="skipped", missing_input=_BACKTEST_MISSING_INPUT, produced_by=_BACKTEST_PRODUCED_BY)

    r2 = con.execute("SELECT * FROM ipca_r2 ORDER BY n_factors").fetchdf()
    if r2.empty:
        manifest = _latest_backtest_manifest(con)
        return FigureResult(
            name=name, status="skipped",
            missing_input=f"ipca_r2 table is empty -- the IPCA fit itself failed: {manifest.get('ipca_error', 'unknown reason')}",
            produced_by=_BACKTEST_PRODUCED_BY,
        )

    manifest = _latest_backtest_manifest(con)
    pca_r2 = manifest.get("pca_r2") or {}
    ff5_r2 = manifest.get("ff5_r2")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=r2["n_factors"], y=r2["total_r2"], mode="lines+markers", name="IPCA total R²",
                              line={"color": ACCENT_BLUE, "width": 2}, marker={"size": 8}))
    fig.add_trace(go.Scatter(x=r2["n_factors"], y=r2["predictive_r2"], mode="lines+markers", name="IPCA predictive R²",
                              line={"color": ACCENT_YELLOW, "width": 2}, marker={"size": 8}))
    if pca_r2:
        pca_x = sorted(int(k) for k in pca_r2)
        pca_y = [pca_r2[str(k)] if str(k) in pca_r2 else pca_r2.get(k) for k in pca_x]
        fig.add_trace(go.Scatter(x=pca_x, y=pca_y, mode="lines+markers", name="naive PCA-on-returns",
                                  line={"color": MUTED, "width": 2, "dash": "dash"}, marker={"size": 7}))
    if ff5_r2 is not None:
        fig.add_hline(y=ff5_r2, line={"color": ACCENT_GREEN, "width": 1.5, "dash": "dot"},
                       annotation_text=f"FF5 regression R²={ff5_r2:.2f}", annotation_font={"color": ACCENT_GREEN})

    fig.update_layout(
        xaxis={"title": "K (number of latent factors)", "gridcolor": GRID, "color": TEXT, "dtick": 1},
        yaxis={"title": "R²", "gridcolor": GRID, "color": TEXT},
        legend={"bgcolor": "rgba(0,0,0,0)", "font": {"color": TEXT}},
    )
    _dark_layout(
        fig,
        "IPCA: total vs. predictive R² by number of latent factors",
        "Real fit on one-per-correlation-cluster characteristics (not the full 35-factor registry -- "
        "confirmed live that feeding IPCA every near-duplicate characteristic makes its ALS estimator hit a "
        "singular matrix). No real q-factor benchmark is wired into this project, so that reference line is "
        "omitted rather than estimated.",
        snapshot_footer_text(con),
        height=650,
    )
    png, html = _save_figure(fig, name)
    return FigureResult(name=name, status="generated", output_png=png, output_html=html)


# --- orchestrator ------------------------------------------------------

FIGURE_FUNCTIONS = (
    hurdle_surface,
    dsr_surface,
    dsr_surface_rotating_gif,
    cluster_heatmap,
    universe_by_year,
    ff_validation,
    cumulative_ls_returns,
    taming_funnel,
    ipca_r2,
)


def run_all(con: duckdb.DuckDBPyConnection) -> list[FigureResult]:
    results = []
    for fn in FIGURE_FUNCTIONS:
        try:
            results.append(fn(con))
        except Exception as exc:  # noqa: BLE001 -- one broken figure must not take the rest down
            results.append(
                FigureResult(name=fn.__name__, status="skipped", missing_input=f"raised {type(exc).__name__}: {exc}")
            )
    return results


def print_status_table(results: list[FigureResult]) -> None:
    print()
    print(f"{'figure':<26} {'status':<10} missing input")
    print("-" * 100)
    for r in results:
        print(f"{r.name:<26} {r.status:<10} {r.missing_input or ''}")
    print()
    n_gen = sum(1 for r in results if r.status == "generated")
    n_skip = sum(1 for r in results if r.status == "skipped")
    print(f"{n_gen} generated, {n_skip} skipped (of {len(results)})")


def main(db_path: Path | None = None) -> int:
    con = pit_store.init_db(db_path)
    try:
        ok, reason = core_tables_present(con)
        if not ok:
            print(f"ERROR: {reason}")
            return 1
        results = run_all(con)
        print_status_table(results)
        return 0
    finally:
        con.close()


if __name__ == "__main__":
    import sys

    sys.exit(main())
