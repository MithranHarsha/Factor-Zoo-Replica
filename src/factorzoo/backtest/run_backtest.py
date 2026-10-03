"""Runs the real backtest: point-in-time factor panels across every
monthly formation date in range, decile-sorted long-short return series
per factor, the FF3 replication against Ken French, and the full taming
pipeline (t-hurdles, BH/BY, DSR, correlation clustering, LASSO spanning,
IPCA). Every other module this orchestrates (portfolios/, eval/, taming/)
was built and unit-tested against synthetic fixtures with known answers;
this module is the first real-data wiring of that machinery -- it adds no
new statistical methodology of its own, only the point-in-time assembly
of cross-sections and the sequencing of calls into those existing,
already-tested functions.

Run with: uv run factorzoo run-backtest
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import factorzoo.factors as f
from factorzoo.eval.benchmarks import ff3_series, validate_against_french
from factorzoo.eval.performance import newey_west_tstat, sharpe_ratio
from factorzoo.factors.momentum import attach_industry
from factorzoo.factors.panel import (
    build_monthly_price_panel,
    get_annual_factor_panel,
    get_monthly_market_return,
)
from factorzoo.portfolios.returns import factor_return_series
from factorzoo.taming.dimension_reduction import (
    correlation_clusters,
    lasso_spanning_test,
    select_cluster_representatives,
)
from factorzoo.taming.multiple_testing import deflated_sharpe_ratio, multiple_testing_report

FINANCIAL_SIC_RANGE = (6000, 6999)


@dataclass
class BacktestResult:
    dates: list
    n_companies_median: int
    factor_vw_returns: pd.DataFrame  # date-indexed, one column per factor
    factor_ew_returns: pd.DataFrame
    factor_tstats: pd.Series  # Newey-West t-stat on VW returns, per factor
    taming_report: pd.DataFrame  # multiple_testing_report() output, plus DSR/cluster/LASSO columns
    cluster_labels: pd.Series
    lasso_survivors: list
    ff3_replica: pd.DataFrame
    french_validation: pd.DataFrame | None
    ipca_r2: pd.DataFrame | None
    ipca_error: str | None = None
    pca_r2: dict = field(default_factory=dict)
    ff5_r2: float | None = None


def _attach_monthly_factor_values(con, monthly_panel: pd.DataFrame) -> pd.DataFrame:
    """Every monthly-panel factor's value, computed ONCE across the whole
    continuous (entity, month) series -- these are rolling computations,
    so they must run on the full time series per entity, not be
    recomputed from scratch at every formation date.
    """
    mkt = get_monthly_market_return(con)
    mkt["month"] = pd.to_datetime(mkt["month"])
    panel = monthly_panel.merge(mkt, on="month", how="left")
    panel = attach_industry(con, panel)

    specs = [s for s in f.FACTOR_REGISTRY.values() if s.panel == "monthly"]
    for spec in specs:
        try:
            panel[spec.name] = spec.compute(panel)
        except Exception:  # noqa: BLE001, S112 -- a factor that can't compute on this data just isn't included
            continue
    return panel


def build_cross_sections(
    con, dates: list[pd.Timestamp], min_companies: int = 20
) -> dict[pd.Timestamp, pd.DataFrame]:
    """One real cross-section per formation date: every computable
    factor's value (annual ones point-in-time-correct via
    get_annual_factor_panel, monthly ones from the precomputed rolling
    panel), market_cap, sic, and the FORWARD monthly return used to score
    a decile sort formed at this date. Dates with fewer than
    `min_companies` scored names are dropped -- too thin a cross-section
    to sort into deciles meaningfully.
    """
    # One ticker per entity_id, same tie-break as factors/panel.py's
    # attach_market_cap -- without this, a multi-class-share company
    # (confirmed live: GOOG/GOOGL, FOX/FOXA, both one CIK under two
    # tickers) gets merged into TWO cross-sectional rows sharing identical
    # fundamentals, double-counting that one company in every decile sort.
    crosswalk = con.execute(
        "SELECT ticker, entity_id, sic FROM entity_crosswalk "
        "QUALIFY ROW_NUMBER() OVER (PARTITION BY entity_id ORDER BY ticker) = 1"
    ).fetchdf()
    monthly_panel = build_monthly_price_panel(con)
    monthly_panel["month"] = pd.to_datetime(monthly_panel["month"])
    monthly_panel = _attach_monthly_factor_values(con, monthly_panel)
    monthly_by_ticker_month = monthly_panel.set_index(["entity_id_hint", "month"])

    annual_specs = [s for s in f.FACTOR_REGISTRY.values() if s.panel == "annual"]
    monthly_specs = [s for s in f.FACTOR_REGISTRY.values() if s.panel == "monthly"]

    panel_by_date: dict[pd.Timestamp, pd.DataFrame] = {}
    for d in dates:
        annual_panel = get_annual_factor_panel(con, d)
        if annual_panel.empty:
            continue
        annual_panel = annual_panel.merge(crosswalk, on="entity_id", how="left")
        for spec in annual_specs:
            try:
                annual_panel[spec.name] = spec.compute(annual_panel)
            except Exception:  # noqa: BLE001, S112 -- a factor that can't compute for this date just isn't included
                continue

        cur_period = d.to_period("M")
        next_period = cur_period + 1
        cur_month = cur_period.to_timestamp()
        next_month = next_period.to_timestamp()

        for spec in monthly_specs:
            values = []
            for ticker in annual_panel["ticker"]:
                try:
                    values.append(monthly_by_ticker_month.loc[(ticker, cur_month), spec.name])
                except KeyError:
                    values.append(np.nan)
            annual_panel[spec.name] = values

        fwd_returns = []
        for ticker in annual_panel["ticker"]:
            try:
                fwd_returns.append(monthly_by_ticker_month.loc[(ticker, next_month), "monthly_return"])
            except KeyError:
                fwd_returns.append(np.nan)
        annual_panel["forward_return"] = fwd_returns

        n_scored = annual_panel["forward_return"].notna().sum()
        if n_scored < min_companies:
            continue
        panel_by_date[d] = annual_panel

    return panel_by_date


def _exclude_financials(panel_by_date: dict, exclude: bool) -> dict:
    if not exclude:
        return panel_by_date
    lo, hi = FINANCIAL_SIC_RANGE
    out = {}
    for d, df in panel_by_date.items():
        sic_numeric = pd.to_numeric(df["sic"], errors="coerce")
        out[d] = df.loc[~sic_numeric.between(lo, hi)]
    return out


def run_full_backtest(con, dates: list[pd.Timestamp], min_companies: int = 20) -> BacktestResult:
    panel_by_date = build_cross_sections(con, dates, min_companies=min_companies)
    if not panel_by_date:
        raise RuntimeError(
            f"No formation date in range had >= {min_companies} scored companies -- "
            "nothing to backtest. Check price/EDGAR coverage for this date range."
        )

    # build_cross_sections keys each cross-section by its FORMATION date,
    # but every value in it (forward_return, and therefore every return
    # series/correlation computed below) is the NEXT month's realized
    # return -- confirmed live: correlating the replica's market return
    # against Ken French's under the formation-date label gave r=-0.09,
    # visually inconsistent with how closely the two lines actually
    # tracked on the chart; shifting the label forward one calendar month
    # (to "the month this return was EARNED in", which is also Ken
    # French's own convention) brought it to r=0.92. Relabeling once here
    # means every return-series consumer below (factor returns, FF3,
    # IPCA's time index) is correct and mutually consistent, rather than
    # patching each call site separately.
    panel_by_date = {(d.to_period("M") + 1).to_timestamp(): df for d, df in panel_by_date.items()}

    specs = sorted(f.FACTOR_REGISTRY.values(), key=lambda s: (s.category, s.name))
    vw_cols, ew_cols = {}, {}
    for spec in specs:
        pbd = _exclude_financials(panel_by_date, spec.exclude_financials)
        try:
            series = factor_return_series(pbd, spec.name, spec.direction)
        except Exception:  # noqa: BLE001, S112 -- one bad factor must not kill the whole backtest
            continue
        if series.empty:
            continue
        vw_cols[spec.name] = series.set_index("formation_date")["vw_long_short"]
        ew_cols[spec.name] = series.set_index("formation_date")["ew_long_short"]

    vw_returns = pd.DataFrame(vw_cols).sort_index()
    ew_returns = pd.DataFrame(ew_cols).sort_index()

    tstats = pd.Series({name: newey_west_tstat(vw_returns[name]) for name in vw_returns.columns}).dropna()
    taming = multiple_testing_report(tstats)

    clean_returns = vw_returns[tstats.index].dropna(axis=1, thresh=max(1, int(0.5 * len(vw_returns))))
    cluster_labels = (
        correlation_clusters(clean_returns) if clean_returns.shape[1] >= 2 else pd.Series(dtype="int64")
    )
    if len(cluster_labels):
        reps = select_cluster_representatives(cluster_labels, tstats.reindex(cluster_labels.index).abs())
    else:
        reps = list(clean_returns.columns)
    taming["one_per_cluster"] = taming.index.isin(reps)
    taming["cluster_label"] = cluster_labels.reindex(taming.index)

    # LASSO spanning test: for each one-per-cluster representative, does it
    # add explanatory power (a significant alpha) beyond the OTHER
    # representatives already in the set? lasso_spanning_test takes one
    # candidate factor at a time against a spanning set, not a whole
    # matrix at once (see its docstring) -- so this is reps-many calls,
    # each testing one representative against all the others.
    lasso_survivors: list[str] = []
    rep_returns = clean_returns[[c for c in reps if c in clean_returns.columns]].dropna(how="all")
    for candidate_name in rep_returns.columns:
        spanning = rep_returns.drop(columns=[candidate_name])
        if spanning.shape[1] == 0:
            continue
        try:
            result = lasso_spanning_test(rep_returns[candidate_name], spanning)
        except Exception:  # noqa: BLE001, S112 -- LASSO is a nice-to-have stage, not load-bearing for the funnel
            continue
        if result.get("adds_value"):
            lasso_survivors.append(candidate_name)
    taming["lasso_survives"] = taming.index.isin(lasso_survivors)

    sharpes = vw_returns[tstats.index].apply(sharpe_ratio) * np.sqrt(12)  # annualized
    n_obs_by_factor = vw_returns[tstats.index].notna().sum()
    dsr = {}
    for name in tstats.index:
        n_obs = int(n_obs_by_factor.get(name, 0))
        if n_obs < 3:
            continue
        dsr[name] = deflated_sharpe_ratio(sharpes.dropna(), sharpes.get(name, np.nan), n_obs=n_obs)
    taming["annualized_sharpe"] = sharpes
    taming["dsr"] = pd.Series(dsr)

    ff3_pbd = _exclude_financials(panel_by_date, False)
    ff3_replica = ff3_series(ff3_pbd, size_col="market_cap", value_col="book_to_market", return_col="forward_return")
    ff3_replica = ff3_replica.set_index("date") if not ff3_replica.empty else ff3_replica

    french_validation = None
    try:
        ff5 = con.execute("SELECT * FROM ff5_daily").fetchdf()
        if not ff5.empty and not ff3_replica.empty:
            ff5["date"] = pd.to_datetime(ff5["date"])
            ff5_monthly = (
                ff5.set_index("date")[["mkt_rf", "smb", "hml", "rf"]]
                .resample("ME")
                .apply(lambda s: (1 + s).prod() - 1)
            )
            ff5_monthly.index = ff5_monthly.index.to_period("M").to_timestamp()
            replica_for_validation = ff3_replica.copy()
            replica_for_validation.index = pd.to_datetime(replica_for_validation.index).to_period("M").to_timestamp()
            french_validation = validate_against_french(replica_for_validation.reset_index(), ff5_monthly.reset_index())
    except Exception:  # noqa: BLE001 -- FF3 validation is reported as unavailable, not fatal
        french_validation = None

    ipca_r2 = None
    ipca_error = None
    pca_r2: dict = {}
    ff5_r2_value = None
    try:
        annual_names = [s.name for s in specs if s.panel == "annual"]
        ipca_characteristics = [r for r in reps if r in annual_names]
        ipca_r2, pca_r2, ff5_r2_value = _run_ipca(con, panel_by_date, ipca_characteristics, vw_returns)
    except Exception as exc:  # noqa: BLE001 -- IPCA is one figure's input, not load-bearing for the rest
        ipca_error = f"{type(exc).__name__}: {exc}"

    return BacktestResult(
        dates=list(panel_by_date.keys()),
        n_companies_median=int(np.median([len(df) for df in panel_by_date.values()])),
        factor_vw_returns=vw_returns,
        factor_ew_returns=ew_returns,
        factor_tstats=tstats,
        taming_report=taming,
        cluster_labels=cluster_labels,
        lasso_survivors=lasso_survivors,
        ff3_replica=ff3_replica,
        french_validation=french_validation,
        ipca_r2=ipca_r2,
        ipca_error=ipca_error,
        pca_r2=pca_r2,
        ff5_r2=ff5_r2_value,
    )


def _run_ipca(con, panel_by_date: dict, characteristic_names: list[str], vw_returns: pd.DataFrame):
    """`characteristic_names` should be one representative per
    correlation cluster (see run_full_backtest's `reps`), not every
    annual factor -- confirmed live: feeding IPCA's ALS estimator the
    full 35-factor registry (many of them near-duplicates, e.g. three
    different accruals measures) made its inner least-squares step hit a
    singular matrix on every K. A decorrelated characteristic set is
    both what the method expects and what makes the fit numerically
    stable.
    """
    from factorzoo.taming.ipca_model import (
        fit_ipca_for_k_range,
        pca_benchmark_r2,
        standardize_characteristics,
    )

    rows = []
    for d, df in panel_by_date.items():
        sub = df[["ticker", "forward_return", *[c for c in characteristic_names if c in df.columns]]].copy()
        sub = sub.dropna(subset=["forward_return"])
        sub["time"] = d
        rows.append(sub)
    long_panel = pd.concat(rows, ignore_index=True)
    char_cols = [c for c in characteristic_names if c in long_panel.columns and long_panel[c].notna().sum() > 30]
    if len(char_cols) < 2 or long_panel.empty:
        raise RuntimeError("not enough populated characteristics for an IPCA fit")

    # standardize_characteristics rank-standardizes WITHIN each time
    # period (needs 'time' as an actual column, not index level), so a
    # missing value for one company at one date must not silently drop
    # that (entity, time) row -- it's filled with that period's
    # cross-sectional median first, same convention as the panel's own
    # "unknown gets excluded, not a crash" rule elsewhere in this project.
    long_panel = long_panel.rename(columns={"ticker": "entity"})
    long_panel[char_cols] = long_panel.groupby("time")[char_cols].transform(lambda s: s.fillna(s.median()))
    long_panel = long_panel.dropna(subset=char_cols, how="any").reset_index(drop=True)

    characteristics = standardize_characteristics(long_panel[[*char_cols, "time"]], time_col="time")
    characteristics.index = pd.MultiIndex.from_frame(long_panel[["entity", "time"]])
    returns = long_panel.set_index(["entity", "time"])["forward_return"]

    r2_df = fit_ipca_for_k_range(characteristics, returns, k_range=range(1, 7))

    returns_wide = returns.reset_index().pivot(index="time", columns="entity", values="forward_return")
    pca_r2 = {k: pca_benchmark_r2(returns_wide, k) for k in range(1, 4)}

    ff5_r2 = None
    try:
        ff5 = con.execute("SELECT * FROM ff5_daily").fetchdf()
        if not ff5.empty:
            ff5["date"] = pd.to_datetime(ff5["date"])
            ff5_monthly = ff5.set_index("date")[["mkt_rf", "smb", "hml", "rmw", "cma"]].resample("ME").apply(lambda s: (1 + s).prod() - 1)
            ff5_monthly.index = ff5_monthly.index.to_period("M").to_timestamp()
            mkt_series = vw_returns.mean(axis=1).dropna()
            joined = ff5_monthly.reindex(mkt_series.index).dropna()
            if len(joined) > 10:
                import statsmodels.api as sm

                X = sm.add_constant(joined.to_numpy())
                y = mkt_series.reindex(joined.index).to_numpy()
                model = sm.OLS(y, X).fit()
                ff5_r2 = float(model.rsquared)
    except Exception:  # noqa: BLE001 -- the FF5 reference line is optional context, not required
        ff5_r2 = None

    return r2_df, pca_r2, ff5_r2


def persist_backtest_result(con, result: BacktestResult) -> None:
    """Writes every table a figure might read from, then one manifest row
    with the scalars that don't fit a table (pca_r2 per K, the FF5
    reference R^2, lasso survivor names, IPCA's error if it didn't fit) --
    see pit_store.py's schema comment: these tables are fully REPLACED,
    not appended, since they're this run's result, not history.
    """
    from factorzoo.data import pit_store

    pit_store.write_factor_returns(con, result.factor_vw_returns, result.factor_ew_returns)
    pit_store.write_taming_report(con, result.taming_report)
    if not result.ff3_replica.empty:
        pit_store.write_ff3_replica(con, result.ff3_replica.reset_index())
    if result.french_validation is not None and not result.french_validation.empty:
        pit_store.write_french_validation(con, result.french_validation)
    if result.ipca_r2 is not None and not result.ipca_r2.empty:
        pit_store.write_ipca_r2(con, result.ipca_r2)

    pit_store.record_manifest(
        con,
        "run-backtest",
        {
            "n_dates": len(result.dates),
            "date_min": str(min(result.dates)) if result.dates else None,
            "date_max": str(max(result.dates)) if result.dates else None,
            "n_companies_median": result.n_companies_median,
            "n_factors_with_returns": int(result.factor_vw_returns.shape[1]),
            "lasso_survivors": result.lasso_survivors,
            "ipca_error": result.ipca_error,
            "pca_r2": result.pca_r2,
            "ff5_r2": result.ff5_r2,
        },
    )
