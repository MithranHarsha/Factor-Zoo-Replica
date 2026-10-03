"""Builds the two panels factor formulas compute against, directly off the
point-in-time store -- this is where "point-in-time correct" stops being a
promise and becomes the actual shape of the data a factor function sees.

Two panels:

- **Annual snapshot** (`build_point_in_time_snapshot`): for ONE as-of date,
  one row per company with each fundamental tag's most recently *known*
  annual value, plus the prior fiscal year's value for the same tag (also
  respecting availability) so growth/diff factors are one-line formulas.
  This is deliberately an as-of QUERY, not a static pivot of "the" annual
  figure per company-year -- a static pivot would have to pick one vintage
  per period in advance, which either leaks a later restatement backward
  or has no principled way to decide. Querying per as-of date instead means
  a factor computed for a 2020 portfolio and a factor computed for the same
  company-year recomputed in 2026 can legitimately differ (the 2026 run
  sees a restatement the 2020 run couldn't have), which is correct, not a
  bug -- see the point-in-time rule in data/pit_store.py.

- **Monthly price panel** (`build_monthly_price_panel`): from daily prices,
  one row per company per month with the return, a volatility measure, and
  the maximum daily return -- what the momentum and trading-frictions
  factors (Section 5) need.

Both are read-only queries over the DuckDB store; neither mutates it.
"""

from __future__ import annotations

import duckdb
import pandas as pd

from factorzoo.data.edgar import CORE_DEI_TAGS, CORE_USGAAP_TAGS, TAG_FALLBACKS

# Which tags are point-in-time "instant" facts (balance-sheet, as of
# period_end, period_start is null) vs "duration" facts (income/cash-flow
# statement, accumulated over period_start..period_end). This distinction
# is what keeps a quarterly duration fact from being mistaken for an
# annual one just because its period_end happens to land on a fiscal
# year-end -- the day-count filter in the SQL below is the actual annual/
# quarterly discriminator, not the tag name or form alone.
INSTANT_TAGS: frozenset[str] = frozenset(
    {
        "Assets", "AssetsCurrent", "Liabilities", "LiabilitiesCurrent", "StockholdersEquity",
        "CommonStockSharesOutstanding", "CashAndCashEquivalentsAtCarryingValue",
        "PropertyPlantAndEquipmentNet", "Goodwill", "IntangibleAssetsNetExcludingGoodwill",
        "LongTermDebtNoncurrent", "LongTermDebtCurrent", "ShortTermBorrowings", "InventoryNet",
        "AccountsReceivableNetCurrent", "AccountsPayableCurrent", "EntityCommonStockSharesOutstanding",
    }
)
DURATION_TAGS: frozenset[str] = frozenset(set(CORE_USGAAP_TAGS) | set(CORE_DEI_TAGS)) - INSTANT_TAGS

# The panel's output column name for each tag, after fallback resolution
# (e.g. Revenues and RevenueFromContractWithCustomerExcludingAssessedTax
# both land in the "revenues" column -- a fallback
# tag order, applied here rather than left to each factor to remember).
TAG_TO_COLUMN: dict[str, str] = {
    "Assets": "assets",
    "AssetsCurrent": "assets_current",
    "Liabilities": "liabilities",
    "LiabilitiesCurrent": "liabilities_current",
    "StockholdersEquity": "stockholders_equity",
    "CommonStockSharesOutstanding": "shares_outstanding_gaap",
    "EntityCommonStockSharesOutstanding": "shares_outstanding_dei",
    "Revenues": "revenues",
    "RevenueFromContractWithCustomerExcludingAssessedTax": "revenues",
    "CostOfGoodsAndServicesSold": "cost_of_goods_sold",
    "GrossProfit": "gross_profit",
    "OperatingIncomeLoss": "operating_income",
    "NetIncomeLoss": "net_income",
    "IncomeTaxExpenseBenefit": "income_tax_expense",
    "InterestExpense": "interest_expense",
    "CashAndCashEquivalentsAtCarryingValue": "cash",
    "PropertyPlantAndEquipmentNet": "ppe_net",
    "Goodwill": "goodwill",
    "IntangibleAssetsNetExcludingGoodwill": "intangibles_net",
    "LongTermDebtNoncurrent": "lt_debt_noncurrent",
    "LongTermDebtCurrent": "lt_debt_current",
    "ShortTermBorrowings": "st_borrowings",
    "DepreciationDepletionAndAmortization": "depreciation",
    "ResearchAndDevelopmentExpense": "rd_expense",
    "SellingGeneralAndAdministrativeExpense": "sga_expense",
    "NetCashProvidedByUsedInOperatingActivities": "cfo",
    "PaymentsOfDividends": "dividends_paid",
    "PaymentsForRepurchaseOfCommonStock": "buybacks",
    "PaymentsToAcquirePropertyPlantAndEquipment": "capex",
    "InventoryNet": "inventory",
    "AccountsReceivableNetCurrent": "receivables",
    "AccountsPayableCurrent": "payables",
}

ANNUAL_MIN_DAYS = 300
ANNUAL_MAX_DAYS = 400


def build_point_in_time_snapshot(
    con: duckdb.DuckDBPyConnection, as_of: str | pd.Timestamp, entity_ids: list[str] | None = None
) -> pd.DataFrame:
    """One row per entity_id, wide-pivoted: `<column>` = latest known
    annual value as of `as_of`, `<column>_prior` = the fiscal year before
    that (also only using what was knowable by `as_of`). A factor like
    asset growth is then literally `(assets / assets_prior) - 1` against
    this frame -- no groupby/shift bug class possible, because there is no
    shift: both years are already columns on the same row.
    """
    as_of = pd.Timestamp(as_of)
    entity_filter = ""
    if entity_ids:
        placeholders = ", ".join(["?"] * len(entity_ids))
        entity_filter = f"AND entity_id IN ({placeholders})"

    # Instant facts: rank by period_end desc (ties broken by the latest
    # filed vintage, so a later amendment wins once it's actually
    # available, never before).
    instant_tags = sorted(INSTANT_TAGS)
    instant_q = f"""
        WITH ranked AS (
            SELECT *, ROW_NUMBER() OVER (
                PARTITION BY entity_id, tag ORDER BY period_end DESC, filed DESC
            ) AS rn
            FROM xbrl_facts
            WHERE available_date <= ? AND tag IN ({", ".join(["?"] * len(instant_tags))}) {entity_filter}
        )
        SELECT entity_id, tag, val, period_end, 1 AS vintage_rank FROM ranked WHERE rn = 1
        UNION ALL
        SELECT entity_id, tag, val, period_end, 2 AS vintage_rank FROM ranked WHERE rn = 2
    """
    instant_params = [as_of, *instant_tags] + (entity_ids or [])

    # Duration facts: same ranking, additionally restricted to rows whose
    # period actually spans ~a year -- this is what stops a quarterly
    # figure from masquerading as the annual one.
    duration_tags = sorted(DURATION_TAGS)
    duration_q = f"""
        WITH ranked AS (
            SELECT *, ROW_NUMBER() OVER (
                PARTITION BY entity_id, tag ORDER BY period_end DESC, filed DESC
            ) AS rn
            FROM xbrl_facts
            WHERE available_date <= ?
              AND tag IN ({", ".join(["?"] * len(duration_tags))})
              AND period_start IS NOT NULL
              AND date_diff('day', period_start, period_end) BETWEEN {ANNUAL_MIN_DAYS} AND {ANNUAL_MAX_DAYS}
              {entity_filter}
        )
        SELECT entity_id, tag, val, period_end, 1 AS vintage_rank FROM ranked WHERE rn = 1
        UNION ALL
        SELECT entity_id, tag, val, period_end, 2 AS vintage_rank FROM ranked WHERE rn = 2
    """
    duration_params = [as_of, *duration_tags] + (entity_ids or [])

    instant_df = con.execute(instant_q, instant_params).fetchdf()
    duration_df = con.execute(duration_q, duration_params).fetchdf()
    long_df = pd.concat([instant_df, duration_df], ignore_index=True)
    if long_df.empty:
        return pd.DataFrame(columns=["entity_id"])

    long_df["column"] = long_df["tag"].map(TAG_TO_COLUMN).fillna(long_df["tag"])
    long_df["column_full"] = long_df.apply(
        lambda r: r["column"] if r["vintage_rank"] == 1 else f"{r['column']}_prior", axis=1
    )

    # Fallback resolution: where two tags map to the same column (e.g.
    # Revenues vs the post-2018 contract-revenue tag), prefer whichever
    # appears first in TAG_FALLBACKS for that logical column, per company.
    long_df = _resolve_tag_fallbacks(long_df)

    wide = long_df.pivot_table(index="entity_id", columns="column_full", values="val", aggfunc="first")
    wide = wide.reset_index()
    return wide


def _resolve_tag_fallbacks(long_df: pd.DataFrame) -> pd.DataFrame:
    """Where multiple raw tags resolve to the same output column (the
    Revenues / RevenueFromContractWithCustomerExcludingAssessedTax case),
    keep one row per (entity_id, column_full) using TAG_FALLBACKS' stated
    priority order rather than an arbitrary pandas tie-break.
    """
    tag_priority: dict[str, int] = {}
    for fallback_tags in TAG_FALLBACKS.values():
        for i, tag in enumerate(fallback_tags):
            tag_priority[tag] = min(i, tag_priority.get(tag, i))
    long_df = long_df.copy()
    long_df["_priority"] = long_df["tag"].map(tag_priority).fillna(0)
    long_df = long_df.sort_values("_priority")
    return long_df.drop_duplicates(subset=["entity_id", "column_full"], keep="first")


def attach_market_cap(
    con: duckdb.DuckDBPyConnection, snapshot: pd.DataFrame, as_of: str | pd.Timestamp, price_source: str | None = None
) -> pd.DataFrame:
    """Joins market cap (= shares outstanding x latest known price on or
    before `as_of`) onto an annual snapshot, via entity_crosswalk's
    ticker <-> entity_id mapping. Needed by every value-category ratio
    (book-to-market, EBITDA/EV, R&D-to-market, ...) -- those are the
    factors that genuinely need both a fundamentals panel and a price
    panel, not an artifact of this module's split.
    """
    if snapshot.empty:
        out = snapshot.copy()
        out["market_cap"] = pd.Series(dtype="float64")
        return out

    as_of = pd.Timestamp(as_of)
    source_filter = "AND p.source = ?" if price_source else ""
    params: list = [as_of]
    if price_source:
        params.append(price_source)

    # entity_crosswalk can have more than one ticker per entity_id --
    # confirmed live with real data: Alphabet (GOOG/GOOGL) and Fox
    # (FOX/FOXA) each file as one CIK under multiple share classes. A
    # plain join fans that out to multiple price rows per entity_id,
    # which downstream silently double- (or quadruple-, once this merges
    # back into a multi-ticker caller) counts that one company's
    # fundamentals in any cross-sectional sort. QUALIFY picks exactly one
    # ticker per entity_id, deterministically (alphabetically first --
    # for both real cases found so far that's also the unsuffixed, more
    # commonly quoted class).
    price_q = f"""
        SELECT c.entity_id, p.close AS price_at_formation
        FROM entity_crosswalk c
        JOIN prices_daily p ON p.entity_id_hint = c.ticker
        WHERE p.date = (
            SELECT MAX(p2.date) FROM prices_daily p2
            WHERE p2.entity_id_hint = c.ticker AND p2.date <= ? {source_filter.replace('p.', 'p2.')}
        )
        {source_filter}
        QUALIFY ROW_NUMBER() OVER (PARTITION BY c.entity_id ORDER BY c.ticker) = 1
    """
    prices = con.execute(price_q, params + ([price_source] if price_source else [])).fetchdf()

    out = snapshot.merge(prices, on="entity_id", how="left")
    shares = resolve_shares_outstanding(out)
    out["market_cap"] = shares * out["price_at_formation"]
    return out


# Every column a factor formula might reference, current + prior-year
# variants, plus the two price-derived columns. Pivoting only creates a
# column when at least one company in the snapshot actually reported that
# tag; without this, a factor referencing a tag nobody in a small pilot
# happened to report would raise KeyError instead of producing NaN for
# every row (the correct behavior -- "unknown" is not "crash").
_BASE_COLUMNS = sorted(set(TAG_TO_COLUMN.values()))
EXPECTED_ANNUAL_COLUMNS: tuple[str, ...] = tuple(
    ["entity_id", "market_cap", "price_at_formation"]
    + _BASE_COLUMNS
    + [f"{c}_prior" for c in _BASE_COLUMNS]
)


def ensure_columns(df: pd.DataFrame, columns: tuple[str, ...] = EXPECTED_ANNUAL_COLUMNS) -> pd.DataFrame:
    df = df.copy()
    for c in columns:
        if c not in df.columns:
            df[c] = pd.NA
    return df


def get_annual_factor_panel(
    con: duckdb.DuckDBPyConnection,
    as_of: str | pd.Timestamp,
    entity_ids: list[str] | None = None,
    price_source: str | None = None,
) -> pd.DataFrame:
    """The single entry point factor code (and Phase 3 portfolio
    formation) should call: snapshot + market cap + guaranteed columns,
    in one place, so every factor's compute() can assume every column it
    might need exists (possibly all-NaN) rather than each one defending
    against a KeyError individually.
    """
    snapshot = build_point_in_time_snapshot(con, as_of, entity_ids=entity_ids)
    with_cap = attach_market_cap(con, snapshot, as_of, price_source=price_source)
    return ensure_columns(with_cap)


def resolve_shares_outstanding(snapshot: pd.DataFrame) -> pd.Series:
    """Fallback order: dei:EntityCommonStockSharesOutstanding
    first, then us-gaap:CommonStockSharesOutstanding."""
    dei = snapshot.get("shares_outstanding_dei")
    gaap = snapshot.get("shares_outstanding_gaap")
    if dei is None and gaap is None:
        return pd.Series(index=snapshot.index, dtype="float64")
    if dei is None:
        return gaap
    if gaap is None:
        return dei
    return dei.fillna(gaap)


# --- monthly price panel ----------------------------------------------------


def build_monthly_price_panel(
    con: duckdb.DuckDBPyConnection, source: str | None = None, entity_id_hints: list[str] | None = None
) -> pd.DataFrame:
    """One row per (entity_id_hint, month): the month-end close, the
    monthly return, within-month daily volatility, and the maximum daily
    return in the month -- the inputs Section 5's momentum and
    trading-frictions factors need. entity_id_hint is the ticker as pulled
    (see prices_daily schema); joining to a stable entity_id happens via
    entity_crosswalk where needed, kept separate so this function has no
    dependency on EDGAR data at all.
    """
    source_filter = "AND source = ?" if source else ""
    hint_filter = ""
    params: list = []
    if source:
        params.append(source)
    if entity_id_hints:
        placeholders = ", ".join(["?"] * len(entity_id_hints))
        hint_filter = f"AND entity_id_hint IN ({placeholders})"
        params.extend(entity_id_hints)

    q = f"""
        WITH daily AS (
            SELECT
                entity_id_hint,
                date,
                close,
                volume,
                close / NULLIF(LAG(close) OVER (PARTITION BY entity_id_hint ORDER BY date), 0) - 1 AS daily_return,
                date_trunc('month', date) AS month
            FROM prices_daily
            WHERE 1=1 {source_filter} {hint_filter}
        )
        SELECT
            entity_id_hint,
            month,
            LAST(close ORDER BY date) AS close_month_end,
            AVG(volume) AS volume_avg,
            STDDEV_SAMP(daily_return) AS volatility_daily_std,
            MAX(daily_return) AS max_daily_return,
            EXP(SUM(LN(1 + daily_return))) - 1 AS monthly_return
        FROM daily
        WHERE daily_return IS NOT NULL OR date = (SELECT MIN(date) FROM prices_daily)
        GROUP BY entity_id_hint, month
        ORDER BY entity_id_hint, month
    """
    return con.execute(q, params).fetchdf()


def get_monthly_market_return(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Monthly market return, compounded from the daily Fama-French
    Mkt-RF + RF series (validation benchmark) --
    used by the market-beta factor (trading_frictions.py) rather than
    building a second, redundant market-return series from this project's
    own (currently much smaller) price panel.
    """
    q = """
        SELECT
            date_trunc('month', date) AS month,
            EXP(SUM(LN(1 + mkt_rf + rf))) - 1 AS mkt_return
        FROM ff5_daily
        GROUP BY date_trunc('month', date)
        ORDER BY month
    """
    return con.execute(q).fetchdf()
