"""Replicates the market, SMB, and HML factors from this project's own
point-in-time data and compares them to Kenneth French's published series
-- Phase 2's actual gate (build guide Section 10): "FF3 correlation above
0.9 vs. Ken French." Below 0.9, Section 8 says to treat it as a
methodology bug in the replica, not a finding.

SMB and HML use the real Fama-French 2x3 double sort (size median split x
book-to-market 30/70 split, both computed from the large-cap breakpoint
subset per Section 6), not a simplified single-factor decile spread --
the double sort is what the published series itself is built from, so
it's the only construction with a real chance of clearing the 0.9 bar.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _large_cap_subset(df: pd.DataFrame, market_cap_col: str, pct: float = 0.25) -> pd.DataFrame:
    n = max(1, int(np.ceil(len(df) * pct)))
    return df.nlargest(n, market_cap_col)


def fama_french_2x3_breakpoints(
    df: pd.DataFrame, size_col: str, value_col: str, market_cap_col: str = "market_cap"
) -> dict:
    """Size median and B/M 30th/70th percentile breakpoints, computed
    from the large-cap subset (build guide Section 6's NYSE-breakpoint
    proxy), applied to the full universe. Returned separately from the
    bucket assignment so tests can check the breakpoint VALUES directly.
    """
    valid = df.dropna(subset=[size_col, value_col, market_cap_col])
    if valid.empty:
        return {"size_median": np.nan, "bm_30": np.nan, "bm_70": np.nan}
    large = _large_cap_subset(valid, market_cap_col)
    return {
        "size_median": float(np.median(large[size_col])),
        "bm_30": float(np.quantile(large[value_col], 0.30)),
        "bm_70": float(np.quantile(large[value_col], 0.70)),
    }


def assign_2x3_portfolio(
    df: pd.DataFrame, size_col: str, value_col: str, market_cap_col: str = "market_cap"
) -> pd.Series:
    """Returns a Series of portfolio labels: S/L, S/M, S/H, B/L, B/M, B/H
    (Small/Big x Low/Medium/High book-to-market), NaN for rows missing an
    input."""
    bp = fama_french_2x3_breakpoints(df, size_col, value_col, market_cap_col)
    valid = df[size_col].notna() & df[value_col].notna() & df[market_cap_col].notna()

    size_label = np.where(df[size_col] <= bp["size_median"], "S", "B")
    bm_label = np.select(
        [df[value_col] <= bp["bm_30"], df[value_col] >= bp["bm_70"]],
        ["L", "H"],
        default="M",
    )
    label = pd.Series(
        [f"{s}/{b}" if ok else np.nan for s, b, ok in zip(size_label, bm_label, valid)], index=df.index
    )
    return label


def compute_smb_hml(
    df: pd.DataFrame, size_col: str, value_col: str, return_col: str, market_cap_col: str = "market_cap"
) -> dict:
    """One cross-section in, {smb, hml, portfolio_returns} out. Each of
    the six portfolios is value-weighted (Fama-French convention)."""
    from factorzoo.portfolios.weighting import value_weights

    labels = assign_2x3_portfolio(df, size_col, value_col, market_cap_col)
    work = df.assign(_label=labels)
    work["_vw"] = value_weights(work, "_label", market_cap_col)

    eligible = work["_label"].notna() & work[return_col].notna() & (work["_vw"] > 0)
    sub = work.loc[eligible].copy()
    if sub.empty:
        return {"smb": np.nan, "hml": np.nan, "portfolio_returns": {}}

    sub["_w_renorm"] = sub.groupby("_label")["_vw"].transform(lambda w: w / w.sum())
    sub["_contrib"] = sub["_w_renorm"] * sub[return_col]
    port_ret = sub.groupby("_label")["_contrib"].sum().to_dict()

    small = [port_ret.get(f"S/{g}") for g in "LMH"]
    big = [port_ret.get(f"B/{g}") for g in "LMH"]
    smb = np.nanmean(small) - np.nanmean(big) if any(v is not None for v in small + big) else np.nan

    high = [port_ret.get("S/H"), port_ret.get("B/H")]
    low = [port_ret.get("S/L"), port_ret.get("B/L")]
    hml = np.nanmean(high) - np.nanmean(low) if any(v is not None for v in high + low) else np.nan

    return {"smb": smb, "hml": hml, "portfolio_returns": port_ret}


def compute_market_return(df: pd.DataFrame, return_col: str, market_cap_col: str = "market_cap") -> float:
    """Value-weighted return of the whole eligible universe -- the
    replica's own market factor, compared against Ken French's Mkt-RF
    (plus RF) in validate_against_french below."""
    eligible = df[return_col].notna() & df[market_cap_col].notna() & (df[market_cap_col] > 0)
    sub = df.loc[eligible]
    if sub.empty or sub[market_cap_col].sum() == 0:
        return np.nan
    weights = sub[market_cap_col] / sub[market_cap_col].sum()
    return float((weights * sub[return_col]).sum())


def ff3_series(panel_by_date: dict, size_col: str, value_col: str, return_col: str) -> pd.DataFrame:
    """Runs compute_market_return + compute_smb_hml across every
    formation date in panel_by_date (date -> cross-sectional DataFrame).
    This is the time series validate_against_french correlates against
    the real Ken French data.
    """
    rows = []
    for date, df in sorted(panel_by_date.items()):
        if df.empty:
            continue
        mkt = compute_market_return(df, return_col)
        smb_hml = compute_smb_hml(df, size_col, value_col, return_col)
        rows.append({"date": date, "mkt": mkt, "smb": smb_hml["smb"], "hml": smb_hml["hml"]})
    return pd.DataFrame(rows)


def validate_against_french(replica: pd.DataFrame, french: pd.DataFrame, min_overlap_months: int = 6) -> pd.DataFrame:
    """Correlates the replica's mkt/smb/hml monthly series against Ken
    French's published ones over their date overlap. `french` must have
    columns [date, mkt_rf, smb, hml, rf] (as riskfree.fetch_ff5_daily
    produces at daily frequency -- resample to monthly before calling
    this, since the replica is monthly by construction).

    Returns one row per factor with the correlation and whether it clears
    the 0.9 gate (build guide Section 10); below that, Section 8 says
    treat it as a methodology bug, not a finding -- this function reports
    the number, the call on what it means is the caller's.
    """
    pairs = [("mkt", "mkt_rf", "market"), ("smb", "smb", "SMB"), ("hml", "hml", "HML")]
    # replica and french both use "smb"/"hml" as column names, so a bare
    # merge() would silently suffix both to smb_x/smb_y and every lookup
    # below would KeyError (or worse, silently compare the wrong pair, if
    # pandas' suffixing ever changed) -- rename to disjoint names first so
    # there is no collision regardless of what the caller names things.
    replica_renamed = replica.rename(columns={rc: f"_replica_{rc}" for rc, _, _ in pairs if rc in replica.columns})
    french_renamed = french.rename(columns={fc: f"_french_{fc}" for _, fc, _ in pairs if fc in french.columns})
    merged = replica_renamed.merge(french_renamed, on="date", how="inner")

    rows = []
    for replica_col, french_col, label in pairs:
        r_col, f_col = f"_replica_{replica_col}", f"_french_{french_col}"
        sub = merged[[r_col, f_col]].dropna()
        if len(sub) < min_overlap_months:
            rows.append({"factor": label, "n_months": len(sub), "correlation": np.nan, "clears_0.9_gate": False})
            continue
        corr = sub[r_col].corr(sub[f_col])
        rows.append({"factor": label, "n_months": len(sub), "correlation": corr, "clears_0.9_gate": bool(corr >= 0.9)})
    return pd.DataFrame(rows)
