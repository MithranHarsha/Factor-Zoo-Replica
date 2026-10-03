"""SEC EDGAR client: company lookup, submissions (for SIC code), Company Facts
(XBRL fundamentals), and the point-in-time availability rule.

Every function that hits the network is paired with a pure function that
does the same parsing/logic off already-downloaded JSON, so the point-in-time
rule and tag extraction can be unit tested without a network call.

Endpoints used (all free, no API key):
  - https://www.sec.gov/files/company_tickers.json           (ticker <-> CIK)
  - https://data.sec.gov/submissions/CIK##########.json       (SIC code, filing history)
  - https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json  (XBRL facts)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
import pandas as pd

from factorzoo.config import CACHE_DIR, EDGAR_COMPANYFACTS_DIR, EDGAR_SUBMISSIONS_DIR, Settings

log = logging.getLogger(__name__)

COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik10}.json"
COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik10}.json"

# The starter factor library needs these us-gaap
# tags. Kept here, not buried in a notebook, because which tags a factor
# needs is itself part of the spec.
CORE_USGAAP_TAGS: tuple[str, ...] = (
    "Assets",
    "AssetsCurrent",
    "Liabilities",
    "LiabilitiesCurrent",
    "StockholdersEquity",
    "CommonStockSharesOutstanding",
    "Revenues",
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "CostOfGoodsAndServicesSold",
    "GrossProfit",
    "OperatingIncomeLoss",
    "NetIncomeLoss",
    "IncomeTaxExpenseBenefit",
    "InterestExpense",
    "CashAndCashEquivalentsAtCarryingValue",
    "PropertyPlantAndEquipmentNet",
    "Goodwill",
    "IntangibleAssetsNetExcludingGoodwill",
    "LongTermDebtNoncurrent",
    "LongTermDebtCurrent",
    "ShortTermBorrowings",
    "DepreciationDepletionAndAmortization",
    "ResearchAndDevelopmentExpense",
    "SellingGeneralAndAdministrativeExpense",
    "NetCashProvidedByUsedInOperatingActivities",
    "PaymentsOfDividends",
    "PaymentsForRepurchaseOfCommonStock",
    "PaymentsToAcquirePropertyPlantAndEquipment",
    "InventoryNet",
    "AccountsReceivableNetCurrent",
    "AccountsPayableCurrent",
)

# dei-taxonomy tags (cover-page facts, not financial-statement facts).
CORE_DEI_TAGS: tuple[str, ...] = ("EntityCommonStockSharesOutstanding",)

# Where a concept changed tag names (e.g. the 2018 revenue-recognition
# standard), try the newer tag first, then fall back -- see Section 4's
# "fallback tag order" fix.
TAG_FALLBACKS: dict[str, tuple[str, ...]] = {
    "Revenues": ("Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax"),
    "SharesOutstanding": ("EntityCommonStockSharesOutstanding", "CommonStockSharesOutstanding"),
}


class RateLimiter:
    """A simple sleep-based limiter. SEC's fair-use guidance is roughly
    10 requests/second; default here is more conservative on purpose."""

    def __init__(self, requests_per_second: float = 5.0) -> None:
        self.min_interval = 1.0 / max(requests_per_second, 0.1)
        self._last_call: float = 0.0

    def wait(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_call
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last_call = time.monotonic()


@dataclass
class EdgarClient:
    settings: Settings
    rate_limiter: RateLimiter | None = None
    client: httpx.Client | None = None

    def __post_init__(self) -> None:
        if self.rate_limiter is None:
            self.rate_limiter = RateLimiter(self.settings.sec_requests_per_second)
        if self.client is None:
            self.client = httpx.Client(
                headers={"User-Agent": self.settings.sec_user_agent, "Accept-Encoding": "gzip, deflate"},
                timeout=30.0,
            )

    def _get_json(self, url: str) -> dict:
        self.rate_limiter.wait()
        resp = self.client.get(url)
        resp.raise_for_status()
        return resp.json()

    # -- ticker <-> CIK -----------------------------------------------------

    def fetch_company_tickers(self, cache_path: Path | None = None, max_age_days: int = 7) -> pd.DataFrame:
        """Returns DataFrame[cik, ticker, title]. This is SEC's list of
        *current* filers only -- a CIK lookup aid, never the primary
        universe source (that would reintroduce survivorship bias by
        construction)."""
        cache_path = cache_path or (CACHE_DIR / "company_tickers.json")
        data = self._load_or_fetch(cache_path, COMPANY_TICKERS_URL, max_age_days)
        rows = [
            {"cik": int(v["cik_str"]), "ticker": v["ticker"], "title": v["title"]}
            for v in data.values()
        ]
        return pd.DataFrame(rows)

    def _load_or_fetch(self, cache_path: Path, url: str, max_age_days: int) -> dict:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if cache_path.exists():
            age = time.time() - cache_path.stat().st_mtime
            if age < max_age_days * 86400:
                return json.loads(cache_path.read_text())
        data = self._get_json(url)
        cache_path.write_text(json.dumps(data))
        return data

    # -- submissions (SIC code etc.) ----------------------------------------

    def fetch_submissions(self, cik: int, force: bool = False) -> dict:
        cik10 = f"{cik:010d}"
        cache_path = EDGAR_SUBMISSIONS_DIR / f"{cik10}.json"
        if cache_path.exists() and not force:
            return json.loads(cache_path.read_text())
        data = self._get_json(SUBMISSIONS_URL.format(cik10=cik10))
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(data))
        return data

    # -- company facts (XBRL) ------------------------------------------------

    def fetch_company_facts(self, cik: int, force: bool = False) -> dict | None:
        cik10 = f"{cik:010d}"
        cache_path = EDGAR_COMPANYFACTS_DIR / f"{cik10}.json"
        if cache_path.exists() and not force:
            return json.loads(cache_path.read_text())
        try:
            data = self._get_json(COMPANYFACTS_URL.format(cik10=cik10))
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                log.info("No company facts for CIK %s (likely no XBRL filings)", cik10)
                return None
            raise
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(data))
        return data

    def close(self) -> None:
        if self.client is not None:
            self.client.close()


def extract_sic(submissions_json: dict) -> dict:
    """Pure function: pull industry classification out of a submissions
    payload. Needed for the industry-momentum factor (Section 5), which has
    no SIC field in XBRL facts at all."""
    return {
        "cik": submissions_json.get("cik"),
        "sic": submissions_json.get("sic"),
        "sic_description": submissions_json.get("sicDescription"),
        "entity_name": submissions_json.get("name"),
    }


def extract_xbrl_facts(
    facts_json: dict,
    tags: tuple[str, ...] = CORE_USGAAP_TAGS,
    dei_tags: tuple[str, ...] = CORE_DEI_TAGS,
) -> pd.DataFrame:
    """Pure function: flatten a Company Facts payload into a tidy long
    DataFrame, one row per reported fact. No network call, no filesystem --
    this is what makes the point-in-time rule unit-testable.

    Columns: cik, entity_id, tag, taxonomy, unit, val, period_start,
    period_end, fy, fp, form, filed, accn.
    """
    cik = facts_json.get("cik")
    entity_id = f"CIK{int(cik):010d}" if cik is not None else None
    rows: list[dict] = []

    def _collect(taxonomy: str, tag_set: tuple[str, ...]) -> None:
        taxonomy_facts = facts_json.get("facts", {}).get(taxonomy, {})
        for tag in tag_set:
            tag_data = taxonomy_facts.get(tag)
            if not tag_data:
                continue
            for unit, observations in tag_data.get("units", {}).items():
                for obs in observations:
                    rows.append(
                        {
                            "cik": cik,
                            "entity_id": entity_id,
                            "taxonomy": taxonomy,
                            "tag": tag,
                            "unit": unit,
                            "val": obs.get("val"),
                            "period_start": obs.get("start"),
                            "period_end": obs.get("end"),
                            "fy": obs.get("fy"),
                            "fp": obs.get("fp"),
                            "form": obs.get("form"),
                            "filed": obs.get("filed"),
                            "accn": obs.get("accn"),
                            "frame": obs.get("frame"),
                        }
                    )

    _collect("us-gaap", tags)
    _collect("dei", dei_tags)

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["period_end"] = pd.to_datetime(df["period_end"], errors="coerce")
    df["period_start"] = pd.to_datetime(df["period_start"], errors="coerce")
    df["filed"] = pd.to_datetime(df["filed"], errors="coerce")
    df["available_date"] = df.apply(
        lambda r: compute_available_date(r["form"], r["period_end"], r["filed"]), axis=1
    )
    return df


# --- the point-in-time rule --------------------------------------------

ANNUAL_FLOOR_DAYS = 91
QUARTERLY_FLOOR_DAYS = 45


def compute_available_date(form: str | None, period_end, filed) -> pd.Timestamp | None:
    """available_date = max(filed, period_end + floor_days)

    floor_days is 91 for annual (10-K family) facts and 45 for quarterly
    (10-Q family) facts. The floor exists because `filed` is occasionally
    anomalously early (data errors, voluntary early disclosure); it
    reproduces the conservative convention Fama-French use in their own
    construction. A fact is never usable before BOTH its filing date and
    this floor have passed.

    Pure function, no I/O -- this is what Section 8's leakage tests exercise
    directly without needing a live EDGAR pull.
    """
    if pd.isna(filed) or pd.isna(period_end):
        return pd.NaT
    form = (form or "").upper()
    is_annual = form.startswith("10-K")
    floor_days = ANNUAL_FLOOR_DAYS if is_annual else QUARTERLY_FLOOR_DAYS
    floor_date = pd.Timestamp(period_end) + pd.Timedelta(days=floor_days)
    filed_ts = pd.Timestamp(filed)
    return max(filed_ts, floor_date)
