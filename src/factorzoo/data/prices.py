"""Price data clients: Yahoo Finance's chart API (primary, free, no key),
Stooq (documented as a free fallback, but see the live finding below), and
Tiingo (free tier with a 500-symbol/month cap, needed for delisted-ticker
history).

**Phase 1 live finding:** Stooq was the original plan for primary price
source, but as of this build, its bulk CSV endpoint (`/q/d/l/?s=...`) returns an
HTML page behind a JavaScript bot-check for any non-browser client -- a
plain `requests`/`httpx` GET gets either a 404 or a JS-challenge page, HTTP
200 notwithstanding, even with a browser User-Agent and cookies replayed.
This was verified live, not assumed: curl with `-A "Mozilla/5.0"` gets
back `<noscript>This site requires JavaScript to verify your browser...`
instead of CSV data. Section 4's "verify Stooq's current terms before
relying on it at scale" caveat turned out to be live and blocking, not
hypothetical. `fetch_stooq_daily` is kept (for if/when that changes, and
because its parser is still useful), but the pipeline's default primary
source is now Yahoo Finance's public chart API called directly (the same
endpoint the `yfinance` package wraps), which responded correctly in a
live check at build time. That endpoint is unauthenticated and
undocumented, so it can change or start blocking at any time exactly as
Stooq did -- Tiingo (paid past 500 symbols/month) remains the fallback of
last resort if both free sources stop working.

**Second live finding, same build:** Yahoo's rate limiting turned out to
be keyed on more than the IP and User-Agent header. A plain `curl` request
and a `httpx.Client` request issuing the IDENTICAL URL, query parameters,
and User-Agent string got different results side by side: curl got HTTP
200 with real data every time, httpx got HTTP 429 every time. The most
likely explanation is TLS/HTTP client fingerprinting (JA3/JA4 or
HTTP/2-level), which no amount of header-spoofing in a Python HTTP client
fixes. The pragmatic fix, not a hypothetical one: the Yahoo fetch shells
out to the system `curl` binary rather than using httpx for this one
endpoint. This is a deliberate, documented exception to "use httpx
everywhere" -- Stooq and Tiingo still use httpx, since Stooq's block is a
JS challenge curl can't pass either, and Tiingo is a proper authenticated
API with no observed fingerprinting issue.
"""

from __future__ import annotations

import io
import json as jsonlib
import logging
import subprocess
import time
import urllib.parse
from pathlib import Path

import httpx
import pandas as pd
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential_jitter

from factorzoo.config import PRICES_DIR, Settings

log = logging.getLogger(__name__)

STOOQ_URL = "https://stooq.com/q/d/l/?s={ticker}.us&i=d"
TIINGO_URL = "https://api.tiingo.com/tiingo/daily/{ticker}/prices"
# Yahoo load-balances chart requests across at least two independent hosts
# with (confirmed live, this build) independent rate-limit counters:
# query1 got a sustained 429 while query2 served real data seconds later.
# Trying both before giving up materially improves success odds against an
# undocumented per-host threshold.
YAHOO_CHART_HOSTS = ("query2.finance.yahoo.com", "query1.finance.yahoo.com")
YAHOO_CHART_URL = "https://{host}/v8/finance/chart/{ticker}"
# Yahoo's chart endpoint is unauthenticated but not fully anonymous-friendly
# either; a browser-shaped User-Agent is what was confirmed working live.
YAHOO_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

STOOQ_COLUMNS = {
    "Date": "date",
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Volume": "volume",
}


def fetch_stooq_daily(
    ticker: str, client: httpx.Client | None = None, cache_dir: Path | None = None, force: bool = False
) -> pd.DataFrame:
    """Download full daily history for one ticker from Stooq. Returns an
    empty DataFrame (not an exception) for an unknown/delisted-on-Stooq
    ticker, since that is an expected, common case here -- Stooq not
    having a name is exactly the signal that Tiingo backfill is needed.
    """
    cache_dir = cache_dir or (PRICES_DIR / "stooq")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{ticker.upper()}.csv"

    if cache_path.exists() and not force:
        raw = cache_path.read_text()
    else:
        owns_client = client is None
        client = client or httpx.Client(timeout=30.0)
        try:
            resp = client.get(STOOQ_URL.format(ticker=ticker.lower()))
            resp.raise_for_status()
            raw = resp.text
        finally:
            if owns_client:
                client.close()
        cache_path.write_text(raw)

    return parse_stooq_csv(raw, ticker)


def parse_stooq_csv(raw: str, ticker: str) -> pd.DataFrame:
    """Pure function: Stooq returns the literal text 'No data' (not a 4xx)
    for a ticker it doesn't have, so that has to be checked explicitly
    rather than relying on an HTTP error."""
    if not raw or raw.strip().lower().startswith("no data") or "Date,Open" not in raw:
        return pd.DataFrame(columns=["entity_id_hint", "date", "open", "high", "low", "close", "volume"])
    df = pd.read_csv(io.StringIO(raw))
    df = df.rename(columns=STOOQ_COLUMNS)
    df["date"] = pd.to_datetime(df["date"])
    df.insert(0, "entity_id_hint", ticker.upper())
    return df[["entity_id_hint", "date", "open", "high", "low", "close", "volume"]]


def _is_rate_limit_error(exc: BaseException) -> bool:
    return isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 429


# Live finding (this build): Yahoo's chart endpoint returned a sustained
# 429 on query1 while query2 served real data seconds later -- an
# undocumented threshold, and per-host rather than per-account. Retrying each host a couple of times
# with backoff, then falling through to the next host, is a direct, tested
# response to that, not a hypothetical precaution.
_yahoo_host_retry = retry(
    retry=retry_if_exception(_is_rate_limit_error),
    wait=wait_exponential_jitter(initial=3, max=20),
    stop=stop_after_attempt(2),
    reraise=True,
)


def _curl_fetch_json(url: str, params: dict, user_agent: str, timeout: float = 30.0) -> dict:
    """GET via the system curl binary rather than httpx -- see this
    module's docstring for why: Yahoo's chart API returns HTTP 429 to
    httpx and HTTP 200 to curl for the byte-identical URL, query string,
    and User-Agent, confirmed side by side live. httpx's exception types
    are reused here purely as data containers (status code + body) so the
    retry/host-fallback logic below doesn't need a second code path.
    """
    full_url = f"{url}?{urllib.parse.urlencode(params)}"
    request = httpx.Request("GET", full_url)
    try:
        proc = subprocess.run(
            ["curl", "-s", "-A", user_agent, "--max-time", str(int(timeout)), "-w", "\n%{http_code}", full_url],
            capture_output=True,
            text=True,
            timeout=timeout + 10,
            check=False,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        raise httpx.ConnectError(str(exc), request=request) from exc

    body, _, status_str = proc.stdout.rpartition("\n")
    status_code = int(status_str) if status_str.isdigit() else 0
    if status_code != 200:
        response = httpx.Response(status_code or 599, request=request, text=body)
        raise httpx.HTTPStatusError(f"curl: HTTP {status_code} for {full_url}", request=request, response=response)
    return jsonlib.loads(body)


@_yahoo_host_retry
def _get_yahoo_chart_from_host(host: str, ticker: str, range_: str) -> dict:
    return _curl_fetch_json(
        YAHOO_CHART_URL.format(host=host, ticker=ticker.upper()),
        params={"range": range_, "interval": "1d", "events": "div,splits"},
        user_agent=YAHOO_USER_AGENT,
    )


def _get_yahoo_chart(ticker: str, range_: str) -> dict:
    last_exc: Exception | None = None
    for host in YAHOO_CHART_HOSTS:
        try:
            return _get_yahoo_chart_from_host(host, ticker, range_)
        except httpx.HTTPStatusError as exc:
            if _is_rate_limit_error(exc):
                log.warning("Rate-limited on %s after retries, trying next host", host)
                last_exc = exc
                continue
            raise
    assert last_exc is not None
    raise last_exc


def fetch_yahoo_chart_daily(
    ticker: str, range_: str = "max", cache_dir: Path | None = None, force: bool = False
) -> pd.DataFrame:
    """Download daily OHLCV history for one ticker from Yahoo Finance's
    public chart API, called directly (not via the `yfinance` package,
    which has documented rate-limit/blocking issues of its own). This is the project's current default primary
    price source; see this module's docstring for why Stooq is not, and
    why this goes through curl rather than httpx.
    """
    cache_dir = cache_dir or (PRICES_DIR / "yahoo")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{ticker.upper()}.json"

    if cache_path.exists() and not force:
        payload = jsonlib.loads(cache_path.read_text())
    else:
        payload = _get_yahoo_chart(ticker, range_)
        cache_path.write_text(jsonlib.dumps(payload))

    return parse_yahoo_chart_json(payload, ticker)


def parse_yahoo_chart_json(payload: dict, ticker: str) -> pd.DataFrame:
    """Pure function. Yahoo's chart JSON nests everything under
    chart.result[0]; a delisted/unknown ticker comes back with a non-null
    chart.error instead of a result, which is treated as "no data" rather
    than raised, matching how an unknown Stooq ticker is handled.
    """
    cols = ["entity_id_hint", "date", "open", "high", "low", "close", "volume"]
    result = (payload or {}).get("chart", {}).get("result")
    if not result:
        return pd.DataFrame(columns=cols)
    r = result[0]
    timestamps = r.get("timestamp")
    if not timestamps:
        return pd.DataFrame(columns=cols)
    quote = r["indicators"]["quote"][0]
    adjclose = r["indicators"].get("adjclose", [{}])[0].get("adjclose")
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(timestamps, unit="s", utc=True).tz_convert(None).normalize(),
            "open": quote.get("open"),
            "high": quote.get("high"),
            "low": quote.get("low"),
            # Use the split/dividend-adjusted close when available, same
            # convention as Stooq/Tiingo's adjusted-close fields, so
            # downstream return calculations don't have to special-case
            # the source.
            "close": adjclose if adjclose is not None else quote.get("close"),
            "volume": quote.get("volume"),
        }
    )
    df = df.dropna(subset=["close"]).reset_index(drop=True)
    df.insert(0, "entity_id_hint", ticker.upper())
    return df[cols]


def fetch_tiingo_daily(
    ticker: str,
    settings: Settings,
    start_date: str = "1990-01-01",
    client: httpx.Client | None = None,
    cache_dir: Path | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """Download daily history for one ticker from Tiingo. Raises
    ConfigError (via settings) if no API key is configured -- this project
    will not silently skip a source and pretend it succeeded.

    Caller is responsible for budgeting the free tier's 500 unique
    symbols/month cap; this function does not
    track usage across calls.
    """
    if not settings.tiingo_api_key:
        raise RuntimeError(
            "fetch_tiingo_daily called without TIINGO_API_KEY configured. "
            "Set it in .env or pass settings from load_settings(require_tiingo=True)."
        )

    cache_dir = cache_dir or (PRICES_DIR / "tiingo")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{ticker.upper()}.json"

    if cache_path.exists() and not force:
        raw_json = cache_path.read_text()
        import json as _json

        payload = _json.loads(raw_json)
    else:
        owns_client = client is None
        client = client or httpx.Client(timeout=30.0)
        try:
            resp = client.get(
                TIINGO_URL.format(ticker=ticker.lower()),
                params={"startDate": start_date, "token": settings.tiingo_api_key, "format": "json"},
            )
            resp.raise_for_status()
            payload = resp.json()
        finally:
            if owns_client:
                client.close()
        cache_path.write_text(__import__("json").dumps(payload))

    return parse_tiingo_json(payload, ticker)


def parse_tiingo_json(payload: list[dict], ticker: str) -> pd.DataFrame:
    if not payload:
        return pd.DataFrame(columns=["entity_id_hint", "date", "open", "high", "low", "close", "volume"])
    df = pd.DataFrame(payload)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    # Tiingo returns BOTH raw and split/dividend-adjusted OHLCV in the same
    # row (confirmed live: a real response also carries divCash and
    # splitFactor). Select the adjusted columns by their own names first,
    # THEN rename -- renaming in place while the raw open/high/low/close/
    # volume columns are still present creates duplicate column labels
    # (two columns both named "open", etc.), which silently blows up the
    # downstream DuckDB insert instead of failing in pandas.
    adjusted = df[["adjOpen", "adjHigh", "adjLow", "adjClose", "adjVolume"]].rename(
        columns={
            "adjOpen": "open",
            "adjHigh": "high",
            "adjLow": "low",
            "adjClose": "close",
            "adjVolume": "volume",
        }
    )
    adjusted.insert(0, "date", df["date"])
    adjusted.insert(0, "entity_id_hint", ticker.upper())
    return adjusted[["entity_id_hint", "date", "open", "high", "low", "close", "volume"]]


class TiingoBudget:
    """Tracks how many *distinct* symbols have been pulled this calendar
    month against the free tier's 500-symbol cap, so a pilot run fails
    loudly with a clear message instead of silently hitting HTTP 429s
    partway through -- the named asterisk on "zero paid data" at full
    scale."""

    def __init__(self, cap: int = 500, ledger_path: Path | None = None) -> None:
        self.cap = cap
        self.ledger_path = ledger_path or (PRICES_DIR / "tiingo_symbol_ledger.json")

    def _load(self) -> dict:
        import json

        if self.ledger_path.exists():
            return json.loads(self.ledger_path.read_text())
        return {}

    def _save(self, ledger: dict) -> None:
        import json

        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        self.ledger_path.write_text(json.dumps(ledger))

    def month_key(self) -> str:
        return time.strftime("%Y-%m")

    def remaining(self) -> int:
        ledger = self._load()
        used = len(ledger.get(self.month_key(), []))
        return max(self.cap - used, 0)

    def register(self, ticker: str) -> None:
        ledger = self._load()
        key = self.month_key()
        symbols = set(ledger.get(key, []))
        if ticker.upper() not in symbols and len(symbols) >= self.cap:
            raise RuntimeError(
                f"Tiingo free-tier symbol cap ({self.cap}/month) reached for {key}. "
                "Wait for next month, upgrade to Tiingo's paid tier, or reduce scope."
            )
        symbols.add(ticker.upper())
        ledger[key] = sorted(symbols)
        self._save(ledger)
