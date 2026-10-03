"""Central configuration: paths, environment variables, pull parameters.

Phase 1 note: SEC EDGAR requires a
descriptive User-Agent on every request. There is no safe default for this
-- making one up risks looking like abusive/anonymous traffic -- so
``SEC_USER_AGENT`` is a required environment variable and the app fails
fast with a clear message if it is missing, rather than silently sending a
bad header.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # loads .env if present; real environment variables still win

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CACHE_DIR = DATA_DIR / "cache"
DB_PATH = DATA_DIR / "factorzoo.duckdb"

EDGAR_COMPANYFACTS_DIR = RAW_DIR / "edgar" / "companyfacts"
EDGAR_SUBMISSIONS_DIR = RAW_DIR / "edgar" / "submissions"
PRICES_DIR = RAW_DIR / "prices"


class ConfigError(RuntimeError):
    """Raised when required configuration (e.g. an API key) is missing."""


@dataclass(frozen=True)
class Settings:
    sec_user_agent: str
    tiingo_api_key: str | None
    sec_requests_per_second: float = 5.0  # conservative; SEC's own guidance is ~10/s
    tiingo_free_tier_symbol_cap_per_month: int = 500


def load_settings(require_tiingo: bool = False) -> Settings:
    """Load settings from the environment, failing fast with an actionable
    message rather than letting a missing key surface later as a cryptic
    HTTP 403 or None-type error deep in a pull.
    """
    user_agent = os.environ.get("SEC_USER_AGENT", "").strip()
    if not user_agent or "@" not in user_agent:
        raise ConfigError(
            "SEC_USER_AGENT is not set (or doesn't look like 'Name email@domain').\n"
            "SEC EDGAR requires a descriptive User-Agent on every request.\n"
            "Copy .env.example to .env and fill it in, e.g.:\n"
            '  SEC_USER_AGENT="Jane Doe jane@example.com"'
        )

    tiingo_key = os.environ.get("TIINGO_API_KEY", "").strip() or None
    if require_tiingo and not tiingo_key:
        raise ConfigError(
            "TIINGO_API_KEY is not set. Tiingo is only needed for delisted-ticker "
            "price backfill. Register for a free "
            "key at https://www.tiingo.com/ and add it to .env, or skip Tiingo pulls."
        )

    return Settings(sec_user_agent=user_agent, tiingo_api_key=tiingo_key)


def ensure_data_dirs() -> None:
    for d in (DATA_DIR, RAW_DIR, CACHE_DIR, EDGAR_COMPANYFACTS_DIR, EDGAR_SUBMISSIONS_DIR, PRICES_DIR):
        d.mkdir(parents=True, exist_ok=True)
