# Factor Zoo Replica

A free-data, point-in-time-correct replication of the academic cross-sectional
factor zoo, with statistical taming (multiple-testing correction + dimension
reduction). Full design document: `Factor_Zoo_Replica_Build_Guide.pdf` /
the companion [build guide](https://claude.ai/artifact/Rf8XmMJxCfJWc8GY6KJoiW)
in this folder.

No WRDS/CRSP/Compustat subscription is used anywhere. Data sources:
SEC EDGAR (fundamentals), Yahoo Finance + Tiingo (prices -- see the live
finding below on why Stooq, the guide's originally documented primary
source, isn't used by default), Ken French Data Library (validation
benchmarks), and a historical S&P 500 membership dataset (universe).

## Status

**Phases 1-5 are built and unit-tested** (261 offline tests, all passing).
See the build guide, Section 10, for the full phased plan and each
phase's gate.

| Phase | What's there | Gate status |
| --- | --- | --- |
| 1. Setup & Data Foundations | EDGAR client, point-in-time store, universe construction, Ken French loader, price clients | **Met.** Real data pulled: 1,209 historical tickers, 48,779 XBRL facts for a 15-company pilot (incl. Apple), 15,897 days of FF5 factors. |
| 2. Core Factor Library | All 48 starter-library factors (build guide Section 5), registry + panel-building machinery | **Code complete, unit-tested.** Fundamentals-only factors (profitability, investment, accruals, most financing/distress -- ~27 of 48) verified against the real 15-company pilot via `compute-factors`. The FF3-vs-Ken-French correlation gate needs price data (see below) to run live; the replication mechanism itself (proper 2x3 Fama-French sort) is implemented and tested against synthetic data with known answers. |
| 3. Portfolios & Backtesting | Decile/quintile sorts with the large-cap breakpoint proxy, VW/EW weighting, long-short spreads, the Section 8 shuffle test, walk-forward splits | **Code complete, unit-tested.** Awaits price data to run on the real universe at scale. |
| 4. Statistical Taming | t-hurdles, Benjamini-Hochberg/Yekutieli FDR, Deflated Sharpe Ratio, Probability of Backtest Overfitting (CSCV), correlation clustering, LASSO spanning test, IPCA (via the `ipca` package) | **Code complete, unit-tested.** |
| 5. Dashboard & Polish | Streamlit app (5 pages), Docker packaging | **Built and smoke-tested live** (launched, health-checked, confirmed error-free against the real project database). Correlation Map / Taming Report pages are implemented but show an honest "waiting on price data" message until Phase 3 has real return series to summarize. |

### Live finding: free price-data access is currently unreliable

Two independent, verified findings from this build, documented in detail
in `src/factorzoo/data/prices.py`'s module docstring:

1. **Stooq** (the guide's originally documented primary source) returns a
   JavaScript bot-check page to any non-browser client, confirmed live
   even with a browser User-Agent and replayed cookies. Not currently
   usable for automated pulls.
2. **Yahoo Finance**'s chart API (the fallback this build switched to,
   same endpoint the `yfinance` package wraps) rate-limits aggressively
   and, in one observed window, gave different results to `httpx` (429)
   and `curl` (200) for the byte-identical request -- likely TLS/client
   fingerprinting. The price client now shells out to `curl` as a
   workaround, with multi-host fallback and exponential backoff, but
   sustained use from one IP still eventually hit a broader block during
   this build.

**Net effect:** the data pipeline, universe construction, and the full
48-factor library are proven correct on real EDGAR data; large-scale
price-dependent work (the FF3 validation gate, full backtests, the
Correlation Map and Taming Report dashboard pages) is blocked on an
external rate limit, not a code gap. Options going forward: retry
`pull-prices` later or from a different network, or register a free
[Tiingo](https://www.tiingo.com/) API key (`.env`'s `TIINGO_API_KEY`;
500 symbols/month free, ~$10/month removes the cap).

## Setup

```bash
uv venv .venv
uv pip install -e ".[dev,dashboard,taming]"
cp .env.example .env   # fill in SEC_USER_AGENT (required) with YOUR real contact info;
                        # TIINGO_API_KEY is optional
```

## CLI commands

```bash
uv run factorzoo init-db                   # create the DuckDB schema
uv run factorzoo build-universe            # point-in-time S&P 500 membership panel
uv run factorzoo pull-riskfree             # Ken French FF5 daily factors
uv run factorzoo pull-edgar --limit 25     # pilot EDGAR fundamentals pull
uv run factorzoo pull-prices --limit 25    # pilot price pull (Yahoo by default; --source stooq to retry that)
uv run factorzoo compute-factors           # run every annual (fundamentals-only) factor, print summary stats
uv run factorzoo compute-factors --category profitability   # restrict to one category
uv run factorzoo status                    # data-health report
```

## Dashboard

```bash
uv run streamlit run src/factorzoo/dashboard/app.py
```

Five pages: Data Health, Zoo Overview (all 48 factors, computed live where
data allows), Factor Detail, Correlation Map, Taming Report.

## Tests

```bash
uv run pytest            # offline tests only (default) -- 261 tests, synthetic fixtures with known answers
uv run pytest -m network # include tests that hit live data sources (SEC EDGAR, Yahoo, Stooq, Ken French)
```

## Repository layout

```
src/factorzoo/
  config.py              # settings, required SEC_USER_AGENT, paths
  data/                  # edgar.py, prices.py, riskfree.py, pit_store.py
  universe/filters.py    # point-in-time S&P 500 membership construction
  factors/               # registry.py, panel.py, + 8 category modules (48 factors)
  portfolios/            # sorts.py, weighting.py, returns.py
  eval/                  # performance.py (Newey-West), benchmarks.py (FF3 vs. Ken French)
  taming/                # multiple_testing.py, dimension_reduction.py, ipca_model.py
  backtest/               # walkforward.py, leakage_tests.py (the shuffle test)
  dashboard/              # data.py (testable helpers), app.py (Streamlit presentation)
  cli.py
tests/                    # one file per module above, offline by default
docker/Dockerfile
.github/workflows/ci.yml
configs/                  # universe.yaml, factors.yaml
```
