"""Distress / quality factors (build guide Section 5, factors #44-48)."""

from __future__ import annotations

import numpy as np

from factorzoo.factors.registry import FactorSpec, register


def _total_debt(p, suffix=""):
    cols = [f"lt_debt_noncurrent{suffix}", f"lt_debt_current{suffix}", f"st_borrowings{suffix}"]
    return p[cols].fillna(0).sum(axis=1)


def _ohlson_o_score(p):
    """Ohlson (1980)'s original logit coefficients, applied to this
    panel's available inputs. A few of Ohlson's original terms (e.g. a
    two-year net-income-decline indicator) need a third year of history
    this panel doesn't carry (see profitability.py's Piotroski note for
    the same limitation) and are approximated with what two vintages
    allow; this is a standard practical compromise, not the literal
    nine-term formula.
    """
    size = np.log(p["assets"].clip(lower=1))
    tlta = _total_debt(p) / p["assets"]  # total liabilities proxy / assets
    wcta = (p["assets_current"] - p["liabilities_current"]) / p["assets"]
    clca = p["liabilities_current"] / p["assets_current"]
    oeneg = (p["liabilities_current"] > p["assets_current"]).astype(float)
    nita = p["net_income"] / p["assets"]
    futl = p["cfo"] / _total_debt(p).replace(0, np.nan)
    intwo = (p["net_income"] < 0).astype(float)
    chin = (p["net_income"] - p["net_income_prior"]) / (p["net_income"].abs() + p["net_income_prior"].abs())

    o_score = (
        -1.32
        - 0.407 * size
        + 6.03 * tlta
        - 1.43 * wcta
        + 0.0757 * clca
        - 2.37 * nita
        - 1.83 * futl.fillna(0)
        + 0.285 * intwo
        - 1.72 * oeneg
        - 0.521 * chin.fillna(0)
    )
    return o_score


o_score = register(
    FactorSpec(
        name="o_score",
        category="distress",
        description="Ohlson (1980) bankruptcy probability score (logit index; higher = more "
        "distressed). Negative weight: high-O-Score firms have historically earned lower returns.",
        source="Ohlson (1980)",
        lag_days=91,
        direction=-1,
        exclude_financials=True,
        panel="annual",
        compute=_ohlson_o_score,
    )
)

altman_z_score = register(
    FactorSpec(
        name="altman_z_score",
        category="distress",
        description="Altman (1968) Z-Score (higher = safer). The original uses retained earnings "
        "and market value of equity in two of its five terms; retained earnings is approximated "
        "here with stockholders' equity (not separately tagged in this panel's starter tag list).",
        source="Altman (1968)",
        lag_days=91,
        direction=+1,
        exclude_financials=True,
        panel="annual",
        compute=lambda p: (
            1.2 * (p["assets_current"] - p["liabilities_current"]) / p["assets"]
            + 1.4 * p["stockholders_equity"] / p["assets"]
            + 3.3 * p["operating_income"] / p["assets"]
            + 0.6 * p["market_cap"] / _total_debt(p).replace(0, np.nan)
            + 1.0 * p["revenues"] / p["assets"]
        ),
    )
)

distress_risk = register(
    FactorSpec(
        name="distress_risk",
        category="distress",
        description="Simplified distress proxy: leverage (total debt / assets) times the firm's "
        "O-Score-implied distress level, standing in for Campbell-Hilscher-Szilagyi's full "
        "hazard-model failure probability (which additionally needs equity volatility history).",
        source="Campbell, Hilscher & Szilagyi (2008)",
        lag_days=91,
        direction=-1,
        exclude_financials=True,
        panel="annual",
        compute=lambda p: (_total_debt(p) / p["assets"]) * _ohlson_o_score(p).clip(lower=0),
    )
)

leverage = register(
    FactorSpec(
        name="leverage",
        category="distress",
        description="Total debt (long-term + short-term) divided by assets.",
        source="Bhandari (1988)",
        lag_days=91,
        direction=-1,
        exclude_financials=True,
        panel="annual",
        compute=lambda p: _total_debt(p) / p["assets"],
    )
)

current_ratio = register(
    FactorSpec(
        name="current_ratio",
        category="distress",
        description="Current assets divided by current liabilities. A weaker, more debated signal "
        "than the other quality factors here -- kept with a positive direction (more liquid = "
        "safer) as the conventional default; verify sign empirically before relying on it.",
        source="Ou & Penman (1989)",
        lag_days=91,
        direction=+1,
        exclude_financials=True,
        panel="annual",
        compute=lambda p: p["assets_current"] / p["liabilities_current"],
    )
)
