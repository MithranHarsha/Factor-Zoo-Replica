"""The factor registry: one FactorSpec per signal, so adding factor 49
never touches existing code (build guide Section 5).

Every category module (value.py, profitability.py, ...) registers its
factors as an import side effect; factors/__init__.py imports all of them
so that importing `factorzoo.factors` populates FACTOR_REGISTRY completely.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class FactorSpec:
    name: str
    category: str
    description: str
    source: str  # citation
    lag_days: int  # minimum point-in-time lag beyond filing date (0 for pure price factors)
    direction: int  # +1 if a high value predicts high returns, -1 if low does
    exclude_financials: bool
    panel: str  # "annual" (fundamentals) or "monthly" (price-based) -- which panel compute() expects
    compute: Callable[[pd.DataFrame], pd.Series]  # takes the panel, returns the signal column


FACTOR_REGISTRY: dict[str, FactorSpec] = {}


def register(spec: FactorSpec) -> FactorSpec:
    if spec.name in FACTOR_REGISTRY:
        raise ValueError(f"Factor '{spec.name}' is already registered -- names must be unique.")
    FACTOR_REGISTRY[spec.name] = spec
    return spec


def factors_by_category(category: str) -> list[FactorSpec]:
    return [f for f in FACTOR_REGISTRY.values() if f.category == category]


def summarize_values(values: pd.Series) -> dict:
    """Median, not mean, and infinities treated as missing. Several
    ratio-style factors (percent operating accruals, O-Score, R&D-to-
    sales) can have a near-zero or exactly-zero denominator for a handful
    of companies -- real pulled data surfaced this directly: one bad
    denominator sent percent_operating_accruals' mean to -5327 and
    rd_to_sales' mean to +inf across an otherwise sane 296-company panel.
    A single extreme ratio is exactly what a mean is not robust to; a
    divide-by-zero result also isn't a real economic number, so it's
    dropped the same way a NaN would be rather than left to corrupt the
    statistic.
    """
    clean = values.replace([np.inf, -np.inf], np.nan)
    n_valid = int(clean.notna().sum())
    return {"n_valid": n_valid, "median": float(clean.median()) if n_valid else None}


def registry_summary() -> pd.DataFrame:
    rows = [
        {
            "name": f.name,
            "category": f.category,
            "direction": f.direction,
            "panel": f.panel,
            "exclude_financials": f.exclude_financials,
            "source": f.source,
        }
        for f in FACTOR_REGISTRY.values()
    ]
    return pd.DataFrame(rows).sort_values(["category", "name"]).reset_index(drop=True)
