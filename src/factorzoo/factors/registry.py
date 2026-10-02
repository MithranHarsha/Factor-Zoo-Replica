"""The factor registry: one FactorSpec per signal, so adding factor 49
never touches existing code (build guide Section 5).

Every category module (value.py, profitability.py, ...) registers its
factors as an import side effect; factors/__init__.py imports all of them
so that importing `factorzoo.factors` populates FACTOR_REGISTRY completely.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

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
