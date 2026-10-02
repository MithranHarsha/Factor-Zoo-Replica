"""Importing this package registers all 48 starter-library factors
(build guide Section 5) into FACTOR_REGISTRY as a side effect."""

from factorzoo.factors import (
    accruals,
    distress,
    financing,
    intangibles,
    investment,
    momentum,
    profitability,
    trading_frictions,
    value,
)
from factorzoo.factors.registry import (
    FACTOR_REGISTRY,
    FactorSpec,
    factors_by_category,
    register,
    registry_summary,
)

__all__ = [
    "FACTOR_REGISTRY",
    "FactorSpec",
    "accruals",
    "distress",
    "factors_by_category",
    "financing",
    "intangibles",
    "investment",
    "momentum",
    "profitability",
    "register",
    "registry_summary",
    "trading_frictions",
    "value",
]
