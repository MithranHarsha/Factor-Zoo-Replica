"""Financing / external-finance factors (build guide Section 5, factors
#28-31). All negative-weight: firms raising more capital (equity or debt)
have historically earned LOWER subsequent returns."""

from __future__ import annotations

import numpy as np

from factorzoo.factors.registry import FactorSpec, register


def _shares(p, suffix=""):
    dei = p[f"shares_outstanding_dei{suffix}"]
    gaap = p[f"shares_outstanding_gaap{suffix}"]
    return dei.fillna(gaap)


net_stock_issuance = register(
    FactorSpec(
        name="net_stock_issuance",
        category="financing",
        description="Change in log shares outstanding (split-adjusted via the shares-outstanding "
        "tags themselves, which report split-adjusted counts).",
        source="Pontiff & Woodgate (2008)",
        lag_days=91,
        direction=-1,
        exclude_financials=True,
        panel="annual",
        compute=lambda p: np.log(_shares(p)) - np.log(_shares(p, "_prior")),
    )
)

# Daniel & Titman's original composite equity issuance is log market-cap
# growth minus the pure price-return component of that growth -- which
# needs the monthly price panel joined against the annual one at the same
# formation date. This annual-only panel can't resolve that correctly, so
# rather than ship a formula that's silently wrong, this factor is
# approximated as net share-count growth (the same signal
# net_stock_issuance already captures at this panel's resolution) and
# documented as such, not left out silently.
composite_equity_issuance = register(
    FactorSpec(
        name="composite_equity_issuance",
        category="financing",
        description="Approximated as net share-count growth (log shares / log shares_prior) -- "
        "the same signal as net_stock_issuance at this panel's annual resolution. Daniel & "
        "Titman's original construction additionally nets out the pure price-return component of "
        "market-cap growth, which needs the monthly price panel joined against the annual one at "
        "the same formation date; left as a documented simplification, revisit once Phase 3's "
        "portfolio code merges both panels at formation time.",
        source="Daniel & Titman (2006)",
        lag_days=91,
        direction=-1,
        exclude_financials=True,
        panel="annual",
        compute=lambda p: np.log(_shares(p)) - np.log(_shares(p, "_prior")),
    )
)

debt_issuance = register(
    FactorSpec(
        name="debt_issuance",
        category="financing",
        description="Change in total debt (long-term + short-term) divided by lagged assets.",
        source="Bradshaw, Richardson & Sloan (2006)",
        lag_days=91,
        direction=-1,
        exclude_financials=True,
        panel="annual",
        compute=lambda p: (
            p[["lt_debt_noncurrent", "lt_debt_current", "st_borrowings"]].fillna(0).sum(axis=1)
            - p[["lt_debt_noncurrent_prior", "lt_debt_current_prior", "st_borrowings_prior"]].fillna(0).sum(axis=1)
        )
        / p["assets_prior"],
    )
)

net_external_financing = register(
    FactorSpec(
        name="net_external_financing",
        category="financing",
        description="Net stock issuance plus net debt issuance, scaled by assets (both legs "
        "approximated at the annual panel's resolution -- see composite_equity_issuance's note).",
        source="Bradshaw, Richardson & Sloan (2006)",
        lag_days=91,
        direction=-1,
        exclude_financials=True,
        panel="annual",
        compute=lambda p: (
            (_shares(p) - _shares(p, "_prior")) / _shares(p, "_prior")
            + (
                p[["lt_debt_noncurrent", "lt_debt_current", "st_borrowings"]].fillna(0).sum(axis=1)
                - p[["lt_debt_noncurrent_prior", "lt_debt_current_prior", "st_borrowings_prior"]]
                .fillna(0)
                .sum(axis=1)
            )
            / p["assets_prior"]
        ),
    )
)
