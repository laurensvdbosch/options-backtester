"""Greek-exposure analytics over the equity curve.

The engine records aggregate portfolio Greeks every day:

* ``delta``          share-equivalent exposure (100 = long 100 shares)
* ``delta_dollars``  delta * spot
* ``gamma``          change in delta per $1 move in spot
* ``theta``          $ per calendar day of time decay (positive = collecting)
* ``vega``           $ per 1 vol-point move in implied vol
"""

from __future__ import annotations

from typing import Any

import pandas as pd

GREEK_COLUMNS = ("delta", "delta_dollars", "gamma", "theta", "vega")


def greek_exposure_summary(equity: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for g in GREEK_COLUMNS:
        s = equity[g]
        out[g] = {
            "mean": float(s.mean()),
            "abs_mean": float(s.abs().mean()),
            "min": float(s.min()),
            "max": float(s.max()),
            "final": float(s.iloc[-1]),
        }
    out["pct_days_short_vega"] = float((equity["vega"] < 0).mean())
    out["pct_days_short_gamma"] = float((equity["gamma"] < 0).mean())
    out["pct_days_with_options"] = float((equity["n_option_positions"] > 0).mean())
    return out


def rolling_greeks(equity: pd.DataFrame, window: int = 21) -> pd.DataFrame:
    return equity[list(GREEK_COLUMNS)].rolling(window, min_periods=1).mean()
