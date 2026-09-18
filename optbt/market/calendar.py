"""Trading calendar helpers.

Simplifications (documented in the README):

* Trading days are Monday-Friday; exchange holidays are ignored.
* Listed expiries are the third Friday of each month (standard monthlies).
* Time to expiry for pricing is calendar days / 365 (act/365).
"""

from __future__ import annotations

from datetime import date as Date
from datetime import timedelta

import pandas as pd


def trading_days(start: Date, end: Date) -> list[Date]:
    return [d.date() for d in pd.bdate_range(start, end)]


def third_friday(year: int, month: int) -> Date:
    first = Date(year, month, 1)
    # weekday(): Monday=0 ... Friday=4
    offset = (4 - first.weekday()) % 7
    return first + timedelta(days=offset + 14)


def monthly_expiries(start: Date, end: Date, months_ahead: int = 3) -> list[Date]:
    """Third-Friday expiries from ``start`` through ``end`` plus ``months_ahead`` extra months."""
    out: list[Date] = []
    y, m = start.year, start.month
    # extend the horizon so contracts listed near the end of the run still exist
    horizon_y, horizon_m = end.year, end.month + months_ahead
    while horizon_m > 12:
        horizon_m -= 12
        horizon_y += 1
    while (y, m) <= (horizon_y, horizon_m):
        exp = third_friday(y, m)
        if exp >= start:
            out.append(exp)
        m += 1
        if m > 12:
            m = 1
            y += 1
    return out


def year_fraction(d0: Date, d1: Date) -> float:
    return max((d1 - d0).days, 0) / 365.0
