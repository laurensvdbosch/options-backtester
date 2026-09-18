"""Performance metrics computed from the daily NAV series and the closed-trade log.

Conventions
-----------
* Returns are simple daily NAV returns; annualisation uses 252 periods.
* Sharpe and Sortino subtract the (continuously quoted) risk-free rate
  divided by 252 per day. Sortino's downside deviation is the RMS of the
  negative excess returns (target = 0), computed over *all* days.
* Max drawdown is the worst peak-to-trough decline of NAV; its duration is
  the longest stretch of trading days spent below a prior peak.
* Trade statistics group closed lifecycles by strategy ``tag`` so that a
  two-leg spread counts as one trade. Stock lifecycles tagged ``*#shares``
  count individually. Positions still open at the end are *not* in the
  trade stats (their P&L is unrealised, visible in NAV only).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

PERIODS_PER_YEAR = 252


def drawdown_series(nav: pd.Series) -> pd.Series:
    peak = nav.cummax()
    return nav / peak - 1.0


def _max_drawdown_duration(dd: pd.Series) -> int:
    longest = current = 0
    for v in dd.values:
        current = current + 1 if v < 0 else 0
        longest = max(longest, current)
    return longest


def trade_groups(trades: pd.DataFrame) -> pd.DataFrame:
    """Collapse closed lifecycles into strategy-level trades keyed by tag."""
    if trades.empty:
        return pd.DataFrame(columns=["tag", "open_date", "close_date", "pnl", "commissions", "legs"])
    df = trades.copy()
    generic = df["tag"].eq("") | df["tag"].str.endswith("#shares")
    df["group"] = df["tag"]
    df.loc[generic, "group"] = df.loc[generic, "tag"] + "@" + df.loc[generic].index.astype(str)
    out = (
        df.groupby("group", sort=False)
        .agg(
            tag=("tag", "first"),
            open_date=("open_date", "min"),
            close_date=("close_date", "max"),
            pnl=("pnl", "sum"),
            commissions=("commissions", "sum"),
            legs=("symbol", "count"),
        )
        .reset_index(drop=True)
    )
    return out.sort_values("open_date").reset_index(drop=True)


def compute_metrics(nav: pd.Series, trades: pd.DataFrame, rf: float = 0.0) -> dict[str, Any]:
    nav = nav.astype(float)
    n = len(nav)
    rets = nav.pct_change().dropna()
    years = max(n - 1, 1) / PERIODS_PER_YEAR
    start, end = float(nav.iloc[0]), float(nav.iloc[-1])

    total_return = end / start - 1.0
    cagr = (end / start) ** (1.0 / years) - 1.0 if start > 0 and end > 0 else float("nan")
    vol = float(rets.std(ddof=1)) if len(rets) > 1 else float("nan")
    ann_vol = vol * math.sqrt(PERIODS_PER_YEAR)
    excess = rets - rf / PERIODS_PER_YEAR
    sharpe = float(excess.mean() / vol * math.sqrt(PERIODS_PER_YEAR)) if vol and vol > 0 else float("nan")
    downside = float(np.sqrt(np.mean(np.minimum(excess.values, 0.0) ** 2))) if len(excess) else float("nan")
    sortino = float(excess.mean() / downside * math.sqrt(PERIODS_PER_YEAR)) if downside and downside > 0 else float("nan")

    dd = drawdown_series(nav)
    max_dd = float(dd.min()) if len(dd) else 0.0
    calmar = cagr / abs(max_dd) if max_dd < 0 else float("nan")

    groups = trade_groups(trades)
    n_trades = int(len(groups))
    wins = groups.loc[groups["pnl"] > 0, "pnl"] if n_trades else pd.Series(dtype=float)
    losses = groups.loc[groups["pnl"] <= 0, "pnl"] if n_trades else pd.Series(dtype=float)
    gross_profit = float(wins.sum())
    gross_loss = float(-losses.sum())

    return {
        "start_nav": start,
        "end_nav": end,
        "total_pnl": end - start,
        "total_return": total_return,
        "cagr": cagr,
        "ann_volatility": ann_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "max_drawdown_days": _max_drawdown_duration(dd),
        "calmar": calmar,
        "n_trades": n_trades,
        "win_rate": float(len(wins) / n_trades) if n_trades else float("nan"),
        "avg_win": float(wins.mean()) if len(wins) else float("nan"),
        "avg_loss": float(losses.mean()) if len(losses) else float("nan"),
        "profit_factor": gross_profit / gross_loss if gross_loss > 0 else float("inf") if gross_profit > 0 else float("nan"),
        "expectancy": float(groups["pnl"].mean()) if n_trades else float("nan"),
        "trading_days": n,
    }
