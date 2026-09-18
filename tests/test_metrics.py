import math
from datetime import date

import numpy as np
import pandas as pd
import pytest

from optbt.analytics.metrics import compute_metrics, drawdown_series, trade_groups


def _nav(values):
    idx = pd.bdate_range("2024-01-01", periods=len(values))
    return pd.Series(values, index=idx, dtype=float)


def test_drawdown_and_duration():
    nav = _nav([100, 110, 99, 104.5, 121, 120])
    dd = drawdown_series(nav)
    assert dd.min() == pytest.approx(-0.10)
    m = compute_metrics(nav, pd.DataFrame())
    assert m["max_drawdown"] == pytest.approx(-0.10)
    assert m["max_drawdown_days"] == 2
    assert m["total_return"] == pytest.approx(0.20)


def test_sharpe_and_sortino_on_known_returns():
    rets = np.array([0.01, -0.005, 0.02, -0.01, 0.015] * 50)
    nav = _nav(100 * np.cumprod(np.concatenate([[1.0], 1 + rets])))
    m = compute_metrics(nav, pd.DataFrame(), rf=0.0)
    exp_sharpe = rets.mean() / rets.std(ddof=1) * math.sqrt(252)
    downside = math.sqrt(np.mean(np.minimum(rets, 0) ** 2))
    assert m["sharpe"] == pytest.approx(exp_sharpe, rel=1e-6)
    assert m["sortino"] == pytest.approx(rets.mean() / downside * math.sqrt(252), rel=1e-6)
    assert m["ann_volatility"] == pytest.approx(rets.std(ddof=1) * math.sqrt(252), rel=1e-6)


def test_flat_nav_has_nan_sharpe():
    m = compute_metrics(_nav([100.0] * 10), pd.DataFrame())
    assert math.isnan(m["sharpe"])
    assert m["max_drawdown"] == 0.0


def test_trade_stats_group_legs_by_tag():
    d = date(2024, 1, 2)
    trades = pd.DataFrame(
        [
            {"symbol": "A 100P", "tag": "s#1", "open_date": d, "close_date": d, "quantity": 1, "direction": "short", "entry_price": 3.0, "realized_pnl": 300.0, "commissions": 1.0, "pnl": 299.0},
            {"symbol": "A 90P", "tag": "s#1", "open_date": d, "close_date": d, "quantity": 1, "direction": "long", "entry_price": 1.0, "realized_pnl": -100.0, "commissions": 1.0, "pnl": -101.0},
            {"symbol": "A 100P", "tag": "s#2", "open_date": d, "close_date": d, "quantity": 1, "direction": "short", "entry_price": 3.0, "realized_pnl": -500.0, "commissions": 1.0, "pnl": -501.0},
            {"symbol": "A", "tag": "s#shares", "open_date": d, "close_date": d, "quantity": 100, "direction": "long", "entry_price": 100.0, "realized_pnl": 50.0, "commissions": 0.0, "pnl": 50.0},
            {"symbol": "A", "tag": "s#shares", "open_date": d, "close_date": d, "quantity": 100, "direction": "long", "entry_price": 100.0, "realized_pnl": -20.0, "commissions": 0.0, "pnl": -20.0},
        ]
    )
    g = trade_groups(trades)
    assert len(g) == 4  # spread #1 (2 legs), spread #2, and two separate share lifecycles
    assert g.loc[g["tag"] == "s#1", "pnl"].iloc[0] == pytest.approx(198.0)
    m = compute_metrics(_nav([100, 101, 102]), trades)
    assert m["n_trades"] == 4
    assert m["win_rate"] == pytest.approx(0.5)
    assert m["profit_factor"] == pytest.approx((198 + 50) / (501 + 20))
