"""The backtest engine: wires market, broker and strategy together in a daily loop.

Per trading day::

    chain   = builder.build(t, date, spot[t])      # price every listed contract
    orders  = strategy.on_bar(ctx)                 # strategy sees the chain and its positions
    fills   = [broker.execute(o, chain) ...]       # fills at bid/ask + slippage + commission
    events  = broker.settle_expirations(chain)     # contracts expiring today, by moneyness
    record(nav, cash, greeks, ...)

The same ``MarketConfig`` (and seed) always produces the same path and vol
surface regardless of the execution settings, so a frictionless and a
realistic run differ *only* in execution costs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

import numpy as np
import pandas as pd

from optbt.config import BacktestConfig, ExecutionConfig, MarketConfig
from optbt.execution.broker import SimulatedBroker
from optbt.execution.costs import ExecutionCosts
from optbt.market.calendar import monthly_expiries, trading_days
from optbt.market.chain import ChainBuilder
from optbt.market.simulation import GBMSimulator, HestonSimulator, PathResult
from optbt.market.vol_surface import ConstantVol, HestonImpliedVol, SeriesVol, VolSurface, simulate_ou_log_iv
from optbt.strategy.base import Context, Strategy


# --------------------------------------------------------------------------- market
def build_market(cfg: MarketConfig) -> tuple[PathResult, VolSurface]:
    """Simulate the underlying path and construct the implied-vol surface on top of it."""
    rng = np.random.default_rng(cfg.seed)
    dates = trading_days(cfg.start, cfg.end)
    if len(dates) < 2:
        raise ValueError("Backtest window must contain at least two trading days")

    if cfg.model == "heston":
        sim = HestonSimulator(mu=cfg.mu, v0=cfg.v0, kappa=cfg.kappa, theta=cfg.theta, xi=cfg.xi, rho=cfg.rho)
        path = sim.simulate(dates, cfg.s0, rng)
        surface: VolSurface = HestonImpliedVol(
            skew=cfg.skew, smile=cfg.smile, variance=path.variance, kappa=cfg.kappa, theta=cfg.theta, premium=cfg.iv_premium
        )
    elif cfg.model == "gbm":
        sim = GBMSimulator(mu=cfg.mu, sigma=cfg.sigma)
        path = sim.simulate(dates, cfg.s0, rng)
        level = cfg.base_iv if cfg.base_iv is not None else cfg.sigma * (1.0 + cfg.iv_premium)
        if cfg.iv_process == "ou":
            levels = simulate_ou_log_iv(
                len(dates), level, cfg.iv_ou_kappa, cfg.iv_ou_vol, path.shocks, cfg.iv_spot_corr, rng
            )
            surface = SeriesVol(skew=cfg.skew, smile=cfg.smile, levels=levels, term_slope=cfg.term_slope)
        else:
            surface = ConstantVol(skew=cfg.skew, smile=cfg.smile, level=level, term_slope=cfg.term_slope)
    else:
        raise ValueError(f"unknown model {cfg.model!r}")
    return path, surface


# --------------------------------------------------------------------------- result
@dataclass
class BacktestResult:
    strategy_name: str
    strategy_description: str
    cost_label: str
    config: BacktestConfig
    equity: pd.DataFrame  # indexed by date
    fills: pd.DataFrame
    trades: pd.DataFrame
    settlements: pd.DataFrame
    cost_summary: dict[str, float]
    path: PathResult = field(repr=False)

    @property
    def nav(self) -> pd.Series:
        return self.equity["nav"]

    @property
    def initial_cash(self) -> float:
        return self.config.execution.initial_cash

    @property
    def final_nav(self) -> float:
        return float(self.equity["nav"].iloc[-1])

    @property
    def total_pnl(self) -> float:
        return self.final_nav - self.initial_cash

    @cached_property
    def metrics(self) -> dict[str, Any]:
        from optbt.analytics.metrics import compute_metrics

        return compute_metrics(self.equity["nav"], self.trades, rf=self.config.market.r)

    @cached_property
    def greek_summary(self) -> dict[str, Any]:
        from optbt.analytics.greeks import greek_exposure_summary

        return greek_exposure_summary(self.equity)


# --------------------------------------------------------------------------- engine
class Backtester:
    def __init__(
        self,
        strategy: Strategy,
        config: BacktestConfig | None = None,
        execution: ExecutionConfig | None = None,
    ) -> None:
        self.strategy = strategy
        self.config = config or BacktestConfig()
        if execution is not None:
            self.config = BacktestConfig(market=self.config.market, execution=execution)

    def run(self) -> BacktestResult:
        mcfg, ecfg = self.config.market, self.config.execution
        path, surface = build_market(mcfg)
        costs = ExecutionCosts.from_config(ecfg)
        expiries = monthly_expiries(mcfg.start, mcfg.end, months_ahead=mcfg.n_expiries + 1)
        builder = ChainBuilder(
            symbol=mcfg.symbol,
            surface=surface,
            expiries=expiries,
            r=mcfg.r,
            q=mcfg.q,
            half_spread_fn=costs.spread.half_spread,
            strike_step_pct=mcfg.strike_step_pct,
            n_strikes=mcfg.n_strikes,
            n_expiries=mcfg.n_expiries,
        )
        broker = SimulatedBroker(
            initial_cash=ecfg.initial_cash,
            costs=costs,
            settlement=ecfg.settlement,
            exercise_threshold=ecfg.exercise_threshold,
        )

        rows: list[dict[str, Any]] = []
        started = False
        cash_rate = mcfg.r if ecfg.cash_interest else 0.0
        for t, date in enumerate(path.dates):
            if t > 0:
                broker.accrue_interest(cash_rate, (date - path.dates[t - 1]).days)
            spot = float(path.spot[t])
            chain = builder.build(t, date, spot)
            ctx = Context(
                date=date,
                t_index=t,
                symbol=mcfg.symbol,
                spot=spot,
                chain=chain,
                broker=broker,
                nav=broker.nav(chain),
                greeks=broker.portfolio_greeks(chain),
            )
            if not started:
                self.strategy.on_start(ctx)
                started = True

            for order in self.strategy.on_bar(ctx):
                broker.execute(order, chain)

            events = broker.settle_expirations(chain)
            if events:
                self.strategy.on_settlement(ctx, events)

            greeks = broker.portfolio_greeks(chain)
            nav = broker.nav(chain)
            rows.append(
                {
                    "date": date,
                    "nav": nav,
                    "cash": broker.cash,
                    "positions_value": nav - broker.cash,
                    "interest_cum": broker.interest_total,
                    "spot": spot,
                    "atm_iv": chain.atm_iv,
                    "n_option_positions": len(broker.option_positions()),
                    **greeks,
                }
            )

        equity = pd.DataFrame(rows).set_index("date")
        equity.index = pd.to_datetime(equity.index)
        return BacktestResult(
            strategy_name=self.strategy.name,
            strategy_description=self.strategy.describe(),
            cost_label=costs.label,
            config=self.config,
            equity=equity,
            fills=_fills_frame(broker),
            trades=_trades_frame(broker),
            settlements=_settlements_frame(broker),
            cost_summary=_cost_summary(broker),
            path=path,
        )


# --------------------------------------------------------------------------- frames
def _fills_frame(broker: SimulatedBroker) -> pd.DataFrame:
    cols = ["date", "symbol", "quantity", "price", "mid", "commission", "spread_cost", "slippage_cost", "tag", "reason"]
    rows = [
        {
            "date": f.date,
            "symbol": str(f.instrument),
            "quantity": f.quantity,
            "price": f.price,
            "mid": f.mid,
            "commission": f.commission,
            "spread_cost": f.spread_cost,
            "slippage_cost": f.slippage_cost,
            "tag": f.tag,
            "reason": f.reason,
        }
        for f in broker.fills
    ]
    return pd.DataFrame(rows, columns=cols)


def _trades_frame(broker: SimulatedBroker) -> pd.DataFrame:
    cols = ["symbol", "tag", "open_date", "close_date", "quantity", "direction", "entry_price", "realized_pnl", "commissions", "pnl"]
    rows = [
        {
            "symbol": t.symbol,
            "tag": t.tag,
            "open_date": t.open_date,
            "close_date": t.close_date,
            "quantity": t.quantity,
            "direction": t.direction,
            "entry_price": t.entry_price,
            "realized_pnl": t.realized_pnl,
            "commissions": t.commissions,
            "pnl": t.pnl,
        }
        for t in broker.trades
    ]
    return pd.DataFrame(rows, columns=cols)


def _settlements_frame(broker: SimulatedBroker) -> pd.DataFrame:
    cols = ["date", "symbol", "quantity", "spot", "intrinsic", "outcome", "shares_delta"]
    rows = [
        {
            "date": e.date,
            "symbol": str(e.contract),
            "quantity": e.quantity,
            "spot": e.spot,
            "intrinsic": e.intrinsic,
            "outcome": e.outcome,
            "shares_delta": e.shares_delta,
        }
        for e in broker.settlements
    ]
    return pd.DataFrame(rows, columns=cols)


def _cost_summary(broker: SimulatedBroker) -> dict[str, float]:
    commission = sum(f.commission for f in broker.fills)
    spread = sum(f.spread_cost for f in broker.fills)
    slippage = sum(f.slippage_cost for f in broker.fills)
    return {
        "commission": commission,
        "spread": spread,
        "slippage": slippage,
        "total": commission + spread + slippage,
        "interest": broker.interest_total,
        "n_fills": float(sum(1 for f in broker.fills if f.reason == "order")),
        "contracts_traded": float(sum(abs(f.quantity) for f in broker.fills if f.reason == "order" and f.instrument.multiplier > 1)),
    }
