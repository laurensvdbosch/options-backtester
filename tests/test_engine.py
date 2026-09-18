from datetime import date

import numpy as np
import pytest

from optbt import Backtester, BacktestConfig, ExecutionConfig, MarketConfig
from optbt.market.calendar import trading_days
from optbt.strategy import BullPutSpread, CoveredCall, ShortStrangle

MARKET = MarketConfig(model="gbm", start=date(2023, 1, 2), end=date(2023, 6, 30), seed=7)


def _run(strategy, frictionless: bool, settlement="cash", cash_interest=False):
    ex = ExecutionConfig(initial_cash=100_000.0, settlement=settlement, frictionless=frictionless, cash_interest=cash_interest)
    return Backtester(strategy, BacktestConfig(market=MARKET, execution=ex)).run()


@pytest.mark.parametrize(
    "factory,settlement",
    [
        (lambda: CoveredCall(lots=2), "physical"),
        (lambda: BullPutSpread(contracts=3), "cash"),
        (lambda: ShortStrangle(contracts=2), "cash"),
    ],
)
def test_strategies_run_and_accounting_is_consistent(factory, settlement):
    res = _run(factory(), frictionless=True, settlement=settlement)
    eq = res.equity
    assert len(eq) == len(trading_days(MARKET.start, MARKET.end))
    assert np.allclose(eq["nav"], eq["cash"] + eq["positions_value"])
    assert res.cost_summary["total"] == 0.0
    assert res.metrics["n_trades"] > 0
    assert (eq["n_option_positions"] > 0).any()
    assert set(["delta", "gamma", "theta", "vega"]).issubset(eq.columns)


def test_same_seed_gives_identical_market_in_both_runs():
    f = _run(ShortStrangle(contracts=2), frictionless=True)
    r = _run(ShortStrangle(contracts=2), frictionless=False)
    assert np.array_equal(f.equity["spot"].values, r.equity["spot"].values)
    assert np.array_equal(f.equity["atm_iv"].values, r.equity["atm_iv"].values)


def test_realistic_costs_are_positive_and_fills_are_off_mid():
    r = _run(BullPutSpread(contracts=3), frictionless=False)
    c = r.cost_summary
    assert c["spread"] > 0 and c["slippage"] > 0 and c["commission"] > 0
    orders = r.fills[r.fills["reason"] == "order"]
    assert (orders["price"] != orders["mid"]).all()
    buys, sells = orders[orders["quantity"] > 0], orders[orders["quantity"] < 0]
    assert (buys["price"] >= buys["mid"]).all() and (sells["price"] <= sells["mid"]).all()


def test_short_premium_strategy_is_short_vega_and_theta_positive():
    r = _run(ShortStrangle(contracts=2), frictionless=True)
    eq = r.equity[r.equity["n_option_positions"] > 0]
    assert (eq["vega"] < 0).all()
    assert (eq["theta"] > 0).all()
    assert (eq["gamma"] < 0).all()


def test_covered_call_delta_is_below_share_count():
    r = _run(CoveredCall(lots=2), frictionless=True, settlement="physical")
    eq = r.equity[r.equity["n_option_positions"] > 0]
    assert (eq["delta"] < 200).all() and (eq["delta"] > 0).all()


def test_settlement_events_are_recorded():
    r = _run(BullPutSpread(contracts=3, take_profit=1.0, stop_loss_multiple=100.0), frictionless=True)  # hold to expiry
    assert not r.settlements.empty
    assert set(r.settlements["outcome"]).issubset({"expired", "exercised", "assigned"})


def test_cash_interest_accrues_when_enabled():
    without = _run(CoveredCall(lots=1), frictionless=True, settlement="physical", cash_interest=False)
    with_ = _run(CoveredCall(lots=1), frictionless=True, settlement="physical", cash_interest=True)
    assert with_.cost_summary["interest"] > 0
    assert without.cost_summary["interest"] == 0.0
    assert with_.final_nav > without.final_nav
