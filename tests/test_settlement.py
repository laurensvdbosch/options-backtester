import pytest

from optbt.execution.broker import SimulatedBroker
from optbt.execution.orders import Order, Side
from optbt.instruments import OptionContract
from optbt.pricing import OptionType
from tests.conftest import EXPIRY, SYMBOL, TODAY, make_builder


def _open_then_settle(costs, kind, strike, qty, spot_at_expiry, settlement):
    """Trade ``qty`` contracts today at mid, then settle on expiry day at ``spot_at_expiry``."""
    builder = make_builder(costs)
    broker = SimulatedBroker(100_000.0, costs, settlement=settlement)
    entry = builder.build(0, TODAY, 100.0)
    contract = OptionContract(SYMBOL, kind, strike, EXPIRY)
    side = Side.BUY if qty > 0 else Side.SELL
    broker.execute(Order(contract, side, abs(qty), tag="x"), entry)
    cash_before = broker.cash
    expiry_chain = builder.build(1, EXPIRY, spot_at_expiry)
    events = broker.settle_expirations(expiry_chain)
    return broker, contract, events, cash_before


def test_long_call_itm_physical_exercise(frictionless_costs):
    broker, c, events, cash_before = _open_then_settle(frictionless_costs, OptionType.CALL, 100.0, +1, 110.0, "physical")
    assert len(events) == 1 and events[0].outcome == "exercised"
    assert events[0].shares_delta == 100
    assert broker.stock_quantity(SYMBOL) == 100
    assert broker.position(c) is None
    # exercising pays the strike for the shares: cash falls by K * 100
    assert broker.cash == pytest.approx(cash_before - 100.0 * 100)
    # shares are worth spot, so NAV reflects the intrinsic gain
    nav_after = broker.cash + 100 * 110.0
    assert nav_after == pytest.approx(cash_before + 10.0 * 100)


def test_short_call_itm_physical_assignment(frictionless_costs):
    broker, c, events, cash_before = _open_then_settle(frictionless_costs, OptionType.CALL, 100.0, -2, 107.0, "physical")
    assert events[0].outcome == "assigned"
    assert broker.stock_quantity(SYMBOL) == -200  # shares called away (short if none held)
    assert broker.cash == pytest.approx(cash_before + 100.0 * 200)


def test_short_put_itm_physical_assignment_delivers_shares(frictionless_costs):
    broker, c, events, cash_before = _open_then_settle(frictionless_costs, OptionType.PUT, 100.0, -1, 90.0, "physical")
    assert events[0].outcome == "assigned"
    assert broker.stock_quantity(SYMBOL) == 100
    assert broker.cash == pytest.approx(cash_before - 100.0 * 100)  # bought at the strike


def test_short_put_itm_cash_settlement(frictionless_costs):
    broker, c, events, cash_before = _open_then_settle(frictionless_costs, OptionType.PUT, 100.0, -1, 90.0, "cash")
    assert events[0].outcome == "assigned"
    assert broker.stock_quantity(SYMBOL) == 0
    assert broker.cash == pytest.approx(cash_before - 10.0 * 100)
    assert broker.position(c) is None


def test_otm_expires_worthless_keeps_premium(frictionless_costs):
    broker, c, events, cash_before = _open_then_settle(frictionless_costs, OptionType.CALL, 105.0, -1, 100.0, "physical")
    assert events[0].outcome == "expired" and events[0].shares_delta == 0
    assert broker.cash == pytest.approx(cash_before)
    assert broker.stock_quantity(SYMBOL) == 0
    trade = broker.trades[-1]
    assert trade.symbol == c.symbol
    assert trade.pnl == pytest.approx(trade.entry_price * 100)  # full premium realised


def test_exercise_threshold(frictionless_costs):
    # intrinsic 0.005 < threshold 0.01 -> treated as expired worthless
    broker, c, events, cash_before = _open_then_settle(frictionless_costs, OptionType.CALL, 100.0, +1, 100.005, "physical")
    assert events[0].outcome == "expired"
    assert broker.stock_quantity(SYMBOL) == 0


def test_contracts_not_expiring_are_untouched(frictionless_costs):
    builder = make_builder(frictionless_costs)
    broker = SimulatedBroker(100_000.0, frictionless_costs)
    chain = builder.build(0, TODAY, 100.0)
    c = OptionContract(SYMBOL, OptionType.CALL, 100.0, EXPIRY)
    broker.execute(Order(c, Side.BUY, 1), chain)
    assert broker.settle_expirations(chain) == []
    assert broker.position(c) is not None


def test_physical_equals_cash_plus_stock_trade(frictionless_costs):
    """Physical settlement must leave NAV identical to cash settlement (shares valued at spot)."""
    spot = 112.0
    b_phys, _, _, _ = _open_then_settle(frictionless_costs, OptionType.CALL, 100.0, +3, spot, "physical")
    b_cash, _, _, _ = _open_then_settle(frictionless_costs, OptionType.CALL, 100.0, +3, spot, "cash")
    nav_phys = b_phys.cash + b_phys.stock_quantity(SYMBOL) * spot
    assert nav_phys == pytest.approx(b_cash.cash)
