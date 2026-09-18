import pytest

from optbt.execution.broker import SimulatedBroker
from optbt.execution.costs import SlippageModel, SpreadModel
from optbt.execution.orders import Order, Side
from optbt.instruments import OptionContract, Stock
from optbt.pricing import OptionType
from tests.conftest import EXPIRY, SPOT, SYMBOL, TODAY


# ------------------------------------------------------------------ spread model
def test_otm_relative_spread_wider_than_atm():
    m = SpreadModel()
    atm_mid, otm_mid = 3.00, 0.30
    atm_rel = m.full_spread(atm_mid, 0.0, 0.50, 30) / atm_mid
    otm_rel = m.full_spread(otm_mid, 0.0, 0.08, 30) / otm_mid
    assert otm_rel > atm_rel * 3


def test_tick_floor_dominates_for_cheap_options():
    m = SpreadModel(min_abs=0.05)
    assert m.full_spread(0.10, 0.0, 0.05, 30) == pytest.approx(0.05)


def test_deep_itm_spread_is_wide_in_dollars_but_narrow_relative():
    m = SpreadModel()
    itm = m.full_spread(30.2, 30.0, 0.98, 30)  # $30 intrinsic, $0.20 extrinsic
    atm = m.full_spread(3.0, 0.0, 0.50, 30)
    assert itm > atm
    assert itm / 30.2 < atm / 3.0


def test_zero_spread_model():
    z = SpreadModel.zero()
    assert z.half_spread(3.0, 0.0, 0.5, 30) == 0.0
    assert z.stock_half_spread(100.0) == 0.0


# ---------------------------------------------------------------- slippage model
def test_slippage_grows_with_size_and_shrinks_with_liquidity():
    s = SlippageModel(impact=0.5, depth=50, max_multiple=2.0)
    assert s.slippage(0.05, 1, 1.0) < s.slippage(0.05, 10, 1.0) < s.slippage(0.05, 100, 1.0)
    assert s.slippage(0.05, 5, 0.1) > s.slippage(0.05, 5, 1.0)
    assert s.slippage(0.05, 10_000, 0.01) == pytest.approx(0.05 * 2.0)  # capped
    assert s.slippage(0.0, 10, 1.0) == 0.0
    assert SlippageModel.zero().slippage(0.05, 10, 1.0) == 0.0


# ------------------------------------------------------------------ broker fills
def _atm_call(chain):
    return chain.find(OptionType.CALL, strike=SPOT, expiry=EXPIRY)


def test_realistic_fill_crosses_spread_and_pays_slippage(realistic_builder, realistic_costs):
    chain = realistic_builder.build(0, TODAY, SPOT)
    q = _atm_call(chain)
    assert q.bid < q.mid < q.ask
    broker = SimulatedBroker(10_000.0, realistic_costs)
    buy = broker.execute(Order(q.contract, Side.BUY, 3), chain)
    assert buy.price > q.ask  # ask plus slippage
    assert buy.slippage_cost > 0 and buy.spread_cost == pytest.approx(3 * 100 * (q.ask - q.mid))
    assert buy.commission == pytest.approx(3 * 0.65)
    assert broker.cash == pytest.approx(10_000.0 - 3 * 100 * buy.price - buy.commission)

    sell = broker.execute(Order(q.contract, Side.SELL, 3), chain)
    assert sell.price < q.bid
    assert broker.position(q.contract) is None  # lifecycle closed
    assert len(broker.trades) == 1
    assert broker.trades[0].pnl < 0  # round trip at the same mid loses the full spread + costs


def test_frictionless_fill_is_at_mid(frictionless_builder, frictionless_costs):
    chain = frictionless_builder.build(0, TODAY, SPOT)
    q = _atm_call(chain)
    assert q.bid == q.mid == q.ask
    broker = SimulatedBroker(10_000.0, frictionless_costs)
    f = broker.execute(Order(q.contract, Side.BUY, 2), chain)
    assert f.price == pytest.approx(q.mid)
    assert f.total_cost == 0.0
    assert broker.nav(chain) == pytest.approx(10_000.0)  # marking at mid: no phantom P&L


def test_average_cost_and_realized_pnl(frictionless_costs):
    c = OptionContract(SYMBOL, OptionType.CALL, 100.0, EXPIRY)
    broker = SimulatedBroker(0.0, frictionless_costs)
    broker._apply_fill(c, +2, 1.00, 1.00, 0.0, 0.0, 0.0, TODAY, "t", "order")
    broker._apply_fill(c, +2, 2.00, 2.00, 0.0, 0.0, 0.0, TODAY, "t", "order")
    pos = broker.position(c)
    assert pos.quantity == 4 and pos.avg_price == pytest.approx(1.50)
    broker._apply_fill(c, -1, 2.50, 2.50, 0.0, 0.0, 0.0, TODAY, "t", "order")
    assert pos.realized_pnl == pytest.approx(100.0)  # (2.50 - 1.50) * 100 * 1
    broker._apply_fill(c, -3, 1.00, 1.00, 0.0, 0.0, 0.0, TODAY, "t", "order")
    assert broker.position(c) is None
    t = broker.trades[-1]
    assert t.realized_pnl == pytest.approx(100.0 - 150.0)
    assert t.direction == "long" and t.quantity == 4
    assert broker.cash == pytest.approx(-200 - 400 + 250 + 300)


def test_short_round_trip_and_position_flip(frictionless_costs):
    c = OptionContract(SYMBOL, OptionType.PUT, 95.0, EXPIRY)
    broker = SimulatedBroker(0.0, frictionless_costs)
    broker._apply_fill(c, -2, 3.00, 3.00, 1.30, 0.0, 0.0, TODAY, "s", "order")
    assert broker.cash == pytest.approx(600.0 - 1.30)
    # buy 5: closes the short 2 (realised (3.00-2.00)*100*2 = +200) and opens long 3
    broker._apply_fill(c, +5, 2.00, 2.00, 0.0, 0.0, 0.0, TODAY, "s2", "order")
    assert len(broker.trades) == 1
    assert broker.trades[0].direction == "short"
    assert broker.trades[0].pnl == pytest.approx(200.0 - 1.30)
    pos = broker.position(c)
    assert pos.quantity == 3 and pos.avg_price == 2.00 and pos.tag == "s2"


def test_stock_fill_uses_bps_spread(realistic_builder, realistic_costs):
    chain = realistic_builder.build(0, TODAY, SPOT)
    broker = SimulatedBroker(50_000.0, realistic_costs)
    f = broker.execute(Order(Stock(SYMBOL), Side.BUY, 100), chain)
    assert f.price == pytest.approx(SPOT * (1 + 0.5e-4))
    assert broker.stock_quantity(SYMBOL) == 100
    assert broker.portfolio_greeks(chain)["delta"] == pytest.approx(100.0)


def test_interest_accrual(frictionless_costs):
    broker = SimulatedBroker(100_000.0, frictionless_costs)
    earned = broker.accrue_interest(0.05, 365)
    assert earned == pytest.approx(100_000.0 * (2.718281828**0.05 - 1), rel=1e-6)
    assert broker.accrue_interest(0.05, 0) == 0.0
