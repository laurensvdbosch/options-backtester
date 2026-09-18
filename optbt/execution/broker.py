"""Simulated broker: order execution, position accounting, expiry settlement.

Accounting conventions
----------------------
* Cash is debited/credited at the executed price times the multiplier plus
  commission. Cash may go negative (physical assignment, debit trades larger
  than the balance); no buying-power or margin checks are modelled. The
  engine optionally accrues the risk-free rate on cash (positive balances
  earn it, negative balances pay it) so that idle capital is not penalised
  when Sharpe is computed against the same rate.
* Positions use average-cost accounting. A *lifecycle* runs from the first
  fill that takes quantity away from zero to the fill that returns it to
  zero; on close a :class:`TradeRecord` with realised P&L net of commissions
  is emitted. Strategies attach a ``tag`` to orders so multi-leg trades can be
  grouped in analytics.
* NAV marks every open position at the theoretical **mid**. The cost of
  crossing the spread therefore shows up at trade time, never as a phantom
  gain/loss on positions that were not traded.

Expiry settlement
-----------------
On a contract's expiry date, after the strategy has acted, every open
position in it is settled against the closing spot:

* intrinsic <= ``exercise_threshold``  -> expires worthless (closed at 0.00)
* otherwise -> closed at intrinsic value (cash settlement), and if the broker
  is in ``physical`` mode the corresponding share transfer at the *strike* is
  booked. That transfer is implemented as an internal stock fill at spot with
  no costs, which is economically identical: receiving (S - K) in cash and
  buying shares at S leaves you long shares having paid K net.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date as Date
from typing import Literal

from optbt.execution.costs import ExecutionCosts
from optbt.execution.orders import Fill, Order, Side
from optbt.instruments import Instrument, OptionContract, Position, Stock
from optbt.market.chain import OptionChain


@dataclass(frozen=True)
class SettlementEvent:
    date: Date
    contract: OptionContract
    quantity: int  # signed position quantity before settlement
    spot: float
    intrinsic: float
    outcome: Literal["expired", "exercised", "assigned"]
    shares_delta: int  # shares transferred (physical settlement only)


@dataclass(frozen=True)
class TradeRecord:
    symbol: str
    instrument: Instrument
    tag: str
    open_date: Date
    close_date: Date
    quantity: int  # peak absolute quantity in the lifecycle
    direction: Literal["long", "short"]
    entry_price: float
    realized_pnl: float  # before commissions
    commissions: float

    @property
    def pnl(self) -> float:
        return self.realized_pnl - self.commissions


class SimulatedBroker:
    def __init__(
        self,
        initial_cash: float,
        costs: ExecutionCosts,
        settlement: Literal["cash", "physical"] = "physical",
        exercise_threshold: float = 0.01,
    ) -> None:
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.costs = costs
        self.settlement = settlement
        self.exercise_threshold = exercise_threshold
        self.positions: dict[Instrument, Position] = {}
        self.fills: list[Fill] = []
        self.trades: list[TradeRecord] = []
        self.settlements: list[SettlementEvent] = []
        self.interest_total = 0.0

    def accrue_interest(self, rate: float, days: int) -> float:
        """Compound the cash balance at ``rate`` (continuous, act/365) over ``days`` calendar days."""
        if days <= 0 or rate == 0.0:
            return 0.0
        interest = self.cash * (math.exp(rate * days / 365.0) - 1.0)
        self.cash += interest
        self.interest_total += interest
        return interest

    # ------------------------------------------------------------------ queries
    def position(self, instrument: Instrument) -> Position | None:
        return self.positions.get(instrument)

    def open_positions(self) -> list[Position]:
        return [p for p in self.positions.values() if p.is_open]

    def option_positions(self) -> list[Position]:
        return [p for p in self.open_positions() if p.is_option]

    def stock_quantity(self, symbol: str) -> int:
        p = self.positions.get(Stock(symbol))
        return p.quantity if p else 0

    # ---------------------------------------------------------------- execution
    def execute(self, order: Order, chain: OptionChain) -> Fill:
        inst = order.instrument
        qty = order.quantity
        c = self.costs
        if isinstance(inst, OptionContract):
            q = chain.quote(inst)
            half = 0.5 * (q.ask - q.bid)
            touch = q.ask if order.side is Side.BUY else q.bid
            slip = c.slippage.slippage(half, qty, q.liquidity)
            price = touch + slip if order.side is Side.BUY else max(touch - slip, 0.0)
            commission = c.commission.option(qty)
            mid = q.mid
        else:
            mid = chain.spot
            half = c.spread.stock_half_spread(mid)
            touch = mid + half if order.side is Side.BUY else mid - half
            slip = 0.0
            price = touch
            commission = c.commission.stock(qty)
        mult = inst.multiplier
        return self._apply_fill(
            instrument=inst,
            signed_qty=order.signed_quantity,
            price=price,
            mid=mid,
            commission=commission,
            spread_cost=qty * mult * abs(touch - mid),
            slippage_cost=qty * mult * abs(price - touch),
            date=chain.date,
            tag=order.tag,
            reason="order",
        )

    def _apply_fill(
        self,
        instrument: Instrument,
        signed_qty: int,
        price: float,
        mid: float,
        commission: float,
        spread_cost: float,
        slippage_cost: float,
        date: Date,
        tag: str,
        reason: str,
    ) -> Fill:
        mult = instrument.multiplier
        self.cash -= signed_qty * price * mult + commission

        pos = self.positions.get(instrument)
        if pos is None:
            pos = Position(instrument)
            self.positions[instrument] = pos

        if pos.quantity == 0:
            self._open_lifecycle(pos, signed_qty, price, date, tag)
        elif (pos.quantity > 0) == (signed_qty > 0):
            new_q = pos.quantity + signed_qty
            pos.avg_price = (pos.avg_price * pos.quantity + price * signed_qty) / new_q
            pos.quantity = new_q
            pos.peak_quantity = max(pos.peak_quantity, abs(new_q))
        else:
            direction = 1 if pos.quantity > 0 else -1
            closing = min(abs(signed_qty), abs(pos.quantity))
            pos.realized_pnl += closing * (price - pos.avg_price) * mult * direction
            remainder = pos.quantity + signed_qty
            pos.quantity = 0 if remainder * direction <= 0 else remainder
            if remainder * direction < 0:
                # position flipped: close the old lifecycle, open a new one with the remainder
                pos.commissions += commission
                commission = 0.0
                self._close_lifecycle(pos, date)  # removes the entry from self.positions
                self._open_lifecycle(pos, remainder, price, date, tag)
                self.positions[instrument] = pos

        pos.commissions += commission
        if pos.quantity == 0:
            self._close_lifecycle(pos, date)

        fill = Fill(
            date=date,
            instrument=instrument,
            quantity=signed_qty,
            price=price,
            mid=mid,
            commission=commission,
            spread_cost=spread_cost,
            slippage_cost=slippage_cost,
            tag=tag,
            reason=reason,
        )
        self.fills.append(fill)
        return fill

    @staticmethod
    def _open_lifecycle(pos: Position, signed_qty: int, price: float, date: Date, tag: str) -> None:
        pos.quantity = signed_qty
        pos.avg_price = price
        pos.open_date = date
        pos.tag = tag
        pos.realized_pnl = 0.0
        pos.commissions = 0.0
        pos.peak_quantity = abs(signed_qty)
        pos.direction = 1 if signed_qty > 0 else -1

    def _close_lifecycle(self, pos: Position, date: Date) -> None:
        if pos.open_date is None:
            return
        self.trades.append(
            TradeRecord(
                symbol=pos.symbol,
                instrument=pos.instrument,
                tag=pos.tag,
                open_date=pos.open_date,
                close_date=date,
                quantity=pos.peak_quantity,
                direction="long" if pos.direction > 0 else "short",
                entry_price=pos.avg_price,
                realized_pnl=pos.realized_pnl,
                commissions=pos.commissions,
            )
        )
        pos.open_date = None
        pos.realized_pnl = 0.0
        pos.commissions = 0.0
        pos.peak_quantity = 0
        pos.quantity = 0
        pos.direction = 0
        if pos.instrument in self.positions:
            del self.positions[pos.instrument]

    # ------------------------------------------------------------- settlement
    def settle_expirations(self, chain: OptionChain) -> list[SettlementEvent]:
        date, spot = chain.date, chain.spot
        events: list[SettlementEvent] = []
        for pos in list(self.option_positions()):
            contract: OptionContract = pos.instrument  # type: ignore[assignment]
            if contract.expiry > date:
                continue
            qty = pos.quantity
            intrinsic = contract.intrinsic(spot)
            if intrinsic > self.exercise_threshold:
                outcome = "exercised" if qty > 0 else "assigned"
                fee = self.costs.commission.exercise(qty)
                self._apply_fill(contract, -qty, intrinsic, intrinsic, fee, 0.0, 0.0, date, pos.tag, outcome)
                shares_delta = 0
                if self.settlement == "physical":
                    shares_delta = contract.kind.sign * qty * contract.multiplier
                    self._apply_fill(
                        Stock(contract.underlying), shares_delta, spot, spot, 0.0, 0.0, 0.0, date, pos.tag, "settlement_stock"
                    )
            else:
                outcome = "expired"
                shares_delta = 0
                self._apply_fill(contract, -qty, 0.0, 0.0, 0.0, 0.0, 0.0, date, pos.tag, outcome)
            ev = SettlementEvent(date, contract, qty, spot, intrinsic, outcome, shares_delta)
            events.append(ev)
            self.settlements.append(ev)
        return events

    # ------------------------------------------------------------ valuation
    def position_value(self, pos: Position, chain: OptionChain) -> float:
        if pos.is_option:
            return pos.quantity * pos.multiplier * chain.quote(pos.instrument).mid  # type: ignore[arg-type]
        return pos.quantity * chain.spot

    def nav(self, chain: OptionChain) -> float:
        return self.cash + sum(self.position_value(p, chain) for p in self.open_positions())

    def portfolio_greeks(self, chain: OptionChain) -> dict[str, float]:
        """Aggregate Greeks: delta in share-equivalents, theta in $/day, vega in $/vol-point."""
        delta = gamma = theta = vega = 0.0
        for pos in self.open_positions():
            if pos.is_option:
                q = chain.quote(pos.instrument)  # type: ignore[arg-type]
                n = pos.quantity * pos.multiplier
                delta += n * q.delta
                gamma += n * q.gamma
                theta += n * q.theta
                vega += n * q.vega
            else:
                delta += pos.quantity
        return {
            "delta": delta,
            "gamma": gamma,
            "theta": theta,
            "vega": vega,
            "delta_dollars": delta * chain.spot,
        }
