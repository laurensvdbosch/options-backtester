"""The strategy interface.

A strategy is called once per trading day with a :class:`Context` and returns
a list of market :class:`Order` objects. It never touches the broker directly;
everything it needs to know (today's chain, its open positions, portfolio
Greeks, NAV) is on the context. Orders are filled in the order returned.

Strategies are deliberately stateless where possible: they re-derive what
they hold from ``ctx.option_positions()`` each day, which makes them robust
to positions being closed by expiry settlement.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date as Date
from typing import TYPE_CHECKING

from optbt.execution.orders import Order, Side
from optbt.instruments import OptionContract, Position, Quote, Stock
from optbt.market.chain import OptionChain

if TYPE_CHECKING:
    from optbt.execution.broker import SettlementEvent, SimulatedBroker


@dataclass
class Context:
    date: Date
    t_index: int
    symbol: str
    spot: float
    chain: OptionChain
    broker: "SimulatedBroker"
    nav: float
    greeks: dict[str, float]

    # convenience accessors -------------------------------------------------
    @property
    def stock(self) -> Stock:
        return Stock(self.symbol)

    def positions(self) -> list[Position]:
        return self.broker.open_positions()

    def option_positions(self) -> list[Position]:
        return self.broker.option_positions()

    def stock_quantity(self) -> int:
        return self.broker.stock_quantity(self.symbol)

    def quote(self, contract: OptionContract) -> Quote:
        return self.chain.quote(contract)

    def close_order(self, pos: Position) -> Order:
        """An order that flattens ``pos`` entirely, keeping its tag."""
        side = Side.SELL if pos.quantity > 0 else Side.BUY
        return Order(pos.instrument, side, abs(pos.quantity), tag=pos.tag)


class Strategy(ABC):
    name: str = "strategy"

    def __init__(self) -> None:
        self._trade_counter = 0

    def next_tag(self) -> str:
        self._trade_counter += 1
        return f"{self.name}#{self._trade_counter}"

    def on_start(self, ctx: Context) -> None:  # noqa: B027 - optional hook
        """Called once before the first bar."""

    @abstractmethod
    def on_bar(self, ctx: Context) -> list[Order]:
        """Called once per trading day. Return the orders to execute today."""

    def on_settlement(self, ctx: Context, events: list["SettlementEvent"]) -> None:  # noqa: B027
        """Called after expiry settlement on days where something settled."""

    def describe(self) -> str:
        return self.name
