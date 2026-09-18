"""Orders submitted by strategies and the fills the broker returns."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date as Date
from enum import Enum

from optbt.instruments import Instrument


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"

    @property
    def sign(self) -> int:
        return 1 if self is Side.BUY else -1


@dataclass(frozen=True)
class Order:
    """Market order for ``quantity`` units (contracts or shares), always positive."""

    instrument: Instrument
    side: Side
    quantity: int
    tag: str = ""  # strategies use this to group the legs of one trade

    def __post_init__(self) -> None:
        if self.quantity <= 0:
            raise ValueError("Order quantity must be positive")

    @property
    def signed_quantity(self) -> int:
        return self.side.sign * self.quantity


@dataclass(frozen=True)
class Fill:
    date: Date
    instrument: Instrument
    quantity: int  # signed: +buy / -sell
    price: float  # executed price per unit
    mid: float  # theoretical mid at the time
    commission: float
    spread_cost: float  # $ paid crossing half the spread (|qty| * mult * |touch - mid|)
    slippage_cost: float  # $ paid beyond the touch
    tag: str = ""
    reason: str = "order"  # "order" | "expired" | "exercised" | "assigned" | "settlement_stock"

    @property
    def notional(self) -> float:
        return abs(self.quantity) * self.instrument.multiplier * self.price

    @property
    def total_cost(self) -> float:
        return self.commission + self.spread_cost + self.slippage_cost
