"""Tradeable instruments, market quotes and portfolio positions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date as Date
from typing import Union

from optbt.pricing.black_scholes import Greeks, OptionType


@dataclass(frozen=True)
class Stock:
    symbol: str

    @property
    def multiplier(self) -> int:
        return 1

    def __str__(self) -> str:
        return self.symbol


@dataclass(frozen=True)
class OptionContract:
    """A single listed option series. Frozen so it can key dictionaries."""

    underlying: str
    kind: OptionType
    strike: float
    expiry: Date
    multiplier: int = 100

    @property
    def is_call(self) -> bool:
        return self.kind is OptionType.CALL

    @property
    def symbol(self) -> str:
        letter = "C" if self.is_call else "P"
        return f"{self.underlying} {self.expiry:%Y-%m-%d} {self.strike:g}{letter}"

    def dte(self, on: Date) -> int:
        """Calendar days to expiry as seen from ``on`` (0 on expiry day)."""
        return (self.expiry - on).days

    def intrinsic(self, spot: float) -> float:
        return max(self.kind.sign * (spot - self.strike), 0.0)

    def __str__(self) -> str:
        return self.symbol


Instrument = Union[Stock, OptionContract]


@dataclass(frozen=True)
class Quote:
    """A priced option on a given day: theoretical mid, executable bid/ask, Greeks, liquidity."""

    contract: OptionContract
    date: Date
    spot: float
    mid: float
    bid: float
    ask: float
    iv: float
    tte: float  # years
    greeks: Greeks
    liquidity: float  # relative depth score in (0, 1]; 1 = ATM front-month

    @property
    def strike(self) -> float:
        return self.contract.strike

    @property
    def expiry(self) -> Date:
        return self.contract.expiry

    @property
    def kind(self) -> OptionType:
        return self.contract.kind

    @property
    def dte(self) -> int:
        return self.contract.dte(self.date)

    @property
    def delta(self) -> float:
        return self.greeks.delta

    @property
    def gamma(self) -> float:
        return self.greeks.gamma

    @property
    def theta(self) -> float:
        return self.greeks.theta

    @property
    def vega(self) -> float:
        return self.greeks.vega

    @property
    def spread(self) -> float:
        return self.ask - self.bid

    @property
    def intrinsic(self) -> float:
        return self.contract.intrinsic(self.spot)

    @property
    def extrinsic(self) -> float:
        return max(self.mid - self.intrinsic, 0.0)


@dataclass
class Position:
    """Signed holding in one instrument, with average-cost accounting for one 'lifecycle'.

    A lifecycle starts when quantity leaves zero and ends when it returns to zero.
    ``realized_pnl`` is net of commissions.
    """

    instrument: Instrument
    quantity: int = 0
    avg_price: float = 0.0
    open_date: Date | None = None
    tag: str = ""
    realized_pnl: float = 0.0
    commissions: float = 0.0
    peak_quantity: int = 0
    direction: int = 0  # +1 long / -1 short for the current lifecycle

    @property
    def is_option(self) -> bool:
        return isinstance(self.instrument, OptionContract)

    @property
    def is_open(self) -> bool:
        return self.quantity != 0

    @property
    def multiplier(self) -> int:
        return self.instrument.multiplier

    @property
    def symbol(self) -> str:
        return str(self.instrument)
