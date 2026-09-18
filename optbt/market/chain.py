"""Synthetic option-chain construction.

Each trading day the ``ChainBuilder`` lists a grid of strikes around spot for
the next ``n_expiries`` monthly expiries, prices every contract with
Black-Scholes at the surface's implied vol, attaches Greeks, a liquidity score
and executable bid/ask prices (via an injected half-spread function so the
market layer does not depend on the execution layer).

Contracts that are held but no longer on the listed grid (e.g. the spot has
drifted away) are priced on demand through ``OptionChain.quote``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date as Date
from typing import Callable

from optbt.instruments import OptionContract, Quote
from optbt.market.vol_surface import VolSurface
from optbt.pricing.black_scholes import OptionType, bs_greeks, bs_price

# half_spread(mid, intrinsic, abs_delta, dte) -> half spread in $/share
HalfSpreadFn = Callable[[float, float, float, int], float]

_NICE_STEPS = (0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 100.0)


def nice_strike_step(spot: float, step_pct: float) -> float:
    target = spot * step_pct
    return min(_NICE_STEPS, key=lambda s: abs(math.log(s / target)))


def liquidity_score(abs_delta: float, dte: int) -> float:
    """Relative market depth in (0, 1]: 1 for ATM front-month, decaying for far OTM/ITM and far-dated.

    ``x`` = 2 * min(|delta|, 1 - |delta|) is 1 at the money and 0 deep in/out of the money.
    """
    x = 2.0 * min(abs_delta, 1.0 - abs_delta)
    x = max(0.0, min(1.0, x))
    moneyness_part = 0.05 + 0.95 * x**1.5
    tenor_part = math.exp(-max(dte - 45, 0) / 180.0)
    return moneyness_part * tenor_part


@dataclass
class OptionChain:
    date: Date
    t_index: int
    spot: float
    atm_iv: float
    quotes: list[Quote] = field(default_factory=list)
    _pricer: Callable[[OptionContract], Quote] | None = field(default=None, repr=False)
    _by_contract: dict[OptionContract, Quote] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        for q in self.quotes:
            self._by_contract[q.contract] = q

    @property
    def expiries(self) -> list[Date]:
        return sorted({q.expiry for q in self.quotes})

    def quote(self, contract: OptionContract) -> Quote:
        q = self._by_contract.get(contract)
        if q is None:
            if self._pricer is None:
                raise KeyError(f"{contract} not in chain and no on-demand pricer attached")
            q = self._pricer(contract)
            self._by_contract[contract] = q
        return q

    def strikes(self, expiry: Date, kind: OptionType | None = None) -> list[float]:
        return sorted({q.strike for q in self.quotes if q.expiry == expiry and (kind is None or q.kind is kind)})

    def select_expiry(self, min_dte: int, max_dte: int) -> Date | None:
        """Earliest listed expiry whose days-to-expiry lies in [min_dte, max_dte]."""
        for e in self.expiries:
            dte = (e - self.date).days
            if min_dte <= dte <= max_dte:
                return e
        return None

    def find(
        self,
        kind: OptionType,
        target_delta: float | None = None,
        strike: float | None = None,
        min_dte: int | None = None,
        max_dte: int | None = None,
        expiry: Date | None = None,
    ) -> Quote | None:
        """Locate a listed contract by expiry window and either target |delta| or strike.

        When ``expiry`` is omitted the earliest expiry inside ``[min_dte, max_dte]`` is used.
        Returns ``None`` if nothing matches.
        """
        if expiry is None:
            if min_dte is None or max_dte is None:
                raise ValueError("find() needs either expiry or both min_dte and max_dte")
            expiry = self.select_expiry(min_dte, max_dte)
            if expiry is None:
                return None
        candidates = [q for q in self.quotes if q.kind is kind and q.expiry == expiry]
        if not candidates:
            return None
        if strike is not None:
            return min(candidates, key=lambda q: abs(q.strike - strike))
        if target_delta is not None:
            tgt = abs(target_delta)
            return min(candidates, key=lambda q: abs(abs(q.delta) - tgt))
        raise ValueError("find() needs target_delta or strike")


class ChainBuilder:
    def __init__(
        self,
        symbol: str,
        surface: VolSurface,
        expiries: list[Date],
        r: float,
        q: float = 0.0,
        half_spread_fn: HalfSpreadFn | None = None,
        strike_step_pct: float = 0.025,
        n_strikes: int = 15,
        n_expiries: int = 3,
        multiplier: int = 100,
    ) -> None:
        self.symbol = symbol
        self.surface = surface
        self.expiries = sorted(expiries)
        self.r = r
        self.q = q
        self.half_spread_fn = half_spread_fn or (lambda mid, intr, d, dte: 0.0)
        self.strike_step_pct = strike_step_pct
        self.n_strikes = n_strikes
        self.n_expiries = n_expiries
        self.multiplier = multiplier

    # -- pricing ------------------------------------------------------------------
    def price_contract(self, t_index: int, date: Date, spot: float, contract: OptionContract) -> Quote:
        dte = contract.dte(date)
        tte = max(dte, 0) / 365.0
        iv = self.surface.iv(t_index, spot, contract.strike, tte)
        mid = bs_price(spot, contract.strike, tte, self.r, iv, contract.kind, self.q)
        greeks = bs_greeks(spot, contract.strike, tte, self.r, iv, contract.kind, self.q)
        intr = contract.intrinsic(spot)
        abs_delta = abs(greeks.delta)
        half = self.half_spread_fn(mid, intr, abs_delta, dte)
        bid = max(mid - half, 0.0)
        ask = mid + half
        return Quote(
            contract=contract,
            date=date,
            spot=spot,
            mid=mid,
            bid=bid,
            ask=ask,
            iv=iv,
            tte=tte,
            greeks=greeks,
            liquidity=liquidity_score(abs_delta, max(dte, 0)),
        )

    # -- listing --------------------------------------------------------------------
    def listed_expiries(self, date: Date) -> list[Date]:
        return [e for e in self.expiries if e >= date][: self.n_expiries]

    def listed_strikes(self, spot: float) -> list[float]:
        step = nice_strike_step(spot, self.strike_step_pct)
        centre = round(spot / step) * step
        return [round(centre + k * step, 4) for k in range(-self.n_strikes, self.n_strikes + 1) if centre + k * step > 0]

    def build(self, t_index: int, date: Date, spot: float) -> OptionChain:
        quotes: list[Quote] = []
        for expiry in self.listed_expiries(date):
            for strike in self.listed_strikes(spot):
                for kind in (OptionType.CALL, OptionType.PUT):
                    c = OptionContract(self.symbol, kind, strike, expiry, self.multiplier)
                    quotes.append(self.price_contract(t_index, date, spot, c))
        chain = OptionChain(
            date=date,
            t_index=t_index,
            spot=spot,
            atm_iv=self.surface.atm(t_index, 30.0 / 365.0),
            quotes=quotes,
            _pricer=lambda c, _t=t_index, _d=date, _s=spot: self.price_contract(_t, _d, _s, c),
        )
        return chain
