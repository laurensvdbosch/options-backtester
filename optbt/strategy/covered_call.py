"""Covered call: long 100 shares per lot, short one ~30-delta monthly call against them.

Rules
-----
* Buy shares on the first bar (and re-buy whenever they were called away).
* If no short call is open, sell ``lots`` calls at ``target_delta`` with
  ``min_dte``-``max_dte`` days to expiry.
* Optionally buy the call back once it has decayed to ``(1 - take_profit)``
  of the premium received and immediately write a new one.
* Otherwise hold to expiry; with physical settlement an ITM call means the
  shares are assigned away at the strike.
"""

from __future__ import annotations

from optbt.execution.orders import Order, Side
from optbt.pricing.black_scholes import OptionType
from optbt.strategy.base import Context, Strategy


class CoveredCall(Strategy):
    name = "covered_call"

    def __init__(
        self,
        lots: int = 1,
        target_delta: float = 0.30,
        min_dte: int = 25,
        max_dte: int = 45,
        take_profit: float | None = 0.5,
    ) -> None:
        super().__init__()
        self.lots = lots
        self.target_delta = target_delta
        self.min_dte = min_dte
        self.max_dte = max_dte
        self.take_profit = take_profit

    def on_bar(self, ctx: Context) -> list[Order]:
        orders: list[Order] = []
        need = self.lots * 100
        shares = ctx.stock_quantity()
        if shares < need:
            orders.append(Order(ctx.stock, Side.BUY, need - shares, tag=f"{self.name}#shares"))

        short_calls = [p for p in ctx.option_positions() if p.quantity < 0 and p.instrument.is_call]  # type: ignore[union-attr]
        if short_calls:
            pos = short_calls[0]
            q = ctx.quote(pos.instrument)  # type: ignore[arg-type]
            hit_target = self.take_profit is not None and q.ask <= (1.0 - self.take_profit) * pos.avg_price
            if q.dte >= 1 and hit_target:
                orders.append(ctx.close_order(pos))
            else:
                return orders

        q = ctx.chain.find(OptionType.CALL, target_delta=self.target_delta, min_dte=self.min_dte, max_dte=self.max_dte)
        if q is not None and q.bid > 0.0:
            orders.append(Order(q.contract, Side.SELL, self.lots, tag=self.next_tag()))
        return orders

    def describe(self) -> str:
        return f"Covered call: {self.lots} lot(s), sell {self.target_delta:.2f}-delta call, {self.min_dte}-{self.max_dte} DTE"
