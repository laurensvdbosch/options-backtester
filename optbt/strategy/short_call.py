"""Short call: sell one ~20-delta monthly call, buy it back at 50% of the premium.

Rules
-----
* When no call is open, sell ``contracts`` calls at ``target_delta`` with
  ``min_dte``-``max_dte`` days to expiry (the earliest listed monthly in that window).
* Buy it back once its ask has fallen to ``(1 - take_profit)`` of the credit
  received, then re-enter on the next bar.
* Otherwise hold to expiry and let the broker settle by moneyness.

This is an *uncovered* short call: risk is open-ended on the upside. Pair it
with ``CoveredCall(target_delta=0.20)`` to see the same rule with shares behind it.
"""

from __future__ import annotations

from optbt.execution.orders import Order, Side
from optbt.pricing import OptionType
from optbt.strategy.base import Context, Strategy


class ShortCall(Strategy):
    name = "short_call"

    def __init__(
        self,
        contracts: int = 1,
        target_delta: float = 0.20,
        min_dte: int = 25,
        max_dte: int = 45,
        take_profit: float = 0.5,
    ) -> None:
        super().__init__()
        self.contracts = contracts
        self.target_delta = target_delta
        self.min_dte = min_dte
        self.max_dte = max_dte
        self.take_profit = take_profit

    def on_bar(self, ctx: Context) -> list[Order]:
        orders: list[Order] = []
        shares = ctx.stock_quantity()
        if shares != 0:  # only with physical settlement: flatten assigned shares
            orders.append(Order(ctx.stock, Side.SELL if shares > 0 else Side.BUY, abs(shares), tag=f"{self.name}#shares"))

        open_calls = [p for p in ctx.option_positions() if p.quantity < 0]
        if open_calls:
            pos = open_calls[0]
            q = ctx.quote(pos.instrument)  # type: ignore[arg-type]
            if q.dte >= 1 and q.ask <= (1.0 - self.take_profit) * pos.avg_price:
                orders.append(ctx.close_order(pos))
            return orders

        q = ctx.chain.find(OptionType.CALL, target_delta=self.target_delta, min_dte=self.min_dte, max_dte=self.max_dte)
        if q is not None and q.bid > 0.0:
            orders.append(Order(q.contract, Side.SELL, self.contracts, tag=self.next_tag()))
        return orders

    def describe(self) -> str:
        return (
            f"Short call: {self.contracts}x sell {self.target_delta:.2f}-delta call, {self.min_dte}-{self.max_dte} DTE, "
            f"buy back at {self.take_profit:.0%} profit"
        )
