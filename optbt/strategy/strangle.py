"""Short strangle: sell a ~16-delta call and a ~16-delta put in the same monthly expiry.

Rules
-----
* Enter when flat, ``min_dte``-``max_dte`` days out.
* Close at ``take_profit`` of the credit, at ``stop_loss_multiple`` times
  the credit, or when ``manage_dte`` days remain (gamma risk management),
  and re-enter on the next bar.
* Lone legs left after settlement are flattened; delivered shares are sold.

This strategy is short vega and short gamma, which makes the Greek exposure
report interesting: watch delta swing as spot moves toward either strike.
"""

from __future__ import annotations

from optbt.execution.orders import Order, Side
from optbt.pricing.black_scholes import OptionType
from optbt.strategy.base import Context, Strategy


class ShortStrangle(Strategy):
    name = "short_strangle"

    def __init__(
        self,
        contracts: int = 1,
        target_delta: float = 0.16,
        min_dte: int = 30,
        max_dte: int = 45,
        take_profit: float = 0.5,
        stop_loss_multiple: float = 2.0,
        manage_dte: int = 21,
    ) -> None:
        super().__init__()
        self.contracts = contracts
        self.target_delta = target_delta
        self.min_dte = min_dte
        self.max_dte = max_dte
        self.take_profit = take_profit
        self.stop_loss_multiple = stop_loss_multiple
        self.manage_dte = manage_dte

    def on_bar(self, ctx: Context) -> list[Order]:
        orders: list[Order] = []
        shares = ctx.stock_quantity()
        if shares != 0:
            orders.append(Order(ctx.stock, Side.SELL if shares > 0 else Side.BUY, abs(shares), tag=f"{self.name}#shares"))

        legs = ctx.option_positions()
        if legs:
            calls = [p for p in legs if p.instrument.is_call]  # type: ignore[union-attr]
            puts = [p for p in legs if not p.instrument.is_call]  # type: ignore[union-attr]
            if not calls or not puts:
                return orders + [ctx.close_order(p) for p in legs]
            c, p = calls[0], puts[0]
            cq, pq = ctx.quote(c.instrument), ctx.quote(p.instrument)  # type: ignore[arg-type]
            if cq.dte <= 0:
                return orders
            credit = c.avg_price + p.avg_price
            cost_to_close = cq.ask + pq.ask
            take = cost_to_close <= (1.0 - self.take_profit) * credit
            stop = cost_to_close >= self.stop_loss_multiple * credit
            manage = cq.dte <= self.manage_dte
            if take or stop or manage:
                orders += [ctx.close_order(c), ctx.close_order(p)]
            return orders

        cq = ctx.chain.find(OptionType.CALL, target_delta=self.target_delta, min_dte=self.min_dte, max_dte=self.max_dte)
        if cq is None:
            return orders
        pq = ctx.chain.find(OptionType.PUT, target_delta=self.target_delta, expiry=cq.expiry)
        if pq is None or pq.strike >= cq.strike or cq.bid <= 0.0 or pq.bid <= 0.0:
            return orders
        tag = self.next_tag()
        orders.append(Order(cq.contract, Side.SELL, self.contracts, tag=tag))
        orders.append(Order(pq.contract, Side.SELL, self.contracts, tag=tag))
        return orders

    def describe(self) -> str:
        return (
            f"Short strangle: {self.contracts}x {self.target_delta:.2f}-delta call & put, {self.min_dte}-{self.max_dte} DTE, "
            f"TP {self.take_profit:.0%}, SL {self.stop_loss_multiple:.1f}x, manage at {self.manage_dte} DTE"
        )
