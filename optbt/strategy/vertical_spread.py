"""Bull put spread (short put vertical): sell a ~30-delta put, buy a ~15-delta put, same expiry.

Rules
-----
* Enter when flat: pick the earliest monthly expiry ``min_dte``-``max_dte``
  days out, short the ``short_delta`` put and long the ``long_delta`` put.
* Close both legs when the cost to close (short ask - long bid) falls to
  ``(1 - take_profit)`` of the credit received, or rises to
  ``stop_loss_multiple`` times the credit.
* Otherwise hold to expiry and let the broker settle by moneyness.
* If a single leg is left over (one leg settled, the other did not) it is
  flattened, and any shares delivered by physical settlement are sold.
"""

from __future__ import annotations

from optbt.execution.orders import Order, Side
from optbt.pricing.black_scholes import OptionType
from optbt.strategy.base import Context, Strategy


class BullPutSpread(Strategy):
    name = "bull_put_spread"

    def __init__(
        self,
        contracts: int = 1,
        short_delta: float = 0.30,
        long_delta: float = 0.15,
        min_dte: int = 30,
        max_dte: int = 45,
        take_profit: float = 0.5,
        stop_loss_multiple: float = 2.0,
    ) -> None:
        super().__init__()
        self.contracts = contracts
        self.short_delta = short_delta
        self.long_delta = long_delta
        self.min_dte = min_dte
        self.max_dte = max_dte
        self.take_profit = take_profit
        self.stop_loss_multiple = stop_loss_multiple

    def on_bar(self, ctx: Context) -> list[Order]:
        orders: list[Order] = []
        shares = ctx.stock_quantity()
        if shares != 0:  # only happens with physical settlement of a lone ITM leg
            orders.append(Order(ctx.stock, Side.SELL if shares > 0 else Side.BUY, abs(shares), tag=f"{self.name}#shares"))

        legs = ctx.option_positions()
        if legs:
            shorts = [p for p in legs if p.quantity < 0]
            longs = [p for p in legs if p.quantity > 0]
            if not shorts or not longs:
                return orders + [ctx.close_order(p) for p in legs]
            s, l = shorts[0], longs[0]
            sq, lq = ctx.quote(s.instrument), ctx.quote(l.instrument)  # type: ignore[arg-type]
            if sq.dte <= 0:
                return orders  # let settlement handle it today
            credit = s.avg_price - l.avg_price
            cost_to_close = sq.ask - lq.bid
            if cost_to_close <= (1.0 - self.take_profit) * credit or cost_to_close >= self.stop_loss_multiple * credit:
                orders += [ctx.close_order(s), ctx.close_order(l)]
            return orders

        sq = ctx.chain.find(OptionType.PUT, target_delta=self.short_delta, min_dte=self.min_dte, max_dte=self.max_dte)
        if sq is None:
            return orders
        lq = ctx.chain.find(OptionType.PUT, target_delta=self.long_delta, expiry=sq.expiry)
        if lq is None or lq.strike >= sq.strike or sq.bid - lq.ask <= 0.0:
            return orders
        tag = self.next_tag()
        orders.append(Order(sq.contract, Side.SELL, self.contracts, tag=tag))
        orders.append(Order(lq.contract, Side.BUY, self.contracts, tag=tag))
        return orders

    def describe(self) -> str:
        return (
            f"Bull put spread: {self.contracts}x short {self.short_delta:.2f}d / long {self.long_delta:.2f}d put, "
            f"{self.min_dte}-{self.max_dte} DTE, TP {self.take_profit:.0%}, SL {self.stop_loss_multiple:.1f}x credit"
        )
