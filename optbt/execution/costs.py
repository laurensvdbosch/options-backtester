"""Transaction-cost models: bid-ask spread, liquidity-based slippage, commissions.

Design notes
------------
* **Spread** is modelled on the option's *extrinsic* value plus a small
  fraction of intrinsic, with an absolute tick floor. That reproduces the
  three stylised facts we care about:

    - ATM options have the tightest *relative* spreads;
    - far-OTM options are cheap, so the tick floor makes their relative
      spread very wide (a $0.05 spread on a $0.15 option is 33%);
    - deep-ITM options have wide *absolute* but narrow *relative* spreads.

  ``x = 2 * min(|delta|, 1 - |delta|)`` is used as the "ATM-ness" measure:
  1.0 at the money, 0.0 deep in/out of the money.

* **Slippage** is the additional price concession beyond the touch. It grows
  with the square root of order size relative to a liquidity-scaled depth,
  so trading 10 contracts of a deep-OTM back-month option costs far more
  than 10 ATM front-month contracts. It is expressed in half-spreads and
  capped.

* **Commissions** are per contract plus an optional per-order fee; stock
  trades can carry a per-share fee. An optional exercise/assignment fee
  applies at settlement.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from optbt.config import ExecutionConfig


@dataclass
class SpreadModel:
    base_rel: float = 0.03
    otm_rel: float = 0.15
    intrinsic_rel: float = 0.005
    min_abs: float = 0.05
    stock_bps: float = 1.0

    @staticmethod
    def atmness(abs_delta: float) -> float:
        x = 2.0 * min(abs_delta, 1.0 - abs_delta)
        return max(0.0, min(1.0, x))

    def full_spread(self, mid: float, intrinsic: float, abs_delta: float, dte: int) -> float:
        extrinsic = max(mid - intrinsic, 0.0)
        x = self.atmness(abs_delta)
        rel = self.base_rel + self.otm_rel * (1.0 - x)
        spread = rel * extrinsic + self.intrinsic_rel * intrinsic
        return max(self.min_abs, spread)

    def half_spread(self, mid: float, intrinsic: float, abs_delta: float, dte: int) -> float:
        return 0.5 * self.full_spread(mid, intrinsic, abs_delta, dte)

    def stock_half_spread(self, price: float) -> float:
        return 0.5 * price * self.stock_bps / 10_000.0

    @classmethod
    def zero(cls) -> "SpreadModel":
        return cls(base_rel=0.0, otm_rel=0.0, intrinsic_rel=0.0, min_abs=0.0, stock_bps=0.0)


@dataclass
class SlippageModel:
    impact: float = 0.5
    depth: float = 50.0
    max_multiple: float = 2.0

    def slippage(self, half_spread: float, quantity: int, liquidity: float) -> float:
        """Extra $/share paid beyond the touch for ``quantity`` contracts."""
        if self.impact <= 0.0 or half_spread <= 0.0 or quantity <= 0:
            return 0.0
        ratio = quantity / (self.depth * max(liquidity, 1e-3))
        return min(half_spread * self.impact * math.sqrt(ratio), half_spread * self.max_multiple)

    @classmethod
    def zero(cls) -> "SlippageModel":
        return cls(impact=0.0)


@dataclass
class CommissionModel:
    per_contract: float = 0.65
    per_order: float = 0.0
    per_share: float = 0.0
    exercise_fee: float = 0.0

    def option(self, contracts: int) -> float:
        return self.per_contract * abs(contracts) + self.per_order

    def stock(self, shares: int) -> float:
        return self.per_share * abs(shares) + self.per_order

    def exercise(self, contracts: int) -> float:
        return self.exercise_fee * abs(contracts)

    @classmethod
    def zero(cls) -> "CommissionModel":
        return cls(0.0, 0.0, 0.0, 0.0)


@dataclass
class ExecutionCosts:
    spread: SpreadModel
    slippage: SlippageModel
    commission: CommissionModel
    label: str = "realistic"

    @classmethod
    def frictionless(cls) -> "ExecutionCosts":
        return cls(SpreadModel.zero(), SlippageModel.zero(), CommissionModel.zero(), label="frictionless")

    @classmethod
    def from_config(cls, cfg: ExecutionConfig) -> "ExecutionCosts":
        if cfg.frictionless:
            return cls.frictionless()
        return cls(
            spread=SpreadModel(
                base_rel=cfg.spread_base_rel,
                otm_rel=cfg.spread_otm_rel,
                intrinsic_rel=cfg.spread_intrinsic_rel,
                min_abs=cfg.spread_min_abs,
                stock_bps=cfg.stock_spread_bps,
            ),
            slippage=SlippageModel(
                impact=cfg.slippage_impact,
                depth=cfg.slippage_depth,
                max_multiple=cfg.slippage_max_multiple,
            ),
            commission=CommissionModel(
                per_contract=cfg.commission_per_contract,
                per_order=cfg.commission_per_order,
                per_share=cfg.commission_per_share,
                exercise_fee=cfg.exercise_fee,
            ),
            label="realistic",
        )
