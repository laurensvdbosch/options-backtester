"""Configuration dataclasses for a backtest.

Kept free of behaviour so they can be serialised, copied and compared easily.
The engine turns them into concrete simulators / cost models.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import date as Date
from typing import Literal


@dataclass
class MarketConfig:
    """How the synthetic market is generated."""

    symbol: str = "SIM"
    start: Date = Date(2022, 1, 3)
    end: Date = Date(2023, 12, 29)
    s0: float = 100.0
    r: float = 0.04  # risk-free rate (continuous)
    q: float = 0.0  # dividend yield

    # --- underlying dynamics ---------------------------------------------------
    model: Literal["gbm", "heston"] = "heston"
    mu: float = 0.07  # real-world drift of the underlying
    sigma: float = 0.20  # GBM volatility
    # Heston: dv = kappa (theta - v) dt + xi sqrt(v) dW_v,  corr(dW_s, dW_v) = rho
    # Defaults satisfy the Feller condition 2*kappa*theta > xi^2 (0.16 > 0.1225) so the
    # variance process stays away from zero and implied vol does not collapse.
    v0: float = 0.04
    kappa: float = 2.0
    theta: float = 0.04
    xi: float = 0.35
    rho: float = -0.7

    # --- implied-vol surface ---------------------------------------------------
    # Implied vol is *not* the realised vol of the path. The surface's ATM level is:
    #   heston: sqrt(expected average variance to expiry) * (1 + iv_premium)
    #   gbm:    base_iv (default sigma * (1 + iv_premium)), optionally driven by an
    #           Ornstein-Uhlenbeck process in log space correlated with spot shocks.
    iv_premium: float = 0.10
    base_iv: float | None = None
    iv_process: Literal["constant", "ou"] = "ou"
    iv_ou_kappa: float = 4.0  # mean-reversion speed of log IV (per year)
    iv_ou_vol: float = 0.6  # vol-of-vol of log IV (per sqrt year)
    iv_spot_corr: float = -0.7  # correlation between spot shocks and IV shocks (leverage effect)
    term_slope: float = 0.05  # gbm only: ATM IV multiplier per unit of (sqrt(T) - sqrt(30/365))
    skew: float = -0.03  # vol points per unit of standardised moneyness (negative => puts richer)
    smile: float = 0.01  # curvature term

    # --- listed chain ----------------------------------------------------------
    strike_step_pct: float = 0.025  # strike spacing as a fraction of spot (rounded to a nice number)
    n_strikes: int = 15  # strikes listed on each side of spot
    n_expiries: int = 3  # monthly expiries listed at any time
    seed: int = 42

    def with_(self, **kwargs) -> "MarketConfig":
        return replace(self, **kwargs)


@dataclass
class ExecutionConfig:
    """Broker and transaction-cost assumptions."""

    initial_cash: float = 100_000.0
    settlement: Literal["cash", "physical"] = "physical"
    exercise_threshold: float = 0.01  # min intrinsic ($/share) for exercise/assignment at expiry
    cash_interest: bool = True  # accrue the risk-free rate on the cash balance (negative cash is charged)

    # Bid-ask spread. Full spread = max(min_abs, base_rel*extrinsic + otm_rel*(1-x)*extrinsic
    #                                       + intrinsic_rel*intrinsic), x = ATM-ness in [0,1].
    spread_base_rel: float = 0.03
    spread_otm_rel: float = 0.15
    spread_intrinsic_rel: float = 0.005
    spread_min_abs: float = 0.05
    stock_spread_bps: float = 1.0

    # Liquidity-based slippage beyond the touch.
    slippage_impact: float = 0.5
    slippage_depth: float = 50.0  # contracts absorbable at the touch for an ATM front-month option
    slippage_max_multiple: float = 2.0  # cap slippage at this many half-spreads

    # Commissions.
    commission_per_contract: float = 0.65
    commission_per_order: float = 0.0
    commission_per_share: float = 0.0
    exercise_fee: float = 0.0

    frictionless: bool = False  # if True all of the above costs are switched off

    def as_frictionless(self) -> "ExecutionConfig":
        return replace(self, frictionless=True)

    def as_realistic(self) -> "ExecutionConfig":
        return replace(self, frictionless=False)

    @property
    def label(self) -> str:
        return "frictionless" if self.frictionless else "realistic"


@dataclass
class BacktestConfig:
    market: MarketConfig = field(default_factory=MarketConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
