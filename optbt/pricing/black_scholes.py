"""Black-Scholes-Merton pricing and Greeks for European options.

All functions are pure and scalar. Conventions:

* ``S``     spot price
* ``K``     strike
* ``T``     time to expiry in **years** (act/365 in the engine)
* ``r``     continuously-compounded risk-free rate
* ``sigma`` implied volatility (annualised, e.g. 0.20)
* ``q``     continuous dividend yield

Greeks are returned in "per-share" units:

* delta  - dPrice/dS
* gamma  - d2Price/dS2
* theta  - dPrice/dt **per calendar day** (already divided by 365)
* vega   - dPrice/dsigma **per 1 vol point** (already divided by 100)
* rho    - dPrice/dr per 1 percentage point of rate

At ``T <= 0`` the functions collapse to intrinsic value / step-function delta
so that the engine can price a chain on expiry day without special casing.

The normal CDF/PDF are implemented with ``math.erf`` rather than SciPy: the
chain builder prices a few hundred contracts per simulated day and scalar
``math`` calls are roughly an order of magnitude faster than SciPy's
array-oriented ``norm.cdf`` for scalar inputs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

_SQRT2 = math.sqrt(2.0)
_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)
_EPS_T = 1e-10


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / _SQRT2))


def norm_pdf(x: float) -> float:
    return _INV_SQRT_2PI * math.exp(-0.5 * x * x)


class OptionType(str, Enum):
    CALL = "call"
    PUT = "put"

    @property
    def sign(self) -> int:
        """+1 for calls, -1 for puts (so payoff = max(sign * (S - K), 0))."""
        return 1 if self is OptionType.CALL else -1


@dataclass(frozen=True)
class Greeks:
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float

    @staticmethod
    def zero() -> "Greeks":
        return Greeks(0.0, 0.0, 0.0, 0.0, 0.0)


def _d1_d2(S: float, K: float, T: float, r: float, sigma: float, q: float):
    sqrt_t = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    return d1, d2


def intrinsic(S: float, K: float, kind: OptionType) -> float:
    return max(kind.sign * (S - K), 0.0)


def bs_price(S: float, K: float, T: float, r: float, sigma: float, kind: OptionType, q: float = 0.0) -> float:
    """Black-Scholes price of a European call or put."""
    if T <= _EPS_T or sigma <= 0.0:
        # Degenerate limit: discounted forward intrinsic. At T ~ 0 this is plain intrinsic.
        fwd = S * math.exp((r - q) * T)
        return math.exp(-r * T) * max(kind.sign * (fwd - K), 0.0)
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    disc_r = math.exp(-r * T)
    disc_q = math.exp(-q * T)
    if kind is OptionType.CALL:
        return S * disc_q * norm_cdf(d1) - K * disc_r * norm_cdf(d2)
    return K * disc_r * norm_cdf(-d2) - S * disc_q * norm_cdf(-d1)


def bs_delta(S: float, K: float, T: float, r: float, sigma: float, kind: OptionType, q: float = 0.0) -> float:
    if T <= _EPS_T or sigma <= 0.0:
        if kind is OptionType.CALL:
            return 1.0 if S > K else 0.0
        return -1.0 if S < K else 0.0
    d1, _ = _d1_d2(S, K, T, r, sigma, q)
    disc_q = math.exp(-q * T)
    if kind is OptionType.CALL:
        return disc_q * norm_cdf(d1)
    return -disc_q * norm_cdf(-d1)


def bs_gamma(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    if T <= _EPS_T or sigma <= 0.0:
        return 0.0
    d1, _ = _d1_d2(S, K, T, r, sigma, q)
    return math.exp(-q * T) * norm_pdf(d1) / (S * sigma * math.sqrt(T))


def bs_vega(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    """Vega per 1 vol point (dPrice/dsigma divided by 100)."""
    if T <= _EPS_T or sigma <= 0.0:
        return 0.0
    d1, _ = _d1_d2(S, K, T, r, sigma, q)
    return S * math.exp(-q * T) * norm_pdf(d1) * math.sqrt(T) / 100.0


def bs_theta(S: float, K: float, T: float, r: float, sigma: float, kind: OptionType, q: float = 0.0) -> float:
    """Theta per calendar day (dPrice/dt divided by 365). Negative for long options normally."""
    if T <= _EPS_T or sigma <= 0.0:
        return 0.0
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    sqrt_t = math.sqrt(T)
    disc_r = math.exp(-r * T)
    disc_q = math.exp(-q * T)
    common = -S * disc_q * norm_pdf(d1) * sigma / (2.0 * sqrt_t)
    if kind is OptionType.CALL:
        annual = common - r * K * disc_r * norm_cdf(d2) + q * S * disc_q * norm_cdf(d1)
    else:
        annual = common + r * K * disc_r * norm_cdf(-d2) - q * S * disc_q * norm_cdf(-d1)
    return annual / 365.0


def bs_rho(S: float, K: float, T: float, r: float, sigma: float, kind: OptionType, q: float = 0.0) -> float:
    """Rho per 1 percentage point move in r."""
    if T <= _EPS_T or sigma <= 0.0:
        return 0.0
    _, d2 = _d1_d2(S, K, T, r, sigma, q)
    disc_r = math.exp(-r * T)
    if kind is OptionType.CALL:
        return K * T * disc_r * norm_cdf(d2) / 100.0
    return -K * T * disc_r * norm_cdf(-d2) / 100.0


def bs_greeks(S: float, K: float, T: float, r: float, sigma: float, kind: OptionType, q: float = 0.0) -> Greeks:
    return Greeks(
        delta=bs_delta(S, K, T, r, sigma, kind, q),
        gamma=bs_gamma(S, K, T, r, sigma, q),
        theta=bs_theta(S, K, T, r, sigma, kind, q),
        vega=bs_vega(S, K, T, r, sigma, q),
        rho=bs_rho(S, K, T, r, sigma, kind, q),
    )


def implied_vol(
    price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    kind: OptionType,
    q: float = 0.0,
    tol: float = 1e-12,
    max_iter: int = 200,
) -> float:
    """Invert Black-Scholes for sigma using Newton steps with a bisection safeguard.

    ``tol`` is on price. For very low-vega contracts (deep ITM/OTM, low vol) a
    price tolerance of 1e-12 still resolves sigma to roughly 1e-6.
    Raises ``ValueError`` when the price is outside the no-arbitrage bounds.
    """
    if T <= _EPS_T:
        raise ValueError("Cannot imply vol at zero time to expiry")
    disc_r = math.exp(-r * T)
    disc_q = math.exp(-q * T)
    lower = max(kind.sign * (S * disc_q - K * disc_r), 0.0)
    upper = S * disc_q if kind is OptionType.CALL else K * disc_r
    if price < lower - 1e-12 or price > upper + 1e-12:
        raise ValueError(f"price {price} outside no-arbitrage bounds [{lower:.6f}, {upper:.6f}]")
    if price - lower < 1e-12:
        return 0.0

    lo, hi = 1e-4, 5.0
    sigma = 0.3
    for _ in range(max_iter):
        diff = bs_price(S, K, T, r, sigma, kind, q) - price
        if abs(diff) < tol:
            return sigma
        if diff > 0:
            hi = min(hi, sigma)
        else:
            lo = max(lo, sigma)
        v = bs_vega(S, K, T, r, sigma, q) * 100.0
        if v > 1e-10:
            step = sigma - diff / v
            if lo < step < hi:
                sigma = step
                continue
        sigma = 0.5 * (lo + hi)
    return sigma
