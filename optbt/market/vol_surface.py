"""Implied-volatility surfaces.

A surface answers ``iv(t_index, spot, strike, tte)``. Every surface here is a
parametric "ATM level + skew + smile" model:

    m   = ln(K/S) / (atm * sqrt(T))          standardised moneyness (clipped to +/-5)
    iv  = atm(t, T) + skew * m + smile * m^2

Only the ATM level differs between implementations:

* ``ConstantVol``     fixed level with an optional term-structure slope
* ``SeriesVol``       a pre-simulated time series of ATM levels (e.g. an OU process)
* ``HestonImpliedVol`` sqrt(expected average variance to T) * (1 + premium) from a
                      Heston variance path, which yields a natural term structure
                      and IV that rises when spot sells off (via rho < 0)

This is the modelling choice that replaces historical option chains: the
underlying path is simulated, and each contract is priced with Black-Scholes at
the surface's vol. See the README for the rationale and limitations.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from optbt.market.simulation import DT

_T_REF = 30.0 / 365.0
_MIN_IV, _MAX_IV = 0.03, 3.0


@dataclass
class VolSurface(ABC):
    skew: float = 0.0
    smile: float = 0.0

    @abstractmethod
    def atm(self, t_index: int, tte: float) -> float:
        """ATM implied vol at day ``t_index`` for an option with ``tte`` years to expiry."""

    def iv(self, t_index: int, spot: float, strike: float, tte: float) -> float:
        atm = self.atm(t_index, tte)
        if tte <= 0.0:
            return atm
        m = math.log(strike / spot) / (atm * math.sqrt(tte))
        m = max(-5.0, min(5.0, m))
        iv = atm + self.skew * m + self.smile * m * m
        return max(_MIN_IV, min(_MAX_IV, iv))


@dataclass
class ConstantVol(VolSurface):
    level: float = 0.20
    term_slope: float = 0.0

    def atm(self, t_index: int, tte: float) -> float:
        return self.level * (1.0 + self.term_slope * (math.sqrt(max(tte, 0.0)) - math.sqrt(_T_REF)))


@dataclass
class SeriesVol(VolSurface):
    levels: np.ndarray = None  # type: ignore[assignment]
    term_slope: float = 0.0

    def atm(self, t_index: int, tte: float) -> float:
        base = float(self.levels[t_index])
        return base * (1.0 + self.term_slope * (math.sqrt(max(tte, 0.0)) - math.sqrt(_T_REF)))


@dataclass
class HestonImpliedVol(VolSurface):
    variance: np.ndarray = None  # type: ignore[assignment]
    kappa: float = 2.0
    theta: float = 0.04
    premium: float = 0.10

    def expected_avg_variance(self, v_t: float, tte: float) -> float:
        if tte <= 1e-8:
            return v_t
        decay = (1.0 - math.exp(-self.kappa * tte)) / (self.kappa * tte)
        return self.theta + (v_t - self.theta) * decay

    def atm(self, t_index: int, tte: float) -> float:
        v_t = float(self.variance[t_index])
        var_bar = max(self.expected_avg_variance(v_t, tte), 1e-6)
        return math.sqrt(var_bar) * (1.0 + self.premium)


def simulate_ou_log_iv(
    n: int,
    level: float,
    kappa: float,
    vol: float,
    spot_shocks: np.ndarray,
    corr: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Mean-reverting log-IV process correlated with the spot shocks (leverage effect).

        d ln(iv) = kappa (ln(level) - ln(iv)) dt + vol dW_iv,  corr(dW_iv, dW_s) = corr
    """
    z_ind = rng.standard_normal(n)
    z_iv = corr * spot_shocks + math.sqrt(1.0 - corr**2) * z_ind
    log_level = math.log(level)
    x = np.empty(n)
    x[0] = log_level
    for i in range(1, n):
        x[i] = x[i - 1] + kappa * (log_level - x[i - 1]) * DT + vol * math.sqrt(DT) * z_iv[i]
    return np.exp(x)
