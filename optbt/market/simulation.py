"""Underlying price simulators.

Both simulators step once per **trading day** with dt = 1/252 and return the
standard-normal shocks they used, so a correlated implied-vol process can be
layered on top (see ``vol_surface``).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date as Date

import numpy as np

TRADING_DAYS_PER_YEAR = 252
DT = 1.0 / TRADING_DAYS_PER_YEAR


@dataclass
class PathResult:
    dates: list[Date]
    spot: np.ndarray  # shape (n,)
    shocks: np.ndarray  # standard normals driving spot, shape (n,), shocks[0] = 0
    variance: np.ndarray | None = None  # instantaneous variance (Heston), shape (n,)

    @property
    def log_returns(self) -> np.ndarray:
        return np.diff(np.log(self.spot))

    @property
    def realized_vol(self) -> float:
        return float(np.std(self.log_returns, ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))


class PriceSimulator(ABC):
    @abstractmethod
    def simulate(self, dates: list[Date], s0: float, rng: np.random.Generator) -> PathResult: ...


@dataclass
class GBMSimulator(PriceSimulator):
    """Geometric Brownian motion: dS/S = mu dt + sigma dW."""

    mu: float = 0.07
    sigma: float = 0.20

    def simulate(self, dates: list[Date], s0: float, rng: np.random.Generator) -> PathResult:
        n = len(dates)
        z = np.zeros(n)
        z[1:] = rng.standard_normal(n - 1)
        drift = (self.mu - 0.5 * self.sigma**2) * DT
        increments = drift + self.sigma * np.sqrt(DT) * z[1:]
        log_path = np.concatenate([[np.log(s0)], np.log(s0) + np.cumsum(increments)])
        spot = np.exp(log_path)
        spot[0] = s0  # avoid exp(log(s0)) rounding
        return PathResult(dates=dates, spot=spot, shocks=z)


@dataclass
class HestonSimulator(PriceSimulator):
    """Heston stochastic volatility, full-truncation Euler scheme.

        dS/S = mu dt + sqrt(v) dW_s
        dv   = kappa (theta - v) dt + xi sqrt(v) dW_v,   corr(dW_s, dW_v) = rho
    """

    mu: float = 0.07
    v0: float = 0.04
    kappa: float = 2.0
    theta: float = 0.04
    xi: float = 0.5
    rho: float = -0.7

    def simulate(self, dates: list[Date], s0: float, rng: np.random.Generator) -> PathResult:
        n = len(dates)
        z_s = np.zeros(n)
        z_v = np.zeros(n)
        z1 = rng.standard_normal(n - 1)
        z2 = rng.standard_normal(n - 1)
        z_s[1:] = z1
        z_v[1:] = self.rho * z1 + np.sqrt(1.0 - self.rho**2) * z2

        spot = np.empty(n)
        var = np.empty(n)
        spot[0] = s0
        var[0] = self.v0
        sqrt_dt = np.sqrt(DT)
        for i in range(1, n):
            v_pos = max(var[i - 1], 0.0)
            spot[i] = spot[i - 1] * np.exp((self.mu - 0.5 * v_pos) * DT + np.sqrt(v_pos) * sqrt_dt * z_s[i])
            var[i] = var[i - 1] + self.kappa * (self.theta - v_pos) * DT + self.xi * np.sqrt(v_pos) * sqrt_dt * z_v[i]
        var = np.maximum(var, 0.0)
        return PathResult(dates=dates, spot=spot, shocks=z_s, variance=var)
