"""Shared fixtures: a tiny deterministic market to exercise the chain, broker and settlement."""

from __future__ import annotations

from datetime import date

import pytest

from optbt.execution.costs import CommissionModel, ExecutionCosts, SlippageModel, SpreadModel
from optbt.market.chain import ChainBuilder
from optbt.market.vol_surface import ConstantVol

SYMBOL = "TST"
TODAY = date(2024, 1, 2)
EXPIRY = date(2024, 1, 19)  # third Friday of Jan 2024
SPOT = 100.0


@pytest.fixture
def realistic_costs() -> ExecutionCosts:
    return ExecutionCosts(SpreadModel(), SlippageModel(), CommissionModel(per_contract=0.65), label="realistic")


@pytest.fixture
def frictionless_costs() -> ExecutionCosts:
    return ExecutionCosts.frictionless()


def make_builder(costs: ExecutionCosts, r: float = 0.0, expiries=(EXPIRY,)) -> ChainBuilder:
    return ChainBuilder(
        symbol=SYMBOL,
        surface=ConstantVol(level=0.25),
        expiries=list(expiries),
        r=r,
        half_spread_fn=costs.spread.half_spread,
        strike_step_pct=0.025,
        n_strikes=12,
        n_expiries=1,
    )


@pytest.fixture
def realistic_builder(realistic_costs) -> ChainBuilder:
    return make_builder(realistic_costs)


@pytest.fixture
def frictionless_builder(frictionless_costs) -> ChainBuilder:
    return make_builder(frictionless_costs)
