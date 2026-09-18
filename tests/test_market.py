import math
from datetime import date

import numpy as np
import pytest

from optbt.market.calendar import monthly_expiries, third_friday, trading_days
from optbt.market.chain import liquidity_score, nice_strike_step
from optbt.market.simulation import GBMSimulator, HestonSimulator
from optbt.market.vol_surface import ConstantVol, HestonImpliedVol, simulate_ou_log_iv
from optbt.pricing import OptionType
from tests.conftest import EXPIRY, SPOT, TODAY


def test_third_friday_and_expiry_listing():
    assert third_friday(2024, 1) == date(2024, 1, 19)
    assert third_friday(2024, 9) == date(2024, 9, 20)
    exps = monthly_expiries(date(2024, 1, 2), date(2024, 3, 31), months_ahead=2)
    assert exps[:3] == [date(2024, 1, 19), date(2024, 2, 16), date(2024, 3, 15)]
    assert exps[-1] == date(2024, 5, 17)


def test_trading_days_are_weekdays():
    days = trading_days(date(2024, 1, 1), date(2024, 1, 14))
    assert all(d.weekday() < 5 for d in days)
    assert len(days) == 10


def test_gbm_is_reproducible_and_has_expected_scale():
    dates = trading_days(date(2020, 1, 1), date(2029, 12, 31))
    a = GBMSimulator(mu=0.05, sigma=0.20).simulate(dates, 100.0, np.random.default_rng(1))
    b = GBMSimulator(mu=0.05, sigma=0.20).simulate(dates, 100.0, np.random.default_rng(1))
    assert np.array_equal(a.spot, b.spot)
    assert a.spot[0] == 100.0
    assert abs(a.realized_vol - 0.20) < 0.03


def test_heston_variance_nonnegative_and_mean_reverting():
    dates = trading_days(date(2020, 1, 1), date(2029, 12, 31))
    p = HestonSimulator(v0=0.09, kappa=3.0, theta=0.04, xi=0.4, rho=-0.7).simulate(dates, 100.0, np.random.default_rng(3))
    assert (p.variance >= 0).all()
    assert abs(p.variance[len(p.variance) // 2 :].mean() - 0.04) < 0.02


def test_ou_log_iv_reverts_to_level():
    n = 252 * 20
    shocks = np.random.default_rng(0).standard_normal(n)
    iv = simulate_ou_log_iv(n, 0.22, kappa=4.0, vol=0.6, spot_shocks=shocks, corr=-0.7, rng=np.random.default_rng(1))
    assert (iv > 0).all()
    assert abs(np.median(iv) - 0.22) < 0.03


def test_skew_makes_puts_richer():
    s = ConstantVol(level=0.20, skew=-0.03, smile=0.01)
    t = 30 / 365
    assert s.iv(0, 100.0, 90.0, t) > s.iv(0, 100.0, 100.0, t) > s.iv(0, 100.0, 110.0, t)
    assert s.iv(0, 100.0, 100.0, t) == pytest.approx(0.20)


def test_heston_surface_term_structure():
    var = np.array([0.09])  # spot variance well above theta -> downward sloping term structure
    s = HestonImpliedVol(variance=var, kappa=2.0, theta=0.04, premium=0.0)
    short, mid, long_ = s.atm(0, 7 / 365), s.atm(0, 1.0), s.atm(0, 20.0)
    assert short > mid > long_
    assert short == pytest.approx(math.sqrt(0.09), abs=0.01)
    assert long_ == pytest.approx(math.sqrt(0.04), abs=0.01)  # converges to the long-run level


def test_nice_strike_step_and_liquidity():
    assert nice_strike_step(100.0, 0.025) == 2.5
    assert nice_strike_step(450.0, 0.025) == 10.0
    assert liquidity_score(0.5, 30) == pytest.approx(1.0)
    assert liquidity_score(0.05, 30) < liquidity_score(0.3, 30) < liquidity_score(0.5, 30)
    assert liquidity_score(0.5, 200) < liquidity_score(0.5, 30)


def test_chain_prices_and_find(realistic_builder):
    chain = realistic_builder.build(0, TODAY, SPOT)
    assert chain.expiries == [EXPIRY]
    assert SPOT in chain.strikes(EXPIRY)
    for q in chain.quotes:
        assert 0.0 <= q.bid <= q.mid <= q.ask
        assert q.ask - q.mid >= 0.025 - 1e-12  # half the tick floor; bid may be floored at zero
    q = chain.find(OptionType.CALL, target_delta=0.30, min_dte=10, max_dte=30)
    assert q is not None and q.strike > SPOT and abs(abs(q.delta) - 0.30) < 0.1
    p = chain.find(OptionType.PUT, target_delta=0.30, expiry=EXPIRY)
    assert p.strike < SPOT and p.delta < 0
    assert chain.find(OptionType.CALL, target_delta=0.3, min_dte=60, max_dte=90) is None


def test_chain_quotes_unlisted_contract_on_demand(realistic_builder):
    from optbt.instruments import OptionContract

    chain = realistic_builder.build(0, TODAY, SPOT)
    far = OptionContract("TST", OptionType.PUT, 12.5, EXPIRY)
    q = chain.quote(far)
    assert q.mid >= 0.0 and q.contract == far
    assert chain.quote(far) is q  # cached
