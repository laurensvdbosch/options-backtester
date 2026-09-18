import math

import pytest

from optbt.pricing import OptionType, bs_delta, bs_gamma, bs_greeks, bs_price, bs_rho, bs_theta, bs_vega, implied_vol

# Textbook reference point (Hull): S=100, K=100, T=1, r=5%, sigma=20%, q=0
S, K, T, R, SIG = 100.0, 100.0, 1.0, 0.05, 0.20


def test_reference_prices():
    assert bs_price(S, K, T, R, SIG, OptionType.CALL) == pytest.approx(10.4506, abs=1e-4)
    assert bs_price(S, K, T, R, SIG, OptionType.PUT) == pytest.approx(5.5735, abs=1e-4)


def test_reference_greeks():
    assert bs_delta(S, K, T, R, SIG, OptionType.CALL) == pytest.approx(0.6368, abs=1e-4)
    assert bs_delta(S, K, T, R, SIG, OptionType.PUT) == pytest.approx(-0.3632, abs=1e-4)
    assert bs_gamma(S, K, T, R, SIG) == pytest.approx(0.018762, abs=1e-5)
    assert bs_vega(S, K, T, R, SIG) == pytest.approx(37.524 / 100.0, abs=1e-4)  # per vol point
    assert bs_theta(S, K, T, R, SIG, OptionType.CALL) == pytest.approx(-6.414 / 365.0, abs=1e-4)  # per day
    assert bs_rho(S, K, T, R, SIG, OptionType.CALL) == pytest.approx(53.232 / 100.0, abs=1e-3)


@pytest.mark.parametrize("K_", [70.0, 90.0, 100.0, 110.0, 140.0])
@pytest.mark.parametrize("T_", [0.05, 0.5, 2.0])
@pytest.mark.parametrize("q", [0.0, 0.02])
def test_put_call_parity(K_, T_, q):
    c = bs_price(S, K_, T_, R, SIG, OptionType.CALL, q)
    p = bs_price(S, K_, T_, R, SIG, OptionType.PUT, q)
    assert c - p == pytest.approx(S * math.exp(-q * T_) - K_ * math.exp(-R * T_), abs=1e-10)


@pytest.mark.parametrize("kind", [OptionType.CALL, OptionType.PUT])
@pytest.mark.parametrize("K_", [85.0, 100.0, 120.0])
def test_greeks_match_finite_differences(kind, K_):
    q = 0.01
    h = 1e-3
    g = bs_greeks(S, K_, T, R, SIG, kind, q)
    price = lambda s=S, t=T, sig=SIG, r=R: bs_price(s, K_, t, r, sig, kind, q)  # noqa: E731
    fd_delta = (price(s=S + h) - price(s=S - h)) / (2 * h)
    fd_gamma = (price(s=S + h) - 2 * price() + price(s=S - h)) / h**2
    fd_vega = (price(sig=SIG + h) - price(sig=SIG - h)) / (2 * h) / 100.0
    fd_theta = -(price(t=T + h) - price(t=T - h)) / (2 * h) / 365.0
    fd_rho = (price(r=R + h) - price(r=R - h)) / (2 * h) / 100.0
    # central differences carry O(h^2) truncation error, so compare at 1e-4 relative
    assert g.delta == pytest.approx(fd_delta, rel=1e-4)
    assert g.gamma == pytest.approx(fd_gamma, rel=1e-3)
    assert g.vega == pytest.approx(fd_vega, rel=1e-4)
    assert g.theta == pytest.approx(fd_theta, rel=1e-4)
    assert g.rho == pytest.approx(fd_rho, rel=1e-4)


def test_expiry_collapses_to_intrinsic():
    assert bs_price(105.0, 100.0, 0.0, R, SIG, OptionType.CALL) == pytest.approx(5.0)
    assert bs_price(95.0, 100.0, 0.0, R, SIG, OptionType.CALL) == 0.0
    assert bs_price(95.0, 100.0, 0.0, R, SIG, OptionType.PUT) == pytest.approx(5.0)
    assert bs_delta(105.0, 100.0, 0.0, R, SIG, OptionType.CALL) == 1.0
    assert bs_delta(95.0, 100.0, 0.0, R, SIG, OptionType.PUT) == -1.0
    assert bs_gamma(100.0, 100.0, 0.0, R, SIG) == 0.0
    assert bs_vega(100.0, 100.0, 0.0, R, SIG) == 0.0


def test_price_bounds_and_monotonicity():
    lo, hi = 0.05, 0.80
    p_lo = bs_price(S, K, T, R, lo, OptionType.CALL)
    p_hi = bs_price(S, K, T, R, hi, OptionType.CALL)
    assert 0 < p_lo < p_hi < S
    put = bs_price(S, 120.0, T, R, SIG, OptionType.PUT)
    assert put >= 120.0 * math.exp(-R * T) - S  # above intrinsic-forward bound


@pytest.mark.parametrize("kind", [OptionType.CALL, OptionType.PUT])
@pytest.mark.parametrize("sigma", [0.08, 0.25, 0.9])
@pytest.mark.parametrize("K_", [80.0, 100.0, 125.0])
def test_implied_vol_round_trip(kind, sigma, K_):
    price = bs_price(S, K_, 0.4, R, sigma, kind, q=0.01)
    assert implied_vol(price, S, K_, 0.4, R, kind, q=0.01) == pytest.approx(sigma, abs=1e-5)


def test_implied_vol_rejects_arbitrage():
    with pytest.raises(ValueError):
        implied_vol(-1.0, S, K, T, R, OptionType.CALL)
    with pytest.raises(ValueError):
        implied_vol(S + 1.0, S, K, T, R, OptionType.CALL)
