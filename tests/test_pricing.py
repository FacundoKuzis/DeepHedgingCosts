import torch

from deephedge import GBMMarket
from deephedge.pricing import (
    BasketCall,
    BlackScholesGreeks,
    LevyGreeks,
    black_scholes_call,
    black_scholes_delta_gamma,
    option_premium,
)


def test_black_scholes_price_and_greeks():
    assert abs(float(black_scholes_call(100.0, 100.0, 1.0, 0.2)) - 7.965567455405804) < 1e-9
    s = torch.tensor([90.0, 100.0, 112.0], dtype=torch.float64)
    h = 1e-3
    delta, gamma = black_scholes_delta_gamma(s, 100.0, 0.1, 0.2)
    up, down = black_scholes_call(s + h, 100.0, 0.1, 0.2), black_scholes_call(s - h, 100.0, 0.1, 0.2)
    assert torch.allclose(delta, (up - down) / (2 * h), atol=1e-7)
    assert torch.allclose(gamma, (up - 2 * black_scholes_call(s, 100.0, 0.1, 0.2) + down) / h ** 2, atol=1e-5)


def test_levy_with_one_asset_is_black_scholes():
    market = GBMMarket((0.2,))
    option = BasketCall(100.0)
    levy, bs = LevyGreeks(market, option), BlackScholesGreeks(market, option)
    prices = torch.tensor([[85.0], [100.0], [117.0]], dtype=torch.float64)
    for t in (0, 25, 49):
        d1, g1 = levy.delta_gamma(t, prices)
        d2, g2 = bs.delta_gamma(t, prices)
        assert torch.allclose(d1, d2, atol=1e-10) and torch.allclose(g1, g2, atol=1e-10)


def test_levy_gammas_are_the_derivatives_of_the_deltas(market3):
    greeks = LevyGreeks(market3, BasketCall(100.0))
    x = torch.tensor([[97.0, 103.0, 110.0]], dtype=torch.float64)
    _, gamma = greeks.delta_gamma(10, x)
    h = 1e-4
    for j in range(3):
        e = torch.zeros_like(x)
        e[0, j] = h
        fd = (greeks.delta(10, x + e) - greeks.delta(10, x - e)) / (2 * h)
        assert torch.allclose(gamma[0, :, j], fd[0], atol=1e-6)


def test_basket_premium_by_monte_carlo(market2):
    # 450,000 paths in blocks of 100,000, so the last block is a partial one
    premium = option_premium(market2, BasketCall(100.0), mc_paths=450_000, mc_block=100_000)
    assert abs(premium.value - 2.7064209904639696) < 4 * premium.standard_error + 1e-3
