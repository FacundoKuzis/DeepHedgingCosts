import torch

from conftest import paths
from deephedge import GBMMarket
from deephedge.pricing import BasketCall, LevyGreeks, black_scholes_delta_gamma
from deephedge.strategies import AtkinsonBox, WhalleyWilmottBand, asymptotic_rule, make_rule


def test_whalley_wilmott_half_width():
    # thesis, eq. (2.19): ((3/2) k S Gamma^2 / g)^(1/3)
    market = GBMMarket((0.2,))
    rule = asymptotic_rule(market, BasketCall(100.0), 0.01, 0.5)
    s = torch.tensor([[100.0]], dtype=torch.float64)
    lo, hi = rule.band(10, s, None)
    _, gamma = black_scholes_delta_gamma(s[:, 0], 100.0, market.time_to_maturity(10), 0.2)
    half = (1.5 * 0.01 * 100.0 * gamma ** 2 / 0.5) ** (1 / 3)
    assert torch.allclose((hi - lo)[:, 0] / 2, half)


def test_atkinson_box_with_one_asset_is_whalley_wilmott(market1):
    levy = LevyGreeks(market1, BasketCall(100.0))
    box, ww = AtkinsonBox(levy, 0.001, 3.0), WhalleyWilmottBand(levy, 0.001, 3.0)
    s = paths(market1, 256)
    for t in (0, 20, 45):
        lo1, hi1 = box.band(t, s[:, t], None)
        lo2, hi2 = ww.band(t, s[:, t], None)
        assert torch.allclose(lo1, lo2, atol=1e-12) and torch.allclose(hi1, hi2, atol=1e-12)


def test_shifted_centre_with_factor_one_is_the_base_rule(market2):
    option = BasketCall(100.0)
    base, shifted = asymptotic_rule(market2, option, 0.001, 2.0), asymptotic_rule(market2, option, 0.001, 2.0, 1.0)
    s = paths(market2, 64)
    assert torch.equal(base.band(5, s[:, 5], None)[0], shifted.band(5, s[:, 5], None)[0])
    moved = asymptotic_rule(market2, option, 0.001, 2.0, 1.3)
    lo_b, hi_b = base.band(5, s[:, 5], None)
    lo_m, hi_m = moved.band(5, s[:, 5], None)
    assert torch.allclose(hi_b - lo_b, hi_m - lo_m)          # the same width
    assert not torch.allclose(lo_b, lo_m)                   # around another centre


def test_band_rules_keep_the_holding_inside(market2):
    # inside the band nothing is traded, and outside the holding moves to the nearest edge
    rule = make_rule("atkinson-box", market2, BasketCall(100.0), 0.001, 1.0)
    s = paths(market2, 128)
    position = torch.rand(128, 2, generator=torch.Generator().manual_seed(0), dtype=torch.float64)
    new = rule.act(10, s[:, 10], position)
    lo, hi = rule.band(10, s[:, 10], position)
    inside = (position >= lo) & (position <= hi)
    assert torch.all(new[inside] == position[inside])
    assert torch.all((new >= lo - 1e-15) & (new <= hi + 1e-15))
