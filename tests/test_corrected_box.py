"""The corrected box: its reductions, its correction against the pair problem solved directly, and the no-trade
property of the rule."""
from __future__ import annotations

import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as sparse_linalg
import torch

from deephedge import BasketCall, GBMMarket
from deephedge.pricing import LevyGreeks, frictionless_greeks
from deephedge.strategies.bands import AtkinsonBox, WhalleyWilmottBand
from deephedge.strategies.corrected_box import CorrectedBox

OPTION = BasketCall(100.0)
DAMPED = {"sweeps": 20, "relaxation": 0.25, "final_clip": True}


def basket(n: int, rho: float = 0.5) -> GBMMarket:
    """Volatilities 15%, 20% and 30% for two and three assets, 15% to 30% for ten."""
    vols = tuple(np.linspace(0.15, 0.30, 10)) if n == 10 else (0.15, 0.2, 0.3)[:n]
    return GBMMarket(vols, rho=rho)


def states(n: int, paths: int, seed: int, spread: float = 0.15, holding: float = 1.2):
    """Prices around the strike and holdings between 0 and holding / n."""
    gen = torch.Generator().manual_seed(seed)
    prices = 100.0 * torch.exp(spread * torch.randn(paths, n, generator=gen, dtype=torch.float64))
    return prices, torch.rand(paths, n, generator=gen, dtype=torch.float64) * holding / n


def max_abs(x: torch.Tensor) -> float:
    return float(x.abs().max())


def direct_pair_shifts(G, H, a, n=81):
    """The pair problem solved in price units, as in the paper (Section 5.3 and Appendix D), on n x n points.

    Returns the shifts of the upper and lower edges of asset 1, as functions of Y_2, and of asset 2, as functions of
    Y_1, from (96)-(97): Y_i^(+-1) = G_ii I_ii / (2 H_ii Y_i^(+-0)).
    """
    h = 2 * a / (n - 1)
    k1, k2 = (index.ravel() for index in np.meshgrid(np.arange(n), np.arange(n), indexing="ij"))

    def reflect(index):
        return np.where(index < 0, -index, np.where(index > n - 1, 2 * (n - 1) - index, index))

    cross = 2 * G[0, 1] / (4 * h[0] * h[1])
    stencil = [(1, 0, G[0, 0] / h[0] ** 2), (-1, 0, G[0, 0] / h[0] ** 2), (0, 1, G[1, 1] / h[1] ** 2),
               (0, -1, G[1, 1] / h[1] ** 2), (0, 0, -2 * G[0, 0] / h[0] ** 2 - 2 * G[1, 1] / h[1] ** 2),
               (1, 1, cross), (-1, -1, cross), (1, -1, -cross), (-1, 1, -cross)]
    values = np.concatenate([np.full(n * n, coefficient) for *_, coefficient in stencil])
    rows = np.tile(k1 * n + k2, len(stencil))
    cols = np.concatenate([reflect(k1 + d1) * n + reflect(k2 + d2) for d1, d2, _ in stencil])
    operator = sparse.csc_matrix((values, (rows, cols)), shape=(n * n, n * n))
    y1, y2 = np.meshgrid(np.linspace(-a[0], a[0], n), np.linspace(-a[1], a[1], n), indexing="ij")
    ones = sparse.csc_matrix(np.ones((n * n, 1)))
    system = sparse.bmat([[operator, ones], [ones.T, None]], format="csc")
    value = sparse_linalg.spsolve(system, np.append(-2 * H[0, 1] * (y1 * y2).ravel(), 0.0))[:-1].reshape(n, n)
    return (G[0, 0] * 2 * (value[-2] - value[-1]) / h[0] ** 2 / (2 * H[0, 0] * a[0]),
            G[0, 0] * 2 * (value[1] - value[0]) / h[0] ** 2 / (2 * H[0, 0] * -a[0]),
            G[1, 1] * 2 * (value[:, -2] - value[:, -1]) / h[1] ** 2 / (2 * H[1, 1] * a[1]),
            G[1, 1] * 2 * (value[:, 1] - value[:, 0]) / h[1] ** 2 / (2 * H[1, 1] * -a[1]))


def test_without_correlation_it_is_the_order_zero_box():
    for n, schedule in ((2, {}), (3, {}), (10, DAMPED)):
        market = basket(n, rho=0.0)
        greeks = LevyGreeks(market, OPTION)
        prices, holdings = states(n, 300 if n < 10 else 200, 11 + n, spread=0.2, holding=1.0)
        for scale in (0.5, 3.0):
            for t in (0, 25, market.dates - 2):
                corrected = CorrectedBox(greeks, 0.01, scale, **schedule).act(t, prices, holdings)
                order_zero = AtkinsonBox(greeks, 0.01, scale).act(t, prices, holdings)
                assert max_abs(corrected - order_zero) < 1e-12, (n, scale, t)


def test_one_asset_is_whalley_wilmott():
    market = GBMMarket((0.2,))
    greeks = frictionless_greeks(market, OPTION)
    prices, holdings = states(1, 400, 3, spread=0.25, holding=1.0)
    for t in (0, 17, market.dates - 1):
        corrected = CorrectedBox(greeks, 0.01, 0.5).act(t, prices, holdings)
        assert max_abs(corrected - WhalleyWilmottBand(greeks, 0.01, 0.5).act(t, prices, holdings)) < 1e-12


def test_shifts_match_the_direct_solution_of_the_pair_problem():
    """The table of the normalised problem, interpolated in the ratio r, against the problem solved in price units,
    for two assets at 100 with 25 days to maturity (the state of the figure of the region in the thesis)."""
    market = basket(2)
    t, prices = market.dates - 25, torch.full((1, 2), 100.0, dtype=torch.float64)
    greeks = LevyGreeks(market, OPTION)
    _, geometry = CorrectedBox(greeks, 0.01, 0.5).geometry(t, prices)
    half = geometry.half_width[0].numpy()
    _, order_zero = AtkinsonBox(greeks, 0.01, 0.5).centre_and_halfwidth(t, prices)
    np.testing.assert_allclose(half, order_zero[0].numpy(), rtol=1e-12)

    _, gamma = greeks.delta_gamma(t, prices)
    sigma = market.step_covariance(prices)[0].numpy() / market.dt
    G = 0.5 * gamma[0].numpy() @ sigma @ gamma[0].numpy().T
    H = -0.5 * 0.5 * sigma
    v = torch.linspace(-1.0, 1.0, 81, dtype=torch.float64)[None]
    for shift, direct, a in zip(geometry.shifts, direct_pair_shifts(G, H, half), half[[0, 0, 1, 1]], strict=True):
        tabulated = (shift.sign * shift.factor * shift.profile(shift.sign * v))[0].numpy()
        assert np.abs(tabulated - direct).max() < 1e-3 * a


def test_acting_again_does_not_trade():
    """After the rule acts the holdings are in the region, so acting again leaves them, and some paths do not trade
    at all. With a shifted centre the same holds for the deviations from that centre."""
    for n in (2, 3):
        market = basket(n)
        prices, holdings = states(n, 400, 21 + n, spread=0.2, holding=1.0)
        greeks = LevyGreeks(market, OPTION)
        centred = CorrectedBox(greeks, 0.01, 0.5)
        shifted = CorrectedBox(greeks, 0.01, 0.5, centre=LevyGreeks(market.scaled(1.1), OPTION))
        for t in (0, 25):
            for rule in (centred, shifted):
                after = rule.act(t, prices, holdings)
                assert max_abs(rule.act(t, prices, after) - after) < 1e-6, (n, t)
                assert int(torch.isclose(after, holdings, atol=1e-12).all(-1).sum()) > 0
            centre, geometry = shifted.geometry(t, prices)
            deviation = shifted.act(t, prices, holdings) - centre
            lower, upper = geometry.edges(deviation)
            assert bool(((deviation >= lower - 1e-12) & (deviation <= upper + 1e-12)).all())
