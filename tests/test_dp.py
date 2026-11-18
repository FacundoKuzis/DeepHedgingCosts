"""The dynamic programmes: without costs they hedge like the delta, with costs they beat the asymptotic band and
predict their own risk, and two identical assets are one asset."""
import math

import pytest
import torch

from deephedge import BasketCall, Entropic, GBMMarket, HedgingProblem
from deephedge.dp import OneAssetDP, OneAssetGrid, OptimalBand, OptimalGridStrategy, TwoAssetDP, TwoAssetGrid
from deephedge.pricing import black_scholes_call, black_scholes_delta_gamma
from deephedge.strategies import asymptotic_rule

G = 0.5  # the risk aversion of the thesis
RISK = Entropic(G)
OPTION = BasketCall(100.0)
MARKET = GBMMarket((0.2,))
PREMIUM = float(black_scholes_call(100.0, 100.0, MARKET.maturity, 0.2))


def problem(cost: float, market: GBMMarket = MARKET) -> HedgingProblem:
    return HedgingProblem(market, OPTION, cost, PREMIUM)


def simulated_risk(strategy, cost: float, paths: torch.Tensor, market: GBMMarket = MARKET) -> float:
    return float(RISK(problem(cost, market).evaluate(strategy, paths).loss))


@pytest.fixture(scope="module")
def band() -> OptimalBand:
    """The optimum with a 1% cost on the coarse grid (its predicted risk is 0.2% above the one of the fine grid)."""
    return OneAssetDP(problem(0.01), G, OneAssetGrid.coarse()).solve()


@pytest.fixture(scope="module")
def identical_assets(tmp_path_factory) -> OptimalGridStrategy:
    """The two-asset optimum for two copies of the asset of the other tests, on a small grid."""
    market = GBMMarket((0.2, 0.2), rho=1.0)
    grid = TwoAssetGrid(x_half=0.3, n_prices=41, phi_min=-0.05, phi_max=0.6, n_positions=14, nodes=8, nodes_last=12)
    strategy = TwoAssetDP(problem(0.01, market), G, grid, table_dir=tmp_path_factory.mktemp("tables")).solve()
    yield strategy
    strategy.delete_tables()


def test_without_costs_the_band_collapses_to_the_delta():
    """With k = 0 both edges minimise the same function, so they coincide. That minimiser is the discrete-time hedge
    under exponential utility, which stays within one step of the holding grid (0.01) of the Black-Scholes delta for
    the first 40 dates."""
    frictionless = OneAssetDP(problem(0.0), G, OneAssetGrid.coarse()).solve()
    assert torch.equal(frictionless.lower, frictionless.upper)
    prices = OPTION.strike * torch.exp(frictionless.log_moneyness)
    for t in range(40):
        delta, _ = black_scholes_delta_gamma(prices, OPTION.strike, MARKET.time_to_maturity(t), 0.2)
        assert float((frictionless.lower[t].double() - delta).abs().max()) < 0.01


def test_with_costs_the_optimum_beats_whalley_wilmott_and_predicts_its_risk(band):
    """The risk the programme predicts against the simulated risk of its strategy, within five standard errors of the
    simulation (0.4% of the risk with 40,000 paths)."""
    assert bool((band.lower <= band.upper + 1e-6).all())
    paths = MARKET.simulate(40_000, torch.Generator().manual_seed(106))
    loss = problem(0.01).evaluate(band, paths).loss
    simulated = float(RISK(loss))
    assert simulated < simulated_risk(asymptotic_rule(MARKET, OPTION, 0.01, G), 0.01, paths)
    weights = torch.exp(G * (loss - loss.max()))
    standard_error = float(weights.std() / (G * weights.mean())) / math.sqrt(len(loss))  # delta method
    assert abs(band.predicted_risk() - simulated) < 5 * standard_error


def test_two_identical_assets_are_one_asset(identical_assets, band):
    """With rho = 1 and equal volatilities both prices are the same, the basket is the asset and holding (f_1, f_2)
    is holding f_1 + f_2 of it, so both programmes solve one problem. They differ by the discretisation of the small
    two-asset grid (thesis, Section 2.9)."""
    assert identical_assets.predicted_risk() == pytest.approx(band.predicted_risk(), rel=0.025)
    market = GBMMarket((0.2, 0.2), rho=1.0)
    paths = market.simulate(20_000, torch.Generator().manual_seed(7))
    assert torch.equal(paths[..., 0], paths[..., 1])
    two_assets = simulated_risk(identical_assets, 0.01, paths, market)
    assert two_assets == pytest.approx(simulated_risk(band, 0.01, paths[..., :1]), rel=0.01)
