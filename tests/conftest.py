import pytest
import torch

from deephedge import GBMMarket, HedgingProblem


@pytest.fixture(autouse=True)
def _threads():
    torch.set_num_threads(2)


@pytest.fixture
def market1():
    return GBMMarket((0.2,))


@pytest.fixture
def market2():
    return GBMMarket((0.15, 0.2), rho=0.5)


@pytest.fixture
def market3():
    return GBMMarket((0.15, 0.2, 0.3), rho=0.5)


@pytest.fixture
def problem1(market1):
    return HedgingProblem.create(market1, 0.01)


def paths(market, n=4096, seed=0, dtype=torch.float64):
    return market.simulate(n, torch.Generator().manual_seed(seed), dtype=dtype)
