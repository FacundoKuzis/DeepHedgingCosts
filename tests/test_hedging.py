import torch

from conftest import paths
from deephedge.strategies import NoHedge, neural_strategy
from deephedge.strategies.base import HedgingStrategy


class Constant(HedgingStrategy):
    def __init__(self, value):
        self.value = value

    def act(self, t, prices, position):
        return torch.full_like(position, self.value)


def test_no_hedge_loses_the_payoff_minus_the_premium(problem1, market1):
    s = paths(market1, 512)
    out = problem1.run(NoHedge(), s)
    assert torch.allclose(out.loss, problem1.option.payoff(s[:, -1]) - problem1.premium)
    assert torch.all(out.cost == 0)


def test_constant_holding_pays_costs_twice(problem1, market1):
    # buys 0.5 at the first date and sells it at maturity, paying 1% of the price each time
    s = paths(market1, 512)
    out = problem1.run(Constant(0.5), s)
    gains = 0.5 * (s[:, -1, 0] - s[:, 0, 0])
    costs = 0.01 * 0.5 * (s[:, 0, 0] + s[:, -1, 0])
    assert torch.allclose(out.loss, problem1.option.payoff(s[:, -1]) - problem1.premium - gains + costs)


def test_gradient_flows_through_the_accounting(problem1, market1):
    strategy = neural_strategy(problem1.market, problem1.option, with_costs=True)
    loss = problem1.run(strategy, paths(market1, 256, dtype=torch.float32)).loss.mean()
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in strategy.network.parameters())
