import pytest
import torch

from conftest import paths
from deephedge.hedging import Batch
from deephedge.risk import CVaR, Entropic
from deephedge.strategies import neural_strategy


def test_entropic_risk():
    # of a constant loss it is the constant, and of a normal loss mu + g sigma^2 / 2
    assert abs(float(Entropic(0.5)(torch.full((1000,), 1.7, dtype=torch.float64))) - 1.7) < 1e-12
    losses = 0.3 + 0.8 * torch.randn(2_000_000, generator=torch.Generator().manual_seed(0), dtype=torch.float64)
    assert abs(float(Entropic(0.5)(losses)) - (0.3 + 0.5 * 0.64 / 2)) < 0.003


def test_cvar_is_the_mean_of_the_worst_paths():
    # at 99% of the 200,000 test paths it is the mean of the worst 2,000
    losses = torch.arange(200_000, dtype=torch.float64)
    assert float(CVaR(0.99)(losses)) == pytest.approx(float(losses[-2000:].mean()), rel=1e-12)


def test_cvar_training_step_has_the_gradient_of_the_batch_cvar(problem1, market1):
    # the step picks the tail without gradient and differentiates only on those paths
    strategy = neural_strategy(problem1.market, problem1.option, with_costs=True, seed=3)
    batch = Batch(paths(market1, 2048, seed=4, dtype=torch.float32))
    cvar = CVaR(0.99)
    cvar.training_loss(problem1, strategy, batch).backward()
    tail = [p.grad.clone() for p in strategy.network.parameters()]
    strategy.network.zero_grad()
    cvar(problem1.run_batch(strategy, batch).loss).backward()
    full = [p.grad for p in strategy.network.parameters()]
    for a, b in zip(tail, full, strict=True):
        assert torch.allclose(a, b, rtol=1e-4, atol=1e-7)
