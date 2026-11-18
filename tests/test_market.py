import math

import torch


def test_prices_are_martingales(market3):
    paths = market3.simulate(200_000, torch.Generator().manual_seed(1))
    final = paths[:, -1]
    se = final.std(0) / math.sqrt(final.shape[0])
    assert torch.all((final.mean(0) - 100.0).abs() < 4 * se)


def test_correlation_of_log_returns(market3):
    paths = market3.simulate(100_000, torch.Generator().manual_seed(2))
    r = torch.log(paths[:, 1:] / paths[:, :-1]).reshape(-1, 3)
    off_diagonal = torch.corrcoef(r.T)[~torch.eye(3, dtype=torch.bool)]
    assert torch.all((off_diagonal - 0.5).abs() < 0.005)


def test_step_covariance_matches_simulation(market2):
    # the covariance of one step, which the no-trade rules use
    start = torch.tensor([[95.0, 110.0]], dtype=torch.float64)
    z = market2.sample_shocks((400_000, 2), torch.Generator().manual_seed(3))
    following = start * torch.exp(market2.drifts + market2.step_vols * (z @ market2.cholesky.T))
    assert torch.allclose(torch.cov((following - start).T), market2.step_covariance(start)[0], rtol=0.01)


def test_same_seed_same_paths(market2):
    a = market2.simulate(100, torch.Generator().manual_seed(5))
    b = market2.simulate(100, torch.Generator().manual_seed(5))
    assert torch.equal(a, b)
    assert a.shape == (100, 51, 2) and torch.all(a[:, 0] == 100.0)
