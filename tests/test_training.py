import pytest
import torch

from conftest import paths
from deephedge.hedging import Batch
from deephedge.risk import Entropic
from deephedge.strategies import neural_strategy
from deephedge.training import StandardStarts, TrainConfig, Trainer


def _fit(problem, market, steps, checkpoint=None, log=None):
    strategy = neural_strategy(market, problem.option, with_costs=True, seed=1)
    cfg = TrainConfig(steps=steps, batch_size=512, warmup=5, validate_every=10, batch_seed=7)
    validation = Batch(paths(market, 1024, seed=99, dtype=torch.float32))
    result = Trainer(problem, Entropic(0.5), cfg, log=log).fit(strategy, StandardStarts(problem), validation,
                                                                checkpoint=checkpoint)
    return strategy, result


def test_training_lowers_the_risk_and_keeps_the_best_step(problem1, market1):
    _, result = _fit(problem1, market1, steps=40)
    values = [sum(h["validation"]) for h in result.history]
    assert min(values) == result.best_validation < values[0]
    assert result.history[values.index(min(values))]["step"] == result.best_step


def test_resuming_gives_the_same_network(problem1, market1, tmp_path):
    straight, _ = _fit(problem1, market1, steps=20)

    def stop(message):
        if message.startswith("step 10/"):
            raise KeyboardInterrupt

    checkpoint = tmp_path / "checkpoint.pt"
    with pytest.raises(KeyboardInterrupt):
        _fit(problem1, market1, steps=20, checkpoint=checkpoint, log=stop)
    resumed, _ = _fit(problem1, market1, steps=20, checkpoint=checkpoint)
    a, b = straight.network.state_dict(), resumed.network.state_dict()
    assert all(torch.equal(a[k], b[k]) for k in a)
