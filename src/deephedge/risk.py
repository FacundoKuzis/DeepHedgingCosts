"""Risk measures: the criteria that train the networks, calibrate the rules and rank the strategies.

Thesis, Sections 2.6, 3.5 and 4.4. A risk measure maps a vector of losses (one per path) to a number.
It is used in three places:

* 'risk(losses)' evaluates it, with gradient when the losses carry one;
* 'risk.training_loss(problem, strategy, batch)' is the objective of one training step; the
  default evaluates the risk of the batch, and :class:`CVaR` overrides it;
* the evaluation module computes all of them on the test sample.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod

import torch

Tensor = torch.Tensor


class RiskMeasure(ABC):
    name: str

    @abstractmethod
    def __call__(self, losses: Tensor) -> Tensor:
        """Risk of a sample of losses [paths] (or [..., paths], along the last dimension)."""

    def training_loss(self, problem, strategy, batch) -> Tensor:
        return self(problem.run_batch(strategy, batch).loss)

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class MeanSquare(RiskMeasure):
    """E[L^2]: the criterion without costs, where the delta hedge is the benchmark."""

    name = "mse"

    def __call__(self, losses: Tensor) -> Tensor:
        return (losses * losses).mean(-1)


class Entropic(RiskMeasure):
    """(1/g) log E[exp(g L)], the certainty equivalent of exponential utility with risk aversion g."""

    name = "entropic"

    def __init__(self, risk_aversion: float = 0.5):
        self.risk_aversion = float(risk_aversion)

    def __call__(self, losses: Tensor) -> Tensor:
        g = self.risk_aversion
        return (torch.logsumexp(g * losses, -1) - math.log(losses.shape[-1])) / g

    def __repr__(self) -> str:
        return f"Entropic({self.risk_aversion})"


class CVaR(RiskMeasure):
    """Mean loss of the worst '1 - level' fraction of the paths (the 2,000 worst of 200,000 at 99%).

    Training uses the tail of each batch: the worst paths are found without gradient, the strategy is
    run again on those paths only, and the step minimises their mean loss. The gradient equals that of
    the CVaR of the whole batch ('tests/test_risk.py') and the graph is a hundred times smaller,
    which is what lets the CVaR networks train on batches of 32,768 paths.
    """

    def __init__(self, level: float = 0.99):
        if not 0.0 < level < 1.0:
            raise ValueError("level must be in (0, 1)")
        self.level = float(level)
        self.name = "cvar" + f"{100 * level:g}".replace(".", "")

    def tail_size(self, n: int) -> int:
        # without the margin, (1 - 0.99) * 200_000 rounds up to 2_001 instead of 2_000
        return max(1, math.ceil((1.0 - self.level) * n - 1e-9))

    def __call__(self, losses: Tensor) -> Tensor:
        return torch.topk(losses, self.tail_size(losses.shape[-1]), dim=-1).values.mean(-1)

    def training_loss(self, problem, strategy, batch) -> Tensor:
        with torch.no_grad():
            worst = problem.run_batch(strategy, batch).loss.topk(self.tail_size(len(batch))).indices
        return problem.run_batch(strategy, batch.subset(worst)).loss.mean()

    def __repr__(self) -> str:
        return f"CVaR({self.level})"


def risk_measure(name: str, risk_aversion: float = 0.5) -> RiskMeasure:
    """'mse', 'entropic' or 'cvar' followed by the level in percent without the point ('cvar99', 'cvar999')."""
    if name == "mse":
        return MeanSquare()
    if name == "entropic":
        return Entropic(risk_aversion)
    if name.startswith("cvar") and name[4:].isdigit():
        digits = name[4:]
        return CVaR(float(f"0.{digits}"))
    raise ValueError(f"unknown risk measure {name!r}")
