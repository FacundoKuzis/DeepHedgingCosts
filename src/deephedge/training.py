"""Training a network strategy (thesis, Sections 3.6 and 4.3).

Every step draws fresh paths (there is no fixed training set), runs the strategy through the
hedging accounting, and takes an Adam step on the risk of the batch. The learning rate warms up
linearly and then decays along a cosine; gradients are clipped. Every 'validate_every' steps the
risk is measured on fixed validation paths, and the network kept at the end is the one with the
lowest validation risk. The test paths are never seen.

The order of every random draw is part of the protocol: with the same seeds, the same device and
the same number of threads, a run reproduces the thesis run that used them.
"""
from __future__ import annotations

import json
import math
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from .hedging import Batch, HedgingProblem
from .pricing import black_scholes_call
from .risk import RiskMeasure

Tensor = torch.Tensor


@dataclass(frozen=True)
class TrainConfig:
    steps: int = 6000
    batch_size: int = 4096
    lr: float = 1e-3
    lr_final: float = 1e-5
    warmup: int = 200
    clip_norm: float = 1.0
    validate_every: int = 250
    validation_chunk: int = 16384
    batch_seed: int = 10_000          # seed of the generator that draws the training batches
    validate_at_start: bool = False   # also measure the starting network (when fine-tuning)

    def replace(self, **changes) -> TrainConfig:
        return TrainConfig(**{**asdict(self), **changes})


def learning_rate(step: int, config: TrainConfig) -> float:
    """Linear warm-up from 10% of 'lr', then cosine decay to 'lr_final' at the last step."""
    if step < config.warmup:
        return config.lr * (0.1 + 0.9 * step / config.warmup)
    progress = (step - config.warmup) / max(1, config.steps - config.warmup)
    return config.lr_final + 0.5 * (config.lr - config.lr_final) * (1.0 + math.cos(math.pi * min(1.0, progress)))


# --- where the training paths come from ------------------------------------------------------------------
class PathSource(ABC):
    @abstractmethod
    def sample(self, n: int, generator: torch.Generator) -> Batch:
        """'n' fresh training paths."""


class StandardStarts(PathSource):
    """Paths from today with no holdings: the problem of the thesis."""

    def __init__(self, problem: HedgingProblem, device: str | torch.device = "cpu", dtype: torch.dtype = torch.float32):
        self.market = problem.market
        self.device, self.dtype = device, dtype

    def sample(self, n: int, generator: torch.Generator) -> Batch:
        return Batch(self.market.simulate(n, generator, device=self.device, dtype=self.dtype))


class RandomStarts(PathSource):
    """Paths that start at a random date, price and inherited holding (thesis, Section 5.2.1).

    The start date is uniform on 0..N-1, the price uniform on 'price_range' and the holding uniform on
    [0, 1]. Before its start date a path keeps the price constant and does not trade, so every path has
    the usual length. Its loss is that of the remaining hedge, centred at the Black-Scholes price of the
    starting state. With the entropic risk the optimal policy is optimal from every state, so these paths
    do not change the optimum; they change the states the network learns from. One asset only.
    """

    def __init__(self, problem: HedgingProblem, price_range: tuple[float, float] = (75.0, 125.0),
                 device: str | torch.device = "cpu", dtype: torch.dtype = torch.float32):
        if problem.n_assets != 1:
            raise ValueError("random starts are defined for a single asset")
        self.market, self.option = problem.market, problem.option
        self.low, self.high = price_range
        self.device, self.dtype = device, dtype

    def sample(self, n: int, generator: torch.Generator) -> Batch:
        m = self.market
        g_dev = generator.device  # the draws are made where the generator lives, in float64
        first = torch.randint(0, m.dates, (n,), generator=generator, device=g_dev)
        start_price = self.low + (self.high - self.low) * torch.rand(n, generator=generator, dtype=torch.float64,
                                                                     device=g_dev)
        holding = torch.rand(n, generator=generator, dtype=torch.float64, device=g_dev)
        z = m.sample_shocks((n, m.dates), generator, device=g_dev)
        steps = torch.where(torch.arange(m.dates, device=g_dev)[None, :] >= first[:, None],
                            m.drifts.item() + m.step_vols.item() * z, torch.zeros_like(z))
        log_paths = torch.cat([torch.zeros(n, 1, dtype=torch.float64, device=g_dev), torch.cumsum(steps, 1)], 1)
        paths = start_price[:, None] * torch.exp(log_paths)
        value = black_scholes_call(start_price, self.option.strike, (m.dates - first).double() * m.dt, m.vols[0])
        return Batch(paths[..., None], holding[:, None], value, first).to(self.device, self.dtype)


# --- the trainer -----------------------------------------------------------------------------------------
@dataclass
class TrainingResult:
    best_step: int
    best_validation: float
    seconds: float
    history: list[dict] = field(default_factory=list)

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=1), encoding="utf-8")


class Trainer:
    """Fits 'strategy.network' to minimise 'risk' on 'problem'.

    'sources' gives the training paths of each step (one source, or several whose risks are added,
    as when training from random states), and 'validation' the fixed validation batches; the
    selection criterion is the sum of their risks.
    """

    def __init__(self, problem: HedgingProblem, risk: RiskMeasure, config: TrainConfig, *,
                 device: str | torch.device = "cpu", log: Callable[[str], None] | None = None):
        self.problem, self.risk, self.config = problem, risk, config
        self.device = torch.device(device)
        self.log = log or (lambda message: None)

    def validation_risk(self, strategy, batch: Batch) -> float:
        network = strategy.network
        network.eval()
        with torch.no_grad():
            chunk = self.config.validation_chunk
            losses = torch.cat([self.problem.run_batch(strategy, batch.subset(slice(i, i + chunk))).loss.double()
                                for i in range(0, len(batch), chunk)])
        network.train()
        return float(self.risk(losses))

    def fit(self, strategy, sources: PathSource | list[PathSource], validation: Batch | list[Batch], *,
            checkpoint: Path | None = None) -> TrainingResult:
        cfg = self.config
        sources = [sources] if isinstance(sources, PathSource) else list(sources)
        validation = [validation] if isinstance(validation, Batch) else list(validation)
        validation = [v.to(self.device) for v in validation]
        network = strategy.network.to(self.device)
        network.train()
        optimizer = torch.optim.Adam(network.parameters(), lr=cfg.lr)
        generator = torch.Generator(device=self.device)
        generator.manual_seed(cfg.batch_seed)

        start, seconds, history = 0, 0.0, []
        best_value, best_step, best_state = float("inf"), -1, None
        if checkpoint is not None and Path(checkpoint).exists():
            state = torch.load(checkpoint, map_location="cpu", weights_only=False)
            network.load_state_dict(state["network"])
            optimizer.load_state_dict(state["optimizer"])
            generator.set_state(state["generator"])
            start, seconds, history = state["step"], state["seconds"], state["history"]
            best_value, best_step, best_state = state["best_value"], state["best_step"], state["best_state"]
            self.log(f"resuming at step {start}")
        elif cfg.validate_at_start:
            values = [self.validation_risk(strategy, v) for v in validation]
            best_value, best_step = sum(values), 0
            best_state = {k: v.detach().clone() for k, v in network.state_dict().items()}
            history.append({"step": 0, "validation": values, "seconds": 0.0})

        clock = time.time()
        for step in range(start + 1, cfg.steps + 1):
            for group in optimizer.param_groups:
                group["lr"] = learning_rate(step, cfg)
            loss = sum(self.risk.training_loss(self.problem, strategy, source.sample(cfg.batch_size, generator))
                       for source in sources)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), cfg.clip_norm)
            optimizer.step()
            if step % cfg.validate_every == 0 or step == cfg.steps:
                values = [self.validation_risk(strategy, v) for v in validation]
                elapsed = seconds + time.time() - clock
                history.append({"step": step, "batch_risk": float(loss), "validation": values,
                                "lr": optimizer.param_groups[0]["lr"], "seconds": elapsed})
                if sum(values) < best_value:
                    best_value, best_step = sum(values), step
                    best_state = {k: v.detach().clone() for k, v in network.state_dict().items()}
                if checkpoint is not None:
                    torch.save({"network": network.state_dict(), "optimizer": optimizer.state_dict(),
                                "generator": generator.get_state(), "step": step, "seconds": elapsed,
                                "history": history, "best_value": best_value, "best_step": best_step,
                                "best_state": best_state}, checkpoint)
                self.log(f"step {step}/{cfg.steps}: validation {sum(values):.6f}, best {best_value:.6f} at step "
                         f"{best_step} ({elapsed:.0f} s)")
        network.load_state_dict(best_state)
        network.eval()
        return TrainingResult(best_step, best_value, seconds + time.time() - clock, history)
