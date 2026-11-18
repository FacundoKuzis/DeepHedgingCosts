"""Metrics, the paired bootstrap and the comparison tables (thesis, Section 4.4).

Every strategy is run on the same 200,000 test paths, so two strategies can be compared path by path.
The paired bootstrap resamples the *indices* of the paths, uses the same indices for both strategies,
and reports the 2.5% and 97.5% percentiles of the difference of the metric over 1,000 resamples. The
difference is expressed as a percentage of the metric of the reference (the network) on the full sample,
a fixed denominator.

Metrics ('METRICS'): entropic risk, mean loss, standard deviation, mean square, CVaR at 95%, 99% and
99.9%, and 'cost', the mean trading cost, computed on the cost of each path instead of its loss.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from .hedging import HedgeOutcome
from .risk import CVaR, Entropic, MeanSquare

Tensor = torch.Tensor
MetricFunction = Callable[[Tensor], Tensor]

# the columns of the thesis tables, with costs and without them
WITH_COSTS = ("entropic", "cost", "std", "cvar95", "cvar99", "cvar999")
WITHOUT_COSTS = ("mse", "entropic", "std", "cvar95", "cvar99", "cvar999")


def metric_function(name: str, risk_aversion: float = 0.5) -> MetricFunction:
    """A function of losses [..., paths] -> [...]. 'cost' is the mean, applied to the costs."""
    functions = {
        "entropic": Entropic(risk_aversion),
        "mean": lambda losses: losses.mean(-1),
        "cost": lambda costs: costs.mean(-1),
        "std": lambda losses: losses.std(-1),
        "mse": MeanSquare(),
        "cvar95": CVaR(0.95),
        "cvar99": CVaR(0.99),
        "cvar999": CVaR(0.999),
    }
    if name not in functions:
        raise ValueError(f"unknown metric {name!r}; choose from {sorted(functions)}")
    return functions[name]


def series(outcome: HedgeOutcome, metric: str) -> Tensor:
    """The per-path quantity a metric is computed on: the cost for 'cost', the loss otherwise."""
    return outcome.cost if metric == "cost" else outcome.loss


def metric_value(outcome: HedgeOutcome, metric: str, risk_aversion: float = 0.5) -> float:
    x = series(outcome, metric).to(torch.float64)
    return float(metric_function(metric, risk_aversion)(x[None])[0])


def summarize(outcome: HedgeOutcome, risk_aversion: float = 0.5,
              metrics: tuple[str, ...] = ("entropic", "mean", "std", "mse", "cvar95", "cvar99", "cvar999", "cost")
              ) -> dict[str, float]:
    out = {m: metric_value(outcome, m, risk_aversion) for m in metrics}
    out["turnover"] = float(outcome.turnover.to(torch.float64).mean())
    return out


# --- the paired bootstrap --------------------------------------------------------------------------------
@dataclass(frozen=True)
class PairedDifference:
    """M(other) - M(reference) on the full sample, with its 95% paired-bootstrap interval."""

    difference: float
    low: float
    high: float
    reference: float  # M(reference) on the full sample

    @property
    def percent(self) -> tuple[float, float, float]:
        """The difference and its interval in % of the reference (negative: 'other' is better)."""
        r = self.reference
        return 100 * self.difference / r, 100 * self.low / r, 100 * self.high / r

    @property
    def significant(self) -> bool:
        return not (self.low <= 0.0 <= self.high)


def _quantiles(draws: Tensor) -> tuple[float, float]:
    lo, hi = torch.quantile(draws, torch.tensor([0.025, 0.975], device=draws.device, dtype=draws.dtype)).tolist()
    return lo, hi


def paired_bootstrap(reference: Tensor, other: Tensor, metric: str = "entropic", *, risk_aversion: float = 0.5,
                     reps: int = 1000, seed: int = 0, device: str | torch.device = "cpu", chunk: int = 20
                     ) -> PairedDifference:
    """Paired bootstrap of M(other) - M(reference); 'reference' and 'other' are per-path series [paths].

    The resampled indices come from 'torch.randint' on a generator of 'device' seeded with 'seed',
    'chunk' resamples at a time: with the same seed and device, the interval is the one of the thesis.
    """
    f = metric_function(metric, risk_aversion)
    a = reference.to(device, torch.float64)
    b = other.to(device, torch.float64)
    n = a.shape[0]
    if b.shape[0] != n:
        raise ValueError("the two series must have one value per path of the same sample")
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    base = float(f(a[None])[0])
    draws = []
    for r0 in range(0, reps, chunk):
        idx = torch.randint(0, n, (min(chunk, reps - r0), n), device=device, generator=generator)
        draws.append(f(b[idx]) - f(a[idx]))
    lo, hi = _quantiles(torch.cat(draws))
    return PairedDifference(float(f(b[None])[0]) - base, lo, hi, base)


def paired_change(before: tuple[Tensor, Tensor], after: tuple[Tensor, Tensor], metric: str = "entropic", *,
                  risk_aversion: float = 0.5, reps: int = 1000, seed: int = 0, device: str | torch.device = "cpu",
                  chunk: int = 20) -> tuple[float, float, float]:
    """How much the gap of a rule over the network, 100 (M(rule) - M(network)) / M(network), changes from one
    experiment to another, in points, with its 95% interval.

    'before' and 'after' are (network, rule) series of the two experiments on test samples of the same size.
    The same resampled indices are used for the four series, which is what makes the interval paired: when the
    two experiments share their test paths the losses are correlated and the interval is narrower than what
    the two separate intervals suggest. Overlapping intervals do not mean that the change is not significant.
    Each gap is divided by the metric of the network of its experiment on the full sample, the fixed denominator
    of :func:`paired_bootstrap`, so a resample changes the differences and not the scale they are measured in.
    """
    f = metric_function(metric, risk_aversion)
    series_ = [tuple(x.to(device, torch.float64) for x in pair) for pair in (before, after)]
    n = series_[0][0].shape[0]
    if any(x.shape[0] != n for pair in series_ for x in pair):
        raise ValueError("the four series must have the same number of paths")
    reference = [float(f(network[None])[0]) for network, _ in series_]

    def gap(i: int, idx: Tensor) -> Tensor:
        network, rule = (x[idx] for x in series_[i])
        return 100 * (f(rule) - f(network)) / reference[i]

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    draws = []
    for r0 in range(0, reps, chunk):
        idx = torch.randint(0, n, (min(chunk, reps - r0), n), device=device, generator=generator)
        draws.append(gap(1, idx) - gap(0, idx))
    lo, hi = _quantiles(torch.cat(draws))
    everything = torch.arange(n, device=device)[None]
    return float(gap(1, everything)[0] - gap(0, everything)[0]), lo, hi


# --- comparison tables -----------------------------------------------------------------------------------
@dataclass
class Row:
    name: str
    outcome: HedgeOutcome
    values: dict[str, float]
    versus_reference: dict[str, PairedDifference] | None  # None in the reference row


class Comparison:
    """One panel of a results table: every strategy against the reference, metric by metric.

    In each metric the rows in bold are the strategy with the lowest value and every strategy that does not
    differ from it significantly (its paired 95% interval contains zero). Against the reference the interval
    is the one already in the table; between two other strategies it is computed with seeds from
    'tie_seed' on, one per comparison (thesis, Section 5).
    """

    def __init__(self, outcomes: dict[str, HedgeOutcome], reference: str = "network",
                 metrics: tuple[str, ...] = WITH_COSTS, *, risk_aversion: float = 0.5, reps: int = 1000,
                 seed: int = 0, tie_seed: int = 9400, device: str | torch.device = "cpu"):
        if reference not in outcomes:
            raise ValueError(f"the reference {reference!r} is not among the outcomes")
        self.metrics, self.risk_aversion, self.reps, self.device = metrics, risk_aversion, reps, device
        self.reference = reference
        self.tie_seed = tie_seed
        ref = outcomes[reference]
        self.rows: list[Row] = []
        next_seed = seed
        for name, outcome in outcomes.items():
            values = {m: metric_value(outcome, m, risk_aversion) for m in metrics}
            versus = None
            if name != reference:
                versus = {}
                for m in metrics:
                    versus[m] = paired_bootstrap(series(ref, m), series(outcome, m), "mean" if m == "cost" else m,
                                                 risk_aversion=risk_aversion, reps=reps, seed=next_seed, device=device)
                    next_seed += 1
            self.rows.append(Row(name, outcome, values, versus))
        self.ties: list[dict] = []

    def bold(self) -> dict[str, set[int]]:
        """For every metric, the indices of the rows in bold ('self.ties' keeps every comparison made)."""
        out: dict[str, set[int]] = {}
        seed = self.tie_seed
        self.ties = []
        for m in self.metrics:
            best = min(range(len(self.rows)), key=lambda i: self.rows[i].values[m])
            chosen = {best}
            for j, row in enumerate(self.rows):
                if j == best:
                    continue
                if self.rows[best].versus_reference is None:      # the best is the reference
                    d = row.versus_reference[m]
                elif row.versus_reference is None:                 # the other one is the reference
                    d = self.rows[best].versus_reference[m]
                else:                                              # two strategies that are not the reference
                    d = paired_bootstrap(series(self.rows[best].outcome, m), series(row.outcome, m),
                                         "mean" if m == "cost" else m, risk_aversion=self.risk_aversion,
                                         reps=self.reps, seed=seed, device=self.device)
                    seed += 1
                tie = not d.significant
                if tie:
                    chosen.add(j)
                self.ties.append({"metric": m, "best": self.rows[best].name, "other": row.name,
                                  "interval": d.percent, "tie": tie})
            out[m] = chosen
        return out

    def as_dict(self) -> dict:
        bold = self.bold()
        return {"reference": self.reference, "metrics": list(self.metrics),
                "rows": [{"name": r.name, "values": r.values,
                          "versus_reference": None if r.versus_reference is None else
                          {m: list(d.percent) for m, d in r.versus_reference.items()},
                          "bold": [m for m in self.metrics if i in bold[m]]} for i, r in enumerate(self.rows)]}
