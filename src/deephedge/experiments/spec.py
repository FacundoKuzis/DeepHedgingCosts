"""An experiment as data.

An :class:`ExperimentSpec` holds everything needed to rerun one experiment of the thesis: the market,
the cost, the criterion, how the validation and test samples are drawn, how the network is trained,
which rules are calibrated and compared, and the grids of the dynamic programme. Specs are immutable;
variants are made with :meth:`ExperimentSpec.replace`, which is how the catalog writes every experiment
as "the base case with one change" (thesis, Table 5).
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import torch

from ..dp import OneAssetGrid, TwoAssetGrid
from ..hedging import HedgingProblem
from ..market import GBMMarket
from ..pricing import BasketCall, option_premium

Grid = OneAssetGrid | TwoAssetGrid


def base_vols(n_assets: int) -> tuple[float, ...]:
    """The volatilities of the thesis: 20% with one asset, 15%, 20%, 30% with two and three, 15% to 30% with ten."""
    if n_assets == 1:
        return (0.2,)
    if n_assets <= 3:
        return (0.15, 0.2, 0.3)[:n_assets]
    return tuple(float(v) for v in np.linspace(0.15, 0.30, n_assets))


@dataclass(frozen=True)
class Sample:
    """'size' paths drawn by a CPU generator seeded with 'seed' or, with 'block', in blocks of 'block'
    paths, block 'j' drawn with the seed '1000 seed + j' (float64, then moved where they are used)."""

    seed: int
    size: int
    block: int | None = None

    def draw(self, market: GBMMarket) -> torch.Tensor:
        if self.block is None:
            return market.simulate(self.size, torch.Generator().manual_seed(self.seed))
        return market.simulate_in_blocks(self.size, self.seed, self.block)

    def smaller(self, size: int) -> Sample:
        return replace(self, size=size, block=None if self.block is None else min(self.block, size))


@dataclass(frozen=True)
class NetworkTraining:
    criterion: str              # 'mse', 'entropic' or 'cvar99'
    steps: int
    warmup: int
    batch_seed: int             # seed of the generator of the training batches
    batch_size: int = 4096
    weight_seed: int = 1        # seed of the initial weights
    lr: float = 1e-3
    lr_final: float = 1e-5
    clip_norm: float = 1.0
    validate_every: int = 250


@dataclass(frozen=True)
class RuleSpec:
    """A rule to calibrate on the validation sample. 'kind' is one of 'whalley-wilmott', 'atkinson-box',
    'corrected-box' and 'damped-corrected-box'; 'search' names a preset of :mod:`deephedge.calibration`."""

    name: str
    kind: str
    search: str
    criterion: str = "entropic"


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    section: str
    description: str
    n_assets: int
    cost: float
    validation: Sample
    test: Sample
    network: NetworkTraining
    vol_scale: float = 1.0
    rho: float = 0.5
    risk_aversion: float = 0.5
    premium: float | None = None            # None: Black-Scholes with one asset, Monte Carlo with a basket
    rules: tuple[RuleSpec, ...] = ()
    theoretical_rule: bool = True           # also evaluate the rule with the risk aversion as its scale
    dp: tuple[Grid, ...] = ()               # grids of the dynamic programme, the last one is the reference
    criterion_rules: str = "entropic"       # criterion the rules are calibrated with
    compare: tuple[str, ...] = ()           # rows of the thesis table, the network first
    tags: tuple[str, ...] = field(default_factory=tuple)

    # --- objects -----------------------------------------------------------------------------------------
    def market(self) -> GBMMarket:
        return GBMMarket(tuple(v * self.vol_scale for v in base_vols(self.n_assets)) if self.vol_scale != 1.0
                         else base_vols(self.n_assets), rho=self.rho)

    def option(self) -> BasketCall:
        return BasketCall(100.0)

    def problem(self, **premium_kwargs) -> HedgingProblem:
        market = self.market()
        premium = self.premium
        if premium is None:
            premium = option_premium(market, self.option(), **premium_kwargs).value
        return HedgingProblem(market, self.option(), self.cost, float(premium))

    def replace(self, **changes) -> ExperimentSpec:
        return replace(self, **changes)

    def smoke(self) -> ExperimentSpec:
        """A version that runs in seconds: small samples, a few training steps, tiny grids."""
        net = replace(self.network, steps=20, warmup=5, batch_size=512, validate_every=10)
        dp = (smoke_grid(self.dp[-1]),) if self.dp else ()
        return replace(self, name=self.name + "/smoke", validation=self.validation.smaller(1024),
                       test=self.test.smaller(2048), network=net, dp=dp, tags=(*self.tags, "smoke"))


def smoke_grid(grid: Grid) -> Grid:
    """A grid with the range of 'grid' and a few nodes, solved in a second."""
    if isinstance(grid, OneAssetGrid):
        return replace(grid, n_prices=61, n_positions=25, nodes_per_half=8)
    step = (grid.phi_max - grid.phi_min) / 12
    return replace(grid, n_prices=15, phi_min=-step, phi_max=-step + 12 * step, n_positions=13, nodes=4, nodes_last=4)
