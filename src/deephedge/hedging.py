"""The hedging problem and its one loss accounting.

Thesis, Section 3.4, equations (3.7) and (3.8). The hedger sells the option for the premium 'p', holds 'phi_t'
units of each asset between dates t and t+1, pays a proportional cost 'k' on every unit
traded, and liquidates the hedge at maturity. The loss of one path is

    L = payoff(S_N) - p - sum_t sum_i phi_t,i (S_t+1,i - S_t,i)
        + k sum_t sum_i S_t,i |phi_t,i - phi_t-1,i| + k sum_i S_N,i |phi_N-1,i|,

with phi_-1 = 0. Every strategy, the networks during training included, is measured by
:meth:`HedgingProblem.run`, so all of them are compared with exactly the same accounting.
"""
from __future__ import annotations

from dataclasses import dataclass, fields

import torch

from .market import GBMMarket
from .pricing import BasketCall, option_premium

Tensor = torch.Tensor


@dataclass(frozen=True)
class HedgeOutcome:
    """Per-path results, each a tensor [paths]."""

    loss: Tensor
    cost: Tensor
    turnover: Tensor  # units traded, summed over assets and dates (the final liquidation included)

    @staticmethod
    def concat(parts: list[HedgeOutcome]) -> HedgeOutcome:
        return HedgeOutcome(*(torch.cat([getattr(p, f.name) for p in parts]) for f in fields(HedgeOutcome)))


@dataclass(frozen=True)
class Batch:
    """Paths to hedge, optionally from a state other than "today, with no holdings".

    'initial_position' [P, n] is the holding carried into date 0, 'premium' [P] replaces the
    premium of the problem path by path, and 'first_date' [P] is the date from which each path
    trades: before it the strategy keeps the initial holding (and the paths keep the price
    constant). The last three are only used to train from random states (Section 5.2.1).
    """

    paths: Tensor
    initial_position: Tensor | None = None
    premium: Tensor | None = None
    first_date: Tensor | None = None

    def __len__(self) -> int:
        return self.paths.shape[0]

    def subset(self, index) -> Batch:
        return Batch(*(None if v is None else v[index] for v in (getattr(self, f.name) for f in fields(Batch))))

    def to(self, device, dtype: torch.dtype | None = None) -> Batch:
        def move(v, is_float):
            if v is None:
                return None
            return v.to(device, dtype) if (is_float and dtype is not None) else v.to(device)
        return Batch(move(self.paths, True), move(self.initial_position, True), move(self.premium, True),
                     move(self.first_date, False))


@dataclass(frozen=True)
class HedgingProblem:
    market: GBMMarket
    option: BasketCall
    cost: float
    premium: float

    @classmethod
    def create(cls, market: GBMMarket, cost: float = 0.0, *, option: BasketCall | None = None,
               premium: float | None = None, **premium_kwargs) -> HedgingProblem:
        """The problem for 'market', pricing the option with :func:`option_premium` unless 'premium' is given."""
        option = option or BasketCall(market.spot)
        if premium is None:
            premium = option_premium(market, option, **premium_kwargs).value
        return cls(market, option, float(cost), float(premium))

    @property
    def n_assets(self) -> int:
        return self.market.n_assets

    def trading_cost(self, prices: Tensor, trade: Tensor) -> Tensor:
        """k sum_i S_i |trade_i| for every path."""
        return (self.cost * prices * trade.abs()).sum(-1)

    def run(self, strategy, paths: Tensor, *, initial_position: Tensor | None = None, premium: Tensor | None = None,
            first_date: Tensor | None = None) -> HedgeOutcome:
        """Hedge every path of 'paths' [P, N + 1, n] to maturity (see :class:`Batch` for the options).

        The computation keeps the autograd graph (the trainer differentiates through it) and works in the
        dtype of 'paths': float32 while training, float64 when strategies are compared.
        """
        n_paths, n_dates = paths.shape[0], paths.shape[1] - 1
        position = (torch.zeros(n_paths, self.n_assets, device=paths.device, dtype=paths.dtype)
                    if initial_position is None else initial_position)
        gains = torch.zeros(n_paths, device=paths.device, dtype=paths.dtype)
        costs = torch.zeros_like(gains)
        turnover = torch.zeros_like(gains)
        for t in range(n_dates):
            prices = paths[:, t]
            new_position = strategy.act(t, prices, position)
            if first_date is not None:
                new_position = torch.where((first_date <= t)[:, None], new_position, position)
            trade = new_position - position
            costs = costs + self.trading_cost(prices, trade)
            turnover = turnover + trade.detach().abs().sum(-1)
            gains = gains + (new_position * (paths[:, t + 1] - prices)).sum(-1)
            position = new_position
        final = paths[:, -1]
        costs = costs + self.trading_cost(final, position)
        turnover = turnover + position.detach().abs().sum(-1)
        loss = self.option.payoff(final) - (self.premium if premium is None else premium) - gains + costs
        return HedgeOutcome(loss, costs, turnover)

    def run_batch(self, strategy, batch: Batch) -> HedgeOutcome:
        return self.run(strategy, batch.paths, initial_position=batch.initial_position, premium=batch.premium,
                        first_date=batch.first_date)

    @torch.no_grad()
    def evaluate(self, strategy, paths: Tensor | Batch, *, chunk: int = 50_000,
                 device: str | torch.device | None = None, dtype: torch.dtype = torch.float64) -> HedgeOutcome:
        """Outcome of 'strategy' on many paths, in blocks of 'chunk', returned as float64 CPU tensors."""
        batch = paths if isinstance(paths, Batch) else Batch(paths)
        device = device or batch.paths.device
        parts = []
        for i in range(0, len(batch), chunk):
            out = self.run_batch(strategy, batch.subset(slice(i, i + chunk)).to(device, dtype))
            parts.append(HedgeOutcome(*(x.to("cpu", torch.float64) for x in (out.loss, out.cost, out.turnover))))
        return HedgeOutcome.concat(parts)
