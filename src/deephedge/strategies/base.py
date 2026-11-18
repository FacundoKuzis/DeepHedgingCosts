"""The interface every hedging strategy implements.

A strategy sees, at each trading date 't', the current prices 'prices' [paths, n] and the
holdings it carries from the previous date 'position' [paths, n] (zero at t = 0), and returns
the holdings it keeps until the next date, on the device and with the dtype of 'prices'. That is all the
information the Markov problem of the thesis needs (Sections 2.7 and 3.2), so the networks, the delta
hedge, the asymptotic rules and the dynamic-programming optimum all fit the same signature and
are evaluated by the same accounting (:class:`deephedge.hedging.HedgingProblem`).
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch

Tensor = torch.Tensor


class HedgingStrategy(ABC):
    name: str = "strategy"

    @abstractmethod
    def act(self, t: int, prices: Tensor, position: Tensor) -> Tensor:
        """Holdings [paths, n] to keep from date 't' to 't + 1'."""

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.name!r})"


class NoTradeBand(HedgingStrategy):
    """A strategy that keeps each holding inside an interval and trades only to its nearest edge.

    With proportional costs the optimal strategy has this form (Davis and Norman, 1990; for one
    asset, Section 2.7 of the thesis): it does not trade while the holding is inside the band
    and, outside, it trades just enough to reach the edge. The Whalley-Wilmott band, the boxes of
    Atkinson and Ingpochai, the band of the dynamic programme and the band network all derive
    from this class and differ only in how they compute the band. Figures that compare bands call
    :meth:`band` on each of them.
    """

    @abstractmethod
    def band(self, t: int, prices: Tensor, position: Tensor) -> tuple[Tensor, Tensor]:
        """Lower and upper edges [paths, n]. They may depend on the position (the boxes do not)."""

    def act(self, t: int, prices: Tensor, position: Tensor) -> Tensor:
        lower, upper = self.band(t, prices, position)
        return torch.minimum(torch.maximum(position, lower), upper)


class NoHedge(HedgingStrategy):
    name = "no-hedge"

    def act(self, t: int, prices: Tensor, position: Tensor) -> Tensor:
        return torch.zeros_like(prices)
