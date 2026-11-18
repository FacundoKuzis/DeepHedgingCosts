"""The frictionless delta hedge: Black-Scholes with one asset, Levy's approximation for a basket."""
from __future__ import annotations

import torch

from ..pricing import Greeks
from .base import HedgingStrategy

Tensor = torch.Tensor


class DeltaHedge(HedgingStrategy):
    """Rebalances to the frictionless delta at every date, whatever it costs.

    Without costs it is the benchmark of Section 5.1 (with one asset it is the optimum of the
    continuous-time problem); with costs it shows how much trading every day costs.
    """

    name = "delta"

    def __init__(self, greeks: Greeks):
        self.greeks = greeks

    def act(self, t: int, prices: Tensor, position: Tensor) -> Tensor:
        return self.greeks.delta(t, prices.to(torch.float64)).to(prices.dtype)
