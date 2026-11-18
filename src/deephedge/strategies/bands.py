"""Asymptotic no-trade rules for small proportional costs.

Whalley and Wilmott (1997) for one asset and the box of Atkinson and Ingpochai (2012) for a
basket share one formula (thesis, Section 2.8, equations (2.19) and (2.20)). Around the frictionless
delta, the holding of asset 'i' may drift by at most

    H_i = ( (3/2) k S_i G_i / g )^(1/3),

where 'g' is the risk aversion and 'G_i' measures how fast the delta of asset 'i' moves:

* one asset (Whalley-Wilmott): G = Gamma^2;
* basket (Atkinson-Ingpochai, order zero in the correlation): G_i = (Gamma C Gamma')_ii / C_ii,
  with Gamma the cross gammas of the Levy price and C = Cov_t(dS) the covariance of one step.

With one asset the two coincide ('tests/test_strategies.py'). The *theoretical* rule uses
the risk aversion of the criterion (g = 0.5); the *calibrated* rule replaces g by a scale
chosen on the validation sample (:mod:`deephedge.calibration`).

The *shifted centre* variant (thesis, Section 5.5) keeps the width and moves the centre to the
delta computed with every volatility multiplied by 'c'; with c = 1 it is the base rule.
"""
from __future__ import annotations

import torch

from ..pricing import Greeks, LevyGreeks, frictionless_greeks
from .base import NoTradeBand

Tensor = torch.Tensor


class AsymptoticBand(NoTradeBand):
    """Centre at a frictionless delta, half-width ((3/2) k S G / scale)^(1/3)."""

    def __init__(self, greeks: Greeks, cost: float, scale: float, centre: Greeks | None = None):
        self.greeks = greeks
        self.cost = float(cost)
        self.scale = float(scale)
        self.centre_greeks = centre  # None: the same Greeks as the width

    def curvature(self, t: int, prices: Tensor, gamma: Tensor) -> Tensor:
        """G_i [paths, n] from the gamma matrix [paths, n, n]."""
        raise NotImplementedError

    def centre_and_halfwidth(self, t: int, prices: Tensor) -> tuple[Tensor, Tensor]:
        x = prices.to(torch.float64)
        delta, gamma = self.greeks.delta_gamma(t, x)
        half = (1.5 * self.cost * x * self.curvature(t, x, gamma) / self.scale) ** (1.0 / 3.0)
        if self.centre_greeks is not None:
            delta = self.centre_greeks.delta(t, x)
        return delta, half

    def band(self, t: int, prices: Tensor, position: Tensor) -> tuple[Tensor, Tensor]:
        centre, half = self.centre_and_halfwidth(t, prices)
        return (centre - half).to(prices.dtype), (centre + half).to(prices.dtype)


class WhalleyWilmottBand(AsymptoticBand):
    name = "whalley-wilmott"

    def curvature(self, t: int, prices: Tensor, gamma: Tensor) -> Tensor:
        g = gamma[..., 0]
        return g * g


class AtkinsonBox(AsymptoticBand):
    name = "atkinson-box"

    def curvature(self, t: int, prices: Tensor, gamma: Tensor) -> Tensor:
        cov = self.greeks.market.step_covariance(prices)
        spread = torch.einsum("bip,bpq,biq->bi", gamma, cov, gamma)
        return spread / torch.diagonal(cov, dim1=-2, dim2=-1)


def asymptotic_rule(market, option, cost: float, scale: float, centre_vol_scale: float = 1.0) -> AsymptoticBand:
    """Whalley-Wilmott with one asset, the Atkinson-Ingpochai box with a basket."""
    greeks = frictionless_greeks(market, option)
    centre = None
    if centre_vol_scale != 1.0:
        shifted = market.scaled(centre_vol_scale)
        centre = frictionless_greeks(shifted, option) if market.n_assets == 1 else LevyGreeks(shifted, option)
    cls = WhalleyWilmottBand if market.n_assets == 1 else AtkinsonBox
    return cls(greeks, cost, scale, centre)
