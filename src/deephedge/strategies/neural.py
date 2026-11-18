"""The hedging networks (thesis, Sections 3.2, 3.3 and 4.3).

Two pieces make a neural strategy:

* the **features**, a fixed map from the state (date, prices) to the network inputs. Nothing
  else is given to the network: no delta, no rule, no optimum;
* the **network**, a multilayer perceptron with three hidden layers of 64 SiLU units, whose
  output is either the holdings (:class:`PositionNetwork`, without costs) or a no-trade band
  (:class:`BandNetwork`, with costs, after Imaki et al. 2021).

Inputs, with x = log(S / K), tau the time to maturity and st = sqrt(tau + dt):

* one asset (3 inputs): x / (sigma st), sqrt(tau / T), 5 x;
* basket of n assets (2n + 3 inputs): x_i / (sigma_i st) and 5 x_i for every asset, the same
  two for the basket level with the Levy volatility of the basket, and sqrt(tau / T). The band
  network of a basket also receives the n holdings (times n), so the region where it does not
  trade need not be a box.

With one asset the basket columns would repeat the asset columns, which is why the single-asset
set is three inputs and keeps the order of the networks trained for the thesis.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from ..market import GBMMarket
from ..pricing import BasketCall, LevyGreeks
from .base import HedgingStrategy, NoTradeBand

Tensor = torch.Tensor


# --- features ------------------------------------------------------------------------------------------
class SingleAssetFeatures:
    size = 3

    def __init__(self, market: GBMMarket, option: BasketCall):
        if market.n_assets != 1:
            raise ValueError("use BasketFeatures for more than one asset")
        self.dates, self.dt, self.maturity = market.dates, market.dt, market.maturity
        self.strike = option.strike
        self.vol = market.vols[0]  # 0.2 in every single-asset experiment of the thesis

    def __call__(self, t: int, prices: Tensor) -> Tensor:
        tau = (self.dates - t) * self.dt
        x = torch.log(prices[..., 0] / self.strike)
        st = math.sqrt(tau + self.dt)
        return torch.stack([x / (self.vol * st), torch.full_like(x, math.sqrt(tau / self.maturity)), 5.0 * x], -1)


class BasketFeatures:
    def __init__(self, market: GBMMarket, option: BasketCall):
        self.dates, self.dt, self.maturity = market.dates, market.dt, market.maturity
        self.option = option
        self.strike = option.strike
        self.vols = torch.tensor(market.vols, dtype=torch.float64)
        self.basket_vol = LevyGreeks(market, option).initial_vol
        self.size = 2 * market.n_assets + 3

    def __call__(self, t: int, prices: Tensor) -> Tensor:
        tau = (self.dates - t) * self.dt
        st = math.sqrt(tau + self.dt)
        x = torch.log(prices / self.strike)
        xb = torch.log(self.option.level(prices) / self.strike)
        sig = self.vols.to(prices.device, prices.dtype)
        time = torch.full_like(xb, math.sqrt(tau / self.maturity))
        return torch.cat([x / (sig * st), 5.0 * x, (xb / (self.basket_vol * st))[..., None], (5.0 * xb)[..., None],
                          time[..., None]], -1)


def features_for(market: GBMMarket, option: BasketCall):
    return SingleAssetFeatures(market, option) if market.n_assets == 1 else BasketFeatures(market, option)


# --- networks ------------------------------------------------------------------------------------------
def mlp(n_inputs: int, n_outputs: int, hidden: int = 64, layers: int = 3) -> nn.Sequential:
    modules, width = [], n_inputs
    for _ in range(layers):
        modules += [nn.Linear(width, hidden), nn.SiLU()]
        width = hidden
    modules.append(nn.Linear(width, n_outputs))
    return nn.Sequential(*modules)


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


class PositionNetwork(nn.Module):
    """Without costs the previous holding does not matter: the network outputs the holdings directly.

    The output starts at the equal split 0.5 / n of a half-delta (last layer initialised at zero) and
    is clipped to [-0.5, 1.5].
    """

    def __init__(self, n_inputs: int, n_assets: int, hidden: int = 64, layers: int = 3):
        super().__init__()
        self.n_assets = n_assets
        self.net = mlp(n_inputs, n_assets, hidden, layers)
        with torch.no_grad():
            self.net[-1].weight.zero_()
            self.net[-1].bias.zero_()

    def forward(self, features: Tensor) -> Tensor:
        return torch.clamp(0.5 / self.n_assets + self.net(features), -0.5, 1.5)


class BandNetwork(nn.Module):
    """With costs the network outputs, for every asset, the centre and half-width of a no-trade band.

    centre = (1.2 sigmoid(a) - 0.1) / n lies in [-0.1, 1.1] / n and half-width = softplus(b) / n is
    positive; the last layer starts small so every band starts near [0.5 - 0.05, 0.5 + 0.05] / n.
    The holding moves to the nearest edge only when it falls outside the band, through
    'position + relu(lower - position) - relu(position - upper)', which equals the clip and keeps a
    gradient on both sides.
    """

    def __init__(self, n_inputs: int, n_assets: int, hidden: int = 64, layers: int = 3, sees_position: bool = True):
        super().__init__()
        self.n_assets = n_assets
        self.sees_position = sees_position
        self.net = mlp(n_inputs + (n_assets if sees_position else 0), 2 * n_assets, hidden, layers)
        with torch.no_grad():
            self.net[-1].weight.mul_(0.1)
            initial_half = math.log(math.expm1(0.05))   # softplus^-1(0.05): every half-width starts at 0.05 / n
            self.net[-1].bias.copy_(torch.cat([torch.zeros(n_assets), torch.full((n_assets,), initial_half)]))

    def band(self, features: Tensor, position: Tensor) -> tuple[Tensor, Tensor]:
        n = self.n_assets
        inputs = torch.cat([features, position * n], -1) if self.sees_position else features
        out = self.net(inputs)
        centre = (1.2 * torch.sigmoid(out[..., :n]) - 0.1) / n
        half = nn.functional.softplus(out[..., n:]) / n
        return centre - half, centre + half

    def forward(self, features: Tensor, position: Tensor) -> Tensor:
        lower, upper = self.band(features, position)
        return position + torch.relu(lower - position) - torch.relu(position - upper)


# --- strategies ----------------------------------------------------------------------------------------
# The network runs in float32 on the device of its weights, and the holdings go back to the device and dtype of the
# prices, as with every other strategy: a network trained on a GPU can hedge paths that live on the CPU.
def _device(network: nn.Module) -> torch.device:
    return next(network.parameters()).device


class NeuralPosition(HedgingStrategy):
    name = "network"

    def __init__(self, network: PositionNetwork, features):
        self.network = network
        self.features = features

    def act(self, t: int, prices: Tensor, position: Tensor) -> Tensor:
        features = self.features(t, prices.float()).to(_device(self.network))
        return self.network(features).to(prices.device, prices.dtype)


class NeuralBand(NoTradeBand):
    name = "network"

    def __init__(self, network: BandNetwork, features):
        self.network = network
        self.features = features

    def _inputs(self, t: int, prices: Tensor, position: Tensor) -> tuple[Tensor, Tensor]:
        device = _device(self.network)
        return self.features(t, prices.float()).to(device), position.float().to(device)

    def band(self, t: int, prices: Tensor, position: Tensor) -> tuple[Tensor, Tensor]:
        lower, upper = self.network.band(*self._inputs(t, prices, position))
        return lower.to(prices.device, prices.dtype), upper.to(prices.device, prices.dtype)

    def act(self, t: int, prices: Tensor, position: Tensor) -> Tensor:
        # the network's own projection, the one it was trained with
        return self.network(*self._inputs(t, prices, position)).to(prices.device, prices.dtype)


def neural_strategy(market: GBMMarket, option: BasketCall, *, with_costs: bool, seed: int = 1, hidden: int = 64,
                    layers: int = 3) -> NeuralPosition | NeuralBand:
    """A fresh network strategy; 'seed' fixes the initial weights (torch's global generator)."""
    features = features_for(market, option)
    n = market.n_assets
    torch.manual_seed(seed)
    if with_costs:
        return NeuralBand(BandNetwork(features.size, n, hidden, layers, sees_position=n > 1), features)
    return NeuralPosition(PositionNetwork(features.size, n, hidden, layers), features)
