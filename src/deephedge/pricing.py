"""The option and the frictionless prices and Greeks the classical rules are built on.

* One asset: Black and Scholes in closed form.
* Baskets: the lognormal approximation of Levy (1992), which gives the equal-weight
  arithmetic basket a lognormal law with its first two moments. Its delta and cross gammas
  come from automatic differentiation of that price (thesis, Section 2.4).

The premium the hedger receives is the Black-Scholes price with one asset and a Monte Carlo
estimate of the basket price otherwise (thesis, Section 4.1).
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

from .market import GBMMarket

Tensor = torch.Tensor
SQRT2 = math.sqrt(2.0)
INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


def normal_cdf(x: Tensor) -> Tensor:
    return 0.5 * (1.0 + torch.erf(x / SQRT2))


def black_scholes_call(spot, strike: float, tau, vol) -> Tensor:
    """Black-Scholes call price with zero rates (float64)."""
    spot = torch.as_tensor(spot, dtype=torch.float64)
    tau = torch.as_tensor(tau, dtype=torch.float64)
    vol = torch.as_tensor(vol, dtype=torch.float64)
    sq = vol * torch.sqrt(tau)
    d1 = (torch.log(spot / strike) + 0.5 * sq * sq) / sq
    return spot * normal_cdf(d1) - strike * normal_cdf(d1 - sq)


def black_scholes_delta_gamma(spot: Tensor, strike: float, tau: float, vol: float) -> tuple[Tensor, Tensor]:
    sq = vol * math.sqrt(tau)
    d1 = (torch.log(spot / strike) + 0.5 * sq * sq) / sq
    delta = normal_cdf(d1)
    gamma = torch.exp(-0.5 * d1 * d1) * INV_SQRT_2PI / (spot * sq)
    return delta, gamma


@dataclass(frozen=True)
class BasketCall:
    """European call on the equal-weight arithmetic basket; with one asset, a plain call."""

    strike: float = 100.0

    def level(self, prices: Tensor) -> Tensor:
        return prices.mean(-1)

    def payoff(self, prices: Tensor) -> Tensor:
        return torch.clamp(self.level(prices) - self.strike, min=0.0)


class Greeks(ABC):
    """Frictionless delta and gammas at trading date 't'; 'prices' is [..., n]."""

    def __init__(self, market: GBMMarket, option: BasketCall):
        self.market = market
        self.option = option

    def tau(self, t: int) -> float:
        return self.market.time_to_maturity(t)

    @abstractmethod
    def delta(self, t: int, prices: Tensor) -> Tensor:
        """Holdings [..., n] that replicate the option without frictions."""

    @abstractmethod
    def delta_gamma(self, t: int, prices: Tensor) -> tuple[Tensor, Tensor]:
        """Delta [..., n] and the gamma matrix [..., n, n] (row i is the gradient of delta_i)."""


class BlackScholesGreeks(Greeks):
    """Closed-form Greeks for a single asset."""

    def __init__(self, market: GBMMarket, option: BasketCall):
        if market.n_assets != 1:
            raise ValueError("Black-Scholes Greeks need a single asset; use LevyGreeks for baskets")
        super().__init__(market, option)
        self.vol = market.vols[0]

    def delta(self, t: int, prices: Tensor) -> Tensor:
        return black_scholes_delta_gamma(prices, self.option.strike, self.tau(t), self.vol)[0]

    def delta_gamma(self, t: int, prices: Tensor) -> tuple[Tensor, Tensor]:
        delta, gamma = black_scholes_delta_gamma(prices, self.option.strike, self.tau(t), self.vol)
        return delta, gamma[..., None]


class LevyGreeks(Greeks):
    """Levy's lognormal approximation of the basket call; reduces to Black-Scholes with one asset."""

    def vol(self, prices: Tensor, tau: float) -> Tensor:
        """Annual volatility of the lognormal law that matches E[B_T] and E[B_T^2]."""
        m = self.market
        mean = prices.mean(-1)
        sig = torch.tensor(m.vols, dtype=prices.dtype, device=prices.device)
        corr = m.correlation.to(prices.device, prices.dtype)
        growth = torch.exp(corr * sig[:, None] * sig[None, :] * tau)
        second = (prices[..., :, None] * prices[..., None, :] * growth).sum((-1, -2)) / m.n_assets ** 2
        return torch.sqrt(torch.clamp(torch.log(second / (mean * mean)), min=1e-16) / tau)

    def price(self, prices: Tensor, tau: float) -> Tensor:
        level = prices.mean(-1)
        sd = self.vol(prices, tau) * math.sqrt(tau)
        d1 = (torch.log(level / self.option.strike) + 0.5 * sd * sd) / sd
        return level * normal_cdf(d1) - self.option.strike * normal_cdf(d1 - sd)

    @property
    def initial_vol(self) -> float:
        """Levy volatility of the basket today, the scale of the basket inputs of the network."""
        start = torch.full((self.market.n_assets,), self.market.spot, dtype=torch.float64)
        return float(self.vol(start, self.market.maturity))

    def delta(self, t: int, prices: Tensor) -> Tensor:
        with torch.enable_grad():
            x = prices.detach().clone().requires_grad_(True)
            (grad,) = torch.autograd.grad(self.price(x, self.tau(t)).sum(), x)
        return grad.detach()

    def delta_gamma(self, t: int, prices: Tensor) -> tuple[Tensor, Tensor]:
        n = prices.shape[-1]
        with torch.enable_grad():
            x = prices.detach().to(torch.float64).clone().requires_grad_(True)
            (delta,) = torch.autograd.grad(self.price(x, self.tau(t)).sum(), x, create_graph=True)
            rows = [torch.autograd.grad(delta[..., i].sum(), x, retain_graph=i < n - 1)[0] for i in range(n)]
        return delta.detach(), torch.stack(rows, -2).detach()


def frictionless_greeks(market: GBMMarket, option: BasketCall) -> Greeks:
    """Black-Scholes for one asset (as in the thesis), Levy for baskets."""
    return BlackScholesGreeks(market, option) if market.n_assets == 1 else LevyGreeks(market, option)


@dataclass(frozen=True)
class Premium:
    value: float
    standard_error: float = 0.0


def option_premium(market: GBMMarket, option: BasketCall, *, mc_paths: int = 4_000_000, mc_block: int = 500_000,
                   mc_seed: int = 950005, device: str | torch.device = "cpu") -> Premium:
    """The price charged for the option: Black-Scholes with one asset, Monte Carlo for a basket.

    The basket estimate draws 'mc_paths' paths in blocks of 'mc_block' (the last one with what is left) from
    one generator seeded with 'mc_seed' on 'device'. Different devices or block sizes draw different paths, so
    to reproduce a thesis premium digit for digit the same device and block size must be used.
    """
    if market.n_assets == 1:
        return Premium(float(black_scholes_call(market.spot, option.strike, market.maturity, market.vols[0])))
    if mc_paths < 1 or mc_block < 1:
        raise ValueError("mc_paths and mc_block must be positive")
    gen = torch.Generator(device=device)
    gen.manual_seed(mc_seed)
    total = squares = 0.0
    count = 0
    for start in range(0, mc_paths, mc_block):
        paths = market.simulate(min(mc_block, mc_paths - start), gen, device=device, dtype=torch.float64)
        pay = option.payoff(paths[:, -1])
        total += float(pay.sum())
        squares += float((pay * pay).sum())
        count += pay.shape[0]
        del paths
    mean = total / count
    return Premium(mean, math.sqrt(max(squares / count - mean * mean, 0.0) / count))
