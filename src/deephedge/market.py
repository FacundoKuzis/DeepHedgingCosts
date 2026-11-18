"""The market: correlated geometric Brownian motions observed once per trading day.

Thesis, Sections 2.3 and 4.1. Interest rates are zero and every asset is a martingale. Between two
trading dates the log price of asset 'i' moves by

    log S_i(t+1) - log S_i(t) = m_i + s_i * eps_i,      eps = L z,

where 'z' are independent standard normal shocks truncated to '|z| <= z_max', 'L' is
the Cholesky factor of the correlation matrix (one common correlation 'rho' between
every pair), 's_i = sigma_i * sqrt(dt)' and 'm_i' is the drift that makes 'E[S_i(t+1)]
= S_i(t)' exactly under the truncated shocks. Truncating at eight standard deviations is
invisible in any simulation of realistic size, but it keeps every moment finite, so the
entropic risk of every strategy is finite and the dynamic programme integrates over a
bounded interval.

A market with one volatility is the single-asset case of the thesis: nothing else in the
package has separate code for it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from functools import cached_property

import torch

Tensor = torch.Tensor


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def truncated_normal_mgf(c: float, z_max: float) -> float:
    """E[exp(c Z)] for a standard normal Z truncated to [-z_max, z_max]."""
    a = z_max
    return math.exp(0.5 * c * c) * (normal_cdf(a - c) - normal_cdf(-a - c)) / (normal_cdf(a) - normal_cdf(-a))


@dataclass(frozen=True)
class GBMMarket:
    """'n' equicorrelated GBM assets, all starting at 'spot', with 'dates' daily steps.

    Prices are returned as tensors of shape '[paths, dates + 1, n_assets]'; date 0 is today.
    """

    vols: tuple[float, ...]
    rho: float = 0.5
    spot: float = 100.0
    dates: int = 50
    dt: float = 1.0 / 252.0
    z_max: float = 8.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "vols", tuple(float(v) for v in self.vols))
        if not self.vols or min(self.vols) <= 0:
            raise ValueError("every asset needs a positive volatility")
        n = len(self.vols)
        if n > 1 and not (-1.0 / (n - 1) < self.rho <= 1.0):
            raise ValueError(f"rho={self.rho} does not give a valid correlation matrix for {n} assets")

    # --- sizes and times ---------------------------------------------------------------------------------
    @property
    def n_assets(self) -> int:
        return len(self.vols)

    @property
    def maturity(self) -> float:
        return self.dates * self.dt

    def time_to_maturity(self, t: int) -> float:
        return (self.dates - t) * self.dt

    # --- the law of one step -----------------------------------------------------------------------------
    @cached_property
    def correlation(self) -> Tensor:
        c = torch.full((self.n_assets, self.n_assets), self.rho, dtype=torch.float64)
        c.fill_diagonal_(1.0)
        return c

    @cached_property
    def cholesky(self) -> Tensor:
        n = self.n_assets
        if self.rho >= 1.0:  # every asset is driven by the same shock
            factor = torch.zeros(n, n, dtype=torch.float64)
            factor[:, 0] = 1.0
            return factor
        return torch.linalg.cholesky(self.correlation)

    @cached_property
    def step_vols(self) -> Tensor:
        """s_i = sigma_i sqrt(dt)."""
        return torch.tensor(self.vols, dtype=torch.float64) * math.sqrt(self.dt)

    @cached_property
    def drifts(self) -> Tensor:
        """m_i = -log E[exp(s_i eps_i)], computed factor by factor because eps_i = sum_j L_ij z_j."""
        s, chol = self.step_vols, self.cholesky
        n = self.n_assets
        return torch.tensor(
            [-sum(math.log(truncated_normal_mgf(float(s[i] * chol[i, j]), self.z_max)) for j in range(n))
             for i in range(n)],
            dtype=torch.float64,
        )

    def step_covariance(self, prices: Tensor) -> Tensor:
        """Cov_t(S(t+1) - S(t)) given S(t) = 'prices' [..., n], as an [..., n, n] tensor.

        Uses the Gaussian second moment; the truncation at 'z_max' changes it by less than 1e-12.
        """
        m = self.drifts.to(prices.device, prices.dtype)
        s = self.step_vols.to(prices.device, prices.dtype)
        corr = self.correlation.to(prices.device, prices.dtype)
        growth = torch.exp(m[:, None] + m[None, :] + 0.5 * (s[:, None] ** 2 + s[None, :] ** 2)
                           + corr * s[:, None] * s[None, :]) - 1.0
        return prices[..., :, None] * prices[..., None, :] * growth

    # --- simulation --------------------------------------------------------------------------------------
    def sample_shocks(self, shape: tuple[int, ...], generator: torch.Generator | None = None, *,
                      device: str | torch.device = "cpu", dtype: torch.dtype = torch.float64) -> Tensor:
        """Independent standard normals truncated to |z| <= z_max (a rejected draw is drawn again)."""
        z = torch.randn(shape, device=device, dtype=dtype, generator=generator)
        rejected = z.abs() > self.z_max
        for _ in range(20):
            if not bool(rejected.any()):
                break
            z = torch.where(rejected, torch.randn(shape, device=device, dtype=dtype, generator=generator), z)
            rejected = z.abs() > self.z_max
        return z

    def simulate(self, n_paths: int, generator: torch.Generator | None = None, *,
                 device: str | torch.device = "cpu", dtype: torch.dtype = torch.float64) -> Tensor:
        """Price paths [n_paths, dates + 1, n_assets]."""
        n = self.n_assets
        z = self.sample_shocks((n_paths, self.dates, n), generator, device=device, dtype=dtype)
        eps = z @ self.cholesky.T.to(device, dtype)
        log_returns = self.drifts.to(device, dtype) + self.step_vols.to(device, dtype) * eps
        if n == 1:  # noqa: SIM108
            # One asset is accumulated as a [paths, dates] tensor. On a GPU its scan rounds differently from
            # the scan of [paths, dates, 1], and this layout gives the training batches of the thesis exactly.
            log_paths = torch.cumsum(log_returns[..., 0], dim=1)[..., None]
        else:
            log_paths = torch.cumsum(log_returns, dim=1)
        start = torch.zeros(n_paths, 1, n, device=device, dtype=dtype)
        return self.spot * torch.exp(torch.cat([start, log_paths], dim=1))

    def simulate_in_blocks(self, n_paths: int, seed: int, block: int) -> Tensor:
        """'n_paths' float64 CPU paths drawn in blocks, block 'j' with its own generator 'seed * 1000 + j'.

        This is how the ten-asset samples of the thesis were drawn (blocks of 25,000 paths).
        """
        parts = [self.simulate(min(block, n_paths - i), torch.Generator().manual_seed(seed * 1000 + j))
                 for j, i in enumerate(range(0, n_paths, block))]
        return torch.cat(parts)

    # --- variants used by the experiments ----------------------------------------------------------------
    def scaled(self, factor: float) -> GBMMarket:
        """Every volatility multiplied by 'factor'."""
        return replace(self, vols=tuple(v * factor for v in self.vols))

    def with_correlation(self, rho: float) -> GBMMarket:
        return replace(self, rho=rho)
