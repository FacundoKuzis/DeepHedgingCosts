"""The dynamic-programming optimum with one asset (thesis, Sections 2.7 and 2.9).

With exponential utility the wealth of the hedger separates out, so the state at date 't' is the price, through
'x = log(S / K)', and the holding 'p' carried from the previous date. Let 'V_t(x, p)' be the smallest
attainable 'log E_t[exp(g L_t)]', where 'L_t' is the part of the loss generated from 't' on and 'g' the risk
aversion. Then (Hodges and Neuberger, 1989; Bellman's recursion)

    V_N(x, p)   = g [(S - K)^+ + k S |p|],
    A_t(x, phi) = log E_t[exp(-g phi (S_t+1 - S_t) + V_t+1(x_t+1, phi))],
    V_t(x, p)   = min_phi [g k S |phi - p| + A_t(x, phi)],

and the entropic risk of the optimal strategy is 'V_0(x_0, 0) / g - premium'. 'A_t' is convex in 'phi', so the
optimum is a no-trade band 'phi_t = min(max(p, l_t), u_t)' with 'l_t = argmin [A_t + g k S phi]' and
'u_t = argmin [A_t - g k S phi]' (Davis, Panas and Zariphopoulou, 1993; thesis, Section 2.7).

Numerics:

* 'x' and the holding live on uniform grids (:class:`OneAssetGrid`), and the expectation is a Gauss-Legendre
  quadrature of the truncated normal shock. From every node the log price moves by the same 'm + s z', so
  'V_t+1' at the destination is a linear interpolation along 'x'.
* The interpolated function is 'W = V - g C', with 'C' the Black-Scholes price (the payoff at maturity). 'W'
  curves far less than 'V' in 'x'; interpolating it and adding 'g C' exactly at the destination removes the
  interpolation error that the curvature of the option would cause. 'C' acts as a control variate.
* 'A_t' is computed in log space (logsumexp), so the exponentials never overflow.
* The minimum over 'phi' is the L1 envelope of 'A_t', exact on the grid, and the band edges are the two argmins
  above, refined between grid nodes with a parabola.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, fields

import torch

from ..hedging import HedgingProblem
from ..pricing import black_scholes_call
from ..strategies.base import NoTradeBand
from .common import l1_envelope, refined_argmin, truncated_normal_quadrature

Tensor = torch.Tensor


@dataclass(frozen=True)
class OneAssetGrid:
    """Grids of the programme: 'x = log(S / K)' on 'n_prices' nodes in '[-x_half, x_half]', the holding on
    'n_positions' nodes in '[phi_min, phi_max]', and '2 * nodes_per_half' quadrature nodes."""

    x_half: float
    n_prices: int
    n_positions: int
    nodes_per_half: int
    phi_min: float = -0.1
    phi_max: float = 1.1

    def __post_init__(self) -> None:
        if min(self.n_prices, self.n_positions) < 3 or not self.phi_min < 0.0 < self.phi_max:
            raise ValueError("each grid needs three nodes or more, and the holdings must include zero")

    @classmethod
    def fine(cls) -> OneAssetGrid:
        """The grid of every one-asset result of the thesis (Section 2.9)."""
        return cls(x_half=0.9, n_prices=721, n_positions=241, nodes_per_half=64)

    @classmethod
    def coarse(cls) -> OneAssetGrid:
        """Half the resolution; the thesis compared it with the fine grid to check convergence."""
        return cls(x_half=0.9, n_prices=361, n_positions=121, nodes_per_half=48)


class OneAssetDP:
    """Backward induction for a one-asset :class:`~deephedge.hedging.HedgingProblem`.

    'risk_aversion' is the 'g' of the entropic risk (0.5 in the thesis) and 'grid' defaults to
    :meth:`OneAssetGrid.fine`. :meth:`solve` returns the optimal strategy as an :class:`OptimalBand`.
    """

    # Price nodes per block of the [rows, quadrature nodes, positions] tensor, which bounds the memory.
    ROWS_PER_BLOCK = 128

    def __init__(self, problem: HedgingProblem, risk_aversion: float = 0.5, grid: OneAssetGrid | None = None):
        if problem.n_assets != 1:
            raise ValueError("OneAssetDP needs a single asset; use TwoAssetDP for a basket of two")
        self.problem = problem
        self.risk_aversion = float(risk_aversion)
        self.grid = grid or OneAssetGrid.fine()

    def control(self, t: int, prices: Tensor) -> Tensor:
        """'C_t': the Black-Scholes price at date 't', and the payoff at maturity."""
        market, option = self.problem.market, self.problem.option
        if t >= market.dates:
            return option.payoff(prices[..., None])
        return black_scholes_call(prices, option.strike, market.time_to_maturity(t),
                                  torch.full_like(prices, market.vols[0]))

    def solve(self, device: str | torch.device = "cpu") -> OptimalBand:
        """Run the recursion from maturity back to date 0 on 'device', in float64.

        With the fine grid it takes about 3 s on a GPU and 10 to 20 s on a CPU.
        """
        start = time.time()
        market, grid = self.problem.market, self.grid
        g, k, strike = self.risk_aversion, self.problem.cost, self.problem.option.strike
        dtype = torch.float64
        nx, dates = grid.n_prices, market.dates
        xs = torch.linspace(-grid.x_half, grid.x_half, nx, device=device, dtype=dtype)
        phi = torch.linspace(grid.phi_min, grid.phi_max, grid.n_positions, device=device, dtype=dtype)
        prices = strike * torch.exp(xs)
        dx, dphi = xs[1] - xs[0], phi[1] - phi[0]
        nodes, weights = truncated_normal_quadrature(grid.nodes_per_half, market.z_max)
        nodes = torch.tensor(nodes, device=device, dtype=dtype)
        log_weights = torch.log(torch.tensor(weights, device=device, dtype=dtype))
        jump = float(market.drifts[0]) + float(market.step_vols[0]) * nodes  # move of x at each node
        growth = torch.exp(jump)  # S_t+1 / S_t
        relative_move = growth - 1.0  # (S_t+1 - S_t) / S_t

        w = g * k * prices[:, None] * phi.abs()[None, :]  # W_N = V_N - g (S - K)^+, on [x, p]
        lower = torch.empty(dates, nx, dtype=dtype, device=device)
        upper = torch.empty_like(lower)
        for t in reversed(range(dates)):
            a = torch.empty(nx, grid.n_positions, dtype=dtype, device=device)  # A_t on [x, phi]
            for first in range(0, nx, self.ROWS_PER_BLOCK):
                rows = slice(first, min(nx, first + self.ROWS_PER_BLOCK))
                cell = (xs[rows, None] + jump[None, :] - xs[0]) / dx  # destination, in grid steps [rows, nodes]
                left = torch.floor(cell).clamp(0, nx - 2).long()
                frac = (cell - left.to(dtype)).clamp(0.0, 1.0)
                w_left, w_right = w[left], w[left + 1]
                v_next = (w_left + (w_right - w_left) * frac[..., None]
                          + g * self.control(t + 1, prices[rows, None] * growth[None, :])[..., None])
                gain = prices[rows, None] * relative_move[None, :]
                exponent = log_weights[None, :, None] - g * phi[None, None, :] * gain[..., None] + v_next
                a[rows] = torch.logsumexp(exponent, dim=1)
            unit_cost = (g * k * prices)[:, None]
            lower[t] = refined_argmin(a + unit_cost * phi[None, :], phi)
            upper[t] = refined_argmin(a - unit_cost * phi[None, :], phi)
            v = l1_envelope(a, unit_cost * dphi)  # V_t on [x, p]
            w = v - g * self.control(t, prices)[:, None]
        return OptimalBand(strike=strike, spot=market.spot, risk_aversion=g, premium=self.problem.premium,
                           log_moneyness=xs.cpu(), positions=phi.cpu(), lower=lower.float().cpu(),
                           upper=upper.float().cpu(), initial_value=v.cpu(), seconds=time.time() - start)


def _cell(grid: Tensor, x: Tensor) -> tuple[Tensor, Tensor]:
    """Left node of the grid cell that holds 'x' and the position of 'x' in it (in [0, 1], flat outside)."""
    position = (x - grid[0]) / (grid[1] - grid[0])
    left = torch.floor(position).clamp(0, grid.shape[0] - 2).long()
    return left, (position - left.to(position.dtype)).clamp(0.0, 1.0)


def _interpolate(table: Tensor, left: Tensor, weight: Tensor) -> Tensor:
    return table[left] * (1 - weight) + table[left + 1] * weight


@dataclass(frozen=True, eq=False, repr=False)
class OptimalBand(NoTradeBand):
    """The band of the one-asset programme, tabulated on its log-moneyness grid with one row per date.

    :meth:`band` interpolates the edges linearly in 'x = log(S / K)' (constant beyond the grid) and 'act',
    inherited from :class:`NoTradeBand`, clips the inherited holding to them, which is the optimal decision. The
    edges are kept in float32 and interpolated in float64. 'initial_value' is the table
    'V_0(x, p)' and 'seconds' the time the solve took.
    """

    strike: float
    spot: float
    risk_aversion: float
    premium: float
    log_moneyness: Tensor  # [n_prices] float64
    positions: Tensor  # [n_positions] float64
    lower: Tensor  # [dates, n_prices] float32
    upper: Tensor  # [dates, n_prices] float32
    initial_value: Tensor  # [n_prices, n_positions] float64
    seconds: float

    name = "dynamic-programming"

    def band(self, t: int, prices: Tensor, position: Tensor) -> tuple[Tensor, Tensor]:
        x = torch.log(prices / self.strike).to(torch.float64)
        left, weight = _cell(self.log_moneyness.to(x.device), x)
        lower = _interpolate(self.lower[t].to(x.device, torch.float64), left, weight)
        upper = _interpolate(self.upper[t].to(x.device, torch.float64), left, weight)
        return lower.to(prices.dtype), upper.to(prices.dtype)

    def predicted_risk(self) -> float:
        """The entropic risk the programme predicts for its own strategy, 'V_0(x_0, 0) / g - premium'.

        'V_0' is interpolated linearly at the initial holding 0 and at 'x_0 = log(spot / strike)'.
        """
        phi = self.positions
        cell = float((0.0 - phi[0]) / (phi[1] - phi[0]))
        j = min(max(math.floor(cell), 0), phi.shape[0] - 2)
        w = cell - j
        value_at_zero = self.initial_value[..., j] * (1.0 - w) + self.initial_value[..., j + 1] * w
        x0 = torch.full((1,), math.log(self.spot / self.strike), dtype=torch.float64)
        v0 = _interpolate(value_at_zero, *_cell(self.log_moneyness, x0))
        g = self.risk_aversion
        return float((v0 - g * self.premium) / g)

    def state_dict(self) -> dict:
        """The fields, for 'torch.save'; :meth:`from_state_dict` rebuilds the strategy from them."""
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_state_dict(cls, state: dict) -> OptimalBand:
        return cls(**{f.name: state[f.name] for f in fields(cls)})
