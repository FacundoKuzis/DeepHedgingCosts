"""The corrected box: the no-trade region of Atkinson and Ingpochai to first order in the correlation.

Atkinson, C. and Ingpochai, P. (2012), The effect of correlation and transaction costs on the pricing of basket
options, Applied Mathematical Finance 19(2), 131-179, Section 5 and Appendices D and E. Thesis, Section 2.8.2
(the box, equation (2.20), and its correction).

Order zero. Measured from the frictionless (Levy) delta, the holding of asset i may deviate by at most a_i, the
half-width of :class:`~deephedge.strategies.bands.AtkinsonBox` (paper, equations (85)-(86)):

    a_i = ( (3/2) k S_i G_ii / |H_ii| )^(1/3),      G = (1/2) Gamma Sigma Gamma',      H = -(g/2) Sigma,

with Sigma the covariance of the price changes, Gamma the cross gammas and g the risk aversion (or the calibrated
scale). The region is a box because this order leaves out H_ij (i != j), the risk of being off the delta in two
correlated assets at once. Sigma is taken as the exact covariance of one step, Cov_t(S(t+1) - S(t)); every formula
below uses ratios of its entries, which do not depend on the unit of time.

First order. With positive correlation, deviations of opposite signs partly offset each other and deviations of the
same sign add up, so the box tilts: the edges of each asset move with the deviations of the others (paper, Section
5.3). The correction splits into one problem per pair of assets (i, j). On the box [-a_i, a_i] x [-a_j, a_j] of their
deviations Y_i and Y_j,

    G_ii I_ii + 2 G_ij I_ij + G_jj I_jj + 2 H_ij Y_i Y_j + H_1 = 0,     zero normal derivative on the four sides,

and each edge of asset i moves by Y_i^(+-1) = G_ii I_ii / (2 H_ii Y_i^(+-0)), with I_ii taken on that side and summed
over the pairs that contain i (paper, equations (96)-(97)).

The pair problem. With Y_i = a_i u, Y_j = a_j v and I = (H_ij a_i^3 a_j / G_ii) J it becomes

    J_uu + 2 c r J_uv + r^2 J_vv = -2 u v - lambda   on [-1, 1]^2,     zero normal derivative on the sides,

which depends on two numbers only. r = sqrt( (G_jj / a_j^2) / (G_ii / a_i^2) ) is the ratio between the volatilities
of the two deltas, each measured in units of its own half-width, and c = G_ij / sqrt(G_ii G_jj) is the correlation
between the changes of the two deltas. The constant lambda (H_1 of the paper) is the one that makes the Neumann
problem solvable. In a basket the deltas move almost together (c lies between 1 - 1e-6 and 1 - 1e-9) and the shifts
no longer depend on c, so :class:`PairCorrection` solves the problem once, with c = 1 - 1e-6, for 65 values of r
from 0.2 to 5 equally spaced in log r, with the nine-point scheme of the paper's Appendix D on 81 x 81 points, and
each state interpolates it, linearly in log r and along the side. Since J(-u, -v) = J(u, v), the upper and lower
edges of i and of j move by

    Y_i^(+1)(v) = f_i J_uu(1, v),    Y_i^(-1)(v) = -f_i J_uu(1, -v),    f_i = (Sigma_ij / Sigma_ii) a_j / 2,
    Y_j^(+1)(u) = f_j J_vv(u, 1),    Y_j^(-1)(u) = -f_j J_vv(-u, 1),    f_j = r^2 (Sigma_ij / Sigma_jj) a_i / 2.

H is proportional to g, so f_i and f_j do not depend on it: the correction deforms a calibrated box in the same
proportion as the theoretical one. Along a side of asset i, the deviation of j runs between the corrected corners,
from -a_j + Y_j^(-1) to a_j + Y_j^(+1), both evaluated on that side, and v is its position in that range mapped to
[-1, 1] (paper, Section 5.4). If the corrected edges of an asset cross, its interval collapses to their midpoint.

The rule. The interval of each asset depends on where the others are, so moving one asset moves the edges of the
rest. A sweep clips every deviation into its interval at once, with the edges computed from the deviations before
the sweep, and the thesis repeats it four times with two and three assets. With ten assets every asset moves the
edges of nine others and clipping all of them at once overshoots, so the damped version moves each deviation a
quarter of the way to its clipped value in each of 20 sweeps and ends with one more clip, which leaves the holding
inside its interval (thesis, Section 2.8.2).
"""
from __future__ import annotations

import math
import os
from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as sparse_linalg
import torch

from ..market import GBMMarket
from ..pricing import BasketCall, Greeks, frictionless_greeks
from .bands import AtkinsonBox

Tensor = torch.Tensor

GRID_POINTS = 81                 # points per side of the pair problem; with 41 the largest shift changes by < 0.5%
DELTA_CORRELATION = 1.0 - 1e-6   # c, the correlation between the changes of the two deltas
RATIOS = np.exp(np.linspace(math.log(0.2), math.log(5.0), 65))   # the values of r in the table


def solve_pair_problem(r: float, c: float = DELTA_CORRELATION, n: int = GRID_POINTS) -> tuple[np.ndarray, np.ndarray]:
    """J_uu(1, v) and J_vv(u, 1) of the normalised pair problem, at n equally spaced points of [-1, 1].

    The nine-point scheme of the paper's Appendix D: central differences on a grid of spacing h, with the normal
    derivative set to zero by reflecting the points outside the square (J(u_-1) = J(u_1)). A Neumann problem fixes J
    only up to a constant and is solvable for a single lambda, so both are found together from
    [A 1; 1' 0] [J; lambda] = [b; 0], which adds the free constant and asks J to have zero mean (the constant does
    not change the second derivatives). The second normal derivative on a side uses the same reflected points.
    """
    h = 2.0 / (n - 1)
    ku, kv = (index.ravel() for index in np.meshgrid(np.arange(n), np.arange(n), indexing="ij"))

    def reflect(index):
        return np.where(index < 0, -index, np.where(index > n - 1, 2 * (n - 1) - index, index))

    a_uu, a_vv, a_uv = 1.0 / h ** 2, r * r / h ** 2, 2 * c * r / (4 * h * h)
    stencil = [(1, 0, a_uu), (-1, 0, a_uu), (0, 1, a_vv), (0, -1, a_vv), (0, 0, -2 * a_uu - 2 * a_vv),
               (1, 1, a_uv), (-1, -1, a_uv), (1, -1, -a_uv), (-1, 1, -a_uv)]
    rows = np.concatenate([ku * n + kv] * len(stencil))          # unknown ku * n + kv is J(u_ku, v_kv)
    cols = np.concatenate([reflect(ku + du) * n + reflect(kv + dv) for du, dv, _ in stencil])
    values = np.concatenate([np.full(n * n, coefficient) for _, _, coefficient in stencil])
    operator = sparse.csc_matrix((values, (rows, cols)), shape=(n * n, n * n))   # reflected entries add up
    u = np.linspace(-1.0, 1.0, n)
    grid_u, grid_v = np.meshgrid(u, u, indexing="ij")
    ones = sparse.csc_matrix(np.ones((n * n, 1)))
    system = sparse.bmat([[operator, ones], [ones.T, None]], format="csc")
    solution = sparse_linalg.spsolve(system, np.append(-2.0 * (grid_u * grid_v).ravel(), 0.0))
    J = solution[:-1].reshape(n, n)
    return 2 * (J[-2, :] - J[-1, :]) / h ** 2, 2 * (J[:, -2] - J[:, -1]) / h ** 2


@dataclass(frozen=True)
class CorrectionProfiles:
    """J_uu(1, .) and J_vv(., 1) of every pair at its own r.

    The table rows 'index' and 'index + 1' around log r, mixed with 'weight': the half of the bilinear
    interpolation that depends on r only, done once per date. Outside the table, log r and the coordinate along the
    side are clamped to its ends.
    """

    j_uu: Tensor    # [n_r, n_side] the table, on the device of the state
    j_vv: Tensor
    index: Tensor   # [P, m]
    weight: Tensor  # [P, m]

    def uu(self, v: Tensor) -> Tensor:
        """J_uu(1, v), which moves the edges of the first asset of each pair; 'v' [P, m] in [-1, 1]."""
        return self._along_side(self.j_uu, v)

    def vv(self, u: Tensor) -> Tensor:
        """J_vv(u, 1), which moves the edges of the second asset of each pair."""
        return self._along_side(self.j_vv, u)

    def _along_side(self, table: Tensor, x: Tensor) -> Tensor:
        points = table.shape[1]
        g = (x.clamp(-1, 1) + 1) / 2 * (points - 1)
        k = g.floor().long().clamp(0, points - 2)
        w = (g - k.to(g.dtype)).clamp(0, 1)
        j, wj = self.index, self.weight
        below = (1 - w) * table[j, k] + w * table[j, k + 1]
        above = (1 - w) * table[j + 1, k] + w * table[j + 1, k + 1]
        return (1 - wj) * below + wj * above


class PairCorrection:
    """The pair problem solved for the table of r, and its interpolation.

    'PairCorrection()' solves the 65 problems (about two seconds). 'PairCorrection(path)' loads the table from
    'path' or, if the file does not exist, solves it and saves it there. The file is an '.npz' archive with the
    keys 'lr' (log r [65]), 'uu' (J_uu(1, v) [65, 81]) and 'vv' (J_vv(u, 1) [65, 81]).
    """

    def __init__(self, path: str | os.PathLike | None = None):
        if path is not None and Path(path).exists():
            with np.load(path) as stored:
                self.log_ratios, self.j_uu, self.j_vv = (torch.from_numpy(stored[k].copy()) for k in ("lr", "uu", "vv"))
        else:
            sides = [solve_pair_problem(r) for r in RATIOS]
            self.log_ratios = torch.tensor(np.log(RATIOS))
            self.j_uu = torch.tensor(np.array([uu for uu, _ in sides]))
            self.j_vv = torch.tensor(np.array([vv for _, vv in sides]))
            if path is not None:
                with open(path, "wb") as file:
                    np.savez(file, lr=self.log_ratios.numpy(), uu=self.j_uu.numpy(), vv=self.j_vv.numpy())
        self._by_device = {self.log_ratios.device: (self.log_ratios, self.j_uu, self.j_vv)}

    def at(self, log_ratio: Tensor) -> CorrectionProfiles:
        """The profiles of pairs with the given log r [P, m]."""
        log_ratios, j_uu, j_vv = self._on(log_ratio.device)
        rows = j_uu.shape[0]
        f = ((log_ratio - log_ratios[0]) / (log_ratios[1] - log_ratios[0])).clamp(0, rows - 1)
        index = f.floor().long().clamp(0, rows - 2)
        return CorrectionProfiles(j_uu, j_vv, index, (f - index.to(f.dtype)).clamp(0, 1))

    def _on(self, device: torch.device) -> tuple[Tensor, Tensor, Tensor]:
        if device not in self._by_device:
            self._by_device[device] = tuple(t.to(device) for t in (self.log_ratios, self.j_uu, self.j_vv))
        return self._by_device[device]


@cache
def default_pair_correction() -> PairCorrection:
    """The table of the thesis, solved on first use and shared by every rule built without one."""
    return PairCorrection()


@dataclass(frozen=True)
class EdgeShift:
    """How one edge of one asset of every pair moves with the deviation of the other asset (paper, (96)-(97)).

    For the upper edge ('sign' = 1) of the first asset of a pair the shift is Y_i^(+1)(v) = f_i J_uu(1, v), for its
    lower edge Y_i^(-1)(v) = -f_i J_uu(1, -v), and likewise for the second asset with f_j and J_vv. Here v is where
    the deviation of the other asset lies between 'start' and 'start + length', the corrected corners of this
    side, mapped to [-1, 1].
    """

    asset: Tensor                          # [m] whose edge moves
    other: Tensor                          # [m] whose deviation moves it
    sign: float                            # +1 for the upper edge, -1 for the lower one
    factor: Tensor                         # [P, m] f_i or f_j
    profile: Callable[[Tensor], Tensor]    # CorrectionProfiles.uu or .vv
    start: Tensor                          # [P, m]
    length: Tensor                         # [P, m]

    def __call__(self, deviation: Tensor) -> Tensor:
        """The shift [P, m] of this edge in every pair, given the deviations [P, n] of the holdings."""
        v = 2 * (deviation[:, self.other] - self.start) / self.length - 1
        return self.sign * self.factor * self.profile(self.sign * v)


@dataclass(frozen=True)
class BoxGeometry:
    """The corrected box at one date for every path, except for what depends on the holdings."""

    half_width: Tensor              # a [P, n], the order-zero box
    shifts: tuple[EdgeShift, ...]   # the upper and lower edges of the first asset of every pair, then of the second

    @classmethod
    def build(cls, prices: Tensor, gamma: Tensor, cov: Tensor, cost: float, scale: float,
              correction: PairCorrection) -> BoxGeometry:
        """From the prices [P, n], the cross gammas [P, n, n] and the covariance of one step [P, n, n] (float64)."""
        spread = gamma @ cov @ gamma.transpose(-1, -2)         # covariance of the changes of the deltas, 2 G
        var_price = torch.diagonal(cov, dim1=-2, dim2=-1)
        var_delta = torch.diagonal(spread, dim1=-2, dim2=-1)
        # the half-width of the box of order zero (thesis, eq. (2.20))
        half = (1.5 * cost * prices * var_delta / var_price / scale) ** (1.0 / 3.0)
        n = prices.shape[-1]
        first, second = torch.triu_indices(n, n, offset=1, device=prices.device)   # pairs i < j, in row order
        safe = half.clamp_min(1e-12)                           # far from the money the gamma and the box vanish
        rate_first = var_delta[:, first] / safe[:, first] ** 2   # G_ii / a_i^2, up to a factor common to the pair
        rate_second = var_delta[:, second] / safe[:, second] ** 2
        ratio = torch.sqrt((rate_second + 1e-300) / (rate_first + 1e-300))
        cov_pair = cov[:, first, second]
        factor_first = cov_pair / var_price[:, first] * half[:, second] / 2
        factor_second = ratio * ratio * cov_pair / var_price[:, second] * half[:, first] / 2
        profiles = correction.at(torch.log(ratio))
        ones = torch.ones_like(ratio)

        def edge(asset, other, sign, factor, profile, other_half, other_factor, other_profile) -> EdgeShift:
            # on this side the deviation of the other asset runs between the two corrected edges of that asset
            start = -other_half - other_factor * other_profile(-sign * ones)
            end = other_half + other_factor * other_profile(sign * ones)
            return EdgeShift(asset, other, sign, factor, profile, start, (end - start).clamp_min(1e-15))

        half_first, half_second = half[:, first], half[:, second]
        shifts = tuple(
            [edge(first, second, s, factor_first, profiles.uu, half_second, factor_second, profiles.vv)
             for s in (1.0, -1.0)]
            + [edge(second, first, s, factor_second, profiles.vv, half_first, factor_first, profiles.uu)
               for s in (1.0, -1.0)])
        return cls(half, shifts)

    def edges(self, deviation: Tensor) -> tuple[Tensor, Tensor]:
        """Lower and upper edges [P, n] of the interval of every asset, given the deviations [P, n] of the holdings."""
        lower, upper = -self.half_width, self.half_width.clone()
        for shift in self.shifts:
            (upper if shift.sign > 0 else lower).index_add_(1, shift.asset, shift(deviation))
        middle = 0.5 * (lower + upper)
        return torch.minimum(lower, middle), torch.maximum(upper, middle)


def _clip(x: Tensor, lower: Tensor, upper: Tensor) -> Tensor:
    return torch.minimum(torch.maximum(x, lower), upper)


class CorrectedBox(AtkinsonBox):
    """The box of Atkinson and Ingpochai corrected to first order in the correlation (see the module docstring).

    The defaults are the rule of the thesis with two and three assets: four simultaneous sweeps. With
    'sweeps=20, relaxation=0.25, final_clip=True' it is the damped rule used with ten assets. 'correction' is the
    tabulated pair problem (the shared :func:`default_pair_correction` if omitted). As in
    :class:`~deephedge.strategies.bands.AsymptoticBand`, 'centre' moves the region to the delta of other Greeks
    without changing its shape (thesis, Section 5.5), and :meth:`centre_and_halfwidth` still gives the order-zero box.
    """

    name = "corrected-box"

    def __init__(self, greeks: Greeks, cost: float, scale: float, correction: PairCorrection | None = None,
                 sweeps: int = 4, relaxation: float = 1.0, final_clip: bool = False, centre: Greeks | None = None):
        super().__init__(greeks, cost, scale, centre)
        if sweeps < 1 or not 0.0 < relaxation <= 1.0:
            raise ValueError(f"need sweeps >= 1 and 0 < relaxation <= 1, got {sweeps} and {relaxation}")
        self.correction = correction if correction is not None else default_pair_correction()
        self.sweeps = int(sweeps)
        self.relaxation = float(relaxation)
        self.final_clip = bool(final_clip)

    def geometry(self, t: int, prices: Tensor) -> tuple[Tensor, BoxGeometry]:
        """The centre [P, n] and the shape of the corrected box at date 't' (neither depends on the holdings)."""
        x = prices.to(torch.float64)
        delta, gamma = self.greeks.delta_gamma(t, x)
        centre = delta if self.centre_greeks is None else self.centre_greeks.delta(t, x)
        cov = self.greeks.market.step_covariance(x)
        return centre, BoxGeometry.build(x, gamma, cov, self.cost, self.scale, self.correction)

    def band(self, t: int, prices: Tensor, position: Tensor) -> tuple[Tensor, Tensor]:
        """The interval of every asset given the current holdings of the others (one evaluation of the edges)."""
        centre, geometry = self.geometry(t, prices)
        lower, upper = geometry.edges(position.to(torch.float64) - centre)
        return (centre + lower).to(prices.dtype), (centre + upper).to(prices.dtype)

    def act(self, t: int, prices: Tensor, position: Tensor) -> Tensor:
        """Bring the deviations from the centre into the region with 'sweeps' sweeps (see the module docstring)."""
        centre, geometry = self.geometry(t, prices)
        deviation = position.to(torch.float64) - centre
        for _ in range(self.sweeps):
            target = _clip(deviation, *geometry.edges(deviation))
            # relaxation 1 is the plain clip
            deviation = target if self.relaxation == 1.0 else deviation + self.relaxation * (target - deviation)
        if self.final_clip:
            deviation = _clip(deviation, *geometry.edges(deviation))
        return (centre + deviation).to(prices.dtype)


def corrected_box(market: GBMMarket, option: BasketCall, cost: float, scale: float, damped: bool = False, *,
                  centre_vol_scale: float = 1.0, correction: PairCorrection | None = None) -> CorrectedBox:
    """The corrected box of the thesis: four simultaneous sweeps, or with 'damped' the rule of the ten-asset runs.

    'centre_vol_scale' = c moves the centre to the delta computed with every volatility multiplied by c, as in
    :func:`~deephedge.strategies.bands.asymptotic_rule`.
    """
    greeks = frictionless_greeks(market, option)
    centre = None if centre_vol_scale == 1.0 else frictionless_greeks(market.scaled(centre_vol_scale), option)
    if damped:
        return CorrectedBox(greeks, cost, scale, correction, sweeps=20, relaxation=0.25, final_clip=True, centre=centre)
    return CorrectedBox(greeks, cost, scale, correction, centre=centre)
