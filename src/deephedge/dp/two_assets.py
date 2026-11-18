"""The dynamic-programming optimum with two assets (thesis, Sections 2.7 and 2.9).

The problem is the one of :mod:`deephedge.dp.one_asset` with the state '(x_1, x_2, p_1, p_2)': 'x_i = log(S_i /
K)' and 'p' the holdings carried from the previous date. With 'B = (S_1 + S_2) / 2' the basket,

    V_N(x, p)   = g [(B - K)^+ + k sum_i S_i |p_i|],
    A_t(x, f)   = log E_t[exp(-g sum_i f_i (S_i,t+1 - S_i,t) + V_t+1(x_t+1, f))],
    V_t(x, p)   = min_f [g k sum_i S_i |f_i - p_i| + A_t(x, f)],

and the entropic risk of the optimal strategy is 'V_0(0, 0) / g - premium'. With two assets the no-trade region is
a two-dimensional set whose shape is only known through the recursion, so the strategy keeps the tables 'A_t' and
takes the minimum over the grid of holdings at every state it meets (:class:`OptimalGridStrategy`).

Numerics:

* Every log price moves by the same amount from every node, so for each quadrature node 'V_t+1(x + d, f)' is the
  table shifted along both price axes: a bilinear interpolation with the same weights everywhere.
* The interpolated function is 'W = V - g C', with 'C' the Levy (1992) price of the basket call (the payoff at
  maturity), and 'g C' is added exactly at the shifted point: a control variate, as with one asset.
* The expectation is a product Gauss-Hermite rule over the two independent shocks, with more nodes in the last
  dates, where the kink of the payoff is close. The mean of the exponentials is accumulated relative to the value
  without moves, so it never overflows. With two identical assets the second shock moves no price and gets one node.
* The cost is separable across assets, so the minimum over '(f_1, f_2)' is two L1 envelopes, one axis at a time.
* The tables are float32. 'A_t' is written to disk for every date with the layout '[m, n, n, m] = (f_1, x_1, x_2,
  f_2)', '4 m^2 n^2' bytes per date: 12.7 GB for the 101-price, 79-position grid of the thesis.
"""
from __future__ import annotations

import contextlib
import json
import tempfile
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch

from ..hedging import HedgingProblem
from ..pricing import LevyGreeks
from ..strategies.base import HedgingStrategy
from .common import gauss_hermite_quadrature, l1_envelope_

Tensor = torch.Tensor


@dataclass(frozen=True)
class TwoAssetGrid:
    """Grids of the programme, the same for both assets.

    'x_i = log(S_i / K)' takes 'n_prices' values in '[-x_half, x_half]', with 'x = 0' at the centre node, and
    each holding 'n_positions' values in '[phi_min, phi_max]', zero among them (the holding before the first
    trade). The expectation uses 'nodes' Gauss-Hermite nodes per shock, 'nodes_last' in the last 'last_dates'
    dates.
    """

    x_half: float
    n_prices: int
    phi_min: float
    phi_max: float
    n_positions: int
    nodes: int = 16
    nodes_last: int = 32
    last_dates: int = 2

    def __post_init__(self) -> None:
        if abs(float(self.log_moneyness()[self.n_prices // 2])) >= 1e-12:
            raise ValueError("x = 0 must be the centre node of the price grid (use an odd n_prices)")
        if abs(float(self.positions()[self.zero_position])) >= 1e-12:
            raise ValueError("the holding 0 must be a node of the position grid")

    def log_moneyness(self, device: str | torch.device = "cpu") -> Tensor:
        return torch.linspace(-self.x_half, self.x_half, self.n_prices, dtype=torch.float64, device=device)

    def positions(self, device: str | torch.device = "cpu") -> Tensor:
        return torch.linspace(self.phi_min, self.phi_max, self.n_positions, dtype=torch.float64, device=device)

    @property
    def zero_position(self) -> int:
        """Index of the holding 0."""
        return int(torch.argmin(self.positions().abs()))

    @property
    def table_bytes(self) -> int:
        """Size of one table A_t on disk."""
        return 4 * self.n_positions ** 2 * self.n_prices ** 2

    @classmethod
    def thesis(cls, cost: float = 0.001) -> TwoAssetGrid:
        """The grid of the thesis results for 'cost' (Section 2.9).

        With 0.1%: 101 prices in [-0.5, 0.5] and 79 holdings from -0.03 in steps of 0.0075. With 1%: 161 prices in
        [-0.6, 0.6] and 53 holdings in [-0.05, 0.6]. 16 nodes per shock and 32 in the last two dates.
        """
        if cost == 0.001:
            return cls(x_half=0.5, n_prices=101, phi_min=-0.03, phi_max=-0.03 + 0.0075 * 78, n_positions=79)
        if cost == 0.01:
            return cls(x_half=0.6, n_prices=161, phi_min=-0.05, phi_max=0.6, n_positions=53)
        raise ValueError(f"the thesis solved the two-asset programme for costs 0.001 and 0.01, not {cost}")

    @classmethod
    def refined(cls) -> TwoAssetGrid:
        """141 prices and 40 holdings (step 0.015 from -0.03): the finest grid of the check of the price grid with
        0.1% cost (Section 5.2), built with the same arithmetic as in that study."""
        step = (0.555 + 0.03) / 39
        return cls(x_half=0.5, n_prices=141, phi_min=-2 * step, phi_max=-2 * step + step * 39, n_positions=40)

    def widened(self, factor: float) -> TwoAssetGrid:
        """The price range times 'factor', with the same nodes: the grid of the volatilities times 2."""
        return replace(self, x_half=self.x_half * factor)


class _Node(NamedTuple):
    """One node of the product quadrature at one date, with everything that does not depend on the table."""

    weight: float
    left_1: Tensor  # [n] left node of the shifted point along x_1
    frac_1: Tensor  # [n] its position in the cell
    left_2: Tensor
    frac_2: Tensor
    control: Tensor  # [n, n] C_t+1 at the shifted points
    gain_1: Tensor  # [n] S_1,t+1 - S_1,t
    gain_2: Tensor


def _shifted_cells(n: int, shift: float, device: str | torch.device) -> tuple[Tensor, Tensor]:
    """For a move of 'shift' grid steps from every node: left node of the destination cell and position in it."""
    cell = torch.arange(n, device=device, dtype=torch.float64) + shift
    left = torch.floor(cell).clamp(0, n - 2).long()
    return left, (cell - left.to(torch.float64)).clamp(0.0, 1.0)


class TwoAssetDP:
    """Backward induction for a two-asset :class:`~deephedge.hedging.HedgingProblem`.

    'grid' defaults to the thesis grid for the cost of the problem and 'table_dir' to a new temporary directory;
    :meth:`solve` writes one table per date there and returns the strategy that reads them. The grid is centred at
    the money, so the market must start at the strike, as in the thesis.
    """

    TABLE_DTYPE = torch.float32
    ENVELOPE_BLOCK = 8  # slices per block of the in-place envelopes, which bounds the GPU memory

    def __init__(self, problem: HedgingProblem, risk_aversion: float = 0.5, grid: TwoAssetGrid | None = None,
                 table_dir: str | Path | None = None):
        if problem.n_assets != 2:
            raise ValueError("TwoAssetDP needs a basket of two assets")
        if problem.market.spot != problem.option.strike:
            raise ValueError("the price grid is centred at the strike, so the market must start there")
        self.problem = problem
        self.risk_aversion = float(risk_aversion)
        self.grid = grid or TwoAssetGrid.thesis(problem.cost)
        self.table_dir = table_dir
        self._levy = LevyGreeks(problem.market, problem.option)

    def control(self, t: int, grid_prices: Tensor, log_shift: Tensor | None = None) -> Tensor:
        """'C_t' on the price grid [n, n, 2], every log price moved by 'log_shift' [2]: Levy's price, and the
        payoff at maturity."""
        prices = grid_prices if log_shift is None else grid_prices * torch.exp(log_shift)
        if t >= self.problem.market.dates:
            return self.problem.option.payoff(prices)
        return self._levy.price(prices, self.problem.market.time_to_maturity(t))

    def positions_per_block(self) -> int:
        """Holdings of the first asset per block of the expectation: all of them, or half with the large grids of
        the thesis (the same rule as its runs), to fit a 4 GB GPU."""
        m, n = self.grid.n_positions, self.grid.n_prices
        return m if n * n * m * m <= 30e6 else (m + 1) // 2

    def solve(self, device: str | torch.device = "cpu") -> OptimalGridStrategy:
        """Run the recursion from maturity back to date 0 on 'device', writing 'A_t' to the table directory.

        The thesis grid takes 28 minutes on the 4 GB GPU of the thesis, with about 1.5 GB of GPU memory, and writes
        12.7 GB of tables; on a CPU it takes hours. Small grids take seconds.
        """
        start = time.time()
        market, grid = self.problem.market, self.grid
        g, k, strike = self.risk_aversion, self.problem.cost, self.problem.option.strike
        dtype, m, n = self.TABLE_DTYPE, grid.n_positions, grid.n_prices
        table_dir = Path(self.table_dir) if self.table_dir else Path(tempfile.mkdtemp(prefix="deephedge-dp2-"))
        table_dir.mkdir(parents=True, exist_ok=True)

        xs, phi = grid.log_moneyness(), grid.positions()
        dx, dphi = float(xs[1] - xs[0]), float(phi[1] - phi[0])
        axis_prices = strike * torch.exp(xs)  # S_i on the grid of x_i, [n]
        grid_prices = torch.stack(torch.meshgrid(axis_prices, axis_prices, indexing="ij"), -1).to(device)  # [n, n, 2]
        s, phi = axis_prices.to(device), phi.to(device)
        phi_t = phi.to(dtype)
        chol = market.cholesky.to(torch.float64)
        identical = bool(torch.all(chol[:, 1] == 0))  # the second shock moves no price

        # W_N = V_N - g (B - K)^+ = g k (S_1 |p_1| + S_2 |p_2|), on [p_1, x_1, x_2, p_2]
        w = (g * k * (phi.abs()[:, None, None, None] * s[None, :, None, None]
                      + phi.abs()[None, None, None, :] * s[None, None, :, None])).to(dtype)
        w = w.expand(m, n, n, m).contiguous()
        unit_cost_1 = (g * k * s).to(dtype)[None, :, None, None] * dphi  # cost of one step of f_1, per x_1
        unit_cost_2 = (g * k * s).to(dtype)[None, None, :, None] * dphi
        block = self.positions_per_block()
        for t in reversed(range(market.dates)):
            nodes = self._nodes(t, grid_prices, s, dx, identical, device)
            base_control = self.control(t + 1, grid_prices).to(dtype)
            a = torch.empty_like(w)  # A_t on [f_1, x_1, x_2, f_2]
            for first in range(0, m, block):
                rows = slice(first, min(m, first + block))
                w_block = w[rows]
                reference = w_block + g * base_control[None, :, :, None]  # the exponent without moves
                total = torch.zeros_like(w_block)
                phi_1 = phi_t[rows][:, None, None, None]
                for node in nodes:
                    shifted = torch.lerp(w_block.index_select(1, node.left_1), w_block.index_select(1, node.left_1 + 1),
                                         node.frac_1.view(1, -1, 1, 1))
                    w_next = torch.lerp(shifted.index_select(2, node.left_2), shifted.index_select(2, node.left_2 + 1),
                                        node.frac_2.view(1, 1, -1, 1))
                    del shifted
                    exponent = w_next + g * node.control[None, :, :, None]
                    del w_next
                    exponent -= g * (phi_1 * node.gain_1[None, :, None, None])
                    exponent -= g * (phi_t[None, None, None, :] * node.gain_2[None, None, :, None])
                    exponent -= reference
                    total += node.weight * torch.exp(exponent)
                    del exponent
                a[rows] = reference + torch.log(total)
                del reference, total
            del w
            np.save(table_dir / f"A_{t:02d}.npy", a.cpu().numpy())
            l1_envelope_(a, unit_cost_2, dim=3, block_dim=0, block=self.ENVELOPE_BLOCK)  # min over f_2 for each f_1
            l1_envelope_(a, unit_cost_1, dim=0, block_dim=1, block=self.ENVELOPE_BLOCK)  # then over f_1: V_t
            a -= g * self.control(t, grid_prices).to(dtype)[None, :, :, None]
            w = a
            del a
        centre, zero = n // 2, grid.zero_position
        initial_value = float(w[zero, centre, centre, zero]) + g * float(self.control(0, grid_prices)[centre, centre])
        meta = {"grid": asdict(grid), "strike": strike, "risk_aversion": g, "cost": k, "premium": self.problem.premium,
                "initial_value": initial_value, "seconds": time.time() - start}
        (table_dir / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
        return OptimalGridStrategy.load(table_dir)

    def _nodes(self, t: int, grid_prices: Tensor, s: Tensor, dx: float, identical: bool,
               device: str | torch.device) -> list[_Node]:
        """The product quadrature of date 't', with the first shock varying slowest."""
        market, grid, dtype = self.problem.market, self.grid, self.TABLE_DTYPE
        count = grid.nodes_last if t >= market.dates - grid.last_dates else grid.nodes
        z_1, w_1 = gauss_hermite_quadrature(count)
        z_2, w_2 = gauss_hermite_quadrature(1 if identical else count)
        chol, vols, drifts = (v.to(torch.float64) for v in (market.cholesky, market.step_vols, market.drifts))
        nodes = []
        for a in range(len(z_1)):
            for b in range(len(z_2)):
                jump = drifts + vols * (chol @ torch.tensor([z_1[a], z_2[b]], dtype=torch.float64))  # move of x
                left_1, frac_1 = _shifted_cells(grid.n_prices, float(jump[0]) / dx, device)
                left_2, frac_2 = _shifted_cells(grid.n_prices, float(jump[1]) / dx, device)
                relative_move = torch.expm1(jump).to(device)
                nodes.append(_Node(float(w_1[a] * w_2[b]), left_1, frac_1.to(dtype), left_2, frac_2.to(dtype),
                                   self.control(t + 1, grid_prices, jump.to(device)).to(dtype),
                                   (s * relative_move[0]).to(dtype), (s * relative_move[1]).to(dtype)))
        return nodes


class OptimalGridStrategy(HedgingStrategy):
    """The strategy of the two-asset programme, read from the tables 'A_t' that :meth:`TwoAssetDP.solve` wrote.

    At date 't' it interpolates 'A_t' bilinearly at the current log prices and picks the pair of grid holdings
    that minimises 'g k sum_i S_i |f_i - p_i| + A_t(x, f)': the plain argmin over the 'm x m' pairs (ties go to the
    first, with 'f_2' varying fastest), in float32 like the tables, in blocks of 'path_block' paths. Each table is
    loaded once onto the device of the prices, so the strategy must be run date by date, as
    :meth:`~deephedge.hedging.HedgingProblem.run` does.
    """

    name = "dynamic-programming"

    def __init__(self, table_dir: str | Path, grid: TwoAssetGrid, *, strike: float, risk_aversion: float,
                 cost: float, premium: float, initial_value: float, seconds: float, path_block: int = 4096):
        self.table_dir = Path(table_dir)
        self.grid = grid
        self.strike = float(strike)
        self.risk_aversion = float(risk_aversion)
        self.cost = float(cost)
        self.premium = float(premium)
        self.initial_value = float(initial_value)  # V_0(0, 0), without the premium
        self.seconds = float(seconds)
        self.path_block = int(path_block)
        self._loaded: tuple[int, torch.device, Tensor] | None = None

    @classmethod
    def load(cls, table_dir: str | Path) -> OptimalGridStrategy:
        """The strategy whose tables and 'meta.json' are in 'table_dir'."""
        meta = json.loads((Path(table_dir) / "meta.json").read_text(encoding="utf-8"))
        return cls(table_dir, TwoAssetGrid(**meta.pop("grid")), **meta)

    def predicted_risk(self) -> float:
        """The entropic risk the programme predicts for its own strategy, 'V_0(0, 0) / g - premium'."""
        return self.initial_value / self.risk_aversion - self.premium

    def delete_tables(self) -> None:
        """Remove the tables and 'meta.json', and the directory if nothing else is left in it."""
        self._loaded = None
        for path in self.table_dir.glob("A_*.npy"):
            path.unlink()
        (self.table_dir / "meta.json").unlink(missing_ok=True)
        with contextlib.suppress(OSError):
            self.table_dir.rmdir()

    def table(self, t: int, device: torch.device) -> Tensor:
        """'A_t' as a [n * n, m * m] float32 matrix on 'device': row 'j_1 n + j_2', column 'f_1 m + f_2'."""
        if self._loaded is None or self._loaded[:2] != (t, device):
            self._loaded = None
            m, n = self.grid.n_positions, self.grid.n_prices
            a = torch.from_numpy(np.load(self.table_dir / f"A_{t:02d}.npy")).to(device)
            self._loaded = (t, device, a.permute(1, 2, 0, 3).reshape(n * n, m * m).contiguous())
        return self._loaded[2]

    def act(self, t: int, prices: Tensor, position: Tensor) -> Tensor:
        device, m, n = prices.device, self.grid.n_positions, self.grid.n_prices
        table = self.table(t, device)
        xs = self.grid.log_moneyness(device)
        phi = self.grid.positions(device)  # built on the device, as when the tables were solved
        phi_32 = phi.to(torch.float32)
        out = torch.empty_like(position)
        for first in range(0, prices.shape[0], self.path_block):
            paths = slice(first, min(prices.shape[0], first + self.path_block))
            s = prices[paths].to(torch.float64)
            x = torch.log(s / self.strike)
            cell_1 = (x[:, 0] - xs[0]) / (xs[1] - xs[0])
            cell_2 = (x[:, 1] - xs[0]) / (xs[1] - xs[0])
            j_1 = torch.floor(cell_1).clamp(0, n - 2).long()
            j_2 = torch.floor(cell_2).clamp(0, n - 2).long()
            f_1 = (cell_1 - j_1).clamp(0, 1).to(torch.float32)[:, None]
            f_2 = (cell_2 - j_2).clamp(0, 1).to(torch.float32)[:, None]
            row = j_1 * n + j_2
            a = ((1 - f_1) * (1 - f_2) * table[row] + f_1 * (1 - f_2) * table[row + n]
                 + (1 - f_1) * f_2 * table[row + 1] + f_1 * f_2 * table[row + n + 1])  # A_t(x, f), [paths, m * m]
            held = position[paths].to(torch.float32)
            unit_cost = (self.risk_aversion * self.cost * s).to(torch.float32)
            trading = (unit_cost[:, 0, None, None] * (phi_32[None, :, None] - held[:, 0, None, None]).abs()
                       + unit_cost[:, 1, None, None] * (phi_32[None, None, :] - held[:, 1, None, None]).abs())
            best = torch.argmin(a + trading.reshape(-1, m * m), dim=1)
            out[paths, 0] = phi[best // m].to(position.dtype)
            out[paths, 1] = phi[best % m].to(position.dtype)
        return out
