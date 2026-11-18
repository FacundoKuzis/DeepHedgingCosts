"""Numerical pieces shared by the one- and two-asset dynamic programmes.

* :func:`l1_envelope` is the trading step of the Bellman recursion (thesis, Section 2.7): with a proportional
  cost, 'V_t(p) = min_phi [c |phi - p| + A_t(phi)]' is the infimal convolution of 'A_t' with 'c |.|', which
  two running minima (one up the grid, one down) give exactly on the grid in linear time.
* :func:`refined_argmin` places a minimum between grid nodes; the one-asset programme uses it for the band edges.
* :func:`truncated_normal_quadrature` (one asset) and :func:`gauss_hermite_quadrature` (two assets, as a product
  rule) approximate the expectation over the shocks of the next day.
"""
from __future__ import annotations

import math

import numpy as np
import torch

Tensor = torch.Tensor


def l1_envelope(values: Tensor, step_cost: Tensor | float, dim: int = -1) -> Tensor:
    """The minimum over j' of values[..., j', ...] + step_cost * |j - j'| along dim, for every node j.

    'step_cost' (non-negative, broadcastable to 'values') is the cost of moving the holding by one grid step.
    The minimum over j' <= j is cummin(values - c j) + c j, and the one over j' >= j is the same scan run down
    the axis. The result is exact on the grid.
    """
    n = values.shape[dim]
    shape = [1] * values.dim()
    shape[dim] = n
    ramp = step_cost * torch.arange(n, device=values.device, dtype=values.dtype).view(shape)
    from_below = torch.cummin(values - ramp, dim=dim).values + ramp
    from_above = torch.flip(torch.cummin(torch.flip(values + ramp, dims=[dim]), dim=dim).values, dims=[dim]) - ramp
    return torch.minimum(from_below, from_above)


def l1_envelope_(values: Tensor, step_cost: Tensor, dim: int, block_dim: int, block: int) -> Tensor:
    """:func:`l1_envelope` written over 'values', 'block' slices of the axis 'block_dim' at a time.

    'block_dim' must differ from 'dim', so the blocks are independent; working by blocks bounds the memory of
    the temporaries, which matters for the [m, n, n, m] tables of the two-asset programme on a 4 GB GPU.
    """
    n = values.shape[block_dim]
    for start in range(0, n, block):
        index = [slice(None)] * values.dim()
        index[block_dim] = slice(start, min(n, start + block))
        index = tuple(index)
        cost = step_cost[index] if step_cost.shape[block_dim] > 1 else step_cost
        values[index] = l1_envelope(values[index], cost, dim)
    return values


def refined_argmin(values: Tensor, grid: Tensor) -> Tensor:
    """Minimiser of 'values' along the last axis, on the uniform 'grid', refined between nodes by a parabola.

    The parabola through the smallest node and its two neighbours moves the minimiser by
    '(v[-1] - v[+1]) / (2 (v[-1] - 2 v[0] + v[+1]))' steps, at most half a step. At either end of the grid, or
    where the three values are not strictly convex, the node itself is returned. Ties go to the first node.
    """
    n = values.shape[-1]
    node = torch.argmin(values, dim=-1)
    step = grid[1] - grid[0]
    inner = (node > 0) & (node < n - 1)
    centre = node.clamp(1, n - 2)
    left = torch.gather(values, -1, (centre - 1).unsqueeze(-1)).squeeze(-1)
    middle = torch.gather(values, -1, centre.unsqueeze(-1)).squeeze(-1)
    right = torch.gather(values, -1, (centre + 1).unsqueeze(-1)).squeeze(-1)
    curvature = left - 2.0 * middle + right
    offset = torch.where(curvature > 1e-14, 0.5 * (left - right) / curvature.clamp(min=1e-14),
                         torch.zeros_like(curvature))
    offset = offset.clamp(-0.5, 0.5)
    base = grid[node]
    return torch.where(inner, base + offset * step, base)


def truncated_normal_quadrature(nodes_per_half: int, z_max: float) -> tuple[np.ndarray, np.ndarray]:
    """Nodes (increasing) and weights for 'E[f(Z)]', 'Z' a standard normal truncated to '[-z_max, z_max]'.

    Gauss-Legendre with 'nodes_per_half' nodes on '[0, z_max]' and its mirror image on '[-z_max, 0]', each
    weight multiplied by the normal density; normalising the weights to one is the truncation. It integrates the
    same law that :class:`deephedge.market.GBMMarket` samples.
    """
    x, w = np.polynomial.legendre.leggauss(nodes_per_half)
    half_nodes = 0.5 * z_max * (x + 1.0)
    half_weights = 0.5 * z_max * w
    z = np.concatenate([-half_nodes[::-1], half_nodes])
    weights = np.concatenate([half_weights[::-1], half_weights]) * np.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)
    return z, weights / weights.sum()


def gauss_hermite_quadrature(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Nodes and weights (summing to one) for 'E[f(Z)]', 'Z' standard normal: probabilists' Gauss-Hermite.

    'n = 1' is the single node at zero, used for a shock that moves no price (two identical assets). The rule
    ignores the truncation of the shocks at 'z_max = 8': with 32 nodes the weight beyond it is 1.4e-15.
    """
    if n == 1:
        return np.array([0.0]), np.array([1.0])
    z, w = np.polynomial.hermite_e.hermegauss(n)
    return z, w / w.sum()
