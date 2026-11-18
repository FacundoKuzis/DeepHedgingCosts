"""Calibrating the no-trade rules on the validation paths (thesis, Section 4.2.2).

The asymptotic rules have one free number, the scale 'g' that replaces the risk aversion in the
half-width. The *calibrated* rule is the one with the lowest validation risk, found by a grid search
that extends the grid when the best value lies on its edge and then refines around the best value.
The test paths are never used. The search evaluates each value once and remembers it, so a search can
be resumed from its 'evaluations'.

The thesis used three variants of the same search, which are the presets below:

* one asset: the fixed grid of 27 values of the first experiments, with no extension or refinement;
* baskets: the grid is extended by x0.6 or x1.6 up to four times, and then refined once at 1/3 and 2/3 of
  the way (in log scale) towards each neighbour of the best value;
* CVaR criterion: a grid from 0.1 to 1,000, extended by /3 or x3 up to five times, and then three rounds
  of geometric midpoints between the best value and its neighbours.

:func:`calibrate_centre_and_scale` searches the scale and the factor 'c' of the shifted centre together, as in
the CVaR experiments, and :func:`refine_centre_and_scale` is the local search of the corrected box with the centre
calibrated, the control of the base case with ten assets. Every value is rounded before it is evaluated, to six
decimals in the searches of the scale and of the shifted centre and to ten in the local search.
"""
from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

Evaluate = Callable[[float], float]


@dataclass(frozen=True)
class ScaleSearch:
    grid: tuple[float, ...]
    down: float = 0.6            # factor applied to the best value when it is the smallest one
    up: float = 1.6              # ... and when it is the largest one
    extensions: int = 4
    refinement: str = "thirds"   # 'none', 'thirds' (one pass) or 'midpoints' ('rounds' passes)
    rounds: int = 1
    decimals: int = 6


SINGLE_ASSET_GRID = (0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.75, 0.9, 1.0,
                     1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0, 20.0)
SINGLE_ASSET = ScaleSearch(SINGLE_ASSET_GRID, extensions=0, refinement="none")
BOX = ScaleSearch((0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0, 20.0, 30.0, 50.0))
CORRECTED_BOX = ScaleSearch((0.3, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 7.5, 10.0))
CVAR = ScaleSearch((0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0), down=1 / 3, up=3.0, extensions=5,
                   refinement="midpoints", rounds=3)
SMOKE = ScaleSearch((1.0, 10.0), extensions=0, refinement="none")


@dataclass
class Calibration:
    scale: float
    value: float
    at_edge: bool
    evaluations: dict[float, float] = field(default_factory=dict)
    centre: float = 1.0

    def as_dict(self) -> dict:
        return {"scale": self.scale, "centre": self.centre, "value": self.value, "at_edge": self.at_edge,
                "evaluations": [[k, v] for k, v in sorted(self.evaluations.items())]}


def calibrate_scale(evaluate: Evaluate, search: ScaleSearch, evaluations: dict[float, float] | None = None
                    ) -> Calibration:
    """The scale with the lowest 'evaluate(scale)'; 'evaluations' holds what was already evaluated."""
    values = {} if evaluations is None else evaluations

    def ev(x: float) -> float:
        x = round(float(x), search.decimals)
        if x not in values:
            values[x] = float(evaluate(x))
        return x

    for x in search.grid:
        ev(x)
    for _ in range(search.extensions):
        best = min(values, key=values.get)
        if best == min(values):
            ev(best * search.down)
        elif best == max(values):
            ev(best * search.up)
        else:
            break
    if search.refinement == "thirds":
        best = min(values, key=values.get)
        keys = sorted(values)
        i = keys.index(best)
        for j in (i - 1, i + 1):
            if 0 <= j < len(keys):
                for f in (1 / 3, 2 / 3):
                    ev(best * (keys[j] / best) ** f)
    elif search.refinement == "midpoints":
        for _ in range(search.rounds):
            ordered = sorted(values)
            best = min(values, key=values.get)
            pos = ordered.index(best)
            for neighbour in ordered[max(0, pos - 1):pos] + ordered[pos + 1:pos + 2]:
                ev(math.sqrt(best * neighbour))
    elif search.refinement != "none":
        raise ValueError(f"unknown refinement {search.refinement!r}")
    best = min(values, key=values.get)
    return Calibration(best, values[best], best in (min(values), max(values)), values)


@dataclass(frozen=True)
class CentreSearch:
    centres: tuple[float, ...] = (0.7, 1.0, 1.15, 1.3, 1.5, 1.75, 2.0, 2.5)
    scales: tuple[float, ...] = (1.0, 10.0, 100.0, 1000.0)   # plus the calibrated scale of the base rule
    extensions: int = 3
    centre_step: float = 1.25
    scale_step: float = 3.0
    refinements: tuple[tuple[float, float], ...] = ((1.1, 2.0), (1.05, 1.4), (1.025, 1.18))
    decimals: int = 6


CENTRE = CentreSearch()
CENTRE_SMOKE = CentreSearch(centres=(1.0, 1.3), scales=(10.0, 100.0), extensions=0, refinements=())


def calibrate_centre_and_scale(evaluate: Callable[[float, float], float], base: Calibration,
                               search: CentreSearch = CENTRE, evaluations: dict | None = None) -> Calibration:
    """Joint search of the centre factor 'c' and the scale 'g' of the shifted-centre rule (Section 5.5).

    'evaluate(g, c)' is the validation risk of the rule; 'base' is the calibration of the base rule (c = 1),
    whose scale joins the initial grid. The rule with c = 1 must reproduce the base rule, which is checked
    before the search goes on. Then the grid is extended while the best pair lies on its edge, and refined
    three times around the best pair with shrinking factors.
    """
    values = {} if evaluations is None else evaluations
    d = search.decimals

    def ev(g: float, c: float) -> float:
        key = (round(float(g), d), round(float(c), d))
        if key not in values:
            values[key] = float(evaluate(*key))
        return values[key]

    scales = sorted(set(search.scales) | {base.scale})
    for c in search.centres:
        for g in scales:
            ev(g, c)
    at_one = ev(base.scale, 1.0)
    if abs(at_one - base.value) > 1e-7 * max(1.0, abs(base.value)):
        raise RuntimeError(f"the shifted rule with c = 1 does not reproduce the base rule ({at_one} vs {base.value})")

    def best_pair():
        return min(values, key=values.get)

    for _ in range(search.extensions):
        g, c = best_pair()
        gs, cs = [k[0] for k in values], [k[1] for k in values]
        edge_c, edge_g = c in (min(cs), max(cs)), g in (min(gs), max(gs))
        if not (edge_c or edge_g):
            break
        new_c = [c / search.centre_step if c == min(cs) else c * search.centre_step] if edge_c else []
        new_g = [g / search.scale_step if g == min(gs) else g * search.scale_step] if edge_g else []
        for cc in sorted(set(new_c + [c])):
            for gg in sorted(set(new_g + [g])):
                ev(gg, cc)
    for fc, fg in search.refinements:
        g, c = best_pair()
        for cc in (c / fc, c, c * fc):
            for gg in (g / fg, g, g * fg):
                ev(gg, cc)
    g, c = best_pair()
    gs, cs = [k[0] for k in values], [k[1] for k in values]
    edge = c in (min(cs), max(cs)) or g in (min(gs), max(gs))
    return Calibration(g, values[(g, c)], edge, values, centre=c)


@dataclass(frozen=True)
class LocalCentreSearch:
    """The local search of the centre and the scale of the corrected box with ten assets and 0.1% (thesis, Section
    5.2). The scale 1.4458571035 is the one a first search of the same control chose on a smaller
    sample, and the thesis kept it in the grid; the calibrated scale of the box with the centre at the delta joins
    it."""

    centres: tuple[float, ...] = (0.94, 0.97, 1.0, 1.03, 1.06)
    scales: tuple[float, ...] = (1.2, 1.4458571035, 1.9, 2.2)
    centre_step: float = 0.03        # added to the centre on the edge of the grid
    scale_step: float = 1.25         # multiplies (or divides) the scale on the edge
    extensions: int = 2
    fixed_centre_refinements: int = 2
    joint_refinements: int = 2
    decimals: int = 10


LOCAL_CENTRE = LocalCentreSearch()
LOCAL_CENTRE_SMOKE = LocalCentreSearch(centres=(1.0, 1.06), scales=(10.0,), extensions=0, fixed_centre_refinements=0,
                                       joint_refinements=0)


def refine_centre_and_scale(evaluate: Callable[[float, float], float], base: Calibration,
                            search: LocalCentreSearch = LOCAL_CENTRE, evaluations: dict | None = None) -> Calibration:
    """Joint local search of the centre factor 'c' and the scale 'g' around a rule calibrated with c = 1.

    'evaluate(g, c)' is the validation risk. The search evaluates the base rule (which must reproduce 'base') and
    the grid 'centres' x 'scales'. While the best pair lies on the edge of what was evaluated (at most
    'extensions' times) it adds the next centre ('centre_step' beyond the edge) with the scales around the best
    one, and the next scale ('scale_step' beyond the edge) with the centres around the best one and c = 1. Then it
    refines the scale with c = 1 at the geometric midpoints next to the best one, and finally evaluates the 3 x 3
    grid of the best pair and its midpoints (arithmetic in c, geometric in g), twice. Values are rounded to
    'decimals' before they are evaluated, and ties go to the pair evaluated first.
    """
    values = {} if evaluations is None else evaluations
    d = search.decimals

    def ev(g: float, c: float) -> float:
        key = (round(float(g), d), round(float(c), d))
        if key not in values:
            values[key] = float(evaluate(*key))
        return values[key]

    def best_pair(pairs=None):
        return min(pairs if pairs is not None else values, key=values.get)

    def around(x: float, axis: int) -> list[float]:
        xs = sorted({k[axis] for k in values})
        j = xs.index(x)
        if j in (0, len(xs) - 1):
            return [x]
        mid = (lambda a, b: math.sqrt(a * b)) if axis == 0 else (lambda a, b: 0.5 * (a + b))
        return [mid(xs[j - 1], x), x, mid(x, xs[j + 1])]

    at_one = ev(base.scale, 1.0)
    if abs(at_one - base.value) > 1e-7 * max(1.0, abs(base.value)):
        raise RuntimeError(f"the rule with c = 1 does not reproduce the base rule ({at_one} vs {base.value})")
    scales = sorted(set(search.scales) | {base.scale})
    for c in search.centres:
        for g in scales:
            ev(g, c)
    for _ in range(search.extensions):
        g, c = best_pair()
        gs, cs = sorted({k[0] for k in values}), sorted({k[1] for k in values})
        new_c = cs[0] - search.centre_step if c == cs[0] else cs[-1] + search.centre_step if c == cs[-1] else None
        new_g = gs[0] / search.scale_step if g == gs[0] else gs[-1] * search.scale_step if g == gs[-1] else None
        if new_c is None and new_g is None:
            break
        if new_c is not None:
            for gg in around(g, 0):
                ev(gg, new_c)
        if new_g is not None:
            for cc in sorted(set(around(c, 1) + [1.0])):
                ev(new_g, cc)
        if new_c is not None and new_g is not None:
            ev(new_g, new_c)
    for _ in range(search.fixed_centre_refinements):
        g, _c = best_pair([k for k in values if k[1] == 1.0])
        gs = sorted(k[0] for k in values if k[1] == 1.0)
        j = gs.index(g)
        if j in (0, len(gs) - 1):
            break
        for gg in (math.sqrt(gs[j - 1] * g), math.sqrt(g * gs[j + 1])):
            ev(gg, 1.0)
    for _ in range(search.joint_refinements):
        g, c = best_pair()
        centres, scales = around(c, 1), around(g, 0)     # both before the new pairs change the axes
        for cc in centres:
            for gg in scales:
                ev(gg, cc)
    g, c = best_pair()
    gs, cs = [k[0] for k in values], [k[1] for k in values]
    edge = c in (min(cs), max(cs)) or g in (min(gs), max(gs))
    return Calibration(g, values[(g, c)], edge, values, centre=c)
