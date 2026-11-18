"""The 24 experiments of the thesis (Table 5), each one the base case with one change.

The seeds and sample sizes are those of the thesis, so the validation and test paths are its paths. The
basket premiums are the Monte Carlo prices (4,000,000 paths) the thesis used, and with one asset the premium
is the Black-Scholes price.

| Section | Experiments | What changes |
|---|---|---|
| 5.1 | 'no-cost/{1,2,3,10}' | no costs; mean-square criterion; the delta as benchmark |
| 5.2 | 'base/{1,2,3,10}' | the base case: 0.1% cost, correlation 0.5, entropic risk (with ten assets, also the
corrected box with its centre calibrated) |
| 5.3 | 'cost-1pct/{1,2,3,10}' | 1% cost |
| 5.4 | 'vol-x2/{2,3,10}', 'vol-x1.5/10', 'vol-x0.5/10' | every volatility multiplied by a factor |
| 5.4 | 'no-correlation/{2,3,10}' | independent assets |
| 5.5 | 'cvar/{1,2,3,10}' | the CVaR at 99% as criterion, for the network and the rules |
"""
from __future__ import annotations

import math

from ..dp import OneAssetGrid, TwoAssetGrid
from .spec import ExperimentSpec, NetworkTraining, RuleSpec, Sample

VALIDATION, TEST = 65_536, 200_000
BLOCK = 25_000   # the ten-asset samples were drawn in blocks


# --- seeds of the thesis --------------------------------------------------------------------------
def single_asset_batch_seed(cost: float, seed: int = 1) -> int:
    return 10_000 * seed + int(round(cost * 1e4))


def basket_batch_seed(n: int, cost: float, with_costs: bool, seed: int = 1) -> int:
    return 20_000 * seed + 97 * n + 13 + int(round(cost * 1e5)) + (7 if with_costs else 0)


def samples(n: int, kind: str) -> tuple[Sample, Sample]:
    """(validation, test) of the thesis. 'kind' is 'no-cost', 'with-costs' or 'variant'.

    The runs of ten assets and those of Section 5.4 (the 'variant' kind, with any number of assets) drew their paths
    in blocks of 25,000; the variants with two assets were tested on the seed 1,040,003 instead of the 1,040,002 of
    the base case, the second test sample on which the two-asset dynamic programme was evaluated."""
    if n == 1:
        return Sample(910_001, VALIDATION), Sample(920_002, TEST)
    block = BLOCK if (n == 10 or kind == "variant") else None
    if kind == "no-cost":
        return Sample(960_001 + n, VALIDATION, block), Sample(970_001 + n, TEST, block)
    test_seed = 1_040_002 if (n == 2 and kind == "with-costs") else 1_040_001 + n
    return Sample(1_030_001 + n, VALIDATION, block), Sample(test_seed, TEST, block)


# Monte Carlo premiums of the thesis, by (number of assets, volatility factor, correlation)
PREMIUMS = {
    (2, 1.0, 0.5): 2.7064209904639696, (3, 1.0, 0.5): 3.1809784787907764, (10, 1.0, 0.5): 2.971654937493223,
    (2, 2.0, 0.5): 5.409448430204254, (3, 2.0, 0.5): 6.355301562617652, (10, 2.0, 0.5): 5.940212348969809,
    (10, 1.5, 0.5): 4.456533682880302, (10, 0.5, 0.5): 1.4859300055700915,
    (2, 1.0, 0.0): 2.2247311339702267, (3, 1.0, 0.0): 2.317703550366149, (10, 1.0, 0.0): 1.2965869967339991,
}


# --- grids of the dynamic programmes (thesis, Section 2.9) -----------------------------------------------------
ONE_ASSET_GRIDS = (OneAssetGrid.coarse(), OneAssetGrid.fine())


def _positions_from(low: float, high: float, count: int) -> tuple[float, float, int]:
    """A uniform grid of holdings that contains zero and covers [low, high] with about 'count' nodes (the rule of
    the grid study of the thesis)."""
    step = (high - low) / (count - 1)
    first = -max(1, round(-low / step)) * step
    n = int(math.ceil((high - first) / step - 1e-9)) + 1
    return first, first + step * (n - 1), n


def two_asset_grid(n_prices: int, n_positions: int, cost: float = 0.001) -> TwoAssetGrid:
    """The grids of the two-asset programme. With 0.1% the 59 and 79 holdings of the results start at -0.03 in steps
    of 0.01 and 0.0075; other counts follow the rule of the grid study. With 1% the holdings cover [-0.05, 0.6]."""
    if cost >= 0.005:
        return TwoAssetGrid(x_half=0.6, n_prices=n_prices, phi_min=-0.05, phi_max=0.6, n_positions=n_positions)
    fixed = {59: (-0.03, -0.03 + 0.01 * 58, 59), 79: (-0.03, -0.03 + 0.0075 * 78, 79)}
    low, high, m = fixed.get(n_positions) or _positions_from(-0.03, 0.555, n_positions)
    return TwoAssetGrid(x_half=0.5, n_prices=n_prices, phi_min=low, phi_max=high, n_positions=m)


def two_asset_grids(cost: float, vol_scale: float = 1.0) -> tuple[TwoAssetGrid, ...]:
    """The two grids of the results: 101 prices with 59 and 79 holdings at 0.1%, 121 x 40 and 161 x 53 at 1%. With the
    volatilities doubled the price range doubles with the same nodes (the same step in daily deviations)."""
    grids = ((two_asset_grid(101, 59), two_asset_grid(101, 79)) if cost < 0.005 else
             (two_asset_grid(121, 40, cost), two_asset_grid(161, 53, cost)))
    return tuple(g.widened(vol_scale) for g in grids) if vol_scale > 1.0 else grids


# the grid study of Sections 2.9 and 5.2: 40 holdings with 23 to 141 prices, and 101 prices with 40 to 79 holdings
GRID_STUDY = {"base/2": {"prices": tuple(two_asset_grid(n, 40) for n in (23, 41, 61, 101, 121, 141)),
                         "positions": tuple(two_asset_grid(101, m) for m in (40, 59, 79))}}


# --- builders --------------------------------------------------------------------------------------------
def _rule(n: int, rho: float, criterion: str = "entropic") -> tuple[RuleSpec, ...]:
    search = "cvar" if criterion != "entropic" else None
    if n == 1:
        return (RuleSpec("whalley-wilmott", "whalley-wilmott", search or "single-asset", criterion),)
    box = RuleSpec("atkinson-box", "atkinson-box", search or "box", criterion)
    if rho == 0.0:   # without correlation the corrected box is the box
        return (box,)
    corrected = "damped-corrected-box" if n == 10 else "corrected-box"
    return (RuleSpec("corrected-box", corrected, search or "corrected-box", criterion), box)


def _comparison(n: int, rho: float, dp: bool) -> tuple[str, ...]:
    rule = "whalley-wilmott" if n == 1 else ("atkinson-box" if rho == 0.0 else "corrected-box")
    return ("network",) + (("dp",) if dp else ()) + (rule,)


def no_cost(n: int) -> ExperimentSpec:
    val, test = samples(n, "no-cost")
    if n == 1:
        net = NetworkTraining("mse", steps=3000, warmup=150, batch_seed=single_asset_batch_seed(0.0))
    else:
        net = NetworkTraining("mse", steps=3000, warmup=150, batch_seed=basket_batch_seed(n, 0.0, False))
    return ExperimentSpec(
        name=f"no-cost/{n}", section="5.1", description=f"{n} asset(s) without costs, mean-square criterion",
        n_assets=n, cost=0.0, validation=val, test=test, network=net, premium=PREMIUMS.get((n, 1.0, 0.5)),
        theoretical_rule=False, compare=("network", "delta"))


def with_costs(n: int, cost: float, *, vol_scale: float = 1.0, rho: float = 0.5, name: str, section: str,
               description: str) -> ExperimentSpec:
    variant = vol_scale != 1.0 or rho != 0.5
    val, test = samples(n, "variant" if variant else "with-costs")
    if n == 1:
        net = NetworkTraining("entropic", steps=12000 if cost < 0.005 else 6000, warmup=200,
                              batch_seed=single_asset_batch_seed(cost))
        dp = ONE_ASSET_GRIDS
    else:
        net = NetworkTraining("entropic", steps=5000, warmup=150, batch_seed=basket_batch_seed(n, cost, True))
        dp = two_asset_grids(cost, vol_scale) if n == 2 else ()
    return ExperimentSpec(
        name=name, section=section, description=description, n_assets=n, cost=cost, validation=val, test=test,
        network=net, vol_scale=vol_scale, rho=rho, premium=PREMIUMS.get((n, vol_scale, rho)), rules=_rule(n, rho),
        dp=dp, compare=_comparison(n, rho, bool(dp)))


def with_calibrated_centre(spec: ExperimentSpec) -> ExperimentSpec:
    """The control of the base case with ten assets (Section 5.2): the damped corrected box with its centre at the
    delta computed with every volatility multiplied by c, and c chosen together with the scale on the validation
    paths by the local search of :func:`~deephedge.calibration.refine_centre_and_scale` (c = 1.056, g = 2.04)."""
    rule = RuleSpec("corrected-box/calibrated-centre", "damped-corrected-box", "local-centre")
    return spec.replace(rules=(*spec.rules, rule), compare=(*spec.compare, rule.name))


def cvar(n: int) -> ExperimentSpec:
    """The network trained on the CVaR at 99% with batches of 32,768 paths, and the rules calibrated on it."""
    base = with_costs(n, 0.001, name=f"cvar/{n}", section="5.5",
                      description=f"{n} asset(s), 0.1% cost, CVaR at 99% as criterion")
    net = NetworkTraining("cvar99", steps=12000 if n == 1 else 5000, warmup=150, batch_seed=31_102_026 + n,
                          batch_size=32_768)
    rules = _rule(n, 0.5, "cvar99")
    # the shifted centre moves the band of Whalley and Wilmott (one asset) or the box of order zero (baskets);
    # with c = 1 it is that rule, calibrated on the same criterion just before
    shifted = RuleSpec("shifted-centre", "whalley-wilmott" if n == 1 else "atkinson-box", "centre", "cvar99")
    return base.replace(network=net, rules=(*rules, shifted), dp=(), criterion_rules="cvar99",
                        compare=("network", "shifted-centre", rules[0].name), tags=("uses:base",))


def build_catalog() -> dict[str, ExperimentSpec]:
    specs = [no_cost(n) for n in (1, 2, 3, 10)]
    specs += [with_costs(n, 0.001, name=f"base/{n}", section="5.2",
                         description=f"base case with {n} asset(s): 0.1% cost") for n in (1, 2, 3)]
    specs += [with_calibrated_centre(with_costs(10, 0.001, name="base/10", section="5.2",
                                                description="base case with 10 asset(s): 0.1% cost"))]
    specs += [with_costs(n, 0.01, name=f"cost-1pct/{n}", section="5.3",
                         description=f"{n} asset(s) with 1% cost") for n in (1, 2, 3, 10)]
    specs += [with_costs(n, 0.001, vol_scale=2.0, name=f"vol-x2/{n}", section="5.4",
                         description=f"{n} assets, every volatility doubled") for n in (2, 3, 10)]
    specs += [with_costs(10, 0.001, vol_scale=f, name=f"vol-x{f:g}/10", section="5.4",
                         description=f"ten assets, every volatility multiplied by {f:g}") for f in (1.5, 0.5)]
    specs += [with_costs(n, 0.001, rho=0.0, name=f"no-correlation/{n}", section="5.4",
                         description=f"{n} independent assets") for n in (2, 3, 10)]
    specs += [cvar(n) for n in (1, 2, 3, 10)]
    return {s.name: s for s in specs}


CATALOG = build_catalog()


def get(name: str, smoke: bool = False) -> ExperimentSpec:
    if name not in CATALOG:
        raise KeyError(f"unknown experiment {name!r}; `deephedge list` shows the catalog")
    spec = CATALOG[name]
    return spec.smoke() if smoke else spec
