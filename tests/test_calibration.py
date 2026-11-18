import math

import pytest

from deephedge.calibration import (
    CENTRE,
    CVAR,
    LOCAL_CENTRE,
    Calibration,
    ScaleSearch,
    calibrate_centre_and_scale,
    calibrate_scale,
    refine_centre_and_scale,
)


def bowl(best):
    return lambda g: (math.log(g) - math.log(best)) ** 2


def test_finds_the_minimum_inside_the_grid():
    result = calibrate_scale(bowl(2.0), ScaleSearch((0.5, 1.0, 2.0, 4.0, 8.0)))
    assert result.scale == 2.0 and not result.at_edge


def test_extends_the_grid_when_the_best_is_on_the_edge():
    result = calibrate_scale(bowl(20.0), ScaleSearch((0.5, 1.0, 2.0), up=1.6, extensions=4, refinement="none"))
    assert result.scale == pytest.approx(2.0 * 1.6 ** 4, rel=1e-6)
    assert result.at_edge          # four extensions were not enough to reach 20


def test_centre_search_finds_a_known_optimum():
    def f(g, c):
        return (math.log(g) - math.log(30.0)) ** 2 + 4 * (c - 1.4) ** 2

    base = calibrate_scale(lambda g: f(g, 1.0), CVAR)
    result = calibrate_centre_and_scale(f, base, CENTRE)
    assert abs(result.centre - 1.4) < 0.05 and abs(math.log(result.scale / 30.0)) < 0.3


def test_local_centre_search_finds_a_known_optimum():
    # the search of the control of ten assets, around the centre at the delta (c = 1) and the scale of the base box
    def f(g, c):
        return (math.log(g) - math.log(1.9)) ** 2 + 4 * (c - 1.04) ** 2

    base = Calibration(scale=1.650964, value=f(1.650964, 1.0), at_edge=False)
    result = refine_centre_and_scale(f, base, LOCAL_CENTRE)
    assert abs(result.centre - 1.04) < 0.01 and abs(math.log(result.scale / 1.9)) < 0.05 and not result.at_edge
