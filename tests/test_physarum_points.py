"""Every behavior point must actually be able to steer.

`sense` and `step` are lengths in grid pixels, and the trail they read has
been box-blurred every frame over a `1/(1-decay)`-frame memory, which
accumulates to roughly SENSE_SIGMA_PX of smoothing. A point whose two side
sensors sit closer together than that smoothing reads the SAME value at both,
so the Jones tie-break resolves the same way every step and the agent turns
one direction forever: a biased random walk wearing a slime mold's name.

That is not hypothetical. `fingers` shipped with 2.21 px of sensor separation
on a field smoothed to 3.33 px, and a stride longer than its own sensing
reach — and it was the default body point, which is a large part of why every
look read as the same electric tangle. These are the floors that keep a point
from being quietly re-broken; widen the point rather than lowering them.
"""
import math

import pytest

from dtouch.physarum import (CPU_SCALE, POINTS, POINT_NAMES, PX_SCALE,
                             SENSE_SIGMA_PX)

# Both engines, and the CPU one binds: the blur radius is the same integer on
# each, so the smoothing is the same number of pixels, but the CPU grid is
# 2.22x narrower. Checking only the GL scale let `fingers` ship at 2.36 px of
# separation on the fallback engine — worse than the 2.21 px this whole
# diagnosis calls blind, on the path every GPU-less machine runs.
SCALES = (("cpu", CPU_SCALE), ("gl", PX_SCALE))


@pytest.mark.parametrize("name", POINT_NAMES)
@pytest.mark.parametrize("engine,scale", SCALES)
def test_point_senses_past_its_own_blur(name, engine, scale):
    """The sensors must reach beyond the smoothing, or they all read one blob."""
    reach = POINTS[name]["sense"] * scale
    assert reach >= SENSE_SIGMA_PX, (
        f"{name} on {engine}: sensor reach {reach:.2f} px is inside the "
        f"{SENSE_SIGMA_PX} px blur — its three sensors sample one smoothed value")


@pytest.mark.parametrize("name", POINT_NAMES)
@pytest.mark.parametrize("engine,scale", SCALES)
def test_point_resolves_a_lateral_gradient(name, engine, scale):
    """Left and right sensors must be far enough apart to disagree."""
    p = POINTS[name]
    sep = 2.0 * p["sense"] * scale * math.sin(p["spread"])
    assert sep >= 1.5 * SENSE_SIGMA_PX, (
        f"{name} on {engine}: left/right separation {sep:.2f} px against "
        f"{SENSE_SIGMA_PX} px of smoothing — fl == fr, so the tie-break turns "
        f"one way every step")


@pytest.mark.parametrize("name", POINT_NAMES)
def test_point_does_not_leap_past_what_it_sensed(name):
    """A stride longer than the sensing reach is a ballistic spray, not a walk."""
    p = POINTS[name]
    assert p["step"] < p["sense"], (
        f"{name}: step {p['step']} >= sense {p['sense']} — the agent lands "
        f"beyond the point it steered toward")


def test_the_points_are_actually_different_from_each_other():
    """Six names have to be six behaviours.

    Distance is taken in log space over all four steering parameters, not on
    any single axis: `cells` and `storm` sit close on sensor separation and
    are still nothing alike, because one barely turns and the other whips.
    The floor is a lower bound on how far apart two points must be before
    they are worth two entries in the menu.
    """
    keys = ("sense", "spread", "turn", "step")

    def dist(a, b):
        return math.sqrt(sum(math.log(POINTS[a][k] / POINTS[b][k]) ** 2
                             for k in keys))

    worst = min(((dist(a, b), a, b)
                 for i, a in enumerate(POINT_NAMES)
                 for b in POINT_NAMES[i + 1:]), key=lambda t: t[0])
    d, a, b = worst
    assert d >= 0.9, (
        f"{a} and {b} are {d:.2f} apart in log-parameter space — too close to "
        f"read as two different behaviours")
