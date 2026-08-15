"""Tests for grid-approximated boids steering.

The three rules are tested by their *effect* rather than by comparing against hardcoded
force values: alignment should make neighbouring velocities agree, cohesion should pull a
neighbourhood together, separation should push it apart. That way the tests survive a change
of formulation and only fail if the behaviour is actually wrong.
"""
import numpy as np
import pytest
from conftest import assert_within, best_ms, skip_if_contended

from dtouch.flock import flock_forces


def _pair(x0=10.0, x1=12.0, v0=(1.0, 0.0), v1=(-1.0, 0.0)):
    """Two particles in the same cell, given positions and opposing velocities."""
    px = np.array([x0, x1], np.float32)
    py = np.array([10.0, 10.0], np.float32)
    vx = np.array([v0[0], v1[0]], np.float32)
    vy = np.array([v0[1], v1[1]], np.float32)
    return px, py, vx, vy


def test_all_gains_zero_is_exactly_zero_force():
    """The off state must cost nothing and change nothing — it is the default."""
    px, py, vx, vy = _pair()
    fx, fy = flock_forces(px, py, vx, vy, 100, 100)
    assert np.all(fx == 0) and np.all(fy == 0)


def test_alignment_pulls_velocities_together():
    px, py, vx, vy = _pair()
    before = abs(vx[0] - vx[1])
    fx, _ = flock_forces(px, py, vx, vy, 100, 100, cell=32, alignment=0.5)
    after = abs((vx[0] + fx[0]) - (vx[1] + fx[1]))
    assert after < before, "alignment must reduce the velocity difference"


def test_cohesion_pulls_positions_together():
    px, py, vx, vy = _pair(x0=10.0, x1=20.0)
    fx, _ = flock_forces(px, py, vx, vy, 100, 100, cell=32, cohesion=1.0)
    # each should be pushed toward the other
    assert fx[0] > 0, "left particle steers right, toward its neighbour"
    assert fx[1] < 0, "right particle steers left, toward its neighbour"


def test_separation_pushes_positions_apart():
    px, py, vx, vy = _pair(x0=10.0, x1=12.0)
    fx, _ = flock_forces(px, py, vx, vy, 100, 100, cell=32, separation=1.0)
    assert fx[0] < 0, "left particle steers away, further left"
    assert fx[1] > 0, "right particle steers away, further right"


def test_separation_is_stronger_when_closer():
    """The 1/r falloff — crowding should hurt more than proximity."""
    near = flock_forces(*_pair(x0=10.0, x1=10.5), gw=100, gh=100, cell=32, separation=1.0)[0]
    far = flock_forces(*_pair(x0=10.0, x1=25.0), gw=100, gh=100, cell=32, separation=1.0)[0]
    assert abs(near[0]) > abs(far[0])


def test_a_particle_alone_in_its_cell_gets_no_force():
    """Self-inclusion would make a lone particle chase (or flee) its own position."""
    px = np.array([5.0, 95.0], np.float32)     # far apart -> separate cells
    py = np.array([5.0, 95.0], np.float32)
    vx = np.array([1.0, -1.0], np.float32)
    vy = np.array([0.0, 0.0], np.float32)
    fx, fy = flock_forces(px, py, vx, vy, 100, 100, cell=8,
                          cohesion=1.0, alignment=1.0, separation=1.0)
    assert np.all(fx == 0) and np.all(fy == 0)


def test_no_nan_or_inf_when_particles_coincide():
    """Two particles at the identical point is a divide-by-zero unless it is softened."""
    px = np.full(4, 10.0, np.float32)
    py = np.full(4, 10.0, np.float32)
    vx = np.zeros(4, np.float32)
    vy = np.zeros(4, np.float32)
    fx, fy = flock_forces(px, py, vx, vy, 100, 100, cell=16,
                          cohesion=1.0, alignment=1.0, separation=1.0)
    assert np.all(np.isfinite(fx)) and np.all(np.isfinite(fy))


def test_out_of_range_positions_are_clipped_not_wrapped():
    """A position past the grid edge must not index into an unrelated cell."""
    px = np.array([-50.0, 500.0], np.float32)
    py = np.array([-50.0, 500.0], np.float32)
    vx = np.zeros(2, np.float32)
    vy = np.zeros(2, np.float32)
    fx, fy = flock_forces(px, py, vx, vy, 100, 100, cell=16,
                          cohesion=1.0, separation=1.0, alignment=1.0)
    assert np.all(np.isfinite(fx)) and np.all(np.isfinite(fy))
    # they are clipped into opposite corners, so they stay in different cells: no force
    assert np.all(fx == 0)


def test_flocking_makes_a_cloud_more_ordered_over_time():
    """End to end: random velocities under alignment should converge toward agreement."""
    rng = np.random.default_rng(0)
    n = 2000
    px = rng.uniform(0, 100, n).astype(np.float32)
    py = rng.uniform(0, 100, n).astype(np.float32)
    vx = rng.standard_normal(n).astype(np.float32)
    vy = rng.standard_normal(n).astype(np.float32)
    def spread(a, b):
        return float(np.mean(np.hypot(a - a.mean(), b - b.mean())))
    start = spread(vx, vy)
    for _ in range(40):
        fx, fy = flock_forces(px, py, vx, vy, 100, 100, cell=25, alignment=0.4)
        vx += fx; vy += fy
        px += vx * 0.1; py += vy * 0.1
        px = np.clip(px, 0, 99.9); py = np.clip(py, 0, 99.9)
    assert spread(vx, vy) < start, "an aligning cloud should become more uniform"


def test_scales_to_the_live_instrument_particle_count(perf_reference):
    """200k is what the live app actually pushes; O(n^2) would never return.

    The one workload in the suite a ratio budget cannot rescue on its own.
    It scatters into a spatial grid, so it is bound by memory LATENCY where
    the reference pass is bound by bandwidth, and the two do not slow down
    together: its ratio went 12.3x quiet -> 76.2x with the load average at 45
    on 14 cores. No margin both survives that and still means anything, so it
    declines to measure on a contended box and asserts a x4 margin (49x) on a
    quiet one. The regression it exists to catch is a complexity change —
    true all-pairs boids at this count is ~4e10 pair terms — which is orders
    of magnitude away, not a factor of four."""
    skip_if_contended()
    rng = np.random.default_rng(1)
    n = 200_000
    px = rng.uniform(0, 416, n).astype(np.float32)
    py = rng.uniform(0, 234, n).astype(np.float32)
    vx = rng.standard_normal(n).astype(np.float32)
    vy = rng.standard_normal(n).astype(np.float32)

    def step():
        flock_forces(px, py, vx, vy, 416, 234, cohesion=0.5, alignment=0.5,
                     separation=0.5)

    step()
    assert_within(best_ms(step, runs=6), 12.3, perf_reference,
                  "flocking at 200k particles", margin=4.0)


def test_forces_are_float32():
    """float64 leaking in would silently double the memory traffic of the hot loop."""
    px, py, vx, vy = _pair()
    fx, fy = flock_forces(px, py, vx, vy, 100, 100, cell=32, cohesion=1.0)
    assert fx.dtype == np.float32 and fy.dtype == np.float32
