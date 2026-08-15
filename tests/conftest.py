"""Shared test helpers — today, the perf-budget machinery.

A perf test has one job: fail when the code gets slower, and only then. The
suite's budgets were absolute milliseconds, which fails that on both sides —
they encode the development machine's speed, and they fire when the box is
merely busy. `test_ordered_dither_720p_within_budget` failed under machine
load on a laptop it passes on when idle, which teaches the reader to re-run
the suite until it is green: the worst thing a test can teach.

The fix is to stop measuring milliseconds and start measuring RATIOS. Every
budget here is a multiple of `reference_ms()` — a fixed, dtouch-free numpy
workload timed in the same process, moments before, on the same cores. A
faster machine speeds up both sides; a busy machine slows down both sides;
an algorithmic regression moves only one of them, which is the whole point.

Two more things keep it honest:

- **best-of-N, not mean-of-N.** A mean is a measure of the machine's spare
  capacity as much as of the code. The minimum over enough runs is the cost
  when nothing else got in the way, which is the number a budget is about.
- **a documented margin.** Budgets are set at 2x the measured ratio, so a 2x
  blowup fails and ordinary scheduling noise does not. The ratios themselves
  are recorded next to each budget, so a drift is visible in the diff rather
  than hidden inside a round number.
"""
import os
import time

import numpy as np
import pytest

# Runs to take the minimum over. Enough that at least one lands in a quiet
# slot on a loaded box; cheap enough that the whole perf set stays under a
# second (the workloads are single-digit ms).
PERF_RUNS = 15

# Every budget in the suite is `MARGIN x the ratio measured on a quiet
# machine`. 2.0 is deliberate: the regressions worth catching here are
# algorithmic (the ASCII renderer's rejected per-cell putText path was 17x),
# and nothing this side of 2x is distinguishable from a scheduler.
#
# Checked against a real load rather than argued. With the box TWICE
# oversubscribed (load average 46 on 14 cores), the measured ratios moved:
#
#   ascii 720p/45    2.10x ->  2.08x     rock stable
#   HUD draw         0.73x ->  0.86x
#   ordered dither  16.40x -> 21.24x
#   flocking 200k   12.30x -> 48.83x     see test_flock.py
#
# The first three stayed inside a x2 budget, while the absolute 12 ms budget
# this replaced measured 15.1 ms and failed. Flocking did not, and that is a
# real property rather than a tuning problem: it scatters into a spatial grid,
# so it is bound by memory LATENCY where the reference is bound by bandwidth,
# and the two do not slow down together. It passes an explicit wider margin
# and says why — which is the point of making the margin a parameter instead
# of quietly raising the default for everyone.
PERF_MARGIN = 2.0

_REF_SHAPE = (720, 1280, 3)


def best_ms(fn, runs=PERF_RUNS):
    """Minimum wall-clock ms over *runs* calls. Warm the caches yourself."""
    best = np.inf
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best * 1000.0


def reference_ms(runs=PERF_RUNS):
    """Cost of a fixed float32 pass over a 720p RGB plane, in ms.

    Deliberately memory-bandwidth-bound and deliberately nothing to do with
    dtouch: it is the unit every perf budget in the suite is denominated in,
    so it must move with the machine and never with the code under test.
    """
    a = np.full(_REF_SHAPE, 0.5, np.float32)
    out = np.empty_like(a)

    def _pass():
        np.multiply(a, np.float32(1.7), out=out)
        np.add(out, np.float32(0.3), out=out)

    _pass()                                   # warm
    return best_ms(_pass, runs)


def skip_if_contended():
    """Skip when the machine is oversubscribed — for latency-bound work only.

    A ratio budget already absorbs a busy box for anything bandwidth-bound.
    It does not for a kernel dominated by memory LATENCY: flocking's ratio to
    the reference went 12.3x -> 76.2x as the load average passed 45 on 14
    cores, because cache-miss cost explodes under contention while the
    reference's streaming cost does not. There is no margin that both survives
    that and still means something, so the honest move is to decline to
    measure. Load average against core count is the direct question — "is
    anything else running?" — and a lagging one-minute EMA is fine here
    because it is a second line of defence, not the only one.
    """
    try:
        load = os.getloadavg()[0]
    except (OSError, AttributeError):        # not POSIX; assume quiet
        return
    cpus = os.cpu_count() or 1
    if load > cpus:
        pytest.skip("load average %.1f on %d cores: a latency-bound timing "
                    "here would measure the box, not the code" % (load, cpus))


def assert_within(measured_ms, ratio, reference, what, margin=PERF_MARGIN):
    """Assert *measured_ms* is inside `margin * ratio * reference` ms.

    Raise *margin* only for a workload whose RATIO is itself load-sensitive —
    a kernel dominated by memory latency rather than bandwidth does not slow
    down in step with the reference — and say so where you raise it. The
    default is the point of the whole exercise; a per-call margin is a claim
    that this workload's ratio was measured under load and found to drift.
    """
    budget = margin * ratio * reference
    assert measured_ms < budget, (
        "%s: %.2f ms = %.1fx the reference pass (%.3f ms); budget is %.1fx "
        "(%.1fx measured on a quiet machine, x%.1f margin)"
        % (what, measured_ms, measured_ms / reference, reference,
           margin * ratio, ratio, margin))


@pytest.fixture
def perf_reference():
    """The reference pass, timed fresh for EVERY perf test.

    Function-scoped on purpose, and it was session-scoped first: a reference
    measured once at the start of a run and compared against a workload timed
    two minutes later is exactly the mismatch this whole file exists to
    remove. On a box whose load was ramping through the run (average 25 -> 59)
    the stale reference made healthy code look like a regression. It costs
    ~6 ms per perf test to re-time it, which is nothing next to being wrong.
    """
    return reference_ms()
