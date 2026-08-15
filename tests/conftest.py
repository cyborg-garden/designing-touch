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
# Checked against a real load rather than argued: with all 14 cores pegged,
# the ordered dither's ratio moved 16.4x -> 21.6x (it is more latency-bound
# than the reference, so it does not slow down in perfect step) and the ASCII
# step's moved 2.03x -> 2.04x. Both stayed inside their budgets, while the
# absolute 12 ms budget this replaced measured 15.1 ms and failed. So the
# margin absorbs a saturated machine and still fails a 2x blowup — which is
# the pair of properties a perf test needs and flat milliseconds cannot have.
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


def assert_within(measured_ms, ratio, reference, what):
    """Assert *measured_ms* is inside `PERF_MARGIN * ratio * reference` ms."""
    budget = PERF_MARGIN * ratio * reference
    assert measured_ms < budget, (
        "%s: %.2f ms = %.1fx the reference pass (%.3f ms); budget is %.1fx "
        "(%.1fx measured on a quiet machine, x%.1f margin)"
        % (what, measured_ms, measured_ms / reference, reference,
           PERF_MARGIN * ratio, ratio, PERF_MARGIN))


@pytest.fixture(scope="session")
def perf_reference():
    """The reference pass, timed once per session."""
    return reference_ms()
