"""Autopilot's contract: it plays, it surprises, and it gets out of the way."""
import pytest

from dtouch.auto import Autopilot, FIRST_DWELL, DWELL

LOOKS = ["veinwork", "amoeba", "ghost", "lightning", "breath"]
MODES = ["particles", "dithergirl", "physarum"]


def _run(a, seconds, dt=1 / 30, mode="physarum", looks=LOOKS, modes=MODES):
    out = []
    for _ in range(int(seconds / dt)):
        out += a.tick(dt, mode, looks, modes)
    return out


def test_off_by_default_and_silent():
    a = Autopilot(seed=1)
    assert not a.on
    assert _run(a, 300) == []


def test_the_first_recast_is_prompt():
    """A control that does nothing for the first 40 seconds reads as broken."""
    a = Autopilot(seed=1)
    a.set(True)
    assert _run(a, FIRST_DWELL - 0.5) == []
    assert _run(a, 1.0), "nothing fired just after the first dwell"


def test_it_keeps_recasting_on_a_randomized_cadence():
    a = Autopilot(seed=3)
    a.set(True)
    n = len([i for i in _run(a, 600) if i[0] in ("preset", "mode")])
    lo, hi = 600 / DWELL[1], 600 / DWELL[0] + 2
    assert lo - 1 <= n <= hi + 1, n


def test_a_human_scene_input_releases_it_at_once():
    a = Autopilot(seed=1)
    a.set(True)
    _run(a, 10)
    assert a.interrupt() is True
    assert not a.on
    assert _run(a, 600) == []
    assert a.interrupt() is False       # idempotent, no double toast


def test_a_stalled_frame_does_not_fire_a_recast_on_resume():
    """The browser build fired instantly when a hidden tab came back, because
    its deadline was absolute wall-clock. dt is clamped instead."""
    a = Autopilot(seed=1)
    a.set(True)
    assert a.tick(600.0, "physarum", LOOKS, MODES) == []


def test_it_visits_every_look_and_does_not_stutter():
    a = Autopilot(seed=5)
    a.set(True)
    picks = [v for k, v in _run(a, 4000) if k == "preset"]
    assert set(picks) == set(LOOKS), sorted(set(picks))


def test_it_walks_between_modes_when_allowed():
    a = Autopilot(seed=5)
    a.set(True)
    hops = [v for k, v in _run(a, 4000) if k == "mode"]
    assert hops, "never left the mode it started in"
    assert set(hops) <= set(MODES)


def test_single_mode_autopilot_never_hops():
    a = Autopilot(seed=5, cross_mode=False)
    a.set(True)
    assert [k for k, _ in _run(a, 3000) if k == "mode"] == []


def test_casts_are_occasional_not_constant():
    """A gesture that happens every time stops reading as a gesture."""
    a = Autopilot(seed=9)
    a.set(True)
    got = _run(a, 6000)
    recasts = len([1 for k, _ in got if k == "preset"])
    casts = len([1 for k, _ in got if k == "command"])
    assert 0 < casts < recasts, (casts, recasts)


def test_toggle_round_trips():
    a = Autopilot(seed=1)
    assert a.toggle() is True and a.on
    assert a.toggle() is False and not a.on
