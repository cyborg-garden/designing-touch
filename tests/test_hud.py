"""HUD layer + overlay state machine — DESIGN.md §5 (visual style), §6.1 (states).

Everything runs on a fake clock so fades are deterministic. The load-bearing pin
is the OBS capture contract: HIDDEN draws NOTHING once toasts expire — the frame
is provably untouched.
"""
import time

import numpy as np
import pytest

from dtouch import hud as H
from dtouch.hud import (Hud, Osd, OverlayState, Toasts, cycle_overlay,
                        esc_overlay, u)
from dtouch.live import _overlay_key


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def _frame(w=1920, h=1080, gray=60):
    return np.full((h, w, 3), gray, np.uint8)


# ---------- u-unit ----------

@pytest.mark.parametrize("h,expect", [(720, 16.0), (1080, 24.0), (2160, 48.0)])
def test_u_unit(h, expect):
    assert u(h) == expect


# ---------- overlay state machine (table-driven) ----------

@pytest.mark.parametrize("state,expect", [
    (OverlayState.HIDDEN, OverlayState.HUD),
    (OverlayState.HUD, OverlayState.PANEL),
    (OverlayState.PANEL, OverlayState.HIDDEN),
])
def test_tab_cycles(state, expect):
    assert cycle_overlay(state) is expect


@pytest.mark.parametrize("state,expect", [
    (OverlayState.PANEL, OverlayState.HUD),
    (OverlayState.HUD, OverlayState.HIDDEN),
    (OverlayState.HIDDEN, OverlayState.HIDDEN),   # in HIDDEN, Esc does nothing
])
def test_esc_steps_toward_hidden(state, expect):
    assert esc_overlay(state) is expect


@pytest.mark.parametrize("key,state,newstate,handled", [
    (9, OverlayState.HUD, OverlayState.PANEL, True),
    (9, OverlayState.PANEL, OverlayState.HIDDEN, True),
    (27, OverlayState.PANEL, OverlayState.HUD, True),
    (27, OverlayState.HIDDEN, OverlayState.HIDDEN, True),   # handled: Esc NEVER
    (ord("q"), OverlayState.HUD, OverlayState.HUD, False),  # falls through to reg
])
def test_overlay_key_routing(key, state, newstate, handled):
    clock = Clock()
    got, h = _overlay_key(key, state, Toasts(clock))
    assert got is newstate and h is handled


def test_entering_hidden_emits_one_final_toast():
    clock = Clock()
    toasts = Toasts(clock)
    _overlay_key(9, OverlayState.PANEL, toasts)     # PANEL -> HIDDEN
    assert toasts.active()
    assert any("TAB to show" in t.text for t in toasts._hints)
    # Esc while already HIDDEN must not re-toast
    toasts2 = Toasts(clock)
    _overlay_key(27, OverlayState.HIDDEN, toasts2)
    assert not toasts2.active()


# ---------- outlined text ----------

def test_put_outlined_draws_ink_and_black_outline():
    img = _frame(640, 360, gray=128)
    H.put_outlined(img, "TEST", (100, 100), 24)
    assert not np.array_equal(img, _frame(640, 360, gray=128))
    # double-drawn: pure-black outline pixels exist under/around the ink
    region = img[60:110, 90:260]
    assert (region == 0).all(axis=2).any(), "no black outline pixels found"


# ---------- toasts ----------

def test_center_flash_draws_then_expires_clean():
    clock = Clock()
    t = Toasts(clock)
    t.flash("EMBERS")
    img = _frame()
    t.draw(img)
    assert not np.array_equal(img, _frame())
    clock.t = 5.0
    img2 = _frame()
    t.draw(img2)
    assert np.array_equal(img2, _frame())
    assert not t.active()


def test_hints_stack_and_cap_at_three():
    clock = Clock()
    t = Toasts(clock)
    for i in range(5):
        t.hint(f"hint {i}")
    assert len(t._hints) == 3
    img = _frame()
    t.draw(img)
    assert not np.array_equal(img, _frame())


# ---------- HIDDEN is provably clean ----------

def test_hidden_draws_nothing_after_toasts_expire():
    clock = Clock()
    hud = Hud(now=clock)
    hud.toasts.hint("overlay hidden - TAB to show")
    img = _frame()
    hud.draw(img, OverlayState.HIDDEN, status="matte=auto")
    assert not np.array_equal(img, _frame()), "the final toast should still show"
    clock.t = 10.0
    img2 = _frame()
    hud.draw(img2, OverlayState.HIDDEN, status="matte=auto",
             debug_status="60fps", recording=True, camera_lost=False)
    assert np.array_equal(img2, _frame()), "HIDDEN must leave the frame untouched"


def test_blackout_tick_shows_even_in_hidden():
    hud = Hud(now=Clock())
    img = _frame()
    hud.draw(img, OverlayState.HIDDEN, blackout=True)
    diff = np.any(img != _frame(), axis=2)
    ys, xs = np.where(diff)
    assert len(xs) > 0
    # confined to the top-right corner (legs 2u = 48px at 1080p)
    assert xs.min() >= 1920 - 60 and ys.max() <= 60
    # and it's amber
    assert (img[ys, xs] == H.AMBER).all(axis=1).any()


# ---------- status line ----------

def test_status_line_draws_top_left_inside_title_safe():
    hud = Hud(now=Clock())
    img = _frame()
    hud.draw(img, OverlayState.HUD, status="matte=auto  color=ice")
    diff = np.any(img != _frame(), axis=2)
    ys, xs = np.where(diff)
    assert len(xs) > 0
    assert xs.min() >= int(1920 * 0.035) - 2, "respects the 3.5% inset"
    assert ys.max() <= int(1080 * 0.15), "status stays in the top band"


def test_debug_toggle_switches_status_variant():
    a, b = _frame(), _frame()
    hud = Hud(now=Clock())
    hud.draw(a, OverlayState.HUD, status="matte=auto", debug_status="60.0fps 16.7ms")
    hud.debug = True
    hud.draw(b, OverlayState.HUD, status="matte=auto", debug_status="60.0fps 16.7ms")
    assert not np.array_equal(a, b)


def test_record_dot_is_red():
    hud = Hud(now=Clock())
    img = _frame()
    hud.draw(img, OverlayState.HUD, status="s", recording=True)
    diff = np.any(img != _frame(), axis=2)
    assert (img[diff] == H.RED).all(axis=1).any()


# ---------- camera-loss note ----------

def test_camera_note_text_is_verbatim():
    # DESIGN.md §6.4: the existing human message, kept verbatim
    assert H.CAMERA_NOTE == "CAMERA IS BLACK - disable iPhone Continuity Camera"
    assert H.CAMERA_FIX == ("iPhone: Settings > General > AirPlay & Handoff > "
                            "Continuity Camera > Off")


def test_camera_note_draws_amber():
    hud = Hud(now=Clock())
    img = _frame()
    hud.draw(img, OverlayState.HUD, camera_lost=True)
    diff = np.any(img != _frame(), axis=2)
    assert (img[diff] == H.AMBER).all(axis=1).any()


def test_camera_note_sits_top_left_under_the_status_line():
    """DESIGN.md §5 center-clear rule: the persistent camera-lost note lives
    top-left under the status line — the frame center stays reserved for
    transient toasts. (The pre-show 'waiting for camera' message is the one
    centered exception.)"""
    hud = Hud(now=Clock())
    img = _frame()
    hud.draw(img, OverlayState.HUD, camera_lost=True)
    diff = np.any(img != _frame(), axis=2)
    ys, xs = np.where(diff)
    assert len(ys) > 0
    assert ys.max() < 1080 * 0.25, "note must sit near the top"
    assert xs.min() >= int(1920 * 0.03), "title-safe left inset"
    assert not diff[int(1080 * 0.4):int(1080 * 0.6)].any(), \
        "the frame center must stay clear"


# ---------- OSD ----------

def test_osd_draws_bottom_left_and_fades():
    clock = Clock()
    osd = Osd(now=clock)
    osd.show("Trails", 0.93, lo=0.5, hi=0.985)
    img = _frame()
    osd.draw(img)
    diff = np.any(img != _frame(), axis=2)
    ys, xs = np.where(diff)
    assert len(xs) > 0
    assert ys.min() >= 1080 // 2, "OSD sits in the bottom half"
    assert xs.max() <= 1920 // 2, "OSD sits on the left"
    clock.t = 2.0
    img2 = _frame()
    osd.draw(img2)
    assert np.array_equal(img2, _frame())


# ---------- budget ----------

def test_hud_draw_stays_within_rough_budget():
    """DESIGN.md §5: overlay budget <= 1 ms/frame. Toasts/status are small-ROI
    ops; typical measure here is well under 0.5 ms. Asserted at 2 ms to absorb
    CI noise while still catching an accidental full-frame op."""
    clock = Clock()
    hud = Hud(now=clock)
    hud.toasts.flash("EMBERS")
    hud.toasts.hint("? for keys")
    img = _frame()
    hud.draw(img, OverlayState.HUD, status="matte=auto  color=ice", recording=True,
             blackout=True)   # warm-up
    n = 100
    t0 = time.perf_counter()
    for _ in range(n):
        clock.t += 0.001     # keep the toasts alive (fading) the whole run
        hud.draw(img, OverlayState.HUD, status="matte=auto  color=ice",
                 recording=True, blackout=True)
    per_frame_ms = (time.perf_counter() - t0) * 1000.0 / n
    assert per_frame_ms < 2.0, f"HUD draw took {per_frame_ms:.2f} ms/frame"
