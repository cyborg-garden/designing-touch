"""HUD layer + overlay state machine — DESIGN.md §5 (visual style), §6.1 (states).

Everything runs on a fake clock so fades are deterministic. The load-bearing pin
is the OBS capture contract: HIDDEN draws NOTHING once toasts expire — the frame
is provably untouched.
"""
import numpy as np
import pytest
from conftest import assert_within, best_ms

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


# ---------- a toast that does not fit is not a message ----------
# The u-unit scales type with the FRAME, not with the string, so a long line
# at a fixed size overflows every resolution equally. The containment flash
# measured 1435 px in a 1280 px frame at 720p, 2152 in 1920 and 4303 in 3840 —
# clipped at both ends, rendering as "mething went wrong - show continu". It is
# the one message the whole containment mechanism exists to show.

CONTAINMENT_FLASH = "something went wrong - show continues"
SIZES = [(1280, 720), (1920, 1080), (3840, 2160)]


def _drawn(monkeypatch, toasts, w, h):
    """Every (text, org, px) blend_outlined lays down for one draw()."""
    calls = []
    real = H.blend_outlined
    monkeypatch.setattr(
        H, "blend_outlined",
        lambda img, text, org, px, color, alpha:
            calls.append((text, org, px)) or real(img, text, org, px, color, alpha))
    toasts.draw(_frame(w, h))
    return calls


@pytest.mark.parametrize("w,h", SIZES)
def test_a_flash_always_fits_inside_the_title_safe_box(monkeypatch, w, h):
    t = Toasts(Clock())
    t.flash(CONTAINMENT_FLASH)
    (text, org, px), = _drawn(monkeypatch, t, w, h)

    inset = int(w * H.TITLE_SAFE)
    tw, _, _ = H.text_size(text, px)
    assert org[0] >= inset, f"{w}x{h}: clipped at the left"
    assert org[0] + tw <= w - inset, f"{w}x{h}: {tw}px of text in a {w}px frame"


@pytest.mark.parametrize("w,h", SIZES)
def test_a_hint_carrying_exception_text_also_fits(monkeypatch, w, h):
    """Hints carry `str(e)[:80]` and file paths — a clipped explanation of a
    failure is not an explanation."""
    t = Toasts(Clock())
    t.hint("ModuleNotFoundError: No module named 'mediapipe' - "
           "reinstall with the [person] extra")
    (text, org, px), = _drawn(monkeypatch, t, w, h)

    inset = int(w * H.TITLE_SAFE)
    tw, _, _ = H.text_size(text, px)
    assert org[0] >= inset and org[0] + tw <= w - inset


def test_a_short_flash_keeps_its_full_3u_size(monkeypatch):
    """Shrink-to-fit only shrinks what does not fit: a mode name is still the
    3.0u type the design specifies (DESIGN.md §5 type table)."""
    t = Toasts(Clock())
    t.flash("EMBERS")
    (_, _, px), = _drawn(monkeypatch, t, 1920, 1080)
    assert px == int(3.0 * u(1080))


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

def test_hud_draw_stays_within_rough_budget(perf_reference):
    """DESIGN.md §5: the overlay is a set of small-ROI ops, and the failure
    worth catching is an accidental FULL-FRAME one — which on a 1080p frame is
    not a percentage, it is a different order of magnitude.

    Measured at 0.73x the reference float32 pass over a 720p RGB plane (see
    tests/conftest.py), so the budget is a ratio rather than the flat 2 ms this
    replaces: mean-of-100 wall clock made a busy machine look like a
    regression, and it duly failed the suite with every core pegged."""
    clock = Clock()
    hud = Hud(now=clock)
    hud.toasts.flash("EMBERS")
    hud.toasts.hint("? for keys")
    img = _frame()

    def draw():
        clock.t += 0.001     # keep the toasts alive (fading) the whole run
        hud.draw(img, OverlayState.HUD, status="matte=auto  color=ice",
                 recording=True, blackout=True)

    draw()                   # warm-up
    assert_within(best_ms(draw, runs=100), 0.73, perf_reference, "HUD draw")
