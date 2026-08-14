"""Collapsible panel sections, and the MOTION / SIGNAL controls they hold.

MOTION (boids) and SIGNAL (circuit-bent) pushed the control column past the 1080p window,
which broke the "everything reachable without scrolling at 1080p" contract that
test_overlay_scroll.py pins. Collapsing is the fix; these tests hold both ends of it —
the panel still fits by default, and the new controls are genuinely reachable once opened.
"""
import cv2
import numpy as np
import pytest

from dtouch.overlay_ui import OverlayUI, DITHERS
from dtouch.particles import PALETTES
from dtouch.live import MATTES

PRESETS = ["abstract", "portrait", "textured", "embers", "aurora", "sigil"]


def _ui(w=1920, h=1080):
    ui = OverlayUI(w, h, PRESETS, list(PALETTES), MATTES)
    ui.draw(np.zeros((h, w, 3), np.uint8), {"status": ""})
    return ui


def _keys(ui, w=1920, h=1080):
    ui.draw(np.zeros((h, w, 3), np.uint8), {"status": ""})
    return [k for _, k, _ in ui._hot]


def _hit(ui, key, w=1920, h=1080):
    """Click the centre of the first control with this key."""
    ui.draw(np.zeros((h, w, 3), np.uint8), {"status": ""})
    rect = next(r for r, k, _ in ui._hot if k == key)
    cx, cy = (rect[0] + rect[2]) // 2, (rect[1] + rect[3]) // 2
    ui.on_mouse(cv2.EVENT_LBUTTONDOWN, cx, cy, 0)


def test_global_rows_match_the_41_sketch():
    """DESIGN.md §4.1: Sound react (A) - Sens - Record (R) - Mirror -
    Menu (M) - Quit. The Menu row is an Action posting menu.open."""
    from dtouch.overlay_ui import build_global_rows
    from dtouch.panelspec import Action
    rows = build_global_rows()
    labels = [getattr(r, "label", None) for r in rows]
    assert labels == ["Sound react (A)", "Sens", "Record (R)", "Mirror",
                      "Menu (M)", "Quit"]
    rec = rows[2]
    assert rec.label_fn(False) == "Record (R)"
    assert rec.label_fn(True) == "Stop recording (R)"
    menu = rows[4]
    assert isinstance(menu, Action) and menu.command == "menu.open"


def test_section_key_hints_declared_and_drawn_right_aligned():
    """DESIGN.md §4.1: MOTION (F), SIGNAL (G) — DIM, right-aligned in the
    header."""
    from dtouch import imgui
    from dtouch.overlay_ui import (build_particles_sections,
                                   build_signal_section)
    secs = build_particles_sections(list(PALETTES), MATTES)
    motion = next(s for s in secs if s.title == "MOTION")
    assert motion.key_hint == "F"
    assert build_signal_section().key_hint == "G"

    a = np.zeros((100, 400, 3), np.uint8)
    b = np.zeros((100, 400, 3), np.uint8)
    ga = imgui.Gui(); ga.begin(1.0, (-1, -1))
    ga.section(a, "MOTION", True, 10, 20, 300)
    gb = imgui.Gui(); gb.begin(1.0, (-1, -1))
    gb.section(b, "MOTION", True, 10, 20, 300, key_hint="F")
    diff = (a != b).any(axis=2)
    assert diff.any(), "the key hint must draw"
    ys, xs = np.where(diff)
    assert xs.min() > 10 + 150, "the key hint must be right-aligned"


def test_particles_status_marks_matte_and_color():
    """DESIGN.md §2.3: the status line derives from status-marked widgets —
    Particles marks its matte and color cycles."""
    from dtouch.overlay_ui import build_particles_sections
    from dtouch.panelspec import Cycle
    secs = build_particles_sections(list(PALETTES), MATTES)
    by = {s.title: s.widgets for s in secs}
    matte = next(w for w in by["SOURCE"] if isinstance(w, Cycle)
                 and w.attr == "matte_idx")
    color = next(w for w in by["LOOK"] if isinstance(w, Cycle))
    assert matte.status == "matte {}"
    assert color.status == "{}"


# ---------- click-flash: border-only + relocation ----------

def _pure_white(img):
    return (img == 255).all(axis=2)


def test_click_flash_is_border_only():
    """The flash must not fill the row (it was hiding the armed delete's
    'sure? x again' content) — border only."""
    ui = OverlayUI(1280, 720, PRESETS, list(PALETTES), MATTES)
    frame = np.zeros((720, 1280, 3), np.uint8)
    ui.draw(frame, {"status": ""})
    rect = next(r for r, k, _ in ui._hot if k == "preset")
    cx, cy = (rect[0] + rect[2]) // 2, (rect[1] + rect[3]) // 2
    ui.on_mouse(cv2.EVENT_LBUTTONDOWN, cx, cy, 0)
    f1 = np.zeros((720, 1280, 3), np.uint8)
    ui.draw(f1, {"status": ""})
    x0, y0, x1, y1 = next(r for r, k, p in ui._hot if k == "preset" and p == 0)
    white = _pure_white(f1)
    assert white[y0, x0 + 4:x1 - 4].any(), "border must flash"
    assert not white[y0 + 4:y1 - 4, x0 + 4:x1 - 4].any(), \
        "the row's interior must stay visible (no fill)"


def test_click_flash_relocates_after_scroll_and_res_change():
    """A stale flash rect after scroll/reflow/res-change was painting a box
    over unrelated pixels — the flash re-locates to the clicked control's
    CURRENT rect, or disappears with it."""
    ui = OverlayUI(1280, 720, PRESETS, list(PALETTES), MATTES)
    frame = np.zeros((720, 1280, 3), np.uint8)
    ui.draw(frame, {"status": ""})
    rect = next(r for r, k, p in ui._hot if k == "preset" and p == 0)
    cx, cy = (rect[0] + rect[2]) // 2, (rect[1] + rect[3]) // 2
    ui.on_mouse(cv2.EVENT_LBUTTONDOWN, cx, cy, 0)
    ui.scroll = 40                            # panel overflows at 720p
    f1 = np.zeros((720, 1280, 3), np.uint8)
    ui.draw(f1, {"status": ""})
    nx0, ny0, nx1, _ = next(r for r, k, p in ui._hot
                            if k == "preset" and p == 0)
    assert ny0 == rect[1] - 40                # the row really moved
    white = _pure_white(f1)
    assert white[ny0, nx0 + 4:nx1 - 4].any(), "flash must follow the row"
    assert not white[rect[1], nx0 + 4:nx1 - 4].any(), "no flash at the stale y"
    # res change mid-flash: relocation against the new-scale hit rects
    ui.scroll = 0
    f2 = np.zeros((2160, 3840, 3), np.uint8)
    ui.w, ui.h = 3840, 2160
    ui.draw(f2, {"status": ""})               # must not crash; flash relocated
    hx0, hy0, hx1, _ = next(r for r, k, p in ui._hot
                            if k == "preset" and p == 0)
    assert _pure_white(f2)[hy0, hx0 + 4:hx1 - 4].any()


def test_rename_hint_and_armed_delete_have_black_underdraw():
    """DESIGN.md §5: all text is double-drawn — these two land OUTSIDE the
    panel scrim, over the live picture, so a white background must not
    swallow them."""
    from dtouch import imgui
    g = imgui.Gui()
    g.begin(1.0, (-1, -1))
    img = np.full((100, 900, 3), 255, np.uint8)      # worst case: white wall
    g.rename_box(img, "name", 0, 700, 40, 180, 700)
    hint_roi = img[40:72, 700 - 245:700 - 60]
    assert (hint_roi < 40).all(axis=2).any(), "rename hint needs under-draw"
    img2 = np.full((100, 900, 3), 255, np.uint8)
    g.manage_buttons(img2, "n", 700, 40, 180, armed="n")
    armed_roi = img2[40:72, 700 - 130:700 - 10]
    assert (armed_roi < 40).all(axis=2).any(), "'sure? x again' needs under-draw"


def test_new_sections_start_closed_so_the_panel_still_fits_1080p():
    ui = _ui()
    assert ui.sections["MOTION"] is False
    assert ui.sections["SIGNAL"] is False
    assert ui._content_h <= 1080, "the default panel must not need scrolling at 1080p"


def test_closed_sections_hide_their_controls():
    ui = _ui()
    keys = _keys(ui)
    assert "flock" not in keys and "glitch" not in keys
    assert "section" in keys, "but the headers themselves are always clickable"


def test_clicking_a_header_reveals_its_controls():
    ui = _ui()
    _hit(ui, "section")            # first header is TEMPLATES
    assert ui.sections["TEMPLATES"] is False, "clicking a header toggles it"

    ui = _ui()
    ui.sections["MOTION"] = True
    keys = _keys(ui)
    for k in ("flock", "cohere", "align", "separate"):
        assert k in keys or any(p and p[0] == k for _, kk, p in ui._hot if kk == "slider"), \
            f"{k} should be reachable when MOTION is open"


def test_motion_controls_are_wired():
    ui = _ui()
    ui.sections["MOTION"] = True
    assert ui.flock is False
    _hit(ui, "flock")
    assert ui.flock is True, "the Flock toggle must actually flip"


def test_signal_controls_are_wired():
    ui = _ui()
    ui.sections["SIGNAL"] = True
    assert ui.glitch is False
    _hit(ui, "glitch")
    assert ui.glitch is True
    assert ui.scanlines is True
    _hit(ui, "scanlines")
    assert ui.scanlines is False


def test_dither_cycles_through_every_mode_and_wraps():
    ui = _ui()
    ui.sections["SIGNAL"] = True
    seen = []
    for _ in range(len(DITHERS) + 1):
        seen.append(ui.dither_name)
        ui.dither_idx = (ui.dither_idx + 1) % len(DITHERS)
    assert seen[:len(DITHERS)] == DITHERS
    assert seen[-1] == DITHERS[0], "cycling must wrap, not run off the end"


def test_off_is_a_real_dither_choice():
    """'off' must be selectable without turning the whole glitch chain off."""
    assert "off" in DITHERS


def test_every_new_slider_has_a_declared_range():
    """A slider without a range raises KeyError at draw time — i.e. in the live app."""
    from dtouch.overlay_ui import _RANGES
    for attr in ("cohere", "align", "separate", "chroma", "drift", "crush"):
        assert attr in _RANGES, f"{attr} has no _RANGES entry"
        lo, hi = _RANGES[attr]
        assert lo < hi
        assert lo <= getattr(_ui(), attr) <= hi, f"{attr}'s default sits outside its range"


def test_all_sections_open_still_reachable_by_scrolling():
    """Opening everything overflows on purpose; scrolling must still get you to Quit."""
    ui = _ui(1280, 720)
    for name in ui.sections:
        ui.sections[name] = True
    # Draw once after opening. The scroll clamp uses the content height measured by the
    # PREVIOUS draw, so until the taller column has been rendered once, scrolling is still
    # clamped to the old (shorter) extent. In the live app that resolves on the next frame,
    # 16 ms later; here nothing would ever redraw unless the test does.
    ui.draw(np.zeros((720, 1280, 3), np.uint8), {"status": ""})
    for _ in range(80):
        ui.on_mouse(cv2.EVENT_MOUSEWHEEL, 1100, 300, -120)
    ui.draw(np.zeros((720, 1280, 3), np.uint8), {"status": ""})
    quit_bottom = next(r for r, k, _ in ui._hot if k == "quit")[3]
    assert quit_bottom <= 720


def test_section_state_survives_a_redraw():
    ui = _ui()
    ui.sections["SIGNAL"] = True
    _keys(ui)
    _keys(ui)
    assert ui.sections["SIGNAL"] is True
