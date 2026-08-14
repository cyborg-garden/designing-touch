"""Panel scrolling — at small outputs (720p) the control column is taller than the
window; wheel / drag-on-empty-panel scrolling must make every control reachable."""
import cv2
import numpy as np

from dtouch.overlay_ui import OverlayUI
from dtouch.particles import PALETTES
from dtouch.live import MATTES

PRESETS = ["abstract", "portrait", "textured", "embers", "aurora", "sigil"]


def _ui(w=1280, h=720):
    ui = OverlayUI(w, h, PRESETS, list(PALETTES), MATTES)
    ui.draw(np.zeros((h, w, 3), np.uint8), {"status": ""})
    return ui


def _quit_bottom(ui, w=1280, h=720):
    ui.draw(np.zeros((h, w, 3), np.uint8), {"status": ""})
    return next(r for r, k, _ in ui._hot if k == "quit")[3]


def test_wheel_scrolls_panel_down():
    ui = _ui()
    before = _quit_bottom(ui)
    ui.on_mouse(cv2.EVENT_MOUSEWHEEL, 1100, 300, -120)   # wheel down (negative delta)
    after = _quit_bottom(ui)
    assert after < before


def test_scroll_clamps_to_zero_at_top():
    ui = _ui()
    ui.on_mouse(cv2.EVENT_MOUSEWHEEL, 1100, 300, +120)    # wheel up at the top
    assert _quit_bottom(ui) == _quit_bottom(_ui())        # unchanged


def test_quit_reachable_at_720p_after_scrolling():
    ui = _ui()
    for _ in range(60):                                   # scroll all the way down
        ui.on_mouse(cv2.EVENT_MOUSEWHEEL, 1100, 300, -120)
    assert _quit_bottom(ui) <= 720


def test_no_scroll_when_content_fits_1080p():
    ui = _ui(1920, 1080)
    before = _quit_bottom(ui, 1920, 1080)
    for _ in range(20):
        ui.on_mouse(cv2.EVENT_MOUSEWHEEL, 1700, 300, -120)
    assert _quit_bottom(ui, 1920, 1080) == before         # clamped: content fits


def test_drag_on_empty_panel_scrolls():
    ui = _ui()
    before = _quit_bottom(ui)
    # a spot inside the panel that hits no control: just under the collapse button
    px = 1280 - 6
    ui.on_mouse(cv2.EVENT_LBUTTONDOWN, px, 400, 0)
    ui.on_mouse(cv2.EVENT_MOUSEMOVE, px, 250, cv2.EVENT_FLAG_LBUTTON)
    ui.on_mouse(cv2.EVENT_LBUTTONUP, px, 250, 0)
    assert _quit_bottom(ui) < before


def test_collapse_button_wins_over_scrolled_row_under_it():
    """DATA-LOSS guard: the collapse button is fixed chrome floating above
    scrolled content. When a user preset's row (and its hover delete button)
    scrolls underneath it, a click must toggle the collapse — never arm the
    delete (two such clicks would silently destroy a saved look)."""
    ui = OverlayUI(1280, 720, PRESETS + ["mine"], list(PALETTES), MATTES)
    ui.user_presets = {"mine"}
    frame = np.zeros((720, 1280, 3), np.uint8)
    ui.draw(frame, {"status": ""})
    collapse = next(r for r, k, _ in ui._hot if k == "collapse")
    row = next(r for r, k, p in ui._hot
               if k == "preset" and p == ui.presets.index("mine"))
    # scroll the panel so the 'mine' row slides under the collapse button
    ui.scroll = row[1] - collapse[1] + 4
    ui.mouse = ((collapse[0] + collapse[2]) // 2,
                (collapse[1] + collapse[3]) // 2)
    ui.draw(frame, {"status": ""})       # hover draw reveals manage buttons
    del_rect = next(r for r, k, p in ui._hot if k == "del" and p == "mine")
    ix0, iy0 = max(del_rect[0], collapse[0]), max(del_rect[1], collapse[1])
    ix1, iy1 = min(del_rect[2], collapse[2]), min(del_rect[3], collapse[3])
    assert ix1 > ix0 and iy1 > iy0, "precondition: delete overlaps collapse"
    cx, cy = (ix0 + ix1) // 2, (iy0 + iy1) // 2
    ui.mouse = (cx, cy)
    ui.draw(frame, {"status": ""})
    ui.on_mouse(cv2.EVENT_LBUTTONDOWN, cx, cy, 0)
    assert ui.open is False              # the collapse click landed
    assert ui._del_armed is None         # the delete never armed
    assert ui.pending_delete is None


def test_slider_drag_still_works_with_scrolling():
    ui = _ui()
    payload = next(p for _, k, p in ui._hot if k == "slider" and p[0] == "video_mix")
    attr, x0, x1, lo, hi = payload
    rect = next(r for r, k, p in ui._hot if k == "slider" and p[0] == "video_mix")
    ymid = (rect[1] + rect[3]) // 2
    ui.on_mouse(cv2.EVENT_LBUTTONDOWN, x1, ymid, 0)
    assert abs(ui.video_mix - hi) < 1e-6
