"""The panel-spec data model + generic walker (DESIGN.md §2.3, §8 step 4).

Two ends held here: a synthetic spec renders and hit-tests correctly through
OverlayUI's generic walker (so a future Mode.panel_spec() can trust it), and
the save/apply preset flags read back as declared — they become the single
schema authority for preset capture at step 7.
"""
import cv2
import numpy as np

from dtouch.imgui import Gui
from dtouch.overlay_ui import OverlayUI, build_particles_spec, _RANGES, _SLIDERS
from dtouch.panelspec import (Slider, Toggle, Cycle, Action, PresetList, Section)
from dtouch.particles import PALETTES
from dtouch.live import MATTES

PRESETS = ["abstract", "portrait", "textured", "embers", "aurora", "sigil"]


def _ui(w=1920, h=1080):
    ui = OverlayUI(w, h, PRESETS, list(PALETTES), MATTES)
    ui.draw(np.zeros((h, w, 3), np.uint8), {"status": ""})
    return ui


def _click(ui, key, w=1920, h=1080, payload=None):
    ui.draw(np.zeros((h, w, 3), np.uint8), {"status": ""})
    rect = next(r for r, k, p in ui._hot
                if k == key and (payload is None or p == payload))
    ui.on_mouse(cv2.EVENT_LBUTTONDOWN, (rect[0] + rect[2]) // 2,
                (rect[1] + rect[3]) // 2, 0)


# ---------- synthetic spec through the generic walker ----------

def _synthetic_ui():
    ui = OverlayUI(1920, 1080, PRESETS, list(PALETTES), MATTES)
    ui.beam = False
    ui.heat = 0.5
    ui.wave_idx = 0
    ui.set_spec([
        Section("ALPHA", [
            Toggle("Beam", "beam"),
            Slider("Heat", "heat", 0.0, 1.0, tip="How hot."),
            Cycle("wave", "wave_idx", ["sine", "saw", "square"]),
        ]),
        Action("Do the thing", "mode.thing"),
    ])
    ui.draw(np.zeros((1080, 1920, 3), np.uint8), {"status": ""})
    return ui


def test_synthetic_spec_renders_and_registers_hits():
    ui = _synthetic_ui()
    kinds = [k for _, k, _ in ui._hot]
    assert "beam" in kinds                      # toggle row, keyed by attr
    assert "mode.thing" in kinds                # action row, keyed by command
    assert kinds.count("cycle") == 2            # < and > arrows
    assert any(k == "slider" and p[0] == "heat" for _, k, p in ui._hot)
    assert kinds[0] == "section" and ui._hot[0][2] == "ALPHA"


def test_synthetic_spec_actually_draws_pixels():
    ui = _synthetic_ui()
    canvas = np.zeros((1080, 1920, 3), np.uint8)
    ui.draw(canvas, {"status": ""})
    assert canvas[:, 1920 - ui._panel_px:].any(), "the panel column must be painted"


def test_toggle_flips_its_attr():
    ui = _synthetic_ui()
    _click(ui, "beam")
    assert ui.beam is True
    _click(ui, "beam")
    assert ui.beam is False


def test_cycle_rotates_and_wraps():
    ui = _synthetic_ui()
    for expect in ("saw", "square", "sine"):    # forward through the > arrow, wrapping
        rects = [(r, p) for r, k, p in ui._hot if k == "cycle" and p == ("wave", +1)]
        r, _ = rects[0]
        ui.on_mouse(cv2.EVENT_LBUTTONDOWN, (r[0] + r[2]) // 2, (r[1] + r[3]) // 2, 0)
        assert ["sine", "saw", "square"][ui.wave_idx] == expect
        ui.draw(np.zeros((1080, 1920, 3), np.uint8), {"status": ""})
    ui._activate("cycle", ("wave", -1), 0)      # backward wraps too
    assert ui.wave_idx == 2


def test_slider_sets_attr_from_track_position():
    ui = _synthetic_ui()
    attr, x0, x1, lo, hi = next(p for _, k, p in ui._hot if k == "slider" and p[0] == "heat")
    rect = next(r for r, k, p in ui._hot if k == "slider" and p[0] == "heat")
    ymid = (rect[1] + rect[3]) // 2
    ui.on_mouse(cv2.EVENT_LBUTTONDOWN, x1, ymid, 0)
    assert abs(ui.heat - hi) < 1e-6
    ui.on_mouse(cv2.EVENT_LBUTTONUP, x1, ymid, 0)


def test_action_without_a_mailbox_posts_to_pending_commands():
    ui = _synthetic_ui()
    _click(ui, "mode.thing")
    assert ui.pending_commands == ["mode.thing"]


def test_set_spec_rebuilds_sections_from_spec_defaults():
    ui = _synthetic_ui()
    assert ui.sections == {"ALPHA": True}
    ui._activate("section", "ALPHA", 0)
    assert ui.sections["ALPHA"] is False


# ---------- save/apply flags (the step-7 schema authority) ----------

def test_dataclass_defaults_match_design():
    s = Slider("X", "x", 0.0, 1.0)
    assert (s.save, s.apply) == (True, "reset")
    t = Toggle("X", "x")
    assert (t.save, t.apply) == (True, "keep")
    c = Cycle("x", "x_idx", ["a", "b"])
    assert (c.save, c.apply) == (True, "reset")


def test_flags_overridable_per_instance():
    assert Slider("X", "x", 0.0, 1.0, apply="keep").apply == "keep"
    assert Toggle("X", "x", save=False).save is False


def test_particles_spec_flags_match_todays_capture():
    """What live.py saves today, declared: mirror/record/res/count stay out,
    the asymmetric video/audio controls apply as 'keep'."""
    ui = _ui()
    widgets = {getattr(w, "attr", None): w for w in ui.iter_widgets()}
    assert widgets["record"].save is False
    assert widgets["mirror"].save is False
    assert widgets["res_idx"].save is False
    assert widgets["video_mix"].apply == "keep"
    assert widgets["sens"].apply == "keep"
    assert widgets["video_bg"].apply == "keep"      # Toggle default: keep
    for _, attr, _tip in _SLIDERS:
        assert widgets[attr].save is True and widgets[attr].apply == "reset"
        lo, hi = _RANGES[attr]
        assert (widgets[attr].lo, widgets[attr].hi) == (lo, hi)


def test_particles_spec_reproduces_shipped_structure():
    spec = build_particles_spec(PRESETS, list(PALETTES), MATTES)
    sections = [s for s in spec if isinstance(s, Section)]
    assert [s.title for s in sections] == ["TEMPLATES", "SOURCE", "LOOK", "MOTION", "SIGNAL"]
    assert [s.open for s in sections] == [True, True, True, False, False]
    assert isinstance(sections[0].widgets[0], PresetList)
    globals_ = spec[len(sections):]
    assert isinstance(globals_[-1], Action) and globals_[-1].command == "quit"


# ---------- the toolkit standalone (no OverlayUI) ----------

def test_gui_toolkit_draws_standalone():
    g = Gui()
    img = np.zeros((200, 400, 3), np.uint8)
    hot = g.begin(1.0, (-1, -1))
    g.row(img, "Hello", "hello", 10, 10, 200)
    g.slider(img, "Amt", "amt", 0.5, 0.0, 1.0, 10, 44, 200)
    y, is_open = g.section(img, "SEC", True, 10, 90, 200)
    assert is_open and y == 106
    assert [k for _, k, _ in hot] == ["hello", "slider", "section"]
    (attr, x0, x1, lo, hi) = next(p for _, k, p in hot if k == "slider")
    assert attr == "amt" and lo == 0.0 and hi == 1.0 and x0 < x1
    assert img.any()
