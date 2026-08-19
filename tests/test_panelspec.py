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
from dtouch.panelspec import (Slider, Toggle, Cycle, Action, PresetList,
                              Section, apply_look)
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


# ---------- unusable VALUES are skipped, never raised ----------
# The store validates a preset file's shape, never its values, and they land
# in float() here — at boot, before any window exists. A well-shaped look
# carrying "contrast": "high" used to end the launch with a traceback.


def test_apply_look_skips_unusable_numbers_and_names_them():
    ui = _ui()
    ui.fade = 0.5
    spec = [Slider("Fade", "fade", 0.0, 1.0), Slider("Spark", "spark", 0.0, 1.0)]
    skipped = apply_look(ui, spec, {"fade": "high", "spark": 0.25})
    assert skipped == ["fade"]
    assert ui.fade == 0.5                    # untouched, not half-written
    assert ui.spark == 0.25                  # the rest of the look still loads


def test_apply_look_rejects_nan_and_infinity():
    """They do not raise — they propagate silently through the render and
    fail somewhere with no connection to the look that introduced them."""
    ui = _ui()
    ui.fade = 0.5
    spec = [Slider("Fade", "fade", 0.0, 1.0)]
    for bad in (float("nan"), float("inf"), None, [1.0]):
        assert apply_look(ui, spec, {"fade": bad}) == ["fade"]
        assert ui.fade == 0.5


def test_apply_look_clamps_a_value_to_the_sliders_declared_range():
    """A Slider's range is the whole truth about what that control can hold.
    `ascii stream` shipped scale=30 under a 45-720 slider: the handle drew off
    the end of its own track and the first click on it destroyed the value.
    Out-of-range is CLAMPED, not skipped — a look that is 90% loadable loads,
    and 30 under a 45 floor means 'as low as this goes', not 'unusable'."""
    ui = _ui()
    ui.fade = 0.5
    spec = [Slider("Fade", "fade", 0.25, 0.75)]
    assert apply_look(ui, spec, {"fade": 2.0}) == []
    assert ui.fade == 0.75
    assert apply_look(ui, spec, {"fade": -3.0}) == []
    assert ui.fade == 0.25
    # in range is still passed through untouched, exactly
    assert apply_look(ui, spec, {"fade": 0.4}) == []
    assert ui.fade == 0.4


def test_a_slider_handle_never_draws_outside_its_track():
    """Defence in depth behind the clamp above: an out-of-range value reaching
    the widget (a hand-edited presets.json, a built-in authored against an
    older range) must still put the handle ON the track. A handle floating
    past the end reads as a broken widget."""
    from dtouch.imgui import HANDLE

    g = Gui()
    for val in (-50.0, 0.0, 0.5, 1.0, 99.0):
        img = np.zeros((200, 400, 3), np.uint8)
        hot = g.begin(1.0, (-1, -1))
        g.slider(img, "Amt", "amt", val, 0.0, 1.0, 10, 44, 200)
        _attr, tx0, tx1, _lo, _hi = next(p for _r, k, p in hot if k == "slider")
        # the handle is the only thing drawn in HANDLE (a fill, so its centre
        # pixels survive the antialiasing exactly)
        cols = np.argwhere((img == np.uint8(HANDLE)).all(axis=2).any(axis=0))
        assert cols.size, "the handle for val=%s must paint something" % val
        # the centre sits ON the track (a handle at an endpoint overhangs by
        # its own radius, which is what an endpoint is supposed to look like)
        centre = (int(cols.min()) + int(cols.max())) / 2.0
        assert tx0 <= centre <= tx1, (
            "handle for val=%s is centred at %s, outside the track %s..%s"
            % (val, centre, tx0, tx1))


def test_apply_look_returns_empty_for_a_clean_look():
    ui = _ui()
    spec = [Slider("Fade", "fade", 0.0, 1.0)]
    assert apply_look(ui, spec, {"fade": 0.75}) == []
    assert ui.fade == 0.75


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


def test_no_shipped_look_sits_outside_the_control_that_edits_it():
    """The clamp above is a safety net, never a silent retune of our own looks.

    Two shipped looks had already drifted outside their sliders: `ascii stream`
    at scale 30 under a 45 floor, and `portrait` at reseed 0.16 under a 0.15
    ceiling — the second only surfaced once apply_look started clamping, which
    turned a look that had rendered at 0.16 since before the panel had ranges
    into one that rendered at 0.15. A built-in is the authority on its own
    value, so the range moved, not the look. This sweeps every mode so the next
    drift cannot land silently.
    """
    from dtouch.modes import REGISTRY

    offenders = []
    for entry in REGISTRY:
        mode = entry() if isinstance(entry, type) else type(entry)()
        ranges = {}
        for section in mode.panel_spec():
            for w in section.widgets:
                if isinstance(w, Slider):
                    ranges[getattr(w, "save_key", None) or w.attr] = (w.lo, w.hi, w.label)
        for look_name, cfg in mode.BUILTIN.items():
            for key, val in cfg.items():
                if key not in ranges or not isinstance(val, (int, float)):
                    continue
                lo, hi, label = ranges[key]
                if not (lo <= float(val) <= hi):
                    offenders.append(
                        "%s/%s: %s=%r outside %s range (%s, %s)"
                        % (mode.id, look_name, key, val, label, lo, hi))
    assert not offenders, ("shipped looks outside their own controls:\n  "
                           + "\n  ".join(offenders))


# ---------- whole-number sliders (DESIGN.md §4.1/§4.2) ----------
#
# The lie being killed: a slider that reads "3.47" while the engine uses 3.
# Every path that can write one of these is pinned, because the value is the
# single thing the panel readout, the nudge OSD, the spec-derived HUD line and
# the engine all read.

def _slider(ui, attr):
    return ui._sliders[attr]


def test_the_quantised_sliders_are_exactly_the_ones_whose_engine_rounds():
    """A whole-number claim has to be checked against the consumer, not
    guessed. Two honesties (panelspec's step docstring): `engine_snaps=True`
    means the engine itself cannot use a fraction (Bits, Crush);
    `engine_snaps=False` means the step is control feel over a continuous
    consumer (Scale — ASCII divides by it before any rounding; Hue — tint_rgb
    takes a float), so a stored look's fraction passes through. Everything
    else in the rack is genuinely continuous and must stay continuous —
    faking coarseness is the same lie as faking precision."""
    from dtouch.modes.dithergirl import DitherGirlMode

    ui = _ui()
    got = {a for a, w in ui._sliders.items() if w.step}
    assert got == {"sig_bits", "crush"}
    assert all(ui._sliders[a].engine_snaps for a in got)
    dg = {w.attr: w for s in DitherGirlMode().panel_spec()
          for w in s.widgets if isinstance(w, Slider)}
    assert {a for a, w in dg.items() if w.step} == {"dg_bits", "dg_scale",
                                                    "dg_hue"}
    assert {a for a, w in dg.items()
            if w.step and not w.engine_snaps} == {"dg_scale", "dg_hue"}
    # continuous by decision, not by omission: Chroma and Drift look like pixel
    # counts and are amplitudes of a per-frame draw, so the fraction is used
    for attr in ("chroma", "drift", "fade", "exposure", "count", "dot", "sens",
                 "cohere", "align", "separate"):
        assert not ui._sliders[attr].step, attr
    for attr in ("chroma", "drift"):
        assert "whole pixel" in ui._sliders[attr].tip, attr


def test_a_quantised_slider_reads_what_the_engine_uses():
    """The specific lie: `"{:.0f}".format(2.92)` is "3" while `int(2.92)` is 2.
    Crush is the one that truncates, so it is the one that could print a
    number the picture was never crushed to."""
    ui = _ui()
    ui.crush = 2.92
    assert ui.crush == 2.0 and int(ui.crush) == 2
    assert f"{ui.crush:{_slider(ui, 'crush').fmt}}" == "2"
    ui.sig_bits = 3.47
    assert ui.sig_bits == 3.0
    assert f"{ui.sig_bits:{_slider(ui, 'sig_bits').fmt}}" == "3"


def test_crush_truncates_and_bits_rounds_because_their_engines_do():
    ui = _ui()
    for raw, want in ((2.99, 2.0), (8.0, 8.0), (0.4, 0.0)):
        ui.crush = 5.0          # far enough that no nudge rule applies
        ui.crush = raw
        assert ui.crush == want, raw
    for raw, want in ((1.4, 1.0), (2.6, 3.0), (4.0, 4.0), (3.47, 3.0)):
        ui.sig_bits = 1.0 if raw > 2.0 else 4.0
        ui.sig_bits = raw
        assert ui.sig_bits == want, raw


def test_a_full_sized_write_sets_a_value_and_does_not_ratchet():
    """Condition 3 of the nudge rule: a write bigger than the `_`/`+` step is
    somebody setting a value. Setting Bits to 3.47 when it already reads 3
    must land on 3, not step to 4."""
    ui = _ui()
    ui.sig_bits = 3.0
    ui.sig_bits = 3.47
    assert ui.sig_bits == 3.0
    ui.crush = 4.0
    ui.crush = 4.9              # 0.9 > the 0.2 press but under the x5 step...
    assert ui.crush == 5.0      # ...so it is still read as a nudge
    ui.crush = 6.5              # 1.5 is over it: a set, floored
    assert ui.crush == 6.0


def test_a_finely_nudgeable_slider_never_ratchets():
    """The rule is gated on `nudge_unreachable`: Scale's quantum is 1 whole
    pixel and one nudge press is 17.25 of them, so it moves on its own and
    must snap plainly — otherwise every sub-pixel write would step it. The
    snap lives on the INPUT surfaces for Scale (engine_snaps=False, so plain
    attribute writes pass through — a stored look's fraction is a real
    picture); `nudge_to` is what those surfaces call."""
    from dtouch.modes.dithergirl import DitherGirlMode
    from dtouch.panelspec import nudge_to, nudge_unreachable

    dg = {w.attr: w for s in DitherGirlMode().panel_spec()
          for w in s.widgets if isinstance(w, Slider)}
    assert not nudge_unreachable(dg["dg_scale"])
    assert not nudge_unreachable(dg["dg_hue"])
    assert nudge_unreachable(dg["dg_bits"])
    assert nudge_to(dg["dg_scale"], 72.4, 72.0) == 72.0   # plain snap, no ratchet
    assert nudge_to(dg["dg_scale"], 89.25, 72.0) == 89.0  # a real press moves it


def test_dragging_a_quantised_slider_snaps_and_stays_put():
    """A drag lands on the nearest step and does NOT ratchet while the finger
    wanders inside it — every mouse-move event is another write."""
    ui = _ui()
    ui.sections["SIGNAL"] = True        # Crush lives in the closed-by-default rack
    ui.draw(np.zeros((1080, 1920, 3), np.uint8), {"status": ""})
    ui.crush = 0.0
    payload = next(p for _r, k, p in ui._hot
                   if k == "slider" and p[0] == "crush")
    _attr, x0, x1, _lo, _hi = payload
    seen = []
    for frac in (0.30, 0.31, 0.32, 0.33):
        ui._set_from_track(payload, int(x0 + frac * (x1 - x0)))
        seen.append(ui.crush)
    assert seen == [2.0, 2.0, 2.0, 2.0], seen
    ui._set_from_track(payload, x1)
    assert ui.crush == 8.0
    ui._set_from_track(payload, x0)
    assert ui.crush == 0.0


def test_the_nudge_keys_move_a_quantised_slider_by_a_whole_step():
    """`-`/`=` step by 1/40 of the range (DESIGN.md §6.2) — 0.075 on Bits,
    whose smallest real step is 1. Quantised naively that is silence on input,
    which §4 calls a bug; it used to 'work' only in that the thirteenth press
    finally moved the picture."""
    ui = _ui()
    w = _slider(ui, "sig_bits")
    ui.sig_bits = 1.0
    step = (w.hi - w.lo) / 40.0           # exactly what shell.py's nudge uses
    for expect in (2.0, 3.0, 4.0, 4.0):   # and it clamps at the top
        ui.sig_bits = min(max(ui.sig_bits + step, w.lo), w.hi)
        assert ui.sig_bits == expect
    for expect in (3.0, 2.0, 1.0, 1.0):
        ui.sig_bits = min(max(ui.sig_bits - step, w.lo), w.hi)
        assert ui.sig_bits == expect


def test_a_stored_look_is_not_re_gridded_by_a_control_the_engine_reads_raw():
    """Scale's step=1.0 must not re-grid stored ASCII looks: under ASCII the
    engine consumes `rows_req` continuously — grid_for divides frame_h by it
    and THEN rounds to a cell — so quantising a stored 45.55 (storable by a
    drag on the previous build) to 46 changes the character grid itself: at
    1080p a 24 px cell becomes 23 (measured 45.7% of pixels moved). The
    branch's own precedent (`portrait`/reseed, pinned two tests up): a stored
    look is the authority; the control adapts, never the look."""
    from dtouch.ascii_art import grid_for
    from dtouch.modes.dithergirl import DitherGirlMode

    # quantising would matter: 45.55 and 46 are different grids at 1080p
    assert grid_for(1920, 1080, 45.55) != grid_for(1920, 1080, 46.0)

    ui = _ui()
    ui.set_spec(DitherGirlMode().panel_spec())    # binds Scale's quantum
    apply_look(ui, ui.spec, {"algorithm": "ASCII", "scale": 45.55})
    assert ui.dg_scale == 45.55
    # ...and a mode switch away and back must not land it either: the
    # set_spec carry-over snap is for engine-snapped sliders only
    ui.set_spec(build_particles_spec(PRESETS, list(PALETTES), MATTES))
    ui.set_spec(DitherGirlMode().panel_spec())
    assert ui.dg_scale == 45.55


def test_a_stored_fraction_reads_faithfully_and_hue_passes_through():
    """The readout follows the value it is showing: `".0f"` on a stored 45.55
    would print "46" over a 45.55-row grid — the readout lie pointed the
    other way — so an off-grid legacy value widens to one honest decimal,
    and an on-grid value keeps the declared fmt. Hue is the same shape:
    tint_rgb takes a float, so the whole-degree step is control feel and a
    stored fractional hue applies exactly."""
    from dtouch.modes.dithergirl import DitherGirlMode
    from dtouch.panelspec import display_fmt

    ui = _ui()
    ui.set_spec(DitherGirlMode().panel_spec())
    apply_look(ui, ui.spec, {"scale": 45.55, "hue": 210.4})
    assert ui.dg_hue == 210.4
    w = ui._sliders["dg_scale"]
    assert display_fmt(w, ui.dg_scale) == ".1f"
    assert display_fmt(w, 46.0) == w.fmt == ".0f"
    cont = ui._sliders["dg_tint"]                 # continuous: never widened
    assert display_fmt(cont, 0.4321) == cont.fmt


def test_quantise_rounds_exactly_as_the_engine_does_on_half_steps():
    """The old lo-anchored grid disagreed with the engines on the half-steps:
    Bits runs 1-4, so `round((2.5 - 1))` said 3 while the engine's
    `int(np.clip(round(2.5), 1, 4))` renders 2 — both roundings are banker's,
    the ANCHOR was the difference — and a stored 2.5 changed bit depth just by
    passing through the quantiser. It must land where the engine lands."""
    from dtouch.modes.dithergirl import DitherGirlMode
    from dtouch.panelspec import quantise

    dg = {w.attr: w for s in DitherGirlMode().panel_spec()
          for w in s.widgets if isinstance(w, Slider)}
    ui = _ui()
    for w in (dg["dg_bits"], ui._sliders["sig_bits"]):
        for v in (1.5, 2.5, 3.5, 1.4, 2.6, 4.0):
            assert quantise(w, v) == int(np.clip(round(v), 1, 4)), (w.attr, v)
    # Crush parity is truncation, already pinned above; what makes the
    # zero anchor a pure parity fix is that every stepped slider's lo sits
    # on the zero-anchored grid
    for w in list(ui._sliders.values()) + list(dg.values()):
        if w.step:
            assert (w.lo / w.step).is_integer(), w.attr


def test_applying_a_look_never_trips_the_nudge_rule():
    """The nudge rule is scoped by who does NOT pre-quantise, so the writer
    that means an exact value must go through the exact path — otherwise a
    stored Crush of 2.10 landing on a live 2.0 would ratchet to 3."""
    ui = _ui()
    spec = [Section("S", [Slider("Crush", "crush", 0.0, 8.0, step=1.0,
                                 snap="floor")])]
    ui.crush = 2.0
    apply_look(ui, spec, {"crush": 2.10})
    assert ui.crush == 2.0
    apply_look(ui, spec, {"crush": 2.99})
    assert ui.crush == 2.0
    apply_look(ui, spec, {"crush": 5.0})
    assert ui.crush == 5.0


def test_binding_a_spec_lands_values_already_on_the_object():
    """A mode switch can bind a quantum to an attr carrying an off-grid value
    from before. The panel must never draw a number the engine is not using,
    including for the one frame after the swap."""
    ui = _ui()
    object.__setattr__(ui, "crush", 3.7)      # smuggle past __setattr__
    ui.set_spec(build_particles_spec(PRESETS, list(PALETTES), MATTES))
    assert ui.crush == 3.0


def test_quantise_leaves_nonsense_and_continuous_values_alone():
    from dtouch.panelspec import nudge_to, quantise

    w = Slider("Bits", "b", 1.0, 4.0, step=1.0)
    cont = Slider("Fade", "f", 0.0, 1.0)
    assert quantise(cont, 0.4321) == 0.4321
    assert quantise(w, "high") == "high"      # apply_look reports it, not us
    assert quantise(w, 9.0) == 4.0 and quantise(w, -3.0) == 1.0
    assert nudge_to(w, 2.0, None) == 2.0      # no current value: plain snap


# ---------- retired option values (DESIGN.md §9) ----------

def test_migrate_legacy_rewrites_a_retired_value_into_several_keys():
    from dtouch.panelspec import migrate_legacy

    spec = [Section("S", [
        Cycle("palette", "pal_idx", ["mono", "amber"], save_key="palette",
              legacy={"old-name": {"palette": "mono", "invert": True}}),
        Toggle("Invert", "inv", save_key="invert"),
    ])]
    cfg = {"palette": "old-name"}
    out = migrate_legacy(spec, cfg)
    assert out == {"palette": "mono", "invert": True}
    assert cfg == {"palette": "old-name"}, "the stored look must not be mutated"
    # a current value is left exactly alone, and identity is preserved so the
    # common case allocates nothing
    same = {"palette": "amber"}
    assert migrate_legacy(spec, same) is same


def test_migrate_legacy_reaches_into_a_nested_store_block():
    from dtouch.panelspec import migrate_legacy

    spec = [Section("SIGNAL", [
        Cycle("dither", "d_idx", ["bayer"], save_key="dither",
              legacy={"ordered": {"dither": "bayer"}}),
    ], store="signal")]
    out = migrate_legacy(spec, {"signal": {"dither": "ordered"}, "bits": 3})
    assert out == {"signal": {"dither": "bayer"}, "bits": 3}


def test_migrate_legacy_survives_a_hand_edited_file():
    """DESIGN.md §7 invites hand-editing, so an unhashable or absent value
    must read as 'not a retired name', never raise."""
    from dtouch.panelspec import migrate_legacy

    spec = [Section("S", [
        Cycle("palette", "pal_idx", ["mono"], save_key="palette",
              legacy={"old-name": {"palette": "mono"}}),
    ], store="signal")]
    for cfg in ({}, {"signal": None}, {"signal": {"palette": ["a", "b"]}},
                {"signal": {"palette": {"x": 1}}}, {"signal": "nope"}):
        assert migrate_legacy(spec, cfg) is cfg
