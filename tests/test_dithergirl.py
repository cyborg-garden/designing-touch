"""Dither Girl — DESIGN.md §4.2, migration step 9.

The mode is pure numpy/cv2 (no GL), so everything here runs fully headless:
protocol conformance, panel-spec structure, palette mapping, matte gating,
swatch caching, preset capture/apply round-trips (incl. the nested "signal"
block and the suppressed dither row), bank seeding, still input, and the
shell-command wiring for menu + direct mode keys.
"""
import time

import cv2
import numpy as np
import pytest
from conftest import best_ms

import dtouch.presets as _presets
from dtouch import imgui
from dtouch.commands import CommandRegistry
from dtouch.modes import REGISTRY, mode_by_id
from dtouch.modes.dithergirl import (ACCENT, ALGOS, AUTHORED_INVERSE,
                                     LEGACY_PALETTES, MATTES_DG, PALETTES,
                                     DitherGirlMode, palette_pair)
from dtouch.modes.particles import MATTE_H, MATTE_W, ParticlesMode
from dtouch.panelspec import Cycle, Readout, Section, Slider, Toggle
from dtouch.shell import Host

RES = (192, 108)


class SyntheticSource:
    def __init__(self, w=64, h=36, on_read=None):
        self.w, self.h = w, h
        self.reads = 0
        self.on_read = on_read
        self.released = False
        self.name = "synthetic"

    def read(self):
        self.reads += 1
        if self.on_read:
            self.on_read(self.reads)
        rng = np.random.default_rng(self.reads)
        return True, rng.integers(0, 256, (self.h, self.w, 3), np.uint8)

    def release(self):
        self.released = True


def _paths(tmp_path):
    return dict(presets_path=str(tmp_path / "presets.json"),
                state_path=str(tmp_path / "state.json"))


def _host(tmp_path, mode=None, src=None, **kw):
    kw.setdefault("res", RES)
    kw.setdefault("show", False)
    kw.setdefault("preset", None)
    return Host(mode or DitherGirlMode(), source=src or SyntheticSource(),
                **_paths(tmp_path), **kw)


def _booted(tmp_path, **kw):
    """A Host that ran one headless frame — ui built, mode configured."""
    host = _host(tmp_path, max_frames=1, **kw)
    host.run()
    return host


def _white(h=36, w=64):
    return np.full((h, w, 3), 255, np.uint8)


def _black(h=36, w=64):
    return np.zeros((h, w, 3), np.uint8)


# ---------- mode protocol conformance (DESIGN.md §2.2) ----------

def test_mode_protocol_attrs():
    m = DitherGirlMode()
    assert m.id == "dithergirl" and m.title == "Dither"
    assert m.accepts_still is True
    assert m.accent == ACCENT
    assert m.safe_look() == "classic"
    assert "dither" in m.claims


def test_registered_in_registry():
    assert DitherGirlMode in REGISTRY
    assert mode_by_id("dithergirl") is DitherGirlMode


def test_start_stop_idempotent_and_headless(tmp_path):
    host = _booted(tmp_path)
    m = host.mode
    m.stop()
    m.stop()                                 # second stop must be a no-op
    m.start(host)
    out = m.step(_white(), None, 1 / 30)
    assert out.shape == (RES[1], RES[0], 3) and out.dtype == np.uint8
    m.stop()


def test_step_without_ui_uses_defaults(tmp_path):
    """The soak path: step before the shell builds the shared UI state."""
    host = _host(tmp_path)
    m = DitherGirlMode()
    m.start(host)
    out = m.step(_white(), None, 1 / 30)
    assert out.shape == (RES[1], RES[0], 3) and out.dtype == np.uint8
    m.stop()


def test_status_line_is_spec_derived_and_ascii(tmp_path):
    """DESIGN.md §2.3: the HUD status renders from the status-marked spec
    widgets — 'MODE TITLE  <values in spec order>  <mode tail>'."""
    host = _booted(tmp_path)
    s = host._status_line()
    assert s == s.encode("ascii", "replace").decode()
    assert s == "DITHER  floyd-steinberg  1-bit  bias auto  src synthetic"
    host.ui.dg_algo_idx = ALGOS.index("Blue noise")
    host.ui.dg_bits = 3.0
    host.ui.input_idx = 1                    # still input -> src tail follows
    assert (host._status_line()
            == "DITHER  blue noise  3-bit  bias auto  src still")


# ---------- panel spec structure (DESIGN.md §4.2) ----------

def test_panel_sections_and_widgets():
    from dtouch.panelspec import PresetList
    spec = DitherGirlMode().panel_spec()
    titles = [s.title for s in spec]
    assert titles == ["TEMPLATES", "SOURCE", "ALGORITHM", "TONE", "PALETTE"]
    # TEMPLATES at the top, like Particles (DESIGN.md §7 — user looks reachable)
    assert isinstance(spec[0].widgets[0], PresetList)
    by = {s.title: s.widgets for s in spec}
    # SOURCE: input cycle, matte cycle, matte-bg toggle, output res, Mirror
    src = by["SOURCE"]
    assert isinstance(src[0], Cycle) and src[0].options == ["camera", "still..."]
    assert isinstance(src[1], Cycle) and src[1].options == MATTES_DG
    assert any(isinstance(w, Toggle) and w.label == "Matte bg black" for w in src)
    assert any(isinstance(w, Toggle) and w.attr == "mirror" for w in src)
    # ALGORITHM: big label readout, algorithm cycle, badge, swatch strip
    alg = by["ALGORITHM"]
    assert isinstance(alg[0], Readout)
    cyc = next(w for w in alg if isinstance(w, Cycle))
    assert list(cyc.options) == ALGOS
    assert sum(isinstance(w, Readout) for w in alg) == 3
    # TONE: Bits 1-4 int-fmt, Gamma toggle, bias cycle, Contrast, Scale 30-720
    tone = by["TONE"]
    bits = next(w for w in tone if isinstance(w, Slider) and w.label == "Bits")
    assert (bits.lo, bits.hi, bits.fmt) == (1.0, 4.0, ".0f")
    assert any(isinstance(w, Toggle) and w.label == "Gamma" for w in tone)
    bias = next(w for w in tone if isinstance(w, Cycle))
    assert list(bias.options) == ["auto", "light", "dark"]
    scale = next(w for w in tone if isinstance(w, Slider) and w.label == "Scale")
    assert (scale.lo, scale.hi) == (30.0, 720.0)
    assert any(isinstance(w, Readout) for w in tone)     # perf-honesty note
    # PALETTE
    pal = by["PALETTE"][0]
    assert list(pal.options) == list(PALETTES)


def test_panel_strings_are_ascii():
    for sec in DitherGirlMode().panel_spec():
        assert sec.title == sec.title.encode("ascii", "replace").decode()
        for w in sec.widgets:
            for s in (getattr(w, "label", ""), getattr(w, "tip", "")):
                assert s == s.encode("ascii", "replace").decode()
            for o in getattr(w, "options", []):
                assert o == o.encode("ascii", "replace").decode()


def test_gamma_default_on(tmp_path):
    host = _booted(tmp_path)
    assert host.ui.dg_gamma is True


# ---------- SIGNAL rack suppression (DESIGN.md §2.4) ----------

def test_rack_hides_all_dither_quality_controls_for_dithergirl(tmp_path):
    """DESIGN.md §2.4: Dither Girl owns ALL dither quality controls — the
    rack loses its dither row AND Bits/Gamma/Bias, keeping the rest."""
    host = _booted(tmp_path)
    rack = next(s for s in host.ui.spec
                if isinstance(s, Section) and s.title == "SIGNAL")
    keys = [getattr(w, "store_key", None) for w in rack.widgets]
    for claimed in ("dither", "bits", "gamma", "bias"):
        assert claimed not in keys
    assert "chroma" in keys and "glitch" in keys        # rack minus claims only


def test_rack_keeps_dither_and_quality_rows_for_particles():
    rack = next(s for s in Host(ParticlesMode()).compose_spec(ParticlesMode())
                if isinstance(s, Section) and s.title == "SIGNAL")
    keys = [getattr(w, "store_key", None) for w in rack.widgets]
    for k in ("dither", "bits", "gamma", "bias"):
        assert k in keys


# ---------- duplicate-control rule (DESIGN.md §4.2: one Mirror) ----------

def test_global_mirror_omitted_when_the_mode_declares_its_own(tmp_path):
    """Dither Girl's SOURCE declares Mirror — the composed spec must carry
    exactly one mirror control (the mode's), never two faces on one attr."""
    host = _booted(tmp_path)
    mirrors = [w for w in host.ui.iter_widgets()
               if getattr(w, "attr", None) == "mirror"]
    assert len(mirrors) == 1
    in_source = next(s for s in host.ui.spec
                     if isinstance(s, Section) and s.title == "SOURCE")
    assert mirrors[0] in in_source.widgets


def test_global_mirror_kept_for_particles():
    spec = Host(ParticlesMode()).compose_spec(ParticlesMode())
    tail = [w for item in spec if not isinstance(item, Section)
            for w in [item]]
    assert any(getattr(w, "attr", None) == "mirror" for w in tail)
    # and the Menu/Quit Actions survive the dedup untouched
    from dtouch.panelspec import Action
    cmds = [w.command for w in tail if isinstance(w, Action)]
    assert cmds == ["menu.open", "quit"]


# ---------- palette mapping (DESIGN.md §4.2) ----------

def test_palette_map_endpoints():
    lv = np.array([[0.0, 1.0]], np.float32)
    out = DitherGirlMode._palette_map(lv, (0, 0, 0), (255, 255, 255))
    assert out[0, 0].tolist() == [0, 0, 0] and out[0, 1].tolist() == [255, 255, 255]


@pytest.mark.parametrize("palette,invert,white_out,black_out", [
    ("mono", False, (255, 255, 255), (0, 0, 0)),
    ("mono", True, (16, 16, 16), (245, 245, 245)),
    ("amber", False, (255, 176, 0), (24, 12, 0)),
    ("amber", True, (24, 12, 0), (255, 176, 0)),
    ("green phosphor", False, (80, 255, 120), (0, 20, 8)),
])
def test_step_maps_on_off_levels_to_the_palette(tmp_path, palette, invert,
                                                white_out, black_out):
    host = _booted(tmp_path)
    ui, m = host.ui, host.mode
    ui.dg_palette_idx = list(PALETTES).index(palette)
    ui.dg_invert = invert
    out = m.step(_white(), None, 1 / 30)
    assert (out.reshape(-1, 3) == np.array(white_out, np.uint8)).all()
    out = m.step(_black(), None, 1 / 30)
    assert (out.reshape(-1, 3) == np.array(black_out, np.uint8)).all()


def test_bits_slider_is_int_snapped(tmp_path):
    host = _booted(tmp_path)
    host.ui.dg_bits = 2.6
    assert host.mode._bits() == 3
    host.ui.dg_bits = 1.4
    assert host.mode._bits() == 1


def test_output_level_count_respects_bits(tmp_path):
    host = _booted(tmp_path)
    ui, m = host.ui, host.mode
    ui.dg_algo_idx = ALGOS.index("Bayer")
    ui.dg_bits = 2.0
    frame = np.tile(np.linspace(0, 255, 64, dtype=np.uint8)[None, :, None],
                    (36, 1, 3))
    out = m.step(frame, None, 1 / 30)
    assert len(np.unique(out[:, :, 0])) <= 4             # 2 bits = 4 levels


# ---------- matte gating (DESIGN.md §4.2) ----------

class FakeMatte:
    """Right half is the subject; left half is background."""

    def compute(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        m = np.zeros((h, w), np.float32)
        m[:, w // 2:] = 1.0
        return m


def _matted_host(tmp_path, black_bg):
    host = _booted(tmp_path)
    ui, m = host.ui, host.mode
    ui.dg_matte_idx = MATTES_DG.index("luma")
    ui.dg_matte_black = black_bg
    m.mat, m._mat_kind = FakeMatte(), "luma"             # inject: no model deps
    return host


def test_matte_gate_dithers_subject_only_black_bg(tmp_path):
    host = _matted_host(tmp_path, black_bg=True)
    out = host.mode.step(_white(), None, 1 / 30)
    w = out.shape[1]
    assert out[:, : w // 4].max() == 0                   # background black
    assert out[:, 3 * w // 4:].min() == 255              # subject dithered white


def test_matte_gate_passes_camera_through_when_bg_not_black(tmp_path):
    host = _matted_host(tmp_path, black_bg=False)
    frame = _white()
    frame[:, : frame.shape[1] // 2] = (0, 0, 255)        # left = pure red (BGR)
    out = host.mode.step(frame, None, 1 / 30)
    w = out.shape[1]
    left = out[:, : w // 4].reshape(-1, 3)
    assert (left == np.array([255, 0, 0], np.uint8)).all(axis=1).all(), \
        "background must be the raw camera picture (RGB), not dithered"
    assert out[:, 3 * w // 4:].min() == 255              # subject dithered white


def test_matte_off_dithers_everything(tmp_path):
    host = _booted(tmp_path)
    frame = _white()
    frame[:, : frame.shape[1] // 2] = (0, 0, 255)
    out = host.mode.step(frame, None, 1 / 30)
    lum = out.reshape(-1, 3)
    assert set(map(tuple, np.unique(lum, axis=0).tolist())) <= \
        {(0, 0, 0), (255, 255, 255)}                     # everything two-tone


# ---------- swatch strip cache (DESIGN.md §4.2) ----------

def _gui():
    g = imgui.Gui()
    g.begin(1.0, (-1, -1))
    return g


def test_swatch_cached_until_a_tone_value_changes(tmp_path):
    host = _booted(tmp_path)
    m, g = host.mode, _gui()
    frame = np.zeros((400, 400, 3), np.uint8)
    m._draw_swatch(frame, g, 10, 10, 200)
    first = m._swatch_cache
    m._draw_swatch(frame, g, 10, 10, 200)
    assert m._swatch_cache is first                      # cache hit — no re-render
    host.ui.dg_bits = 3.0
    m._draw_swatch(frame, g, 10, 10, 200)
    assert m._swatch_cache is not first                  # TONE change re-renders


@pytest.mark.parametrize("attr,value", [
    ("dg_algo_idx", 0), ("dg_gamma", False),
    ("dg_bias_idx", 2), ("dg_palette_idx", 2),
])
def test_swatch_invalidates_on_each_relevant_control(tmp_path, attr, value):
    host = _booted(tmp_path)
    m, g = host.mode, _gui()
    frame = np.zeros((400, 400, 3), np.uint8)
    m._draw_swatch(frame, g, 10, 10, 200)
    first = m._swatch_cache
    setattr(host.ui, attr, value)
    m._draw_swatch(frame, g, 10, 10, 200)
    assert m._swatch_cache is not first


def test_swatch_blit_is_clipped_when_scrolled_off_frame(tmp_path):
    host = _booted(tmp_path)
    m, g = host.mode, _gui()
    hpx = g.S(16)
    frame = np.zeros((100, 400, 3), np.uint8)
    m._draw_swatch(frame, g, 10, -8, 200)                # partially above the top
    cache = m._swatch_cache
    # the visible slice landed (rows 0..hpx-8 = cache rows 8..hpx)...
    assert np.array_equal(frame[0:hpx - 8, 10:210], cache[8:hpx])
    # ...and nothing leaked outside the swatch's rows/columns
    assert not frame[hpx - 8:].any()
    assert not frame[:, :10].any() and not frame[:, 210:].any()

    frame2 = np.zeros((100, 400, 3), np.uint8)
    m._draw_swatch(frame2, g, 10, 95, 200)               # partially below the bottom
    assert np.array_equal(frame2[95:100, 10:210], cache[0:5])
    assert not frame2[:95].any()


# ---------- perf honesty (DESIGN.md §4.2) ----------

def test_slow_warning_only_for_error_diffusion_at_high_scale(tmp_path):
    host = _booted(tmp_path)
    ui, m = host.ui, host.mode
    assert m.slow_warning() is False                     # classic: FS at 72
    ui.dg_scale = 400.0
    assert m.slow_warning() is True                      # FS dragged high
    ui.dg_algo_idx = ALGOS.index("Bayer")
    assert m.slow_warning() is False                     # ordered: full res is fine
    g = _gui()
    frame = np.zeros((400, 400, 3), np.uint8)
    assert m._draw_perf_note(frame, g, 10, 10, 200) == 10   # renders nothing
    ui.dg_algo_idx = ALGOS.index("Riemersma")
    assert m._draw_perf_note(frame, g, 10, 10, 200) > 10    # renders the note


def test_the_slow_threshold_is_inclusive(tmp_path):
    """A threshold named SLOW_SCALE means "this value IS slow" — `>` left the
    boundary itself silently fast, which is where `newsprint` used to sit."""
    from dtouch.modes.dithergirl import SLOW_SCALE

    host = _booted(tmp_path)
    ui, m = host.ui, host.mode
    ui.dg_algo_idx = ALGOS.index("Floyd-Steinberg")
    ui.dg_scale = SLOW_SCALE - 1.0
    assert m.slow_warning() is False
    ui.dg_scale = SLOW_SCALE
    assert m.slow_warning() is True


def test_no_shipped_look_balances_on_the_slow_threshold(tmp_path):
    """`newsprint` shipped at exactly 180.0 against a `>` test, so the one
    built-in the perf line was closest to could never trigger it and the
    coincidence was invisible. Every error-diffusion built-in must now be
    clearly on one side or the other — which side is a design choice per
    look, but balancing on the line is not a choice, it is an accident."""
    from dtouch.modes.dithergirl import ORDERED, SLOW_SCALE

    for name, look in DitherGirlMode.BUILTIN.items():
        if look["algorithm"] in ORDERED:
            continue                     # ordered dithers are fine at full res
        assert abs(look["scale"] - SLOW_SCALE) > SLOW_SCALE * 0.05, (
            "%s sits within 5%% of the perf line - make it deliberate" % name)


def test_no_shipped_look_lands_outside_its_own_control(tmp_path):
    """`ascii stream` shipped with scale=30 under a 45-720 Scale slider. The
    handle drew 3 px past the end of its own track, and the first click
    anywhere on that track snapped 30 -> 45 — taking the look from 12x24 cells
    to 8x16 with no slider position that gets it back. A built-in is part of
    the known-good set: every value in one must be reachable by the control
    that owns it."""
    from dtouch.panelspec import Slider, walk_spec

    sliders = {w.store_key: w for _s, w in walk_spec(DitherGirlMode().panel_spec())
               if isinstance(w, Slider)}
    assert {"bits", "contrast", "scale", "hue", "tint"} <= set(sliders)
    for name, look in DitherGirlMode.BUILTIN.items():
        for key, val in look.items():
            w = sliders.get(key)
            if w is None:
                continue
            assert w.lo <= val <= w.hi, (
                "built-in %r sets %s=%s, outside the slider's %s-%s"
                % (name, key, val, w.lo, w.hi))


def test_which_side_of_the_line_each_shipped_look_is_on(tmp_path):
    """The live-usable built-ins are below it; `riemersma still` is named for
    stills and is deliberately above, so it SHOULD carry the amber note."""
    host = _booted(tmp_path)
    ui, m = host.ui, host.mode
    warned = {}
    for name, look in DitherGirlMode.BUILTIN.items():
        ui.dg_algo_idx = ALGOS.index(look["algorithm"])
        ui.dg_scale = look["scale"]
        warned[name] = m.slow_warning()
    assert warned["newsprint"] is False       # the boundary case, now decided
    assert warned["classic"] is False
    assert warned["riemersma still"] is True


def test_perf_note_wraps_inside_the_panel_column_at_720p(tmp_path):
    """At 720p the one-line note overflowed the sidebar — it must word-wrap
    to the column width (§4.2: the note is part of the panel, not graffiti
    over the picture)."""
    host = _booted(tmp_path)
    ui, m = host.ui, host.mode
    ui.dg_scale = 400.0                                  # FS high: note on
    g = _gui()                                           # s = 1.0 (720p floor)
    frame = np.zeros((720, 1280, 3), np.uint8)
    x, cw = 1030, 258                                    # 720p panel column
    y2 = m._draw_perf_note(frame, g, x, 100, cw)
    assert y2 > 100
    ys, xs = np.where(frame.any(axis=2))
    assert xs.max() <= x + cw, "the note must stay inside the panel column"
    assert xs.min() >= x
    assert len(np.unique(ys // 16)) >= 2 or y2 - 100 > 20   # actually wrapped


# ---------- contrast-reorder canary (perf rewrite, DESIGN.md §4.2) ----------

def _contrast_orders(gray_u8, ww, wh, contrast):
    """(pre-rewrite, shipped) tone planes for the same frame.

    pre-rewrite: float at FULL res -> contrast+clip -> INTER_AREA down.
    shipped:     INTER_AREA down (uint8) -> float -> contrast+clip.
    Mirrors DitherGirlMode.step's ordering."""
    full = gray_u8.astype(np.float32) / 255.0
    old = cv2.resize(np.clip((full - 0.5) * contrast + 0.5, 0.0, 1.0),
                     (ww, wh), interpolation=cv2.INTER_AREA)
    small = (cv2.resize(gray_u8, (ww, wh), interpolation=cv2.INTER_AREA)
             .astype(np.float32) / 255.0)
    new = np.clip((small - 0.5) * contrast + 0.5, 0.0, 1.0)
    return old, new


def _detailed_frame(h=720, w=1280, seed=7):
    """High-frequency texture — the worst case for an area-average reorder."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    g = 0.5 + 0.5 * np.sin(xx / 3.0) * np.cos(yy / 5.0)
    return ((0.6 * g + 0.4 * rng.random((h, w), dtype=np.float32)) * 255
            ).astype(np.uint8)


def test_contrast_reorder_canary_bounds_the_accepted_difference():
    """Moving contrast after the resize is NOT identity — clip does not commute
    with INTER_AREA. The reorder is kept for the ~90x cost reduction, so this
    pins how far it may drift: measured ~1.95/255 mean-abs at Contrast 1.6 and
    the default working height (72), bounded here at 4/255. A regression that
    pushes it past that is a visible tone change, not a rounding artifact."""
    gray = _detailed_frame()
    wh = 72                                              # SCALE_DEFAULT
    ww = max(8, int(round(wh * 1280 / 720.0)))
    old, new = _contrast_orders(gray, ww, wh, 1.6)
    diff = np.abs(old - new).mean() * 255.0
    assert diff < 4.0, f"contrast reorder drifted to {diff:.2f}/255"
    # …and it is genuinely non-zero: the comment must not claim identity again
    assert diff > 0.5


def test_contrast_reorder_is_identity_only_when_nothing_is_resized():
    """The planes agree exactly when the working res IS the frame res — proof
    the divergence comes from the resize, not from the contrast math."""
    gray = _detailed_frame(h=72, w=128)
    old, new = _contrast_orders(gray, 128, 72, 3.0)
    assert np.allclose(old, new, atol=1e-6)


# ---------- audio modulation (minimal — DESIGN.md §4.2) ----------

def test_bass_modulates_contrast_only_when_levels_present(tmp_path):
    host = _booted(tmp_path)
    m = host.mode
    frame = np.full((36, 64, 3), 140, np.uint8)
    quiet = m.step(frame, None, 1 / 30)
    loud = m.step(frame, {"bass": 1.0, "treble": 0.0}, 1 / 30)
    assert not np.array_equal(quiet, loud)
    again = m.step(frame, None, 1 / 30)
    assert np.array_equal(quiet, again)                  # no sticky state


# ---------- presets: capture/apply round-trip + bank (DESIGN.md §7) ----------

def test_boot_applies_the_safe_look_classic(tmp_path):
    host = _booted(tmp_path)
    assert host.ui.preset_name == "classic"
    assert ALGOS[host.ui.dg_algo_idx] == "Floyd-Steinberg"


def test_capture_includes_tone_and_nested_signal_minus_dither(tmp_path):
    host = _booted(tmp_path)
    ui = host.ui
    ui.dg_algo_idx = ALGOS.index("Bayer")
    ui.dg_bits = 3.0
    ui.glitch = True
    ui.chroma = 25.0
    cfg = host._capture_cfg()
    assert cfg["algorithm"] == "Bayer" and cfg["bits"] == 3.0
    assert cfg["palette"] == "mono" and cfg["invert"] is False
    assert cfg["signal"]["chroma"] == 25.0 and cfg["signal"]["glitch"] is True
    assert "dither" not in cfg["signal"], "the suppressed row must not serialize"
    assert "input_idx" not in cfg and "res" not in cfg   # save=False widgets


def test_preset_round_trip_restores_state(tmp_path):
    host = _booted(tmp_path)
    ui = host.ui
    ui.dg_algo_idx = ALGOS.index("Riemersma")
    ui.dg_bits = 4.0
    ui.dg_contrast = 2.0
    ui.dg_palette_idx = list(PALETTES).index("amber")
    ui.glitch = True
    ui.drift = 33.0
    cfg = host._capture_cfg()
    _presets.save("mylook", cfg, path=host.presets_path, mode="dithergirl")
    host._reload_presets()
    # scramble, then recall through the same mailbox a panel click uses
    ui.dg_algo_idx, ui.dg_bits, ui.dg_contrast = 0, 1.0, 0.5
    ui.dg_palette_idx, ui.glitch, ui.drift = 0, False, 8.0
    ui.pending_preset = "mylook"
    host._apply_pending_preset()
    assert ALGOS[ui.dg_algo_idx] == "Riemersma"
    assert (ui.dg_bits, ui.dg_contrast) == (4.0, 2.0)
    assert list(PALETTES)[ui.dg_palette_idx] == "amber"
    assert ui.glitch is True and ui.drift == 33.0        # signal nesting applied


def test_user_look_save_recall_bank_round_trip_via_the_panel_path(tmp_path):
    """DESIGN.md §7 through the TEMPLATES panel path: '+ Save current look'
    click -> auto-name + rename box; recall via a preset-row click; slot
    badge click banks it; everything persists per mode."""
    host = _booted(tmp_path)
    ui = host.ui
    ui.dg_algo_idx = ALGOS.index("Riemersma")
    ui.dg_bits = 4.0
    ui._activate("save", None, 0)                        # panel Save row click
    assert ui.pending_save is True
    host._pump_preset_mailboxes()
    name = next(iter(ui.user_presets))
    assert ui.renaming == name                           # naming is one flow
    ui.renaming = None
    ui._activate("slot", name, 0)                        # slot badge click
    host._pump_preset_mailboxes()
    slot = next(s for s, n in ui.bank.items() if n == name)
    assert _presets.bank(host.presets_path, mode="dithergirl")[slot] == name
    # scramble, then recall through the panel's preset-row click path
    ui.dg_algo_idx, ui.dg_bits = 0, 1.0
    ui._activate("preset", ui.presets.index(name), 0)
    host._pump_preset_mailboxes()
    assert ALGOS[ui.dg_algo_idx] == "Riemersma" and ui.dg_bits == 4.0
    # the look survives a fresh boot (per-mode persistence)
    host2 = _booted(tmp_path)
    assert name in host2.ui.presets
    assert host2.ui.bank[slot] == name


def test_builtins_seed_bank_slots(tmp_path):
    host = _booted(tmp_path)
    builtins = list(DitherGirlMode.BUILTIN)
    assert host.ui.bank == {str(i + 1): n for i, n in enumerate(builtins[:9])}


def test_bank_recall_applies_a_builtin(tmp_path):
    host = _booted(tmp_path)
    ui = host.ui
    slot = next(s for s, n in ui.bank.items() if n == "stream-safe")
    ui.pending_preset = ui.bank[slot]
    host._apply_pending_preset()
    assert ALGOS[ui.dg_algo_idx] == "Blue noise"
    assert ui.dg_contrast == 1.6 and ui.dg_scale == 56.0  # stream-tuned look


# ---------- still input (v1: CLI --still + mailbox) ----------

def test_still_flag_boots_into_still_input_and_feeds_the_still(tmp_path):
    import cv2
    path = str(tmp_path / "still.png")
    still = np.zeros((30, 40, 3), np.uint8)
    still[:, :, 2] = 200                                 # red-ish, distinctive
    cv2.imwrite(path, still)
    seen = []

    class Spy(DitherGirlMode):
        def step(self, frame_bgr, audio, dt):
            seen.append(frame_bgr.copy())
            return super().step(frame_bgr, audio, dt)

    host = _host(tmp_path, mode=Spy(still=True), still=path, max_frames=3)
    host.run()
    assert host.ui.input_idx == 1
    assert len(seen) == 3
    for f in seen:                                       # same frame every tick
        assert np.array_equal(f, seen[0])
    assert np.array_equal(seen[0], cv2.flip(still, 1))   # shell-mirrored still


def test_still_boot_flag_applies_on_first_entry_only():
    """--still boots into still input, but switching away and back must
    preserve the operator's input choice (amended DESIGN.md §6.2 re-entry)."""
    from types import SimpleNamespace
    ui = SimpleNamespace()
    m = DitherGirlMode(still=True)
    m.configure_ui(ui)
    assert ui.input_idx == 1                 # first entry: still input
    ui.input_idx = 0                         # operator switches to camera
    m.configure_ui(ui)                       # mode away and back (re-entry)
    assert ui.input_idx == 0                 # the choice sticks


def test_still_cycle_without_a_still_snaps_back_with_hint(tmp_path):
    host = _host(tmp_path, mode=DitherGirlMode(still=True), max_frames=2)
    host.run()
    assert host.ui.input_idx == 0                        # snapped back to camera
    assert host.still is None
    # ...and the snap-back is TOLD, not silent (DESIGN.md principle 4)
    hints = [t.text for t in host.hud.toasts._hints]
    assert any("no still loaded" in t for t in hints)


def test_pending_still_path_mailbox_loads_the_image(tmp_path):
    import cv2
    path = str(tmp_path / "later.png")
    cv2.imwrite(path, np.full((20, 20, 3), 77, np.uint8))
    host = _booted(tmp_path)
    host.ui.pending_still_path = path
    host._pump_preset_mailboxes()
    assert host.still is not None and host.still[0, 0, 0] == 77
    assert host.ui.pending_still_path is None


# ---------- panel rendering (Readouts through the real walker) ----------

def test_panel_draws_headless_with_readouts(tmp_path):
    """The generic walker renders the whole Dither Girl panel — big algorithm
    label, speed badge, swatch strip, perf note — without a window."""
    host = _booted(tmp_path, res=(1280, 720))
    ui = host.ui
    frame = np.zeros((720, 1280, 3), np.uint8)
    ui.draw(frame, {"status": ""})
    assert frame.any()
    assert host.mode._swatch_cache is not None           # swatch actually drew
    ui.dg_scale = 400.0                                  # FS high: amber note on
    before = frame.copy()
    ui.draw(frame, {"status": ""})
    assert not np.array_equal(before, frame)


# ---------- shell commands: menu + direct mode keys (step 8 wiring) ----------

def test_menu_and_mode_switch_commands(tmp_path):
    host = _booted(tmp_path)
    host.reg = CommandRegistry()
    host._register_shell_commands()
    keys = {k for k, _ in host.reg.table()}
    assert keys >= {"m", "p", "d"}
    host.reg.dispatch(ord("m"))
    assert host.menu.open is True
    assert host.menu.sel == 1                            # active mode's card
    host.reg.dispatch(ord("m"))
    assert host.menu.open is False
    host.reg.dispatch(ord("p"))
    assert host.pending_mode == "particles"
    host.pending_mode = None
    host.reg.dispatch(ord("d"))                          # already in dithergirl
    assert host.pending_mode is None
    assert host.hud.toasts.active()


def test_boot_card_drawn_and_recorded_on_switch(tmp_path):
    """DESIGN.md §3: the recorder captures the static boot card, not a gray
    flash. Headless switch dithergirl -> dithergirl is refused, so drive
    _switch_mode straight at a fake writer."""

    class FakeWriter:
        def __init__(self):
            self.frames = []

        def append_data(self, f):
            self.frames.append(np.asarray(f))

        def close(self):
            pass

    host = _booted(tmp_path)
    host.mode.start(host)                                # run()'s finally stopped it
    host.writer = FakeWriter()
    assert host._switch_mode("dithergirl") is True       # same-id via _switch is fine
    assert len(host.writer.frames) == 1
    card = host.writer.frames[0]
    assert card.shape == (RES[1], RES[0], 3)
    # glyph + name render in the accent (anti-aliased at tiny res, so look
    # for magenta-leaning pixels rather than exact values); ground is black
    r, g_, b = card[:, :, 0].astype(int), card[:, :, 1].astype(int), card[:, :, 2].astype(int)
    assert ((r > 160) & (b > 160) & (g_ < r - 30)).any(), \
        "the boot card must carry the mode accent"
    assert (card == 0).all(axis=2).mean() > 0.5          # mostly black card
    assert host.hud.toasts.active()
    host.writer = None


# ---------- palettes: the shipped set, measured (DESIGN.md §4.2 / §5) ----------

def _rel_luma(rgb):
    from dtouch.dither import srgb_to_linear
    return float((srgb_to_linear(np.float32(rgb) / 255.0)
                  * np.float32([0.2126, 0.7152, 0.0722])).sum())


def _contrast(a, b):
    la, lb = _rel_luma(a), _rel_luma(b)
    return (max(la, lb) + 0.05) / (min(la, lb) + 0.05)


def test_every_palette_reads_at_projector_distance():
    """Colour is state, never decoration (§5) — and a duotone whose two ends
    do not separate is not a look, it is an unreadable picture. 4.5:1 is the
    same floor the mode accent was picked against."""
    for name, (off, on) in PALETTES.items():
        assert _contrast(off, on) >= 4.5, f"{name} is too flat to project"


def test_palettes_are_ascii_named_and_well_formed():
    for name, pair in PALETTES.items():
        assert name == name.encode("ascii", "replace").decode()
        assert len(pair) == 2
        for c in pair:
            assert len(c) == 3 and all(0 <= v <= 255 for v in c)


def test_every_palette_reads_at_projector_distance_inverted_too():
    """Invert must not be able to ship an unreadable pair. For nine palettes
    that is arithmetic (a swap is the same two colours), but `mono`'s inverse
    is authored, so it is a separate measurement, not a corollary."""
    for name in PALETTES:
        off, on = palette_pair(name, True)
        assert _contrast(off, on) >= 4.5, f"{name} inverted is too flat"


# ---------- Invert, and the migration off the retired pair (§4.2) ----------

def test_invert_swaps_the_pair_for_every_palette_but_mono():
    for name, (off, on) in PALETTES.items():
        if name in AUTHORED_INVERSE:
            continue
        assert palette_pair(name, True) == (on, off)
        assert palette_pair(name, False) == (off, on)


def test_mono_inverted_is_the_authored_pair_not_the_naive_swap():
    """The two monochrome palettes this replaced were never each other's
    mirror: white-on-black was pure 0/255, black-on-white deliberately
    softened to 245/16. A naive swap would silently re-tone every look that
    named `black-on-white` — the `newsprint` built-in included."""
    assert palette_pair("mono", True) == ((245, 245, 245), (16, 16, 16))
    assert palette_pair("mono", True) != PALETTES["mono"][::-1]


def test_newsprint_resolves_to_the_authored_dark_ink_pair():
    """`newsprint` is the one built-in whose stored representation the palette
    consolidation changed (`black-on-white` -> `mono` + `invert: True`), and
    nothing else pins that `invert` key: drop it from the BUILTIN dict and the
    suite stays green while the shipped dark-ink-on-paper look silently
    re-tones to light-on-dark. The authority is the RESOLVED pair — the exact
    colours the look shipped with under the retired name."""
    cfg = DitherGirlMode.BUILTIN["newsprint"]
    pair = palette_pair(cfg["palette"], cfg.get("invert", False))
    assert pair == ((245, 245, 245), (16, 16, 16))


def test_the_retired_palette_names_are_still_accepted():
    """DESIGN.md §9: presets are the one user-data-loss surface, and a Cycle
    silently IGNORES a stored value it does not recognise — so without this a
    look naming a retired palette would have loaded with whatever palette
    happened to be live, quietly, forever."""
    assert LEGACY_PALETTES["white-on-black"] == {"palette": "mono",
                                                 "invert": False}
    assert LEGACY_PALETTES["black-on-white"] == {"palette": "mono",
                                                "invert": True}
    pal = next(w for w in DitherGirlMode().panel_spec()[4].widgets
               if getattr(w, "label", None) == "palette")
    assert pal.legacy is LEGACY_PALETTES


@pytest.mark.parametrize("legacy,pair", [
    ("white-on-black", ((0, 0, 0), (255, 255, 255))),
    ("black-on-white", ((245, 245, 245), (16, 16, 16))),
])
def test_a_look_naming_a_retired_palette_renders_identically(tmp_path, legacy,
                                                             pair):
    """The acceptance test for the whole change: a stored look that says
    `white-on-black` / `black-on-white` must reach the exact pair it used to,
    from any live state — including one where Invert is currently the WRONG
    way round, which is the case a `apply="keep"` Invert would have failed."""
    host = _booted(tmp_path)
    ui, m = host.ui, host.mode
    for live_invert in (False, True):
        ui.dg_invert = live_invert
        ui.dg_palette_idx = list(PALETTES).index("amber")
        assert host._apply_look("legacy", {"palette": legacy})
        assert m._palette() == pair
        assert m._palette_name() == "mono"


# The three looks below are the SHAPE of what a real presets.json written by
# the pre-change build contains: a retired palette name, and Bits / Scale /
# Crush carrying the fractional values a mouse drag produces (the track is
# ~118 px wide, so every dragged value is a 118th of the range). Their
# "engine" numbers were read off the PRE-change code and pasted here, so this
# test fails if the collapse or the quantisation ever moves a stored look's
# picture — which is the whole promise of both halves of this change.
_PRE_CHANGE_LOOKS = [
    (dict(algorithm="Bayer", bits=1.076271186440678, gamma=True, bias="auto",
          contrast=0.7627118644067796, scale=508.3474576271186,
          palette="green phosphor", matte="off",
          signal=dict(crush=2.9152542372881354)),
     dict(bits=1, height=508, crush=2, pair=((0, 20, 8), (80, 255, 120)))),
    (dict(algorithm="Floyd-Steinberg", bits=1.8389830508474576, gamma=True,
          bias="auto", contrast=1.0, scale=268.0932203389831,
          palette="white-on-black", matte="off",
          signal=dict(crush=1.7627118644067796)),
     dict(bits=2, height=268, crush=1, pair=((0, 0, 0), (255, 255, 255)))),
    (dict(algorithm="Blue noise", bits=1.0, gamma=True, bias="auto",
          contrast=1.6, scale=56.0, palette="white-on-black", hue=0.0,
          tint=0.0, matte="off", signal=dict(crush=2.1016949152542375)),
     dict(bits=1, height=56, crush=2, pair=((0, 0, 0), (255, 255, 255)))),
]


@pytest.mark.parametrize("cfg,engine", _PRE_CHANGE_LOOKS)
def test_a_pre_change_look_reaches_the_same_engine_values(tmp_path, cfg,
                                                          engine):
    """Migration is lossless where it counts: not "the file still loads" but
    "every number the RENDER consumes is the one it consumed before".

    The stored fractions are deliberately kept in the fixture rather than
    pre-rounded, because that is what is actually on disk: the panel used to
    print `1.84` and `2.92` while the engine had already made them 2 and 1.
    Quantising moved the READOUT onto the engine's number, and this pins that
    it did not move the engine.
    """
    host = _booted(tmp_path)
    ui, m = host.ui, host.mode
    assert host._apply_look("stored", cfg)
    assert m._bits() == engine["bits"]
    assert int(np.clip(round(float(ui.dg_scale)), 30, 720)) == engine["height"]
    assert int(ui.crush) == engine["crush"]
    assert m._palette() == engine["pair"]
    # Bits and Crush land on the engine's own number — their engines cannot
    # use a fraction, so the readout and the engine become one number
    for attr in ("dg_bits", "crush"):
        assert float(getattr(ui, attr)).is_integer(), attr
    # ...but Scale passes through EXACTLY (engine_snaps=False): the pixel
    # dithers round it themselves, and under ASCII the engine divides by it
    # before any rounding — the stored fraction IS the picture, so snapping
    # it here would be the one thing this test exists to forbid
    assert float(ui.dg_scale) == cfg["scale"]


def test_a_look_that_never_heard_of_invert_lands_un_inverted(tmp_path):
    """Invert is apply="reset", not a Toggle's default "keep": it is part of
    the picture, so recalling a look saved before Invert existed must not
    inherit whatever the live toggle happened to be."""
    host = _booted(tmp_path)
    ui, m = host.ui, host.mode
    ui.dg_invert = True
    assert host._apply_look("old", {"palette": "green phosphor",
                                    "algorithm": "Bayer"})
    assert ui.dg_invert is False
    assert m._palette() == PALETTES["green phosphor"]


def test_invert_round_trips_through_capture_and_apply(tmp_path):
    host = _booted(tmp_path)
    ui = host.ui
    ui.dg_invert, ui.dg_palette_idx = True, list(PALETTES).index("ice")
    cfg = host._capture_cfg()
    assert cfg["invert"] is True and cfg["palette"] == "ice"
    ui.dg_invert = False
    assert host._apply_look("x", cfg)
    assert ui.dg_invert is True


def test_invert_composes_with_tint(tmp_path):
    """Tint steers the pair the operator is looking at: inverting then tinting
    is a plain swap for nine palettes, and for mono it must tint the AUTHORED
    inverse rather than the naive one."""
    from dtouch.modes.dithergirl import tinted_palette

    a = tinted_palette("amber", 120.0, 0.6, True)
    b = tinted_palette("amber", 120.0, 0.6, False)
    assert a == b[::-1]
    mono_inv = tinted_palette("mono", 200.0, 0.5, True)
    assert mono_inv != tinted_palette("mono", 200.0, 0.5, False)[::-1]
    assert tinted_palette("mono", 0.0, 0.0, True) == AUTHORED_INVERSE["mono"]


def test_invert_composes_with_the_matte_gate(tmp_path):
    """Invert paints the dithered SUBJECT; the ungated background is still the
    raw camera (or black), which is the matte's own contract."""
    host = _booted(tmp_path)
    ui, m = host.ui, host.mode
    ui.dg_matte_idx = MATTES_DG.index("luma")
    ui.dg_matte_black = True
    ui.dg_palette_idx, ui.dg_invert = list(PALETTES).index("mono"), True
    out = m.step(_white(), None, 1 / 30)
    assert out.shape == (host.res[1], host.res[0], 3)
    assert out.max() > 0


def test_invert_changes_the_swatch_strip(tmp_path):
    """The strip previews the live pair (§4.2), so it must be keyed on Invert
    — a cache that ignored it would show yesterday's ramp."""
    host = _booted(tmp_path)
    ui, m = host.ui, host.mode
    frame = np.zeros((host.res[1], host.res[0], 3), np.uint8)
    ui.open = True
    ui.draw(frame.copy(), {"status": ""})
    before = m._swatch_cache.copy()
    ui.dg_invert = True
    ui.draw(frame.copy(), {"status": ""})
    assert not np.array_equal(before, m._swatch_cache)


def test_invert_flips_ink_and_ground_under_ascii(tmp_path):
    """ASCII needed no code for Invert — it measures the pair it is handed —
    but that is a claim, so it is pinned: the same frame under mono and mono
    inverted must come back with its ink and its ground exchanged."""
    host = _booted(tmp_path)
    ui, m = host.ui, host.mode
    ui.dg_algo_idx = ALGOS.index("ASCII")
    ui.dg_palette_idx = list(PALETTES).index("mono")
    # A white frame picks the DENSEST glyph either way — what changes is which
    # colour that ink is. Un-inverted it is white ink on a black ground, so the
    # frame reads dark; inverted it is black ink on paper, so it reads light.
    ui.dg_invert = False
    ink_on_black = m.step(_white(), None, 1 / 30).mean()
    ui.dg_invert = True
    ink_on_paper = m.step(_white(), None, 1 / 30).mean()
    assert ink_on_paper > ink_on_black, "invert must exchange ink and ground"
    # and the paper is the authored 245, not a naive 255: a black frame picks
    # the sparsest glyph, i.e. bare ground
    out = m.step(_black(), None, 1 / 30)
    assert out.max() == 245


def test_the_first_palettes_are_unchanged():
    """Looks saved before the palette set grew name their palette by VALUE;
    renaming or re-toning one of the originals silently re-colours every look
    that used it. `mono` and `mono` inverted carry the exact pairs the retired
    `white-on-black` / `black-on-white` shipped with."""
    assert list(PALETTES)[:3] == ["mono", "amber", "green phosphor"]
    assert PALETTES["mono"] == ((0, 0, 0), (255, 255, 255))
    assert AUTHORED_INVERSE["mono"] == ((245, 245, 245), (16, 16, 16))
    assert PALETTES["amber"] == ((24, 12, 0), (255, 176, 0))
    assert PALETTES["green phosphor"] == ((0, 20, 8), (80, 255, 120))


# ---------- Hue / Tint (the customisability) ----------

def test_tint_zero_is_an_exact_no_op():
    """Tint COMPOSES with the named palettes rather than replacing them: at 0
    every palette must be bit-identical to its shipped pair, at every hue —
    inverted included, because mono's inverse is an AUTHORED pair
    (245/16, not a swap) and a sweep over PALETTES alone never touches it."""
    from dtouch.modes.dithergirl import tinted_palette

    for name, pair in PALETTES.items():
        for hue in (0.0, 137.0, 359.0):
            assert tinted_palette(name, hue, 0.0) == pair
            assert (tinted_palette(name, hue, 0.0, invert=True)
                    == palette_pair(name, True))


def test_tint_cannot_push_any_palette_below_the_contrast_floor():
    """The luminance floor exists for exactly this: pure blue is 7% of white's
    luminance, so an unfloored hue rotation could quietly turn any palette
    into an unreadable navy-on-black. Swept inverted too: the nine plain
    swaps keep their ratio by symmetry, but mono's AUTHORED inverse
    (245/16) is a pair of its own that a PALETTES-only sweep never
    measures — and Tint steers the pair the operator is looking at."""
    from dtouch.modes.dithergirl import tinted_palette

    worst = min(
        (_contrast(*tinted_palette(name, hue, amt, invert)),
         name, invert, hue, amt)
        for name in PALETTES
        for invert in (False, True)
        for hue in range(0, 360, 5)
        for amt in (0.25, 0.5, 0.75, 1.0))
    assert worst[0] >= 4.5, (
        "tint broke %s (invert=%s) at hue %s tint %s (%.2f:1)"
        % (worst[1], worst[2], worst[3], worst[4], worst[0]))


def test_tint_steers_toward_the_hue_and_keeps_black_black():
    from dtouch.modes.dithergirl import tinted_palette

    off, on = tinted_palette("mono", 120.0, 1.0)
    assert off == (0, 0, 0)                      # a pure black ground stays so
    assert on[1] > on[0] and on[1] > on[2]       # green now leads


def test_tint_moves_further_the_higher_it_goes():
    from dtouch.modes.dithergirl import tinted_palette

    base = PALETTES["mono"][1]
    dist = [sum(abs(a - b) for a, b in
                zip(tinted_palette("mono", 200.0, amt)[1], base))
            for amt in (0.0, 0.25, 0.5, 0.75, 1.0)]
    assert dist[0] == 0
    assert dist == sorted(dist) and dist[-1] > 0


def test_palette_section_carries_hue_and_tint():
    spec = DitherGirlMode().panel_spec()
    pal = next(s for s in spec if s.title == "PALETTE")
    labels = [getattr(w, "label", None) for w in pal.widgets]
    assert labels == ["palette", "Invert", "Hue", "Tint"]
    hue = pal.widgets[2]
    assert (hue.lo, hue.hi, hue.store_key) == (0.0, 360.0, "hue")
    assert pal.widgets[3].store_key == "tint"


def test_hue_and_tint_round_trip_through_a_look(tmp_path):
    host = _booted(tmp_path)
    ui = host.ui
    ui.dg_hue, ui.dg_tint = 210.0, 0.7
    cfg = host._capture_cfg()
    assert cfg["hue"] == 210.0 and cfg["tint"] == 0.7
    ui.dg_hue, ui.dg_tint = 0.0, 0.0
    _presets.save("tinted", cfg, path=host.presets_path, mode="dithergirl")
    host._reload_presets()
    ui.pending_preset = "tinted"
    host._apply_pending_preset()
    assert (ui.dg_hue, ui.dg_tint) == (210.0, 0.7)


def test_a_builtin_recall_resets_tint(tmp_path):
    """Built-ins record no hue/tint, and both are apply="reset" — recalling
    `classic` mid-set must return the named palette, not keep yesterday's
    tint (DESIGN.md §7)."""
    host = _booted(tmp_path)
    ui = host.ui
    ui.dg_hue, ui.dg_tint = 300.0, 1.0
    ui.pending_preset = "classic"
    host._apply_pending_preset()
    assert ui.dg_tint == 0.0


def test_step_applies_the_tint(tmp_path):
    host = _booted(tmp_path)
    ui, m = host.ui, host.mode
    plain = m.step(_white(), None, 1 / 30)
    ui.dg_hue, ui.dg_tint = 120.0, 1.0
    tinted = m.step(_white(), None, 1 / 30)
    assert not np.array_equal(plain, tinted)
    px = tinted.reshape(-1, 3)[0]
    assert px[1] > px[0] and px[1] > px[2]


def test_swatch_invalidates_on_hue_and_tint(tmp_path):
    """The swatch is the panel's answer to 'what am I looking at' — a palette
    change it does not preview is a lying readout."""
    host = _booted(tmp_path)
    m, g = host.mode, _gui()
    frame = np.zeros((400, 400, 3), np.uint8)
    # Tint first: Hue alone is inert at Tint 0, which is the whole point of
    # the pair, so a hue nudge must move the swatch only once Tint is up.
    for attr, val in (("dg_tint", 0.8), ("dg_hue", 200.0)):
        m._draw_swatch(frame, g, 10, 10, 200)
        first = m._swatch_cache
        setattr(host.ui, attr, val)
        m._draw_swatch(frame, g, 10, 10, 200)
        assert m._swatch_cache is not first
        assert not np.array_equal(m._swatch_cache, first)


# ---------- ASCII (DESIGN.md §4.2, the fifth quantiser) ----------

def test_ascii_is_the_fifth_algorithm_and_reads_live(tmp_path):
    from dtouch.modes.dithergirl import ORDERED, SCALE_HI

    assert ALGOS[-1] == "ASCII" and "ASCII" in ORDERED
    host = _booted(tmp_path)
    ui, m = host.ui, host.mode
    ui.dg_algo_idx = ALGOS.index("ASCII")
    ui.dg_scale = SCALE_HI
    assert m.slow_warning() is False          # a table lookup, at any Scale
    g, frame = _gui(), np.zeros((400, 400, 3), np.uint8)
    assert m._draw_perf_note(frame, g, 10, 10, 200) == 10   # no amber note


def test_ascii_renders_and_stays_inside_the_palette(tmp_path):
    host = _booted(tmp_path)
    ui, m = host.ui, host.mode
    ui.dg_algo_idx = ALGOS.index("ASCII")
    ui.dg_bits, ui.dg_scale = 4.0, 45.0
    out = m.step(_white(), None, 1 / 30)
    assert out.shape == (RES[1], RES[0], 3) and out.dtype == np.uint8
    black = m.step(_black(), None, 1 / 30)
    assert (black.reshape(-1, 3) == np.array((0, 0, 0), np.uint8)).all()


def test_ascii_respects_the_palette_cycle(tmp_path):
    host = _booted(tmp_path)
    ui, m = host.ui, host.mode
    ui.dg_algo_idx = ALGOS.index("ASCII")
    ui.dg_palette_idx = list(PALETTES).index("amber")
    black = m.step(_black(), None, 1 / 30)
    assert (black.reshape(-1, 3) == np.array((24, 12, 0), np.uint8)).all()


def test_ascii_bits_set_the_ramp_length(tmp_path):
    host = _booted(tmp_path)
    ui, m = host.ui, host.mode
    ui.dg_algo_idx = ALGOS.index("ASCII")
    for bits, n in ((1.0, 2), (2.0, 4), (3.0, 8), (4.0, 16)):
        ui.dg_bits = bits
        m.step(_white(), None, 1 / 30)
        assert m._ascii.n == n


def test_ascii_rebuilds_only_what_a_change_invalidates(tmp_path):
    """The frame buffer is what a rebuild was really costing: at 4K it is
    24 MB, and allocate-and-clear measured 31.9 ms of a 35.0 ms construction —
    so a Hue drag (which invalidates the ATLAS and nothing else) ran at 46
    ms/frame against 7 ms idle. Retune in place; keep the buffer."""
    host = _booted(tmp_path, res=(640, 360))   # tall enough that Scale moves
                                               # the grid off the cell floor
    ui, m = host.ui, host.mode
    ui.dg_algo_idx = ALGOS.index("ASCII")
    m.step(_white(), None, 1 / 30)
    rend, buf, atlas, lut = m._ascii, m._ascii.out, m._ascii.atlas, m._ascii.pos_lut
    ramp, view = m._ascii.ramp, m._ascii.view

    m.step(_black(), None, 1 / 30)
    assert m._ascii is rend and m._ascii.atlas is atlas   # a frame is not a rebuild
    ui.dg_contrast = 1.8
    m.step(_white(), None, 1 / 30)
    assert m._ascii is rend and m._ascii.atlas is atlas   # nor a per-frame control

    # palette only: the atlas is palette-coloured and the LUT reads the atlas,
    # so both go — the buffer, the strided view and the ramp search do not
    ui.dg_palette_idx = list(PALETTES).index("ice")
    m.step(_white(), None, 1 / 30)
    assert m._ascii is rend
    assert m._ascii.out is buf, "a palette change reallocated the frame buffer"
    assert m._ascii.view is view
    assert m._ascii.ramp == ramp          # the ramp SEARCH did not re-run
    assert m._ascii.atlas is not atlas and m._ascii.pos_lut is not lut

    # gamma only: the LUT inverts the atlas's measured curve, the atlas stands
    atlas = m._ascii.atlas
    ui.dg_gamma = False
    m.step(_white(), None, 1 / 30)
    assert m._ascii.atlas is atlas and m._ascii.out is buf
    assert m._ascii.pos_lut is not lut

    # the grid: new cells, so everything below them — still the same buffer
    ui.dg_scale = 30.0
    m.step(_white(), None, 1 / 30)
    assert m._ascii.out is buf
    assert m._ascii.ramp is not ramp and m._ascii.view is not view


def test_hue_and_tint_do_not_reallocate_the_ascii_buffer(tmp_path):
    """The interaction the rebuild-every-frame cost actually landed on: Hue
    and Tint are sliders, so a drag is one palette change per frame."""
    host = _booted(tmp_path)
    ui, m = host.ui, host.mode
    ui.dg_algo_idx = ALGOS.index("ASCII")
    ui.dg_tint = 0.5
    m.step(_white(), None, 1 / 30)
    rend, buf = m._ascii, m._ascii.out
    for hue in range(0, 360, 7):
        ui.dg_hue = float(hue)
        m.step(_white(), None, 1 / 30)
        assert m._ascii is rend and m._ascii.out is buf


def test_ascii_matte_gate_still_composites(tmp_path):
    host = _booted(tmp_path)
    ui, m = host.ui, host.mode
    ui.dg_algo_idx = ALGOS.index("ASCII")
    ui.dg_matte_idx = 1
    m.mat, m._mat_kind = FakeMatte(), MATTES_DG[1]
    ui.dg_matte_black = True
    out = m.step(_white(), None, 1 / 30)
    w = out.shape[1]
    assert (out[:, :w // 2 - 2] == 0).all()        # background half is black
    assert out[:, w // 2 + 2:].any()               # subject half has glyphs


def test_ascii_survives_a_blackout_of_its_own_buffer(tmp_path):
    """The shell blacks `out` IN PLACE for blackout (§6.2) and the renderer
    hands back a reused buffer."""
    host = _booted(tmp_path)
    ui, m = host.ui, host.mode
    ui.dg_algo_idx = ALGOS.index("ASCII")
    ui.dg_palette_idx, ui.dg_invert = list(PALETTES).index("mono"), True
    m.step(_white(), None, 1 / 30)[:] = 0
    again = m.step(_white(), None, 1 / 30)
    assert again.any(), "the buffer must be wholly rewritten each frame"


def test_dither_helper_refuses_ascii():
    """`_dither` returns a [0,1] plane for `_palette_map`; ASCII has no such
    plane. Falling through to Riemersma would have been a silent wrong
    picture."""
    from dtouch.modes.dithergirl import _dither

    with pytest.raises(ValueError):
        _dither(np.zeros((4, 4), np.float32), "ASCII", 1, True, "auto")


def test_ascii_builtins_are_shipped_and_apply(tmp_path):
    assert set(DitherGirlMode.BUILTIN) >= {"ascii", "ascii stream"}
    host = _booted(tmp_path)
    ui = host.ui
    ui.pending_preset = "ascii stream"
    host._apply_pending_preset()
    assert ALGOS[ui.dg_algo_idx] == "ASCII"
    assert ui.dg_scale == 30.0 and ui.dg_contrast == 1.4   # stream-tuned
    assert list(PALETTES)[ui.dg_palette_idx] == "green phosphor"


def test_the_safe_look_is_still_a_pixel_dither():
    """DESIGN.md §6.2: `0` must walk OUT of ASCII to a known-good picture."""
    assert (DitherGirlMode.BUILTIN[DitherGirlMode().safe_look()]["algorithm"]
            == "Floyd-Steinberg")


def test_nudging_a_stored_fraction_lands_scale_back_on_the_grid(tmp_path):
    """The other half of the pass-through bargain: the stored fraction is the
    look's, not the control's. The look applies exactly, but the first real
    edit — a nudge key here, through the real registry — adapts the control
    back onto its whole-row grid, the same way the range fix let `ascii
    stream` keep 30 until the operator moved the slider."""
    from dtouch.panelspec import nudgeable

    host = _booted(tmp_path)
    host._wire_keys()
    ui = host.ui
    assert host._apply_look("legacy", {"scale": 45.55})
    assert ui.dg_scale == 45.55                  # the stored look applies exactly
    ws = [w for w in ui.iter_widgets() if nudgeable(w)]
    ui.nudge_idx = next(i for i, w in enumerate(ws) if w.attr == "dg_scale")
    host.reg.dispatch(ord("="))                  # one press: +17.25 rows
    assert ui.dg_scale == 63.0                   # 62.8, snapped to the grid


def test_status_line_names_ascii(tmp_path):
    host = _booted(tmp_path)
    host.ui.dg_algo_idx = ALGOS.index("ASCII")
    host.ui.dg_bits = 4.0
    s = host._status_line()
    assert s == "DITHER  ascii  4-bit  bias auto  src synthetic"
    assert s == s.encode("ascii", "replace").decode()


def test_grid_note_renders_only_under_ascii(tmp_path):
    """Scale changes UNIT under ASCII, from working pixels to character rows.
    Hiding that would be dishonest; so would drawing the note for the pixel
    dithers, where it is a lie."""
    host = _booted(tmp_path)
    ui, m, g = host.ui, host.mode, _gui()
    frame = np.zeros((400, 400, 3), np.uint8)
    assert m._draw_grid_note(frame, g, 10, 10, 200) == 10   # not ASCII: silent
    ui.dg_algo_idx = ALGOS.index("ASCII")
    assert m._draw_grid_note(frame, g, 10, 10, 200) == 10   # no renderer yet
    m.step(_white(), None, 1 / 30)
    assert m._draw_grid_note(frame, g, 10, 10, 200) > 10
    assert m._ascii.grid_note().startswith("grid ")


def test_grid_note_says_when_scale_has_run_out_of_room(tmp_path):
    from dtouch.modes.dithergirl import SCALE_HI

    host = _booted(tmp_path, res=(640, 360))
    ui, m, g = host.ui, host.mode, _gui()
    ui.dg_algo_idx = ALGOS.index("ASCII")
    frame = np.zeros((400, 400, 3), np.uint8)
    ui.dg_scale = 30.0            # 12 px cells at 360p: still room to move
    m.step(_white(360, 640), None, 1 / 30)
    short = m._draw_grid_note(frame, g, 10, 10, 200)
    assert m._ascii.clamped is False
    ui.dg_scale = SCALE_HI
    m.step(_white(360, 640), None, 1 / 30)
    # past the floor the GRID stops changing, so nothing is rebuilt — the
    # request is still read, or the note would never appear at all
    assert m._ascii.clamped is True
    assert m._draw_grid_note(frame, g, 10, 10, 200) > short   # one line more


def test_the_ascii_perf_note_is_measured_and_reported(tmp_path):
    """Perf honesty (§4.2) without a guessed per-resolution threshold: the
    mode times its own step. ASCII is in the ordered class at 720p/1080p but a
    4K frame at the cell floor measures ~8.7 ms, and an operator is owed the
    number rather than a quietly halved frame rate."""
    from dtouch.modes.dithergirl import (ASCII_SLOW_CLEAR, ASCII_SLOW_MS,
                                         ASCII_WARMUP_FRAMES)

    # pinned against the LITERALS: every assertion below compares a cost
    # against these, so they all pass vacuously if the thresholds move, and
    # the thresholds ARE the claim the panel makes
    assert (ASCII_SLOW_MS, ASCII_SLOW_CLEAR, ASCII_WARMUP_FRAMES) == (8.0, 0.75, 8)

    host = _booted(tmp_path)
    ui, m, g = host.ui, host.mode, _gui()
    ui.dg_algo_idx = ALGOS.index("ASCII")
    frame = np.zeros((400, 400, 3), np.uint8)

    # driven with numbers, not with a stopwatch: what is pinned here is the
    # rule, and "this machine renders ASCII in under 8 ms" is a perf test
    # wearing a logic test's clothes — it fails on a busy box for reasons that
    # have nothing to do with the rule
    for _ in range(20):
        m._ascii_verdict(1.2)
    assert m._ascii_ms < ASCII_SLOW_MS and m._ascii_slow is False
    m.step(_white(), None, 1 / 30)             # the renderer the note reads
    clean = m._draw_grid_note(frame, g, 10, 10, 200)

    m._ascii_ms, m._ascii_slow = 9.3, True
    assert m._draw_grid_note(frame, g, 10, 10, 200) > clean

    # a warm-up before either verdict is allowed: a cold cell size pays a
    # one-off coverage scan that says nothing about the steady state
    m._ascii_ms, m._ascii_frames, m._ascii_slow = None, 0, False
    for _ in range(ASCII_WARMUP_FRAMES - 1):
        m._ascii_verdict(40.0)
    assert m._ascii_slow is False              # measured slow, not yet said
    m._ascii_verdict(40.0)
    assert m._ascii_slow is True

    # ...and the verdict follows the measurement back DOWN, so a setting that
    # is fast again stops being scolded without needing a rebuild to clear it
    for _ in range(40):
        m._ascii_verdict(1.2)
    assert m._ascii_slow is False and m._ascii_ms < ASCII_SLOW_MS * ASCII_SLOW_CLEAR


def test_the_ascii_perf_note_can_fire_during_the_drag_that_is_slow(tmp_path):
    """The note used to be structurally unable to report the one interaction
    that was slow: the clock started AFTER the rebuild (so it timed render()
    only — the real step cost 28.7 ms while the note read 5.3 ms), and the
    frame counter was cleared on every rebuild, so a drag that rebuilt every
    frame could never reach the 8-frame warm-up gate."""
    from dtouch.ascii_art import AsciiRenderer
    from dtouch.modes.dithergirl import ASCII_SLOW_MS, ASCII_WARMUP_FRAMES

    host = _booted(tmp_path)
    ui, m = host.ui, host.mode
    ui.dg_algo_idx = ALGOS.index("ASCII")
    ui.dg_tint = 0.5
    m.step(_white(), None, 1 / 30)

    real, cost = AsciiRenderer.configure, ASCII_SLOW_MS * 2.0 / 1000.0

    def slow_configure(self, *a, **kw):
        time.sleep(cost)                  # stand-in for a 4K atlas rebuild
        return real(self, *a, **kw)

    AsciiRenderer.configure = slow_configure
    try:
        for i in range(ASCII_WARMUP_FRAMES * 3):
            ui.dg_hue = float((i * 11) % 360)     # a drag: retune every frame
            m.step(_white(), None, 1 / 30)
    finally:
        AsciiRenderer.configure = real

    assert m._ascii_ms > ASCII_SLOW_MS, (
        "the measurement must include what the frame actually cost")
    assert m._ascii_slow is True, (
        "a drag that retunes every frame must still reach the warm-up gate")


def test_the_ascii_notes_wrap_inside_the_panel_column(tmp_path):
    """Same rule as the amber slow-note: a note that runs off the sidebar is
    graffiti over the picture, not part of the panel."""
    host = _booted(tmp_path)
    ui, m, g = host.ui, host.mode, _gui()
    ui.dg_algo_idx = ALGOS.index("ASCII")
    m.step(_white(), None, 1 / 30)
    m._ascii_ms, m._ascii_slow = 9.3, True
    frame = np.zeros((400, 400, 3), np.uint8)
    wide = m._draw_grid_note(frame, g, 10, 10, 400)
    narrow = m._draw_grid_note(frame, g, 10, 10, 120)
    assert narrow > wide
    assert len(m._note_lines(g, "ascii 9.3 ms/frame - lower Scale or output",
                             120)) > 1


def test_hue_alone_does_nothing_until_tint_is_up(tmp_path):
    """The pair is a direction and an amount. A Hue slider that recoloured the
    picture on its own would make Tint 0 unreachable by accident."""
    host = _booted(tmp_path)
    ui, m = host.ui, host.mode
    plain = m.step(_white(), None, 1 / 30)
    ui.dg_hue = 47.0
    assert np.array_equal(plain, m.step(_white(), None, 1 / 30))


def test_ascii_swatch_previews_the_glyph_ramp(tmp_path):
    host = _booted(tmp_path)
    ui, m, g = host.ui, host.mode, _gui()
    frame = np.zeros((400, 400, 3), np.uint8)
    ui.dg_algo_idx = ALGOS.index("ASCII")
    m._draw_swatch(frame, g, 10, 10, 200)
    swatch = m._swatch_cache
    assert swatch is not None and swatch.any()
    # sparse on the left, dense on the right — it is a ramp, not a fill
    lit = swatch.reshape(swatch.shape[0], -1).mean(axis=0)
    quarter = swatch.shape[1] // 4
    assert lit[-quarter:].mean() > lit[:quarter].mean()


def test_ascii_step_is_not_slower_than_the_default_algorithm(tmp_path):
    """ASCII replaces the mode's Floyd-Steinberg default when selected; the
    operator must not pay for the switch.

    Already a ratio between two things timed in the same run, so the machine's
    speed cannot decide it — but a MEAN can still be decided by a load spike
    landing in one half and not the other, so both halves are best-of-N."""
    host = _booted(tmp_path, res=(640, 360))
    ui, m = host.ui, host.mode
    frame = np.random.default_rng(7).integers(0, 256, (360, 640, 3), np.uint8)

    def bench(algo, scale):
        ui.dg_algo_idx = ALGOS.index(algo)
        ui.dg_scale = scale
        for _ in range(3):
            m.step(frame, None, 1 / 30)
        return best_ms(lambda: m.step(frame, None, 1 / 30))

    assert bench("ASCII", 45.0) < bench("Floyd-Steinberg", 72.0)
