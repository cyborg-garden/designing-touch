"""Dither Girl — DESIGN.md §4.2, migration step 9.

The mode is pure numpy/cv2 (no GL), so everything here runs fully headless:
protocol conformance, panel-spec structure, palette mapping, matte gating,
swatch caching, preset capture/apply round-trips (incl. the nested "signal"
block and the suppressed dither row), bank seeding, still input, and the
shell-command wiring for menu + direct mode keys.
"""
import numpy as np
import pytest

import dtouch.presets as _presets
from dtouch import imgui
from dtouch.commands import CommandRegistry
from dtouch.modes import REGISTRY, mode_by_id
from dtouch.modes.dithergirl import (ACCENT, ALGOS, MATTES_DG, PALETTES,
                                     DitherGirlMode)
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
    assert m.id == "dithergirl" and m.title == "Dither Girl"
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
    assert s == "DITHER GIRL  floyd-steinberg  1-bit  bias auto  src synthetic"
    host.ui.dg_algo_idx = ALGOS.index("Blue noise")
    host.ui.dg_bits = 3.0
    host.ui.input_idx = 1                    # still input -> src tail follows
    assert (host._status_line()
            == "DITHER GIRL  blue noise  3-bit  bias auto  src still")


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
    # TONE: Bits 1-4 int-fmt, Gamma toggle, bias cycle, Contrast, Scale 45-720
    tone = by["TONE"]
    bits = next(w for w in tone if isinstance(w, Slider) and w.label == "Bits")
    assert (bits.lo, bits.hi, bits.fmt) == (1.0, 4.0, ".0f")
    assert any(isinstance(w, Toggle) and w.label == "Gamma" for w in tone)
    bias = next(w for w in tone if isinstance(w, Cycle))
    assert list(bias.options) == ["auto", "light", "dark"]
    scale = next(w for w in tone if isinstance(w, Slider) and w.label == "Scale")
    assert (scale.lo, scale.hi) == (45.0, 720.0)
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


@pytest.mark.parametrize("palette,white_out,black_out", [
    ("white-on-black", (255, 255, 255), (0, 0, 0)),
    ("black-on-white", (16, 16, 16), (245, 245, 245)),
    ("amber", (255, 176, 0), (24, 12, 0)),
    ("green phosphor", (80, 255, 120), (0, 20, 8)),
])
def test_step_maps_on_off_levels_to_the_palette(tmp_path, palette,
                                                white_out, black_out):
    host = _booted(tmp_path)
    ui, m = host.ui, host.mode
    ui.dg_palette_idx = list(PALETTES).index(palette)
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
    assert cfg["palette"] == "white-on-black"
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
