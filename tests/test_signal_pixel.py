"""The SIGNAL rack's Pixel slider — dither block size, in output pixels.

The owner's ask (2026-08-24 play session): "Glitch mode doesn't have a
slider for pixel size." The visual parameter that reads as pixel size is the
rack dither's working resolution (`CircuitBent.dither_size`: dither at a
reduced height, nearest-neighbour back up — the block-pixel look). Pixel
exposes it in the unit the eye sees: the block edge in output pixels.

Contract under test:

- `sync_signal` maps ui.sig_px -> cb.dither_size: 0 = auto (the mode's own
  `signal_dither_rows`, or the rack's 72), 1 = full-res dithering, px ->
  round(out_h / px) working rows. Both backends read cb.dither_size, so the
  mapping is the whole cross-backend surface (parity at px-derived sizes is
  held in tests/test_rack_gl.py).
- the blocks really are px pixels tall (the nearest upscale's bands);
- the control is a whole-pixel slider (engine_snaps quantisation);
- it serializes as signal.pixel and round-trips through a look;
- Dither mode claims it (its Scale IS that control there), and it hides
  wherever it cannot act (Glitch off, or the rack's dither off).
"""
from types import SimpleNamespace

import numpy as np
import pytest

from dtouch.circuit_bent import CircuitBent
from dtouch.modes.dithergirl import DitherGirlMode
from dtouch.overlay_ui import DITHERS, OverlayUI, sync_signal
from dtouch.panelspec import Slider, apply_look, capture_look, visible, walk_spec

PRESETS = ["abstract"]
PALETTES = ["ice", "fire"]
MATTES = ["auto"]


def _ui():
    return OverlayUI(1920, 1080, PRESETS, PALETTES, MATTES)


def _stub_ui(sig_px=0.0, dither="bayer"):
    return SimpleNamespace(chroma=10.0, drift=8.0, crush=0.0, scanlines=True,
                           sig_bits=3.0, sig_gamma=True, sig_bias_idx=0,
                           sig_px=sig_px, dither_name=dither)


def _cb():
    return CircuitBent(seed=0)


class _AutoRowsMode:
    claims = ()

    @staticmethod
    def signal_dither_rows(out_h):
        return max(96, out_h // 6)


class _PlainMode:
    claims = ()


# ---------- the sync mapping ----------

def test_px_zero_is_the_mode_auto_rows():
    cb = _cb()
    sync_signal(cb, _stub_ui(sig_px=0.0), _AutoRowsMode(), 1080)
    assert cb.dither_size == 180            # physarum's rule: 1080 // 6
    sync_signal(cb, _stub_ui(sig_px=0.0), _PlainMode(), 1080)
    assert cb.dither_size == 72             # the rack's default block look


def test_px_one_is_full_res():
    cb = _cb()
    sync_signal(cb, _stub_ui(sig_px=1.0), _AutoRowsMode(), 1080)
    # dither_size >= out_h: both backends treat this as full-res dithering
    assert cb.dither_size == 1080


def test_px_maps_to_out_h_over_px_rows():
    cb = _cb()
    sync_signal(cb, _stub_ui(sig_px=8.0), _PlainMode(), 1080)
    assert cb.dither_size == 135
    sync_signal(cb, _stub_ui(sig_px=9.0), _PlainMode(), 288)
    assert cb.dither_size == 32             # the rack-parity test's size
    sync_signal(cb, _stub_ui(sig_px=32.0), _PlainMode(), 1080)
    assert cb.dither_size == 34


def test_px_overrides_the_mode_auto():
    cb = _cb()
    sync_signal(cb, _stub_ui(sig_px=6.0), _AutoRowsMode(), 1080)
    assert cb.dither_size == 180            # = auto here (6 px IS the auto cell)
    sync_signal(cb, _stub_ui(sig_px=12.0), _AutoRowsMode(), 1080)
    assert cb.dither_size == 90             # chunkier than the mode's choice


def test_missing_attr_is_auto():
    """A bare UI state (soak tests, stubs) without sig_px keeps today's
    behaviour — getattr default 0.0 = auto."""
    cb = _cb()
    ui = _stub_ui()
    del ui.sig_px
    sync_signal(cb, ui, _PlainMode(), 1080)
    assert cb.dither_size == 72


# ---------- the blocks are really px pixels tall ----------

def test_blocks_are_px_pixels_tall():
    """dither_size rows nearest-upscaled to h means the dithered picture
    repeats in bands of exactly px rows — the visible 'pixel size'."""
    h, w, px = 96, 128, 8
    cb = CircuitBent(seed=0, chroma_shift=0, scan_drift=0, glitch_prob=0.0,
                     bit_crush=0, scanlines=False, dither_mode="bayer",
                     dither_size=h // px)
    rng = np.random.default_rng(3)
    frame = rng.integers(0, 256, (h, w, 3), np.uint8)
    out = cb.process(frame)
    bands = out.reshape(h // px, px, w, 3)
    assert np.array_equal(bands, np.repeat(bands[:, :1], px, axis=1)), \
        "every band of px rows must be a single repeated dither row"


def test_different_px_render_different_pictures():
    h, w = 96, 128
    rng = np.random.default_rng(3)
    frame = rng.integers(0, 256, (h, w, 3), np.uint8)
    outs = []
    for px in (4, 16):
        cb = CircuitBent(seed=0, chroma_shift=0, scan_drift=0,
                         glitch_prob=0.0, bit_crush=0, scanlines=False,
                         dither_mode="bayer", dither_size=h // px)
        outs.append(cb.process(frame))
    assert np.abs(outs[0].astype(np.int16) - outs[1].astype(np.int16)).mean() > 1.0


# ---------- panel surface ----------

def _pixel_widget(spec):
    return next((w for _s, w in walk_spec(spec)
                 if getattr(w, "attr", None) == "sig_px"), None)


def test_pixel_is_a_whole_pixel_slider():
    ui = _ui()
    w = _pixel_widget(ui.spec)
    assert isinstance(w, Slider)
    assert (w.lo, w.hi, w.step, w.snap) == (0.0, 32.0, 1.0, "round")
    assert w.engine_snaps and w.store_key == "pixel"
    ui.sig_px = 7.4                          # writes land on the engine's grid
    assert ui.sig_px == 7.0


def test_pixel_serializes_as_signal_pixel_and_round_trips():
    ui = _ui()
    ui.sig_px = 12.0
    cfg = capture_look(ui, ui.spec)
    assert cfg["signal"]["pixel"] == 12.0
    ui.sig_px = 0.0
    assert apply_look(ui, ui.spec, cfg, {}) == []
    assert ui.sig_px == 12.0
    # a look without the key leaves the live value alone (like the rest of
    # the rack sliders: reset-flagged, no declared default)
    ui.sig_px = 5.0
    assert apply_look(ui, ui.spec, {"signal": {}}, {}) == []
    assert ui.sig_px == 5.0


def test_pixel_hides_where_it_cannot_act():
    ui = _ui()
    w = _pixel_widget(ui.spec)
    assert not visible(ui, w), "Glitch off: the rack is not running"
    ui.glitch = True
    assert visible(ui, w)
    ui.dither_idx = DITHERS.index("off")
    assert not visible(ui, w), "rack dither off: no blocks to size"


def test_dither_mode_claims_pixel():
    m = DitherGirlMode()
    assert "pixel" in m.claims               # its Scale IS this control
    from dtouch.shell import Host
    spec = Host(m, source=None, res=(192, 108), show=False,
                preset=None).compose_spec(m)
    assert _pixel_widget(spec) is None
