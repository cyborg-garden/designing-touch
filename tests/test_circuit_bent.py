"""Tests for dtouch.circuit_bent — stochastic circuit-bent camera effect."""
import numpy as np
import pytest

from dtouch.circuit_bent import CircuitBent


def _frame(h=72, w=128, seed=7):
    """Random BGR uint8 test frame."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


def _grey_ramp_frame(h=72, w=128):
    """Horizontal luminance ramp as a BGR uint8 frame."""
    row = np.linspace(0, 255, w, dtype=np.uint8)
    frame = np.stack([np.tile(row, (h, 1))] * 3, axis=2)
    return frame


class TestDitherQualityControls:
    """The SIGNAL rack's Bits/Gamma/Bias drive the dither call (DESIGN.md
    §2.4/§4.1) — output density/level counts must actually change."""

    @staticmethod
    def _cb(**kw):
        kw.setdefault("dither_mode", "bayer")
        kw.setdefault("dither_size", None)
        return CircuitBent(seed=0, chroma_shift=0, scan_drift=0,
                           glitch_prob=0.0, bit_crush=0, scanlines=False, **kw)

    def test_bits_changes_the_level_count(self):
        frame = _grey_ramp_frame()
        one = self._cb(dither_bits=1).process(frame)
        three = self._cb(dither_bits=3).process(frame)
        assert len(np.unique(one[:, :, 0])) == 2          # 1 bit = two-tone
        assert len(np.unique(three[:, :, 0])) > 2
        assert not np.array_equal(one, three)

    def test_gamma_changes_the_output(self):
        frame = _grey_ramp_frame()
        lin = self._cb(dither_gamma=True).process(frame)
        crushed = self._cb(dither_gamma=False).process(frame)
        assert not np.array_equal(lin, crushed)
        # both are still valid dithers of the same ramp (same level count)
        assert len(np.unique(lin[:, :, 0])) == len(np.unique(crushed[:, :, 0]))

    def test_bias_flips_the_rounding_direction(self):
        frame = _grey_ramp_frame()
        light = self._cb(dither_bits=1, dither_invert=False).process(frame)
        dark = self._cb(dither_bits=1, dither_invert=True).process(frame)
        assert not np.array_equal(light, dark)

    def test_quality_controls_reach_error_diffusion_too(self):
        frame = _grey_ramp_frame()
        fs_lin = self._cb(dither_mode="fs", dither_gamma=True).process(frame)
        fs_off = self._cb(dither_mode="fs", dither_gamma=False).process(frame)
        assert not np.array_equal(fs_lin, fs_off)

    def test_the_shipped_defaults_are_3_bit_and_gamma_correct(self):
        """DESIGN.md §4.1: "Bits int-snapped 1-4, Gamma default ON". The
        panel seeds sig_bits/sig_gamma to match CircuitBent's own defaults, so
        the two have to agree — and nothing checked the constructor end. A
        default of 1 bit would ship the rack as a two-tone crusher; gamma off
        would ship the crushed retro look as the neutral one."""
        cb = CircuitBent()
        assert cb.dither_bits == 3
        assert cb.dither_gamma is True
        assert cb.dither_invert == "auto"

        # ...and the defaults are load-bearing, not just stored
        frame = _grey_ramp_frame()
        default = self._cb().process(frame)
        assert np.array_equal(default, self._cb(dither_bits=3,
                                                dither_gamma=True).process(frame))
        assert not np.array_equal(default, self._cb(dither_bits=1).process(frame))
        assert not np.array_equal(default,
                                  self._cb(dither_gamma=False).process(frame))

    def test_the_panel_defaults_agree_with_the_engine_defaults(self):
        """The rack's UI seeds and CircuitBent's constructor are two copies of
        the same numbers; drift between them is silent."""
        from dtouch.overlay_ui import OverlayUI

        ui = OverlayUI(64, 36, ["abstract"], ["ice"], ["auto"])
        cb = CircuitBent()
        assert int(ui.sig_bits) == cb.dither_bits
        assert bool(ui.sig_gamma) is cb.dither_gamma


class TestCircuitBentBasics:
    def test_output_shape(self):
        cb = CircuitBent(seed=0)
        frame = _frame()
        out = cb.process(frame)
        assert out.shape == frame.shape

    def test_output_dtype(self):
        cb = CircuitBent(seed=0)
        out = cb.process(_frame())
        assert out.dtype == np.uint8

    def test_output_range(self):
        cb = CircuitBent(seed=0)
        out = cb.process(_frame())
        assert int(out.min()) >= 0 and int(out.max()) <= 255

    def test_does_not_mutate_input(self):
        cb = CircuitBent(seed=0)
        frame = _frame()
        original = frame.copy()
        cb.process(frame)
        assert np.array_equal(frame, original)

    def test_deterministic_with_seed(self):
        """Same seed → identical output for first frame."""
        frame = _frame()
        out_a = CircuitBent(seed=42).process(frame)
        out_b = CircuitBent(seed=42).process(frame)
        assert np.array_equal(out_a, out_b)

    def test_different_seeds_differ(self):
        frame = _frame()
        out_a = CircuitBent(seed=1).process(frame)
        out_b = CircuitBent(seed=2).process(frame)
        assert not np.array_equal(out_a, out_b)

    def test_invalid_frame_raises(self):
        cb = CircuitBent(seed=0)
        with pytest.raises(ValueError):
            cb.process(np.zeros((72, 128), np.uint8))   # 2-D, no channels


class TestEffectsCanBeDisabled:
    def _all_off(self):
        return CircuitBent(
            seed=0,
            chroma_shift=0,
            scan_drift=0,
            glitch_prob=0.0,
            bit_crush=0,
            scanlines=False,
            dither_mode=None,
        )

    def test_all_effects_off_minimal_change(self):
        """With every effect disabled the output should be very close to input."""
        cb = self._all_off()
        frame = _grey_ramp_frame()
        out = cb.process(frame)
        # no effects → output equals input
        assert np.array_equal(out, frame)


class TestChromaShift:
    def test_chroma_shift_changes_output(self):
        cb_on  = CircuitBent(seed=0, chroma_shift=10, scan_drift=0, glitch_prob=0,
                             bit_crush=0, scanlines=False, dither_mode=None)
        cb_off = CircuitBent(seed=0, chroma_shift=0,  scan_drift=0, glitch_prob=0,
                             bit_crush=0, scanlines=False, dither_mode=None)
        frame = _grey_ramp_frame()
        assert not np.array_equal(cb_on.process(frame), cb_off.process(frame))


class TestScanDrift:
    def test_scan_drift_changes_output(self):
        cb_on  = CircuitBent(seed=0, chroma_shift=0, scan_drift=8, glitch_prob=0,
                             bit_crush=0, scanlines=False, dither_mode=None)
        cb_off = CircuitBent(seed=0, chroma_shift=0, scan_drift=0, glitch_prob=0,
                             bit_crush=0, scanlines=False, dither_mode=None)
        frame = _grey_ramp_frame()
        assert not np.array_equal(cb_on.process(frame), cb_off.process(frame))


class TestBitCrush:
    def test_bit_crush_reduces_unique_values(self):
        cb = CircuitBent(seed=0, chroma_shift=0, scan_drift=0, glitch_prob=0,
                         bit_crush=2, scanlines=False, dither_mode=None)
        frame = _grey_ramp_frame()
        out = cb.process(frame)
        unique = len(np.unique(out[:, :, 0]))
        # 2-bit → max 4 levels (0, 85, 170, 255)
        assert unique <= 4

    def test_bit_crush_range_intact(self):
        cb = CircuitBent(seed=0, chroma_shift=0, scan_drift=0, glitch_prob=0,
                         bit_crush=2, scanlines=False, dither_mode=None)
        out = cb.process(_frame())
        assert int(out.min()) >= 0 and int(out.max()) <= 255


class TestScanlines:
    def test_scanlines_darken_alternate_rows(self):
        cb_on  = CircuitBent(seed=0, chroma_shift=0, scan_drift=0, glitch_prob=0,
                             bit_crush=0, scanlines=True,  scanline_strength=0.5,
                             dither_mode=None)
        cb_off = CircuitBent(seed=0, chroma_shift=0, scan_drift=0, glitch_prob=0,
                             bit_crush=0, scanlines=False, dither_mode=None)
        frame = np.full((64, 64, 3), 200, dtype=np.uint8)
        out_on  = cb_on.process(frame)
        out_off = cb_off.process(frame)
        # every other row should be darker when scanlines are on
        even_rows_darker = out_on[::2].mean() < out_off[::2].mean()
        assert even_rows_darker


class TestDithering:
    def test_bayer_dither_mode(self):
        cb = CircuitBent(seed=0, chroma_shift=0, scan_drift=0, glitch_prob=0,
                         bit_crush=0, scanlines=False, dither_mode="bayer",
                         dither_bits=2, dither_size=None)
        out = cb.process(_grey_ramp_frame())
        unique = len(np.unique(out[:, :, 0]))
        assert unique <= 5   # 2-bit = max 4 levels (plus tolerance)

    def test_fs_dither_mode(self):
        cb = CircuitBent(seed=0, chroma_shift=0, scan_drift=0, glitch_prob=0,
                         bit_crush=0, scanlines=False, dither_mode="fs",
                         dither_bits=2, dither_size=None)
        frame = _grey_ramp_frame(h=16, w=32)
        out = cb.process(frame)
        assert out.shape == frame.shape
        assert int(out.min()) >= 0 and int(out.max()) <= 255

    def test_blue_noise_dither_mode(self):
        cb = CircuitBent(seed=0, chroma_shift=0, scan_drift=0, glitch_prob=0,
                         bit_crush=0, scanlines=False, dither_mode="blue",
                         dither_bits=2, dither_size=None)
        out = cb.process(_grey_ramp_frame())
        unique = len(np.unique(out[:, :, 0]))
        assert unique <= 5   # 2-bit = max 4 levels (plus tolerance)

    def test_riemersma_dither_mode(self):
        cb = CircuitBent(seed=0, chroma_shift=0, scan_drift=0, glitch_prob=0,
                         bit_crush=0, scanlines=False, dither_mode="riemersma",
                         dither_bits=2, dither_size=None)
        frame = _grey_ramp_frame(h=16, w=32)
        out = cb.process(frame)
        assert out.shape == frame.shape
        assert int(out.min()) >= 0 and int(out.max()) <= 255


class TestMultiFrame:
    def test_evolves_across_frames(self):
        """Output should differ between consecutive frames (stochastic drift)."""
        cb = CircuitBent(seed=0, chroma_shift=12, scan_drift=8, glitch_prob=0.5,
                         bit_crush=0, scanlines=False, dither_mode=None)
        frame = _frame()
        out0 = cb.process(frame)
        out1 = cb.process(frame)
        # Not identical (effects evolve)
        assert not np.array_equal(out0, out1)

    def test_output_stays_valid_over_many_frames(self):
        cb = CircuitBent(seed=0)
        frame = _frame()
        for _ in range(30):
            out = cb.process(frame)
            assert out.shape == frame.shape
            assert out.dtype == np.uint8
            assert int(out.min()) >= 0 and int(out.max()) <= 255
