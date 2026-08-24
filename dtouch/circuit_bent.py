"""Circuit-bent camera effect — stochastic image-space glitch simulation.

Real circuit-bent cameras have hardware randomness: sync drift, colour bleed,
and bit corruption that is impossible to predict from frame to frame. This
module recreates that aesthetic through controlled stochastic processes:

- Smooth per-frame drift via IIR-filtered noise for slow-moving colour bleed
  and scan-line sync errors.
- Per-frame event sampling for sharp glitch-block corruption.
- Multi-frame hold so glitches "stick" rather than flash once and vanish.

The main interface is :class:`CircuitBent`:

    cb = CircuitBent(seed=42)
    out = cb.process(bgr_frame)   # in: uint8 (H, W, 3), out: uint8 (H, W, 3)

Compose with dtouch.dither for a dithered / bit-reduced overlay:

    from dtouch.dither import bayer_dither
"""
from __future__ import annotations

import cv2
import numpy as np

from .dither import bayer_dither, blue_noise_dither, floyd_steinberg, riemersma_dither

# The CRT look is authored at ~270 scan-line periods per frame: the mask
# pitch is round(h / SCANLINE_ROWS) rows, floored at 2 (so up to 540p this
# is the shipped every-other-row mask; 1080p gets a 4-row pitch, 4K 8-row —
# the lines survive window fit / projector scalers instead of averaging out).
SCANLINE_ROWS = 270


class CircuitBent:
    """Stateful circuit-bent camera effect with controlled stochastic behaviour.

    All effects share a single evolving random state.  A fixed seed makes the
    sequence deterministic and reproducible; pass ``None`` for per-session
    randomness.

    Parameters
    ----------
    seed : int or None
        RNG seed for deterministic output.
    chroma_shift : int
        Maximum horizontal pixel offset applied independently to R and B
        channels (slow IIR drift — simulates colour-decoder sync loss).
    scan_drift : int
        Maximum per-row horizontal displacement in pixels.  Rows are grouped
        into slowly drifting bands with occasional sharp "sync-tear" events.
    glitch_prob : float
        Per-frame probability that a glitch-block event fires.
    glitch_hold : int
        Frames a glitch block persists before the next re-roll.
    bit_crush : int
        Output bit-depth per channel for hard quantisation (0 = off).
    scanlines : bool
        Overlay CRT-style scan lines (darken every other row).
    scanline_strength : float
        Darkening amount for scan lines (0 = invisible, 1 = fully black rows).
    dither_mode : str or None
        ``'bayer'`` = fast ordered dithering, ``'blue'`` = blue-noise ordered
        dithering, ``'fs'`` = Floyd-Steinberg, ``'riemersma'`` =
        Hilbert-curve error diffusion, ``None`` = skip dithering. All modes
        dither in linear light (sRGB gamma-correct) by default.
    dither_bits : int
        Bit depth used for dithering.
    dither_gamma : bool
        Dither in linear light (default). False = the crushed retro look.
    dither_invert : bool or "auto"
        Ordered-dither rounding bias (see dtouch.dither): "auto" flips on
        dark frames; True/False force it. Error diffusion self-corrects and
        ignores it.
    dither_size : int or None
        If set, dithering is computed at this image height (aspect-preserving)
        then up-scaled with nearest-neighbour — cheaper and adds a block-pixel
        aesthetic authentic to lo-fi hardware.
    """

    def __init__(
        self,
        seed: int | None = 0,
        chroma_shift: int = 10,
        scan_drift: int = 8,
        glitch_prob: float = 0.10,
        glitch_hold: int = 4,
        bit_crush: int = 0,
        scanlines: bool = True,
        scanline_strength: float = 0.35,
        dither_mode: str | None = "bayer",
        dither_bits: int = 3,
        dither_size: int | None = 72,
        dither_gamma: bool = True,
        dither_invert: bool | str = "auto",
    ):
        self._rng = np.random.default_rng(seed)
        self.chroma_shift = chroma_shift
        self.scan_drift = scan_drift
        self.glitch_prob = glitch_prob
        self.glitch_hold = glitch_hold
        self.bit_crush = bit_crush
        self.scanlines = scanlines
        self.scanline_strength = scanline_strength
        self.dither_mode = dither_mode
        self.dither_bits = dither_bits
        self.dither_size = dither_size
        self.dither_gamma = dither_gamma
        self.dither_invert = dither_invert

        # IIR-smoothed chroma offsets: target is re-sampled each frame then
        # low-pass filtered so the colour bleed drifts slowly, not jittering.
        self._chroma_r = 0.0
        self._chroma_b = 0.0
        # Accumulating phase for the scan-drift sine envelope (not a simple
        # counter — the increment is itself randomised each frame so it never
        # settles into a perfectly periodic pattern).
        self._scan_phase = 0.0

        # Glitch-block state: a sampled region is held for glitch_hold frames.
        self._glitch_rect: tuple[int, int, int, int] | None = None
        self._glitch_hold_rem: int = 0
        self._glitch_tile: np.ndarray | None = None

        self._frame_idx = 0

    # ------------------------------------------------------------------
    def process(self, frame: np.ndarray) -> np.ndarray:
        """Apply circuit-bent effects to a BGR uint8 frame.

        Parameters
        ----------
        frame : uint8 ndarray (H, W, 3), BGR colour order.

        Returns
        -------
        uint8 ndarray (H, W, 3) BGR, same spatial size as input.
        """
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("expected BGR (H, W, 3) uint8 frame")

        # Work in float32 [0, 1] throughout; single conversion at the end.
        out = frame.astype(np.float32) / 255.0
        h, w = out.shape[:2]

        # 1. Per-row horizontal drift (simulates lost horizontal-sync pulses).
        if self.scan_drift > 0:
            out = self._apply_scan_drift(out, h, w)

        # 2. Independent R / B channel horizontal offsets (colour-decoder bleed).
        if self.chroma_shift > 0:
            out = self._apply_chroma_shift(out, h, w)

        # 3. Glitch blocks: random region corruption that persists N frames.
        # Also call when glitch_prob is 0 but a glitch is still in its hold window
        # so live prob=0 changes drain the active glitch rather than freezing it.
        if self.glitch_prob > 0 or self._glitch_hold_rem > 0 or self._glitch_rect is not None:
            out = self._apply_glitch(out, h, w)

        # 4. Hard bit-depth reduction (quantisation / posterisation).
        if self.bit_crush > 0:
            levels = float((1 << self.bit_crush) - 1)
            out = np.round(out * levels) / levels

        # 5. Dithering (Bayer or Floyd-Steinberg, optionally at reduced res).
        out = self._apply_dither(out, h, w)

        # 6. CRT scan-line overlay. The line pitch scales with the output so
        # the lines stay visible at any resolution: a fixed every-other-row
        # mask is 1px at 1080p/4K and vanishes entirely the moment the frame
        # is scaled down (window fit, projector scaler, screen capture) —
        # adjacent rows average to a uniform dim. Authored as ~SCANLINE_ROWS
        # visible line pairs per frame; a GPU port mirrors this as
        # floor(uv.y * SCANLINE_ROWS * 2) % 2.
        if self.scanlines:
            p = max(2, round(h / SCANLINE_ROWS))
            ys = np.arange(h) % p < (p // 2)
            mask = np.ones((h, 1, 1), np.float32)
            mask[ys] = 1.0 - self.scanline_strength
            out = out * mask

        out = np.clip(out, 0.0, 1.0)
        self._frame_idx += 1
        return (out * 255.0).astype(np.uint8)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _apply_scan_drift(self, out: np.ndarray, h: int, w: int) -> np.ndarray:
        """Drift each row horizontally by a correlated, slowly-evolving amount."""
        rng = self._rng
        # Two overlapping sine waves at incommensurate frequencies create a
        # drift envelope that never repeats. The phase increment is randomised
        # each frame so the pattern wanders rather than oscillating cleanly.
        self._scan_phase += float(rng.uniform(0.01, 0.04))
        ys = np.linspace(0.0, 1.0, h, dtype=np.float32)
        envelope = (
            np.sin(ys * 2.0 * np.pi * 2.3 + self._scan_phase) *
            np.sin(ys * 2.0 * np.pi * 0.7 + self._scan_phase * 0.4)
        )
        # Small per-row noise keeps adjacent rows from being perfectly coupled.
        noise = rng.standard_normal(h).astype(np.float32) * 0.15
        drift_px = ((envelope + noise) * self.scan_drift).astype(np.int32)

        # Occasional "sync tear": a horizontal band of rows slips far at once.
        if rng.random() < 0.04:
            band_h = max(1, h // 8)
            y0 = int(rng.integers(0, max(1, h - band_h)))
            tear_amt = int(rng.integers(-self.scan_drift * 3, self.scan_drift * 3 + 1))
            drift_px[y0:y0 + band_h] += tear_amt

        for y in range(h):
            shift = int(drift_px[y])
            if shift:
                out[y] = np.roll(out[y], shift, axis=0)
        return out

    def _apply_chroma_shift(self, out: np.ndarray, h: int, w: int) -> np.ndarray:
        """Displace R and B channels by slowly-drifting IIR-filtered offsets."""
        rng = self._rng
        alpha = 0.85   # smoothing coefficient: 0 = instant, 1 = frozen
        self._chroma_r = (alpha * self._chroma_r +
                          (1.0 - alpha) * float(rng.uniform(-self.chroma_shift, self.chroma_shift)))
        self._chroma_b = (alpha * self._chroma_b +
                          (1.0 - alpha) * float(rng.uniform(-self.chroma_shift, self.chroma_shift)))
        shift_r = int(round(self._chroma_r))
        shift_b = int(round(self._chroma_b))
        # BGR layout: channel 0 = B, channel 2 = R.
        if shift_r:
            out[:, :, 2] = np.roll(out[:, :, 2], shift_r, axis=1)
        if shift_b:
            out[:, :, 0] = np.roll(out[:, :, 0], shift_b, axis=1)
        return out

    def _apply_glitch(self, out: np.ndarray, h: int, w: int) -> np.ndarray:
        """Corrupt a random rectangular region; hold it for several frames."""
        rng = self._rng
        if self._glitch_hold_rem > 0:
            self._glitch_hold_rem -= 1
        elif rng.random() < self.glitch_prob:
            bh = int(rng.integers(max(1, h // 16), max(2, h // 4)))
            bw = int(rng.integers(max(1, w // 8),  max(2, w // 2)))
            y0 = int(rng.integers(0, max(1, h - bh)))
            x0 = int(rng.integers(0, max(1, w - bw)))
            sy = int(rng.integers(0, max(1, h - bh)))
            sx = int(rng.integers(0, max(1, w - bw)))
            # Freeze a tile from the current frame and replay it at (y0, x0).
            self._glitch_tile = out[sy:sy + bh, sx:sx + bw].copy()
            self._glitch_rect = (y0, x0, bh, bw)
            self._glitch_hold_rem = self.glitch_hold
        else:
            # Hold expired and no new glitch fired: clear so the tile stops replaying.
            self._glitch_rect = None
            self._glitch_tile = None

        if self._glitch_rect is not None and self._glitch_tile is not None:
            y0, x0, bh, bw = self._glitch_rect
            th, tw = self._glitch_tile.shape[:2]
            ah = min(bh, th, h - y0)
            aw = min(bw, tw, w - x0)
            if ah > 0 and aw > 0:
                out[y0:y0 + ah, x0:x0 + aw] = self._glitch_tile[:ah, :aw]
        return out

    def _apply_dither(self, out: np.ndarray, h: int, w: int) -> np.ndarray:
        """Optionally dither, at a reduced resolution for the block-pixel look."""
        if not self.dither_mode:
            return out
        if self.dither_size and self.dither_size < h:
            dh = self.dither_size
            dw = max(1, int(w * dh / h))
            small = cv2.resize(out, (dw, dh), interpolation=cv2.INTER_LINEAR)
            small = self._dither_array(small)
            return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        return self._dither_array(out)

    def _dither_array(self, img: np.ndarray) -> np.ndarray:
        bits, gamma = self.dither_bits, self.dither_gamma
        if self.dither_mode == "bayer":
            return bayer_dither(img, bits=bits, invert=self.dither_invert,
                                gamma=gamma)
        if self.dither_mode == "blue":
            return blue_noise_dither(img, bits=bits, invert=self.dither_invert,
                                     gamma=gamma)
        if self.dither_mode == "riemersma":
            # error diffusion self-corrects — no invert parameter
            return riemersma_dither(img, bits=bits, gamma=gamma)
        if self.dither_mode == "fs":
            result = np.empty_like(img)
            for c in range(img.shape[2]):
                result[:, :, c] = floyd_steinberg(img[:, :, c], bits=bits,
                                                  gamma=gamma)
            return result
        return img
