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

from dataclasses import dataclass

import cv2
import numpy as np

from .dither import bayer_dither, blue_noise_dither, floyd_steinberg, riemersma_dither

# The CRT look is authored at ~270 scan-line periods per frame: the mask
# pitch is round(h / SCANLINE_ROWS) rows, floored at 2 (so up to 540p this
# is the shipped every-other-row mask; 1080p gets a 4-row pitch, 4K 8-row —
# the lines survive window fit / projector scalers instead of averaging out).
SCANLINE_ROWS = 270


@dataclass
class SignalPlan:
    """One frame's stochastic control values, drawn by CircuitBent.plan().

    The rack's randomness (scan-drift envelope, chroma IIR, glitch events)
    lives in plan(); the pixel work is a pure function of the plan. Both the
    numpy path (apply_plan) and the GPU path (dtouch.rack_gl.SignalRackGL)
    consume the same plan, so the two backends bend the picture identically
    for the same seed and settings.
    """
    drift_px: "np.ndarray | None" = None       # int32 (h,) per-row shift, or None
    shift_r: int = 0                           # channel 2 horizontal shift
    shift_b: int = 0                           # channel 0 horizontal shift
    glitch_capture: "tuple[int, int, int, int] | None" = None   # (sy, sx, th, tw)
    glitch_rect: "tuple[int, int, int, int] | None" = None      # (y0, x0, ah, aw)
    glitch_seq: int = 0        # which glitch event the rect belongs to — a
                               # backend replays only a tile it captured for
                               # THIS event (engine switches mid-hold skip)


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
        self._glitch_src: tuple[int, int] | None = None   # captured tile (th, tw)
        self._glitch_seq: int = 0                         # fires seen so far

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

        h, w = frame.shape[:2]
        plan = self.plan(h, w)
        # Work in float32 [0, 1] throughout; single conversion at the end.
        out = self.apply_plan(frame.astype(np.float32) / 255.0, plan)
        return (out * 255.0).astype(np.uint8)

    # ------------------------------------------------------------------
    def plan(self, h: int, w: int) -> SignalPlan:
        """Advance every stochastic control one frame and return the frame's
        control values (per-row drift, chroma offsets, glitch events).

        All RNG draws, IIR filters, and glitch-hold state live HERE, in the
        exact order the classic process() made them, so a fixed seed yields
        the same sequence whichever backend applies the plan. Pixel work is
        a pure function of the plan: apply_plan() is the numpy backend and
        dtouch.rack_gl.SignalRackGL is the fragment-pass backend.
        """
        p = SignalPlan()

        # 1. Per-row horizontal drift (simulates lost horizontal-sync pulses).
        if self.scan_drift > 0:
            p.drift_px = self._plan_scan_drift(h)

        # 2. Independent R / B channel horizontal offsets (colour-decoder bleed).
        if self.chroma_shift > 0:
            p.shift_r, p.shift_b = self._plan_chroma()

        # 3. Glitch blocks: random region corruption that persists N frames.
        # Also planned when glitch_prob is 0 but a glitch is still in its hold
        # window so live prob=0 changes drain the active glitch rather than
        # freezing it.
        if (self.glitch_prob > 0 or self._glitch_hold_rem > 0
                or self._glitch_rect is not None):
            p.glitch_capture, p.glitch_rect = self._plan_glitch(h, w)
            p.glitch_seq = self._glitch_seq

        self._frame_idx += 1
        return p

    def apply_plan(self, out: np.ndarray, plan: SignalPlan) -> np.ndarray:
        """Apply one frame's plan to a float32 [0, 1] (H, W, 3) image — the
        numpy pixel backend. Returns float32 [0, 1], clipped. The stage
        order is the rack's contract: drift, chroma, glitch, crush, dither,
        scanlines (SignalRackGL mirrors it pass for pass)."""
        h, w = out.shape[:2]

        if plan.drift_px is not None:
            for y in range(h):
                shift = int(plan.drift_px[y])
                if shift:
                    out[y] = np.roll(out[y], shift, axis=0)

        # BGR layout: channel 0 = B, channel 2 = R (an RGB caller only swaps
        # which channel drifts left vs right — the draws are symmetric).
        if plan.shift_r:
            out[:, :, 2] = np.roll(out[:, :, 2], plan.shift_r, axis=1)
        if plan.shift_b:
            out[:, :, 0] = np.roll(out[:, :, 0], plan.shift_b, axis=1)

        if plan.glitch_capture is not None:
            sy, sx, th, tw = plan.glitch_capture
            # Freeze a tile from the current (post-chroma) frame.
            self._glitch_tile = out[sy:sy + th, sx:sx + tw].copy()
        if plan.glitch_rect is not None and self._glitch_tile is not None:
            y0, x0, ah, aw = plan.glitch_rect
            out[y0:y0 + ah, x0:x0 + aw] = self._glitch_tile[:ah, :aw]

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

        return np.clip(out, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _plan_scan_drift(self, h: int) -> np.ndarray:
        """Per-row horizontal shift for this frame — a correlated,
        slowly-evolving int32 (h,) array."""
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
        return drift_px

    def _plan_chroma(self) -> tuple[int, int]:
        """(shift_r, shift_b) — slowly-drifting IIR-filtered channel offsets."""
        rng = self._rng
        alpha = 0.85   # smoothing coefficient: 0 = instant, 1 = frozen
        self._chroma_r = (alpha * self._chroma_r +
                          (1.0 - alpha) * float(rng.uniform(-self.chroma_shift, self.chroma_shift)))
        self._chroma_b = (alpha * self._chroma_b +
                          (1.0 - alpha) * float(rng.uniform(-self.chroma_shift, self.chroma_shift)))
        return int(round(self._chroma_r)), int(round(self._chroma_b))

    def _plan_glitch(self, h: int, w: int):
        """Advance the glitch state machine one frame.

        Returns (capture, rect): `capture` = (sy, sx, th, tw) when a NEW
        glitch fires this frame (the backend must freeze that tile from the
        current post-chroma image), else None; `rect` = (y0, x0, ah, aw) to
        overlay the held tile onto, else None. Each backend keeps its own
        frozen tile (numpy array here, a GL texture in SignalRackGL) and
        skips the overlay when it holds none — e.g. after an engine switch
        mid-hold."""
        rng = self._rng
        capture = None
        if self._glitch_hold_rem > 0:
            self._glitch_hold_rem -= 1
        elif rng.random() < self.glitch_prob:
            bh = int(rng.integers(max(1, h // 16), max(2, h // 4)))
            bw = int(rng.integers(max(1, w // 8),  max(2, w // 2)))
            y0 = int(rng.integers(0, max(1, h - bh)))
            x0 = int(rng.integers(0, max(1, w - bw)))
            sy = int(rng.integers(0, max(1, h - bh)))
            sx = int(rng.integers(0, max(1, w - bw)))
            # Freeze a tile from the current frame; replay it at (y0, x0).
            th, tw = min(bh, h - sy), min(bw, w - sx)   # numpy-slice clamp
            capture = (sy, sx, th, tw)
            self._glitch_rect = (y0, x0, bh, bw)
            self._glitch_src = (th, tw)
            self._glitch_seq += 1
            self._glitch_hold_rem = self.glitch_hold
        else:
            # Hold expired and no new glitch fired: clear so the tile stops replaying.
            self._glitch_rect = None
            self._glitch_tile = None
            self._glitch_src = None

        rect = None
        if self._glitch_rect is not None:
            y0, x0, bh, bw = self._glitch_rect
            th, tw = self._glitch_src if self._glitch_src is not None else (bh, bw)
            ah = min(bh, th, h - y0)
            aw = min(bw, tw, w - x0)
            if ah > 0 and aw > 0:
                rect = (y0, x0, ah, aw)
        return capture, rect

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
