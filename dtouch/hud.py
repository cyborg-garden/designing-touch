"""HUD layer — u-unit geometry, outlined text, toasts, OSD, status line, corner tick.

The perform surface (DESIGN.md §5–6): everything here is authored in the u-unit
(`u = frame_height / 45` — ~16 px @720p, 24 @1080p, 48 @4K), double-drawn for
worst-case contrast (black under near-white — legible against a white wall), and
transient except the status line and the blackout corner tick.

Overlay states (DESIGN.md §6.1): HIDDEN (provably clean output — the OBS capture
contract; nothing draws once toasts expire) → HUD (status line + momentary
toasts/OSD) → PANEL (full sidebar; HUD still active). TAB cycles forward, Esc
always steps toward HIDDEN and never quits.

Everything draws onto small ROIs (text boxes, a corner triangle) — no full-frame
ops — to hold the ≤1 ms/frame overlay budget. ASCII text only (Hershey fonts
render non-ASCII as '?').
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np

# BGR — matched to the panel's chrome (overlay_ui) plus the one amber addition.
INK = (215, 222, 218)
DIM = (140, 150, 145)
ACC = (120, 215, 140)
RED = (70, 70, 235)
AMBER = (0, 191, 255)          # armed / warning / transient (DESIGN.md §5)
OUTLINE = (0, 0, 0)

FONT = cv2.FONT_HERSHEY_SIMPLEX
_CAP_PX = 22.0                 # Hershey simplex cap height at fontScale 1.0

TITLE_SAFE = 0.035             # 3.5% title-safe inset (DESIGN.md §5)

# Camera-loss note — the existing human message, kept verbatim (DESIGN.md §6.4).
CAMERA_NOTE = "CAMERA IS BLACK - disable iPhone Continuity Camera"
CAMERA_FIX = "iPhone: Settings > General > AirPlay & Handoff > Continuity Camera > Off"

CENTER_TOAST_S = 1.2           # mode/center flash fade (DESIGN.md §5 type table)
HINT_TOAST_S = 1.5
OSD_FADE_S = 1.5


def u(frame_h):
    """The u-unit: frame-relative geometry base (DESIGN.md §5)."""
    return frame_h / 45.0


class OverlayState(Enum):
    HIDDEN = 0
    HUD = 1
    PANEL = 2


def cycle_overlay(state):
    """TAB: HIDDEN -> HUD -> PANEL -> HIDDEN."""
    return {OverlayState.HIDDEN: OverlayState.HUD,
            OverlayState.HUD: OverlayState.PANEL,
            OverlayState.PANEL: OverlayState.HIDDEN}[state]


def esc_overlay(state):
    """Esc: one step toward HIDDEN; in HIDDEN it does nothing. Esc never quits."""
    return {OverlayState.PANEL: OverlayState.HUD,
            OverlayState.HUD: OverlayState.HIDDEN,
            OverlayState.HIDDEN: OverlayState.HIDDEN}[state]


def _font(px):
    scale = px / _CAP_PX
    thick = max(1, int(round(scale)))
    return scale, thick


def put_outlined(img, text, org, px, color=INK, thick_mul=1.0):
    """Double-drawn Hershey text: black under the ink, so it survives any background."""
    scale, thick = _font(px)
    thick = max(1, int(round(thick * thick_mul)))
    cv2.putText(img, text, org, FONT, scale, OUTLINE, thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, FONT, scale, color, thick, cv2.LINE_AA)


def text_size(text, px):
    scale, thick = _font(px)
    (tw, th), base = cv2.getTextSize(text, FONT, scale, thick + 2)
    return tw, th, base


def blend_outlined(img, text, org, px, color, alpha):
    """Outlined text at partial opacity, blended over a small ROI only (budget)."""
    if alpha <= 0.0:
        return
    if alpha >= 0.999:
        put_outlined(img, text, org, px, color)
        return
    h, w = img.shape[:2]
    tw, th, base = text_size(text, px)
    m = 4  # outline margin
    x0 = max(org[0] - m, 0); y0 = max(org[1] - th - m, 0)
    x1 = min(org[0] + tw + m, w); y1 = min(org[1] + base + m, h)
    if x1 <= x0 or y1 <= y0:
        return
    roi = img[y0:y1, x0:x1]
    over = roi.copy()
    put_outlined(over, text, (org[0] - x0, org[1] - y0), px, color)
    cv2.addWeighted(over, alpha, roi, 1.0 - alpha, 0, dst=roi)


def draw_corner_tick(img, size_px, color=AMBER):
    """Filled corner triangle, top-right, legs `size_px` — the one persistent
    element allowed on a blacked-out frame (DESIGN.md §5: blackout armed)."""
    h, w = img.shape[:2]
    s = int(round(size_px))
    pts = np.array([[w - s, 0], [w, 0], [w, s]], np.int32)
    cv2.fillConvexPoly(img, pts, color, cv2.LINE_AA)


@dataclass
class _Toast:
    text: str
    color: tuple
    t0: float
    ttl: float

    def alpha(self, now):
        left = self.ttl - (now - self.t0)
        if left <= 0:
            return 0.0
        return min(1.0, left / (0.45 * self.ttl))   # hold, then fade out


class Toasts:
    """Momentary feedback: one big center flash + small stacked hints.
    Silence-on-input is a bug (DESIGN.md principle 4)."""

    def __init__(self, now=time.monotonic):
        self._now = now
        self._center = None
        self._hints = []

    def flash(self, text, color=INK):
        """Center 3.0u flash (mode change, RESET, BLACKOUT...). Replaces the last."""
        self._center = _Toast(text, color, self._now(), CENTER_TOAST_S)

    def hint(self, text, color=DIM):
        """Small 0.75u hint ('? for keys', 'q again to quit'...)."""
        self._hints.append(_Toast(text, color, self._now(), HINT_TOAST_S))
        self._hints = self._hints[-3:]

    def active(self):
        now = self._now()
        if self._center is not None and self._center.alpha(now) > 0:
            return True
        return any(t.alpha(now) > 0 for t in self._hints)

    def draw(self, img):
        now = self._now()
        h, w = img.shape[:2]
        uu = u(h)
        if self._center is not None:
            a = self._center.alpha(now)
            if a <= 0:
                self._center = None
            else:
                px = int(3.0 * uu)
                tw, _, _ = text_size(self._center.text, px)
                blend_outlined(img, self._center.text, ((w - tw) // 2, (h + px) // 2),
                               px, self._center.color, a)
        self._hints = [t for t in self._hints if t.alpha(now) > 0]
        px = int(0.75 * uu)
        y = h - int(h * TITLE_SAFE)
        for t in reversed(self._hints):
            tw, _, _ = text_size(t.text, px)
            blend_outlined(img, t.text, ((w - tw) // 2, y), px, t.color, t.alpha(now))
            y -= int(1.5 * px)


class Osd:
    """Param readout: name + value + bar, 1.5u, bottom-left, fades in 1.5 s."""

    def __init__(self, now=time.monotonic):
        self._now = now
        self._show = None      # (name, value_text, fill 0..1 or None, t0)

    def show(self, name, value, lo=None, hi=None, fmt="{:.2f}"):
        fill = None
        if lo is not None and hi is not None and hi > lo:
            fill = min(max((value - lo) / (hi - lo), 0.0), 1.0)
        text = fmt.format(value) if not isinstance(value, str) else value
        self._show = (name, text, fill, self._now())

    def draw(self, img):
        if self._show is None:
            return
        name, text, fill, t0 = self._show
        left = OSD_FADE_S - (self._now() - t0)
        if left <= 0:
            self._show = None
            return
        a = min(1.0, left / (0.45 * OSD_FADE_S))
        h, w = img.shape[:2]
        uu = u(h)
        px = int(1.5 * uu)
        x = int(w * TITLE_SAFE)
        y = h - int(h * TITLE_SAFE) - int(1.2 * uu)
        blend_outlined(img, f"{name}  {text}", (x, y), px, INK, a)
        if fill is not None:
            bw, bh = int(10 * uu), max(2, int(0.5 * uu))
            y0 = y + int(0.4 * uu)
            roi = img[y0:y0 + bh, x:x + bw]
            over = roi.copy()
            cv2.rectangle(over, (0, 0), (bw - 1, bh - 1), OUTLINE, -1)
            cv2.rectangle(over, (0, 0), (int(bw * fill), bh - 1), INK, -1)
            cv2.rectangle(over, (0, 0), (bw - 1, bh - 1), DIM, 1)
            cv2.addWeighted(over, a, roi, 1.0 - a, 0, dst=roi)


class Hud:
    """The perform-surface renderer. Draw AFTER the recorder write — recordings
    never contain HUD/panel (the existing ordering invariant, kept)."""

    def __init__(self, now=time.monotonic):
        self._now = now
        self.toasts = Toasts(now)
        self.osd = Osd(now)
        self.debug = False     # 'i' — status line variant: fps / frame-time / res

    def draw(self, img, state, status="", debug_status="", recording=False,
             blackout=False, camera_lost=False):
        h, w = img.shape[:2]
        uu = u(h)
        if blackout:
            draw_corner_tick(img, 2.0 * uu)   # all states, including HIDDEN
        if state is OverlayState.HIDDEN:
            # Provably clean: only already-ticking toasts, then pixels untouched.
            self.toasts.draw(img)
            return img
        ix, iy = int(w * TITLE_SAFE), int(h * TITLE_SAFE)
        px = int(0.75 * uu)
        line = debug_status if self.debug else status
        if line:
            put_outlined(img, line, (ix, iy + px), px, DIM)
        if recording:
            tw = text_size(line, px)[0] if line else 0
            cv2.circle(img, (ix + tw + int(0.9 * uu), iy + px // 2 + 2),
                       max(2, int(0.3 * uu)), RED, -1, cv2.LINE_AA)
        if camera_lost:
            npx = int(1.0 * uu)
            put_outlined(img, CAMERA_NOTE, (ix, h // 2), npx, AMBER)
            put_outlined(img, CAMERA_FIX, (ix, h // 2 + int(1.4 * npx)),
                         int(0.75 * uu), AMBER)
        self.toasts.draw(img)
        self.osd.draw(img)
        return img
