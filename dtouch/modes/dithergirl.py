"""Dither Girl — the dither pipeline as the primary image (DESIGN.md §4.2).

camera or still → optional matte gate → gamma-correct dither at the chosen
working scale → palette. The SIGNAL rack is available on top, minus its dither
row (§2.4: this mode claims "dither" — the shell hides any rack control the
active mode claims, and runs CircuitBent without its dither stage).

Pure numpy/cv2 — no GL — so the mode runs fully headless (tests, CI, and the
one-hand dark-room rig alike). Engines are trivial here: the only stateful
object is the matte operator, rebuilt when the matte cycle changes.

Perf honesty (§4.2): the default working height (72 px) keeps all four
algorithms fast; ordered dithers are permitted full-res; when Scale is dragged
high while an error-diffusion algorithm is active, an inline amber note reads
'slow - ordered dither recommended live'. Never silent frame drops.

ASCII text only in panel strings (Hershey fonts render non-ASCII as '?').
"""
from __future__ import annotations

import cv2
import numpy as np

from ..dither import (bayer_dither, blue_noise_dither, floyd_steinberg,
                      riemersma_dither)
from ..hud import AMBER, put_outlined, text_size, u as _u
from ..imgui import DIM
from ..matte import make_matte
from ..overlay_ui import RES_OPTIONS
from ..panelspec import Cycle, PresetList, Readout, Section, Slider, Toggle
from .particles import MATTE_H, MATTE_W, MATTES

# Accent (DESIGN.md §5): magenta-leaning, picked by measured WCAG contrast —
# BGR (245, 140, 245) = RGB (245, 140, 245). Against the panel scrim fill
# (PANEL (34,32,30) at 86%) it measures 7.75:1, and 4.99:1 against the worst
# case of a pure-white frame under that scrim — both over the 4.5:1 floor.
# (Pure magenta (255,0,255) fails the bright-scrim case at 3.36:1.)
ACCENT = (245, 140, 245)

ALGOS = ["Bayer", "Blue noise", "Floyd-Steinberg", "Riemersma"]
ORDERED = ("Bayer", "Blue noise")     # per-pixel threshold — safe at full res
BIASES = ["auto", "light", "dark"]    # ordered-dither rounding bias (invert)
_BIAS_INVERT = {"auto": "auto", "light": False, "dark": True}

# Palette (§4.2): map the dither's off/on levels to two colors (multi-bit
# output interpolates between them). RGB (step() returns RGB at host.res).
PALETTES = {
    "white-on-black": ((0, 0, 0), (255, 255, 255)),
    "black-on-white": ((245, 245, 245), (16, 16, 16)),
    "amber": ((24, 12, 0), (255, 176, 0)),
    "green phosphor": ((0, 20, 8), (80, 255, 120)),
}

MATTES_DG = ["off"] + MATTES          # "off" = dither the whole frame

SCALE_LO, SCALE_HI, SCALE_DEFAULT = 45.0, 720.0, 72.0
SLOW_SCALE = 180.0                    # error-diffusion above this = amber note


def _dither(gray, algo, bits, gamma, bias):
    """One grayscale float [0,1] plane through the named algorithm. Bias
    (rounding direction) applies to the ordered dithers only — error diffusion
    self-corrects and takes no invert parameter."""
    if algo == "Bayer":
        return bayer_dither(gray, bits=bits, invert=_BIAS_INVERT[bias], gamma=gamma)
    if algo == "Blue noise":
        return blue_noise_dither(gray, bits=bits, invert=_BIAS_INVERT[bias], gamma=gamma)
    if algo == "Floyd-Steinberg":
        return floyd_steinberg(gray, bits=bits, gamma=gamma)
    return riemersma_dither(gray, bits=bits, gamma=gamma)


class DitherGirlMode:
    """Live + still dithering as a shell plugin (Mode protocol, DESIGN.md §2.2)."""

    id = "dithergirl"
    title = "Dither Girl"
    accent = ACCENT
    accepts_still = True
    blurb = "live + still\ndithering"      # home-menu card copy (DESIGN.md §3)
    # §2.4: the rack hides what we own — Dither Girl owns ALL dither quality
    # controls (dither row + Bits/Gamma/Bias), not just the dither cycle; two
    # visible dither subsystems in one panel is the bolted-features
    # incoherence the overhaul exists to kill.
    claims = frozenset({"dither", "bits", "gamma", "bias"})

    # option lists the shell's boot path reads (OverlayUI ctor)
    palettes = list(PALETTES)
    mattes = list(MATTES_DG)
    matte_kind = "off"

    # Built-in looks (DESIGN.md §7): in code, immune to rename/delete,
    # bankable. 'stream-safe' is the stream-tuned variant — bigger cells
    # (low Scale) + higher contrast survive stream compression (§4.2).
    BUILTIN = {
        "classic": dict(algorithm="Floyd-Steinberg", bits=1.0, gamma=True,
                        bias="auto", contrast=1.0, scale=72.0,
                        palette="white-on-black", matte="off"),
        "newsprint": dict(algorithm="Floyd-Steinberg", bits=1.0, gamma=True,
                          bias="light", contrast=1.15, scale=180.0,
                          palette="black-on-white", matte="off"),
        "phosphor": dict(algorithm="Bayer", bits=2.0, gamma=True,
                         bias="auto", contrast=1.1, scale=144.0,
                         palette="green phosphor", matte="off"),
        "stream-safe": dict(algorithm="Blue noise", bits=1.0, gamma=True,
                            bias="auto", contrast=1.6, scale=56.0,
                            palette="white-on-black", matte="off"),
        "riemersma still": dict(algorithm="Riemersma", bits=2.0, gamma=True,
                                bias="auto", contrast=1.0, scale=240.0,
                                palette="white-on-black", matte="off"),
    }

    # apply="reset" merges a look over these (store keys — DESIGN.md §7)
    DEFAULTS = dict(algorithm="Floyd-Steinberg", bits=1.0, bias="auto",
                    contrast=1.0, scale=SCALE_DEFAULT,
                    palette="white-on-black", matte="off")

    # shared-UI attrs this mode seeds (prefixed to coexist with every mode's
    # attrs on the one OverlayUI state object)
    _UI_DEFAULTS = dict(input_idx=0, dg_matte_idx=0, dg_matte_black=False,
                        dg_algo_idx=ALGOS.index("Floyd-Steinberg"),
                        dg_bits=1.0, dg_gamma=True, dg_bias_idx=0,
                        dg_contrast=1.0, dg_scale=SCALE_DEFAULT,
                        dg_palette_idx=0)

    def __init__(self, still=False):
        self.boot_still = still            # CLI --still: boot with still input
        self.host = None
        self.mat = None
        self._mat_kind = None
        self._swatch_key = None
        self._swatch_cache = None

    # ----- lifecycle -----
    def start(self, host):
        self.host = host

    def stop(self):
        """Idempotent (DESIGN.md §2.2). No GL — just drop the matte."""
        self.mat = None
        self._mat_kind = None

    def on_resize(self, w, h):
        pass                               # everything derives from host.res

    # ----- ui plumbing -----
    def configure_ui(self, ui):
        """Seed this mode's attrs on the shared UI state (first entry only —
        switching away and back preserves the operator's settings, including
        the input cycle: --still applies on the first entry, then the
        operator's choice sticks)."""
        first = not hasattr(ui, "input_idx")
        for k, v in self._UI_DEFAULTS.items():
            if not hasattr(ui, k):
                setattr(ui, k, v)
        if first and self.boot_still:
            ui.input_idx = 1

    def _ui(self, attr, default):
        ui = self.host.ui if self.host is not None else None
        return getattr(ui, attr, default) if ui is not None else default

    # current values (int-snapped / wrapped where the control is continuous)
    def _algo(self):
        return ALGOS[int(self._ui("dg_algo_idx", self._UI_DEFAULTS["dg_algo_idx"]))
                     % len(ALGOS)]

    def _bits(self):
        return int(np.clip(round(self._ui("dg_bits", 1.0)), 1, 4))

    def _bias(self):
        return BIASES[int(self._ui("dg_bias_idx", 0)) % len(BIASES)]

    def _palette_name(self):
        return self.palettes[int(self._ui("dg_palette_idx", 0)) % len(self.palettes)]

    def _matte_name(self):
        return MATTES_DG[int(self._ui("dg_matte_idx", 0)) % len(MATTES_DG)]

    def slow_warning(self):
        """True when the inline amber perf note should render (§4.2): an
        error-diffusion algorithm with Scale dragged high."""
        return (self._algo() not in ORDERED
                and self._ui("dg_scale", SCALE_DEFAULT) > SLOW_SCALE)

    # ----- panel (DESIGN.md §4.2) -----
    def panel_spec(self):
        """TEMPLATES / SOURCE / ALGORITHM / TONE / PALETTE; the shell appends
        the SIGNAL rack (minus our claimed rows) + global rows (minus our own
        Mirror). TEMPLATES is the generic PresetList (DESIGN.md §7): built-ins
        as rows, user looks per mode, recall/save/rename/delete/slot badges
        all shell-owned."""
        return [
            Section("TEMPLATES", [PresetList()]),
            Section("SOURCE", [
                Cycle("input", "input_idx", ["camera", "still..."], save=False),
                Cycle("matte", "dg_matte_idx", list(MATTES_DG), save_key="matte"),
                Toggle("Matte bg black", "dg_matte_black", save_key="matte_black",
                       tip="With a matte on: black outside the subject instead "
                           "of the raw camera picture."),
                Cycle("output", "res_idx", [n for n, _, _ in RES_OPTIONS],
                      key="res", save=False),
                Toggle("Mirror", "mirror", on_text="on", save=False),
            ]),
            Section("ALGORITHM", [
                Readout(self._draw_algo_label),
                Cycle("algorithm", "dg_algo_idx", list(ALGOS), save_key="algorithm",
                      status=str.lower),
                Readout(self._draw_speed_badge),
                Readout(self._draw_swatch),
            ]),
            Section("TONE", [
                Slider("Bits", "dg_bits", 1.0, 4.0, fmt=".0f", save_key="bits",
                       status="{:.0f}-bit",
                       tip="Output bit depth. 1 = pure two-tone; higher keeps "
                           "more shades."),
                Toggle("Gamma", "dg_gamma", save_key="gamma",
                       tip="Dither in linear light so mid-tones keep their "
                           "perceived brightness. Off = the crushed retro look."),
                Cycle("bias", "dg_bias_idx", list(BIASES), save_key="bias",
                      status="bias {}"),
                Slider("Contrast", "dg_contrast", 0.25, 3.0, save_key="contrast",
                       tip="Push tones apart before dithering. High contrast "
                           "survives stream compression."),
                Slider("Scale", "dg_scale", SCALE_LO, SCALE_HI, fmt=".0f",
                       save_key="scale",
                       tip="Working height in pixels. Low = big chunky cells; "
                           "high = fine grain (slow for the diffusion dithers)."),
                Readout(self._draw_perf_note),
            ]),
            Section("PALETTE", [
                Cycle("palette", "dg_palette_idx", list(PALETTES),
                      save_key="palette"),
            ]),
        ]

    def commands(self):
        return {}                          # no mode-local perform keys (v1)

    def safe_look(self):
        """The panic target (DESIGN.md §6.2 '0')."""
        return "classic"

    def status_tail(self, cam_name):
        """The source tail of the spec-derived HUD status (DESIGN.md §2.3 —
        the body renders from the status-marked spec widgets)."""
        src = "still" if self._ui("input_idx", 0) == 1 else cam_name[:16]
        return f"src {src}"

    # ----- panel readouts -----
    def _draw_algo_label(self, frame, g, x, y, cw):
        """Big active-algorithm label — 1.4u in the mode accent (§4.2),
        shrunk to fit the column when the name is long (FLOYD-STEINBERG)."""
        name = self._algo().upper()
        px = int(1.4 * _u(frame.shape[0]))
        tw = text_size(name, px)[0]
        if tw > cw:
            px = max(int(px * cw / tw), 10)
        if 0 <= y and y + px <= frame.shape[0]:
            put_outlined(frame, name, (x, y + px), px, self.accent)
        return y + px + g.S(8)

    def _draw_speed_badge(self, frame, g, x, y, cw):
        """live/slow badge: ordered = 'live'; error diffusion = 'slow at full
        res' (§4.2)."""
        if self._algo() in ORDERED:
            g.text(frame, "live", x, y + g.S(12), DIM, 0.42)
        else:
            g.text(frame, "slow at full res", x, y + g.S(12), AMBER, 0.42)
        return y + g.S(20)

    def _draw_perf_note(self, frame, g, x, y, cw):
        """Inline amber perf note — renders only when Scale is dragged high
        while an error-diffusion algorithm is active (§4.2). Word-wrapped to
        the panel column (at 720p the one-liner overflows the sidebar)."""
        if not self.slow_warning():
            return y
        words = "slow - ordered dither recommended live".split()
        lines, cur = [], ""
        for wd in words:
            cand = (cur + " " + wd).strip()
            tw = cv2.getTextSize(cand, cv2.FONT_HERSHEY_SIMPLEX,
                                 0.42 * g.s, 1)[0][0]
            if cur and tw > cw:
                lines.append(cur)
                cur = wd
            else:
                cur = cand
        if cur:
            lines.append(cur)
        for line in lines:
            g.text(frame, line, x, y + g.S(12), AMBER, 0.42)
            y += g.S(16)
        return y + g.S(4)

    def _draw_swatch(self, frame, g, x, y, cw):
        """Swatch strip: a horizontal 0->1 gradient ramp rendered through the
        CURRENT algorithm+bits+gamma+bias (+palette), cached and re-rendered
        only when one of those changes (§4.2 — instant 'what am I hearing'
        for the eyes)."""
        hpx = g.S(16)
        key = (self._algo(), self._bits(), bool(self._ui("dg_gamma", True)),
               self._bias(), self._palette_name(), cw, hpx)
        if key != self._swatch_key:
            self._swatch_cache = self._render_swatch(*key)
            self._swatch_key = key
        # clipped blit (the panel scrolls; a Readout must not write off-frame)
        y0, y1 = max(y, 0), min(y + hpx, frame.shape[0])
        if y1 > y0:
            frame[y0:y1, x:x + cw] = self._swatch_cache[y0 - y:y1 - y]
        return y + hpx + g.S(8)

    def _render_swatch(self, algo, bits, gamma, bias, palette, w, h):
        ramp = np.tile(np.linspace(0.0, 1.0, w, dtype=np.float32), (h, 1))
        lit = _dither(ramp, algo, bits, gamma, bias)
        off, on = PALETTES[palette]
        rgb = self._palette_map(lit, off, on)
        return rgb[:, :, ::-1].copy()      # panel frames are BGR

    # ----- per-frame -----
    @staticmethod
    def _palette_map(levels, off, on):
        """Map dither levels [0,1] onto the off->on color ramp (uint8 RGB)."""
        off = np.float32(off)
        on = np.float32(on)
        return (off[None, None, :]
                + levels[:, :, None] * (on - off)).astype(np.uint8)

    def step(self, frame_bgr, audio_levels, dt):
        """input frame → optional matte gate → gamma-correct dither at the
        working scale (nearest-neighbor back up) → palette. Returns RGB at
        host.res. The frame is pre-mirrored by the shell and never None."""
        rw, rh = self.host.res
        algo = self._algo()
        bits = self._bits()
        gamma = bool(self._ui("dg_gamma", True))
        bias = self._bias()
        contrast = float(self._ui("dg_contrast", 1.0))
        scale = float(self._ui("dg_scale", SCALE_DEFAULT))
        off, on = PALETTES[self._palette_name()]
        matte_kind = self._matte_name()

        # audio modulation — deliberately minimal (§4.2): bass nudges Contrast
        # when Sound react is on; nothing else moves.
        if audio_levels is not None:
            sens = float(self._ui("sens", 1.0))
            contrast *= 1.0 + 0.35 * sens * float(audio_levels["bass"])

        # perf: all float work happens at WORKING res — grayscale stays uint8
        # through the resize, contrast runs on the small plane, and the palette
        # maps BEFORE the nearest-neighbour upscale (that last move IS bitwise
        # identical: NEAREST replicates pixels and the palette map is
        # per-pixel).
        #
        # Behavior change, accepted for the ~90x cost reduction: contrast now
        # applies at working res. clip does NOT commute with INTER_AREA, so
        # near the clip boundaries at high contrast this differs from the
        # pre-rewrite pipeline (full-res contrast, then resize) — a few
        # 1/255 at Contrast 1.6, tens of 1/255 at 3.0 on detailed frames.
        # test_contrast_reorder_canary in tests/test_dithergirl.py bounds it;
        # post-dither the visual difference is small.
        gray_u8 = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        wh = int(np.clip(round(scale), SCALE_LO, SCALE_HI))
        ww = max(8, int(round(wh * rw / float(rh))))
        small = (cv2.resize(gray_u8, (ww, wh), interpolation=cv2.INTER_AREA)
                 .astype(np.float32) / 255.0)
        if contrast != 1.0:
            small = np.clip((small - 0.5) * contrast + 0.5, 0.0, 1.0)

        lit = _dither(small, algo, bits, gamma, bias)
        out = self._palette_map(lit, off, on)
        out = cv2.resize(out, (rw, rh), interpolation=cv2.INTER_NEAREST)

        if matte_kind != "off":
            # matte gate: dither the matted subject only; elsewhere the raw
            # camera picture, or black (the 'Matte bg black' toggle)
            if self._mat_kind != matte_kind:
                self.mat = make_matte(matte_kind)
                self._mat_kind = matte_kind
            m = self.mat.compute(cv2.resize(frame_bgr, (MATTE_W, MATTE_H)))
            m = np.clip(cv2.resize(m, (rw, rh)), 0.0, 1.0).astype(np.float32)
            if self._ui("dg_matte_black", False):
                bg = np.zeros_like(out)
            else:
                bg = cv2.cvtColor(cv2.resize(frame_bgr, (rw, rh)),
                                  cv2.COLOR_BGR2RGB)
            # perf: uint8 blend in cv2 — no full-res float temporaries
            out = cv2.blendLinear(out, bg, m, 1.0 - m)
        return out
