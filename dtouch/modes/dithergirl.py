"""Dither — the dither pipeline as the primary image (DESIGN.md §4.2).

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

import time

import cv2
import numpy as np

from ..ascii_art import AsciiRenderer
from ..dither import (bayer_dither, blue_noise_dither, floyd_steinberg,
                      linear_to_srgb, riemersma_dither, srgb_to_linear)
from ..hud import AMBER, put_outlined, text_size, u as _u
from ..imgui import DIM
from ..matte import select_matte
from ..overlay_ui import RES_OPTIONS
from ..panelspec import Cycle, PresetList, Readout, Section, Slider, Toggle
from .particles import MATTE_H, MATTE_W, MATTES

# Accent (DESIGN.md §5): magenta-leaning, picked by measured WCAG contrast —
# BGR (245, 140, 245) = RGB (245, 140, 245). Against the panel scrim fill
# (PANEL (34,32,30) at 86%) it measures 7.75:1, and 4.99:1 against the worst
# case of a pure-white frame under that scrim — both over the 4.5:1 floor.
# (Pure magenta (255,0,255) fails the bright-scrim case at 3.36:1.)
ACCENT = (245, 140, 245)

ALGOS = ["Bayer", "Blue noise", "Floyd-Steinberg", "Riemersma", "ASCII"]
# No inter-pixel dependency — safe at full res, badge reads `live`. ASCII
# belongs here for the same reason Bayer does: it is a per-cell table lookup
# against a threshold texture, and it measures in the ordered-dither class
# (0.6-1.2 ms at 720p, vs 3.4 ms for the Floyd-Steinberg default it replaces).
ORDERED = ("Bayer", "Blue noise", "ASCII")
BIASES = ["auto", "light", "dark"]    # ordered-dither rounding bias (invert)
_BIAS_INVERT = {"auto": "auto", "light": False, "dark": True}

# Palette (§4.2): map the dither's off/on levels to two colors (multi-bit
# output interpolates between them). RGB (step() returns RGB at host.res).
#
# Chosen for a projector, not a monitor: every pair was measured for WCAG
# contrast between its off and its on (the numbers below), and the floor for
# shipping one is 4.5:1 — the same floor the mode accent was picked against
# (§5). The weakest shipped pair is `gameboy` at 7.7:1, because Tint (see
# TINT_LUMA_FLOOR) can spend some of a pair's separation and nothing may fall
# under 4.5:1 after it does.
#
# `white-on-black` and `black-on-white` used to sit at the top of this list.
# They were one palette entered twice: which end is the ink is not a property
# of the hue, it is a property orthogonal to every palette here, and spending
# two of eleven slots on it meant the OTHER nine could not be flipped at all.
# They are now `mono` plus the Invert toggle, which flips any of the ten.
PALETTES = {
    # --- shipped since v1 ---
    "mono": ((0, 0, 0), (255, 255, 255)),                    # 21.00:1
    "amber": ((24, 12, 0), (255, 176, 0)),                   # 10.50:1
    "green phosphor": ((0, 20, 8), (80, 255, 120)),          # 14.42:1
    # --- the rest of the phosphor family ---
    "cyan": ((0, 14, 20), (96, 240, 255)),                   # 14.36:1
    "magenta": ((18, 0, 16), (255, 120, 214)),               #  8.62:1
    # --- cold and hot ---
    "ice": ((6, 12, 26), (198, 236, 255)),                   # 15.66:1
    "blood": ((12, 0, 2), (255, 112, 96)),                   #  7.61:1
    # --- objects, not screens ---
    "gameboy": ((10, 34, 10), (155, 188, 15)),               #  7.70:1
    "sepia": ((236, 224, 198), (54, 36, 24)),                # 11.28:1
    # --- the stream-safe pair: luma-dominant, chroma cheap to encode ---
    "hi-vis": ((8, 8, 10), (255, 214, 10)),                  # 14.17:1
    # --- multi-colour (owner's call, 2026-08-25: "opinionated multi color,
    # --- not just a monochrome shade of one colour"). A palette here is a
    # --- tuple of N gradient stops; the dither's quantisation levels walk
    # --- them, so each level lands on its OWN hue instead of a shade of
    # --- one. They come alive from Bits 2 up (1-bit uses only the ends —
    # --- which is why every ramp still keeps ink-dark ends against a
    # --- bright top, same 4.5:1 floor as the duotones). Register models:
    # --- physarum's aurora (the pretty one), and the owner's CCCCCC look
    # --- (magenta steered blue-violet + chroma bleed — a duotone faking
    # --- multi-colour; these stop faking it). Contrast noted is ends.
    "aurora": ((0, 0, 0), (8, 60, 48), (20, 180, 120),       # 18.36:1
               (140, 120, 220), (240, 240, 255)),
    "ultraviolet": ((14, 0, 28), (122, 22, 162),             # 16.32:1
                    (224, 64, 222), (96, 112, 255), (236, 226, 255)),
    "vaporwave": ((16, 8, 28), (96, 42, 160), (244, 86, 184),  # 17.72:1
                  (84, 220, 236), (250, 246, 255)),
    "sunset": ((10, 10, 42), (46, 58, 138), (232, 96, 64),   # 15.85:1
               (255, 182, 72), (255, 240, 198)),
    "oil slick": ((10, 8, 14), (24, 118, 92), (128, 62, 178),  # 18.60:1
                  (226, 178, 64), (244, 244, 234)),
    # the one 4-stop: at Bits 2 each of the four levels IS one of the four
    # classic CGA colours (their canonical RGBI values — hardware history,
    # not anyone's design). Deliberately tone-disordered in the middle,
    # like the real card.
    "cga": ((0, 0, 0), (85, 255, 255), (255, 85, 255),       # 21.00:1
            (255, 255, 255)),
}

# Invert = swap the ink and the ground. One exception, and it is authored, not
# accidental: `mono` inverted is NOT (255,255,255)-on-(0,0,0).
#
# The two monochrome palettes this replaces were never each other's mirror.
# `white-on-black` was pure — 0 and 255, 21.00:1 — and `black-on-white` was
# deliberately softened to 245 paper and 16 ink, 17.45:1, because a full frame
# of pure white is punishing off a projector and pure black ink on it rings.
# A naive swap would have silently re-toned every look that ever said
# `black-on-white`, the shipped `newsprint` built-in included, which is exactly
# the "migration is the hard part" failure. So the softened pair is kept, as
# mono's declared inverse, and the migration is bit-for-bit (pinned in
# tests/test_dithergirl.py).
#
# Every other palette inverts by plain swap: same two colours, so the same
# measured contrast ratio, and nothing to re-check.
AUTHORED_INVERSE = {
    "mono": ((245, 245, 245), (16, 16, 16)),                 # 17.45:1
}

# The retired names, and what they mean now. Kept forever (DESIGN.md §9:
# presets are the one user-data-loss surface, and a Cycle silently ignores a
# value it does not recognise — a look naming a retired palette would have
# loaded with whatever palette happened to be live).
LEGACY_PALETTES = {
    "white-on-black": {"palette": "mono", "invert": False},
    "black-on-white": {"palette": "mono", "invert": True},
}


def palette_stops(name, invert=False):
    """The full stop tuple for a named palette, flipped or not.

    A duotone is a 2-stop ramp; the multi-colour palettes carry 4-5 stops.
    Invert reverses the walk (the same colours, ink and ground swapped),
    except where an inverse is authored (mono — see AUTHORED_INVERSE)."""
    stops = tuple(tuple(c) for c in PALETTES[name])
    if not invert:
        return stops
    auth = AUTHORED_INVERSE.get(name)
    return tuple(tuple(c) for c in auth) if auth else stops[::-1]


# ----- low bit depths ------------------------------------------------------
#
# At Bits 1 a dither emits two values, so a palette renders as two colours —
# and the ramps sample those at their ENDS, which is where every one of them
# is deliberately ink-dark against a bright top for the 4.5:1 floor. The
# result is that at the shipped boot depth `aurora` and `cga` come out
# BYTE-IDENTICAL, and every multi-colour palette collapses to a monochrome
# ramp of one hue. That is exactly the complaint the multi-colour palettes
# were added to answer, reappearing one bit lower down.
#
# So at two levels a ramp keeps its dark ground and takes its most CHROMATIC
# stop as ink, rather than its brightest — provided that stop still clears
# the legibility floor against the ground. aurora goes black/green, cga
# black/cyan, vaporwave black/pink, ultraviolet black/magenta: six palettes
# that are six colours at 1 bit instead of six greyscales.
#
# Duotones are untouched (they have no middle stop to prefer), which keeps
# every look predating the ramps bit-exact.
LEGIBILITY_FLOOR = 4.5


def _rel_luminance(rgb):
    """WCAG relative luminance of an 8-bit sRGB triple."""
    c = [v / 255.0 for v in rgb]
    c = [(x / 12.92) if x <= 0.04045 else (((x + 0.055) / 1.055) ** 2.4)
         for x in c]
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]


def contrast_ratio(a, b):
    """WCAG contrast between two 8-bit sRGB triples."""
    la, lb = _rel_luminance(a), _rel_luminance(b)
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)


def _chroma(rgb):
    """How colourful, 0..1. Plain max-min: cheap, and it ranks the way the
    eye does for the stops these ramps actually contain."""
    return (max(rgb) - min(rgb)) / 255.0


def low_depth_stops(stops, levels):
    """The stops a dither at `levels` quantisation levels should actually use.

    Only the two-level case is special-cased, and only for ramps with a
    middle to prefer; everything else walks the ramp as before.
    """
    stops = tuple(tuple(int(v) for v in c) for c in stops)
    if levels != 2 or len(stops) <= 2:
        return stops
    ground = stops[0]
    ink = max(stops[1:],
              key=lambda c: (contrast_ratio(ground, c) >= LEGIBILITY_FLOOR,
                             _chroma(c)))
    if contrast_ratio(ground, ink) < LEGIBILITY_FLOOR:
        return (ground, stops[-1])      # nothing chromatic is legible here
    return (ground, ink)


def palette_pair(name, invert=False):
    """The (off, on) END pair for a named palette, flipped or not — what a
    1-bit dither renders, what ASCII colours its glyphs with, and what the
    4.5:1 legibility floor is measured on. For a duotone this IS the whole
    palette; a multi-colour ramp's middle stops live in palette_stops."""
    stops = palette_stops(name, invert)
    return stops[0], stops[-1]

# ----- Hue / Tint (the customisability, §4.2 'two-color ramps later') -----
#
# Two sliders instead of a longer list, because a list of named pairs cannot
# be dialled during a set and a per-colour RGB editor is six sliders of
# fiddling in a panel meant to be driven at arm's length in the dark.
#
# Tint is an amount and Hue is a direction: at Tint 0 every named palette is
# bit-identical to its shipped values, so the sliders COMPOSE with the
# palettes rather than replacing them (the palette still owns which end is
# dark, how dark, and the whole tonal structure; Hue/Tint only steer the
# chroma). Both ends are steered, so a duotone stays a duotone — and a pure
# black ground stays pure black for free, because the target is built at the
# source colour's own value.
#
# The steering target is the fully saturated hue at the source's value, but
# lifted toward white until its relative luminance is at least
# TINT_LUMA_FLOOR of the source's. Without that floor a hue near blue costs
# ~14x luminance (pure blue is 7% of white) and Tint could quietly turn any
# palette into an unreadable navy-on-black. With it, every shipped palette
# stays above 4.5:1 at every hue and every tint — measured over 72 hues x 4
# tints in test_dithergirl.py.
TINT_LUMA_FLOOR = 0.65
HUE_LO, HUE_HI = 0.0, 360.0
_LUMA = np.float32([0.2126, 0.7152, 0.0722])
_TINT_CACHE = {}


def _hue_rgb(hue_deg, value):
    """Fully saturated sRGB triple of hue *hue_deg* at max-channel *value*."""
    h6 = (float(hue_deg) / 60.0) % 6.0
    i = int(h6)
    f = h6 - i
    v = float(value)
    q, t = v * (1.0 - f), v * f
    return np.float32([(v, t, 0.0), (q, v, 0.0), (0.0, v, t),
                       (0.0, q, v), (t, 0.0, v), (v, 0.0, q)][i])


def tint_rgb(rgb, hue_deg, amount, floor=TINT_LUMA_FLOOR):
    """Steer one palette colour *amount* of the way toward *hue_deg*.

    amount == 0 returns the colour unchanged (exactly — Tint 0 is a no-op, not
    a round-trip). Mixing toward white happens in linear light, where relative
    luminance is linear in the mix factor and the floor solves in closed form.
    """
    if amount <= 0.0:
        return tuple(int(c) for c in rgb)
    src = np.float32(rgb)
    lin_hue = srgb_to_linear(_hue_rgb(hue_deg, float(src.max())) / 255.0)
    l_hue = float((lin_hue * _LUMA).sum())
    l_src = float((srgb_to_linear(src / 255.0) * _LUMA).sum())
    need = floor * l_src
    if l_hue < need:
        k = (need - l_hue) / max(1.0 - l_hue, 1e-6)
        lin_hue = lin_hue + k * (1.0 - lin_hue)
    target = linear_to_srgb(lin_hue) * 255.0
    out = np.clip(src + float(amount) * (target - src), 0.0, 255.0)
    return tuple(int(round(c)) for c in out)


def tinted_palette(name, hue_deg, amount, invert=False):
    """The stop tuple for a named palette under Invert and the Hue/Tint pair
    (2 stops for a duotone, 4-5 for the multi-colour ramps).

    Cached — this runs per frame and the answer only ever depends on four
    values. Invert resolves FIRST and Tint steers what comes out: for the
    plain-swap palettes the order cannot matter (tint_rgb is per-colour), but
    mono's inverse is an authored pair and Tint must steer the stops the
    operator is actually looking at.
    """
    key = (name, round(float(hue_deg), 2), round(float(amount), 4), bool(invert))
    got = _TINT_CACHE.get(key)
    if got is None:
        got = tuple(tint_rgb(c, hue_deg, amount)
                    for c in palette_stops(name, invert))
        if len(_TINT_CACHE) > 512:       # slider drags are unbounded in theory
            _TINT_CACHE.clear()
        _TINT_CACHE[key] = got
    return got


MATTES_DG = ["off"] + MATTES          # "off" = dither the whole frame

SCALE_LO, SCALE_HI, SCALE_DEFAULT = 30.0, 720.0, 72.0
# The floor is 30, not the 45 this slider shipped with, because Scale changed
# UNIT when ASCII joined the cycle: 45 working PIXELS is a fine coarse floor for
# the four pixel dithers, but 45 character ROWS is an 8x16 cell — the FINEST
# grid ASCII should ever be asked for, not the coarsest. The `ascii stream`
# built-in wants 30 rows (12x24 cells, what survives H.264), and a built-in
# sitting outside its own control's range is a look no slider position can get
# back to. Verified at 30 working pixels for all four pixel dithers (a 53x30
# plane; every algorithm returns in under 0.6 ms and spans 0..1).
# Error diffusion AT OR ABOVE this working height gets the amber note. The
# comparison is >= on purpose: a threshold named SLOW_SCALE should mean "this
# value is slow", and `newsprint` shipped sitting exactly on 180.0 with a `>`
# test, so the one built-in the threshold was closest to could never trigger
# it. `newsprint` now sits at 160.0 — deliberately a step under the line
# rather than balanced on it (a shipped built-in is part of the known-good
# set; it must not boot into a perf warning, and it must not be one float
# nudge away from one either).
SLOW_SCALE = 180.0


ASCII_SLOW_MS = 8.0   # measured ASCII step cost that earns an amber note
ASCII_SLOW_CLEAR = 0.75   # ...and the fraction of it that takes the note away
ASCII_WARMUP_FRAMES = 8   # EMA frames before either verdict is allowed


# ----- panel visibility gates (panelspec.visible; magic-over-control,
# ----- 2026-08-24: a control that does nothing perceptible in the current
# ----- state hides instead of sitting on the panel looking functional) -----

def _matte_on_ui(s):
    """'Matte bg black' only acts while a matte is on (step() consults it
    inside the `matte_kind != "off"` branch only)."""
    return MATTES_DG[int(getattr(s, "dg_matte_idx", 0)) % len(MATTES_DG)] != "off"


def _ordered_ui(s):
    """Bias steers the ordered dithers (and ASCII's ramp) only — error
    diffusion self-corrects and takes no invert parameter (see _dither), so
    under Floyd-Steinberg / Riemersma the row moved nothing at all."""
    return ALGOS[int(getattr(s, "dg_algo_idx", 2)) % len(ALGOS)] in ORDERED


def _tint_on_ui(s):
    """Hue only acts once Tint is up (tint_rgb with amount 0 is the palette
    exactly as named) — the old tooltip even confessed it ('Does nothing
    until Tint is up'). Now the row appears when Tint does."""
    return float(getattr(s, "dg_tint", 0.0)) > 0.0


def _dither(gray, algo, bits, gamma, bias):
    """One grayscale float [0,1] plane through the named algorithm. Bias
    (rounding direction) applies to the ordered dithers only — error diffusion
    self-corrects and takes no invert parameter.

    ASCII is deliberately NOT reachable here: it quantises to glyph tiles, not
    to grey levels, so it never produces a [0,1] plane for `_palette_map` to
    colour. `dtouch.ascii_art` owns that path end to end."""
    if algo == "ASCII":
        raise ValueError("ASCII renders through dtouch.ascii_art, not _dither")
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
    title = "Dither"
    accent = ACCENT
    accepts_still = True
    blurb = "live + still\ndithering"      # home-menu card copy (DESIGN.md §3)
    # §2.4: the rack hides what we own — Dither owns ALL dither quality
    # controls (dither row + Pixel/Bits/Gamma/Bias), not just the dither
    # cycle; two visible dither subsystems in one panel is the
    # bolted-features incoherence the overhaul exists to kill. "pixel" is
    # the rack dither's block size — this mode's Scale IS that control.
    claims = frozenset({"dither", "pixel", "bits", "gamma", "bias"})

    # option lists the shell's boot path reads (OverlayUI ctor)
    palettes = list(PALETTES)
    mattes = list(MATTES_DG)
    matte_kind = "off"

    # Built-in looks (DESIGN.md §7): in code, immune to rename/delete,
    # bankable. 'stream-safe' is the stream-tuned variant — bigger cells
    # (low Scale) + higher contrast survive stream compression (§4.2).
    BUILTIN = {
        # The look the mode selector wears. The home menu renders the live
        # camera through a 1-bit blue-noise dither at half resolution and
        # NEAREST-upscales it, so every dither cell is a 2x2 block — it is
        # the first thing anyone sees in this app and the owner's favourite
        # thing in it, and it was reachable from nowhere. Scale 360 is half
        # of a 720p output, which is what menu.draw_menu computes.
        # Deliberately 1-bit and neutral: see low_depth_stops for why that no
        # longer costs the palettes their colour.
        "menu": dict(algorithm="Blue noise", bits=1.0, gamma=True,
                     bias="auto", contrast=1.0, scale=360.0,
                     palette="mono", matte="off"),
        # The owner's own saved look, promoted from presets.json so it cannot
        # be lost to a rename and so it ships to everyone. Its colours are not
        # the palette's: magenta steered blue-violet gives four stops, and the
        # rack's 1-bit crush then binarises those to black, black, pure blue
        # and pure magenta, with chroma bleed fringing the edges. That
        # accident is the register the multi-colour palettes were written to
        # make native — it belongs in the menu next to them, not in one
        # person's preset file.
        "CCCCCC": dict(algorithm="Riemersma", bits=2.0, gamma=True,
                       bias="auto", contrast=1.60, scale=486.0,
                       palette="magenta", hue=238.0, tint=0.61, matte="off",
                       signal=dict(glitch=True, chroma=55.4, drift=9.49,
                                   crush=1.0, scanlines=True)),
        "classic": dict(algorithm="Floyd-Steinberg", bits=1.0, gamma=True,
                        bias="auto", contrast=1.0, scale=72.0,
                        palette="mono", matte="off"),
        # the only built-in that inverts: dark ink on paper. Its pair is
        # AUTHORED_INVERSE["mono"], the exact colours it shipped with under the
        # retired `black-on-white` name.
        "newsprint": dict(algorithm="Floyd-Steinberg", bits=1.0, gamma=True,
                          bias="light", contrast=1.15, scale=160.0,
                          palette="mono", invert=True, matte="off"),
        "phosphor": dict(algorithm="Bayer", bits=2.0, gamma=True,
                         bias="auto", contrast=1.1, scale=144.0,
                         palette="green phosphor", matte="off"),
        "stream-safe": dict(algorithm="Blue noise", bits=1.0, gamma=True,
                            bias="auto", contrast=1.6, scale=56.0,
                            palette="mono", matte="off"),
        "riemersma still": dict(algorithm="Riemersma", bits=2.0, gamma=True,
                                bias="auto", contrast=1.0, scale=240.0,
                                palette="mono", matte="off"),
        # ASCII reinterprets Scale as character ROWS, so its looks live in a
        # different numeric register: 45 rows is an 8x16 cell at 720p (the
        # recommended default), 30 rows a 12x24 one.
        "ascii": dict(algorithm="ASCII", bits=4.0, gamma=True, bias="auto",
                      contrast=1.0, scale=45.0,
                      palette="mono", matte="off"),
        # ASCII output is maximally high-frequency two-tone content — the
        # worst case for H.264. Big cells and hard contrast are what survives
        # a stream; 4x8 cells turn to mush (§4.2's stream-tuned looks).
        "ascii stream": dict(algorithm="ASCII", bits=3.0, gamma=True,
                             bias="light", contrast=1.4, scale=30.0,
                             palette="green phosphor", matte="off"),
    }

    # apply="reset" merges a look over these (store keys — DESIGN.md §7).
    # `invert` is here, and its Toggle is apply="reset" rather than the Toggle
    # default "keep", because it is part of the PICTURE, not a live rig switch
    # like Glitch or Sound react: a look that does not mention it must land
    # un-inverted, or recalling `phosphor` while Invert happened to be on would
    # render a look nobody saved. It is also what makes the legacy rewrite
    # total — every look predating Invert says nothing about it, and every one
    # of them means False.
    DEFAULTS = dict(algorithm="Floyd-Steinberg", bits=1.0, bias="auto",
                    contrast=1.0, scale=SCALE_DEFAULT,
                    palette="mono", invert=False, hue=0.0, tint=0.0,
                    matte="off")

    # shared-UI attrs this mode seeds (prefixed to coexist with every mode's
    # attrs on the one OverlayUI state object)
    _UI_DEFAULTS = dict(input_idx=0, dg_matte_idx=0, dg_matte_black=False,
                        dg_algo_idx=ALGOS.index("Floyd-Steinberg"),
                        dg_bits=1.0, dg_gamma=True, dg_bias_idx=0,
                        dg_contrast=1.0, dg_scale=SCALE_DEFAULT,
                        dg_palette_idx=0, dg_invert=False,
                        dg_hue=0.0, dg_tint=0.0)

    def __init__(self, still=False):
        self.boot_still = still            # CLI --still: boot with still input
        self.host = None
        self.mat = None
        self._mat_kind = None
        self._swatch_key = None
        self._swatch_cache = None
        self._ascii = None            # AsciiRenderer, rebuilt on any key change
        self._ascii_ms = None         # measured cost, EMA (perf honesty §4.2)
        self._ascii_frames = 0
        self._ascii_slow = False      # latched; cleared on every rebuild

    # ----- lifecycle -----
    def start(self, host):
        self.host = host

    def stop(self):
        """Idempotent (DESIGN.md §2.2). No GL — just drop the matte and the
        ASCII renderer (which holds a full-res output buffer)."""
        self.mat = None
        self._mat_kind = None
        self._ascii = None
        self._ascii_ms, self._ascii_frames, self._ascii_slow = None, 0, False

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

    def _invert(self):
        return bool(self._ui("dg_invert", False))

    def _hue(self):
        return float(np.clip(self._ui("dg_hue", 0.0), HUE_LO, HUE_HI))

    def _tint(self):
        return float(np.clip(self._ui("dg_tint", 0.0), 0.0, 1.0))

    def _palette(self):
        """The live stop tuple (2 for a duotone, 4-5 for the multi-colour
        ramps): the named palette, flipped by Invert if the toggle is on,
        then steered by Hue/Tint."""
        return tinted_palette(self._palette_name(), self._hue(), self._tint(),
                              self._invert())

    def _matte_name(self):
        return MATTES_DG[int(self._ui("dg_matte_idx", 0)) % len(MATTES_DG)]

    def slow_warning(self):
        """True when the inline amber perf note should render (§4.2): an
        error-diffusion algorithm at or above the slow working height."""
        return (self._algo() not in ORDERED
                and self._ui("dg_scale", SCALE_DEFAULT) >= SLOW_SCALE)

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
                       show_when=_matte_on_ui,
                       tip="With a matte on: black outside the subject instead "
                           "of the raw camera picture."),
                Cycle("output", "res_idx", [n for n, _, _ in RES_OPTIONS],
                      key="res", save=False, nudge=False),
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
                       status="{:.0f}-bit", step=1.0,
                       tip="Output bit depth (or, under ASCII, how many "
                           "glyphs). Whole numbers only - there are four "
                           "settings. 1 = pure two-tone."),
                Toggle("Gamma", "dg_gamma", save_key="gamma",
                       tip="Dither in linear light so mid-tones keep their "
                           "perceived brightness. Off = the crushed retro look."),
                Cycle("bias", "dg_bias_idx", list(BIASES), save_key="bias",
                      status="bias {}", show_when=_ordered_ui,
                      tip="Which way the dots lean on a mostly-dark or "
                          "mostly-bright picture. Auto decides per frame. "
                          "The diffusion dithers self-correct and ignore it."),
                Slider("Contrast", "dg_contrast", 0.25, 3.0, save_key="contrast",
                       tip="Push tones apart before dithering. High contrast "
                           "survives stream compression."),
                # engine_snaps=False: the pixel dithers round Scale themselves,
                # but ASCII consumes it continuously (grid_for divides by it
                # before any rounding), so a stored fractional scale is a real
                # grid a look has always rendered — the step is control feel,
                # and apply_look passes stored values through un-snapped
                # (panelspec's step docstring).
                Slider("Scale", "dg_scale", SCALE_LO, SCALE_HI, fmt=".0f",
                       save_key="scale", step=1.0, engine_snaps=False,
                       # "requested", not "whole character rows": the cell
                       # rounds to whole pixels, so the delivered row count
                       # can differ from the request — the grid note below
                       # the slider is the truth about what you got.
                       tip="Working height: whole pixels for the dithers; "
                           "for ASCII, the character rows requested - the "
                           "grid note shows the rows you actually get. Low "
                           "= big chunky cells; high = fine grain (slow for "
                           "the diffusion dithers)."),
                Readout(self._draw_grid_note),
                Readout(self._draw_perf_note),
            ]),
            Section("PALETTE", [
                Cycle("palette", "dg_palette_idx", list(PALETTES),
                      save_key="palette", legacy=LEGACY_PALETTES),
                # next to the cycle it modifies, because it belongs to every
                # entry in it: which end is the ink is orthogonal to the hue.
                Toggle("Invert", "dg_invert", save_key="invert",
                       apply="reset",
                       tip="Swap the ink and the background. Works on any "
                           "palette - dark on light, or light on dark."),
                # Tint before Hue: Hue only exists while Tint is up
                # (show_when=_tint_on_ui), so the reveal unfolds BELOW the
                # slider being dragged instead of shoving it down mid-drag.
                Slider("Tint", "dg_tint", 0.0, 1.0, save_key="tint",
                       tip="How far to steer the palette toward Hue. 0 = the "
                           "palette exactly as named."),
                # engine_snaps=False: tint_rgb consumes hue as a float — the
                # whole-degree step is control feel (a degree is below what an
                # eye can name), not an engine constraint, so a stored
                # fractional hue applies exactly (panelspec's step docstring).
                Slider("Hue", "dg_hue", HUE_LO, HUE_HI, fmt=".0f",
                       save_key="hue", step=1.0, engine_snaps=False,
                       show_when=_tint_on_ui,
                       tip="Which colour Tint steers toward, in whole "
                           "degrees around the colour wheel."),
            ]),
        ]

    def commands(self):
        return {}                          # no mode-local perform keys (v1)

    def safe_look(self):
        """The look the mode opens on, and the panic target (DESIGN.md §6.2
        '0'). The menu's own dither: whoever just pressed `d` was looking at
        it a second ago, so entering the mode continues the picture rather
        than replacing it."""
        return "menu"

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

    @staticmethod
    def _note_lines(g, text, cw):
        """Word-wrap a note to the panel column. Every inline note goes through
        this: the sidebar is narrow at 720p and gets narrower as the frame
        does, and a note that runs off the panel is graffiti over the picture
        rather than part of the panel."""
        lines, cur = [], ""
        for wd in text.split():
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
        return lines

    def _draw_note(self, frame, g, x, y, cw, text, color):
        for line in self._note_lines(g, text, cw):
            g.text(frame, line, x, y + g.S(12), color, 0.42)
            y += g.S(16)
        return y

    def _draw_perf_note(self, frame, g, x, y, cw):
        """Inline amber perf note — renders only when Scale is dragged high
        while an error-diffusion algorithm is active (§4.2). Word-wrapped to
        the panel column (at 720p the one-liner overflows the sidebar)."""
        if not self.slow_warning():
            return y
        y = self._draw_note(frame, g, x, y, cw,
                            "slow - ordered dither recommended live", AMBER)
        return y + g.S(4)

    def _draw_grid_note(self, frame, g, x, y, cw):
        """ASCII-only TONE readout (renders nothing otherwise — the same
        conditional-Readout trick `_draw_perf_note` uses).

        Under ASCII, Scale keeps its store key, its range and its label but
        changes UNIT, from working pixels to character rows. That is a real
        semantic shift inside one control and hiding it would be dishonest, so
        the grid says what it actually is; and when Scale runs past the
        legibility floor and stops doing anything, it says that too."""
        if self._algo() != "ASCII":
            return y
        rend = self._ascii
        if rend is None:
            return y
        y = self._draw_note(frame, g, x, y, cw, rend.grid_note(), DIM)
        if rend.clamped:
            # DIM, not amber: nothing is wrong, the control has run out of room
            y = self._draw_note(frame, g, x, y, cw,
                                "cell floor - characters stay legible", DIM)
        if self._ascii_slow and self._ascii_ms is not None:
            # perf honesty (§4.2): the measured cost, not a guessed threshold —
            # ASCII is in the ordered class at 720p/1080p but a 4K frame at the
            # cell floor is ~8.7 ms, and an operator is owed the number rather
            # than a quietly halved frame rate.
            y = self._draw_note(frame, g, x, y, cw,
                                "ascii %.1f ms/frame - lower Scale or output"
                                % self._ascii_ms, AMBER)
        return y + g.S(4)

    def _draw_swatch(self, frame, g, x, y, cw):
        """Swatch strip: a horizontal 0->1 gradient ramp rendered through the
        CURRENT algorithm+bits+gamma+bias (+palette, +hue/tint), cached and
        re-rendered only when one of those changes (§4.2 — instant 'what am I
        hearing' for the eyes). Under ASCII the strip becomes one row of the
        live glyph ramp, which is the most direct answer the panel can give to
        'what am I looking at'."""
        hpx = g.S(16)
        key = (self._algo(), self._bits(), bool(self._ui("dg_gamma", True)),
               self._bias(), self._palette_name(), self._hue(), self._tint(),
               self._invert(), cw, hpx)
        if key != self._swatch_key:
            self._swatch_cache = self._render_swatch(*key)
            self._swatch_key = key
        # clipped blit (the panel scrolls; a Readout must not write off-frame)
        y0, y1 = max(y, 0), min(y + hpx, frame.shape[0])
        if y1 > y0:
            frame[y0:y1, x:x + cw] = self._swatch_cache[y0 - y:y1 - y]
        return y + hpx + g.S(8)

    def _render_swatch(self, algo, bits, gamma, bias, palette, hue, tint,
                       invert, w, h):
        stops = tinted_palette(palette, hue, tint, invert)
        if algo == "ASCII":
            # one row of characters across the strip: the strip is 16 px tall,
            # so it cannot preview the live CELL size, but it can preview the
            # thing that actually changed — the ramp. ASCII colours its
            # glyphs with the palette's END pair (see step()).
            rend = AsciiRenderer(w, h, rows_req=1, n=1 << bits,
                                 palette=(stops[0], stops[-1]), gamma=gamma,
                                 bias=bias)
            ramp_u8 = np.tile(
                np.linspace(0, 255, w, dtype=np.float32).astype(np.uint8),
                (h, 1))
            rgb = rend.render(ramp_u8)
        else:
            ramp = np.tile(np.linspace(0.0, 1.0, w, dtype=np.float32), (h, 1))
            rgb = self._palette_map(_dither(ramp, algo, bits, gamma, bias),
                                    low_depth_stops(stops, 1 << bits))
        return rgb[:, :, ::-1].copy()      # panel frames are BGR

    # ----- per-frame -----
    @staticmethod
    def _palette_map(levels, stops):
        """Map dither levels [0,1] onto the palette's stop ramp (uint8 RGB).

        Piecewise-linear over the stops: a duotone (2 stops) computes the
        exact same off + levels*(on-off) it always did, bit for bit (pinned
        in tests); a multi-colour palette walks its 4-5 stops, so each
        quantisation level lands on its own hue. With B bits the dither
        emits levels k/(2^B - 1) — at 2 bits a 5-stop ramp is sampled at
        0, 1/3, 2/3, 1: four distinct colours on screen."""
        arr = np.float32(stops)
        n = len(arr)
        if n == 2:
            off, on = arr
            return (off[None, None, :]
                    + levels[:, :, None] * (on - off)).astype(np.uint8)
        pos = np.clip(levels, 0.0, 1.0) * (n - 1)
        i = np.minimum(pos.astype(np.int32), n - 2)
        f = (pos - i)[:, :, None]
        return (arr[i] + f * (arr[i + 1] - arr[i])).astype(np.uint8)

    def _ascii_step(self, gray_u8, rw, rh, bits, gamma, bias, contrast, scale,
                    palette):
        """The ASCII branch of step(): quantise tone to glyph tiles.

        The renderer is RETUNED in place for a settings change and rebuilt only
        for a new output resolution. It used to be rebuilt for both, and the
        frame buffer made that ruinous, because every frame allocated and
        cleared a fresh 24 MB buffer for a change that only invalidates the
        atlas. Both paths measured in one process, mean ms/frame:

                        idle           Hue drag        Scale drag
            720p    0.63 -> 0.64    3.14 ->  0.88    3.38 ->  2.49
            1080p   1.25 -> 1.25    6.59 ->  1.60    7.21 ->  4.23
            4K      4.13 -> 4.15   24.80 ->  4.87   29.11 -> 10.76

        Idle is untouched, as it must be — this changes what an EDIT costs, not
        what a frame costs. A Scale drag still pays a real coverage scan for
        each new cell size, which is why the note below matters.

        The whole step is timed, rebuild included (§4.2 perf honesty). Timing
        only `render()` measured the one part of an edit that was never the
        expensive part: the real step cost 28.7 ms while the note read 5.3 ms.
        """
        t0 = time.perf_counter()
        rend = self._ascii
        if rend is None or (rend.frame_w, rend.frame_h) != (rw, rh):
            rend = self._ascii = AsciiRenderer(rw, rh, rows_req=scale,
                                               n=1 << bits, palette=palette,
                                               gamma=gamma, bias=bias)
            # a new output resolution is a different instrument; nothing
            # measured against the old one carries over
            self._ascii_ms, self._ascii_frames, self._ascii_slow = None, 0, False
        elif rend.key() != AsciiRenderer.make_key(rw, rh, scale, 1 << bits,
                                                  palette, gamma, bias, True):
            rend.configure(scale, 1 << bits, palette, gamma, bias, True)
        rend.set_rows_req(scale)
        out = rend.render(gray_u8, contrast)
        self._ascii_verdict((time.perf_counter() - t0) * 1000.0)
        return out

    def _ascii_verdict(self, ms):
        """Fold one measured step cost into the EMA and update the slow latch.

        Split out of `_ascii_step` so the honesty rule can be tested by feeding
        it numbers instead of by racing a stopwatch — a test that asserts "this
        machine renders ASCII in under 8 ms" is a perf test wearing a logic
        test's clothes, and it fails on a busy box for reasons that have
        nothing to do with the rule it means to pin.

        The verdict follows the measurement in BOTH directions, through a
        hysteresis band so it cannot blink while an operator hovers the
        threshold. It used to be latched-until-rebuild instead, which was the
        same bug as timing only render(): a drag that retunes every frame
        cleared the frame counter every frame, so the warm-up gate was
        unreachable during the one interaction that was actually slow. Warm-up
        still applies, because a cold cell size pays a one-off coverage scan
        that says nothing about the steady state.
        """
        self._ascii_ms = ms if self._ascii_ms is None else (
            0.85 * self._ascii_ms + 0.15 * ms)
        self._ascii_frames += 1
        if self._ascii_frames >= ASCII_WARMUP_FRAMES:
            if self._ascii_ms > ASCII_SLOW_MS:
                self._ascii_slow = True
            elif self._ascii_ms < ASCII_SLOW_MS * ASCII_SLOW_CLEAR:
                self._ascii_slow = False

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
        stops = self._palette()
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
        if algo == "ASCII":
            # ASCII colours its glyph atlas with the palette's END pair —
            # the glyphs themselves carry the tonal ramp, so a multi-colour
            # palette renders as its ink/ground ends there
            out = self._ascii_step(gray_u8, rw, rh, bits, gamma, bias,
                                   contrast, scale, (stops[0], stops[-1]))
        else:
            wh = int(np.clip(round(scale), SCALE_LO, SCALE_HI))
            ww = max(8, int(round(wh * rw / float(rh))))
            small = (cv2.resize(gray_u8, (ww, wh), interpolation=cv2.INTER_AREA)
                     .astype(np.float32) / 255.0)
            if contrast != 1.0:
                small = np.clip((small - 0.5) * contrast + 0.5, 0.0, 1.0)

            lit = _dither(small, algo, bits, gamma, bias)
            out = self._palette_map(lit, low_depth_stops(stops, 1 << bits))
            out = cv2.resize(out, (rw, rh), interpolation=cv2.INTER_NEAREST)

        if matte_kind != "off":
            # matte gate: dither the matted subject only; elsewhere the raw
            # camera picture, or black (the 'Matte bg black' toggle)
            if self._mat_kind != matte_kind:
                # a matte whose optional dependency is missing reverts the
                # cycle + toasts instead of raising out of step() (§6.4);
                # this frame simply renders un-matted
                mat, self._mat_kind = select_matte(
                    self.host.ui, "dg_matte_idx", MATTES_DG,
                    self._mat_kind or "off", self.host.hud.toasts)
                if mat is None:
                    return out
                self.mat = mat
            m = self.mat.compute(cv2.resize(frame_bgr, (MATTE_W, MATTE_H)))
            m = np.clip(cv2.resize(m, (rw, rh)), 0.0, 1.0).astype(np.float32)
            # The matte is NOT snapped to the character grid under ASCII, and
            # that was a decision, not an oversight. Snapping (average the
            # matte per cell, expand back) was built and looked at: it paints a
            # staircase of half-lit grey cells around the subject — off-palette
            # blocks in a two-tone picture — and a hard per-cell threshold
            # instead throws away the feather every shipped matte produces.
            # The unsnapped cut does clip glyphs, but at 8x16 cells that reads
            # as texture, and it is what the pixel dithers already do.
            if self._ui("dg_matte_black", False):
                bg = np.zeros_like(out)
            else:
                bg = cv2.cvtColor(cv2.resize(frame_bgr, (rw, rh)),
                                  cv2.COLOR_BGR2RGB)
            # perf: uint8 blend in cv2 — no full-res float temporaries
            out = cv2.blendLinear(out, bg, m, 1.0 - m)
        return out
