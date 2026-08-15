"""ASCII as a quantiser — the fifth entry in Dither Girl's ALGORITHM cycle.

ASCII art is not a separate engine; it answers the same question the four
dithers answer: *given an output alphabet smaller than the input's tonal
range, how do I spend it to fake continuous tone?* Bayer spends 2-16 grey
levels through a threshold matrix; ASCII spends 2-16 glyphs through the same
threshold matrix. So this module reuses ``dtouch.dither``'s blue-noise texture,
its sRGB transfer functions and its ``invert``/bias semantics — on the glyph
index instead of the grey level — and ``dither.py`` stays a pure-array module
with no font, atlas or cache state in it.

Three things here are load-bearing and were each arrived at by rendering the
alternative and looking at it:

**The ramp is measured, not folklore.** Every candidate glyph is rasterised by
the font that will draw it, at the cell size it will be drawn at, and its ink
coverage is the mean alpha of that render. The folkloric `" .:-=+*#%@"` is not
monotonic under measurement (`-` is lighter than `:`, `%` and `@` are lighter
than `#`): four of its nine steps go backwards, so its upper mid-tones collapse
onto one mark and faces read flat.

**Hershey is a stroke font, so stroke weight is part of the ramp.** At a fixed
thickness the densest glyph covers only ~0.38 of the cell, which caps the
picture at 38% of the ink colour's luminance and clips every highlight above
sRGB ~0.65 onto a single glyph. Searching (glyph x weight) jointly reaches
~0.70 coverage, moving the clip point to sRGB ~0.87 — only true speculars. It
also reads better: the ramp gets *bolder* as it gets denser. The search is
narrowed for a SHORT ramp, though (see :func:`weights_for`): reach is only
worth having when there are enough steps to travel there gradually, and spent
on two or four steps a heavy stroke is a cell-filling blob rather than a glyph.

**Tone is normalised to the INK colour, not to the ramp's range.** Stretching
the ramp over the input range maps sRGB mid-grey to ~0.42 apparent instead of
0.50 and halves the picture's apparent brightness — which is why
"gamma-correct ASCII looks bad" is folklore too. The 256-entry position LUT
inverts the ramp's *measured composited luminance* curve, which also makes
dark-ink palettes (black-on-white) invert for free: the measured luminances run
downwards, the interpolation follows, and a bright input correctly picks a
sparse glyph. No ``if palette ==`` anywhere.

Cost (measured, M-series, cv2 4.13): 0.61-1.17 ms/frame at 720p and 1.29-1.79
ms at 1080p for the whole step — the ordered-dither class, and 3-5x cheaper
than the Floyd-Steinberg default it sits beside. The naive one-``cv2.putText``
-per-cell path measures 20.7 ms at 720p and scales with *ink* rather than with
pixels (a bright frame costs more than a dark one), so the glyph atlas is not
an optimisation, it is the only viable shape.

ASCII text only (the pool, and every panel string built from it): Hershey
renders non-ASCII as '?'.
"""
from __future__ import annotations

from collections import OrderedDict

import cv2
import numpy as np

from .dither import _MID_GREY_LINEAR, _blue_noise_matrix, srgb_to_linear

# ---------------------------------------------------------------------------
# Geometry + rasterisation constants
# ---------------------------------------------------------------------------

FONT = cv2.FONT_HERSHEY_SIMPLEX
SS = 4                 # supersample factor for the coverage/atlas raster
FILL = 0.82            # glyph box height as a fraction of the cell
MIN_CELL_H = 8         # legibility floor: below this a glyph is a smudge
ASPECT = 0.5           # cell_w / cell_h — a character cell is ~2:1 tall

# Rec.709 luminance weights (the frame is RGB here, as step() returns RGB).
_LUMA = np.float32([0.2126, 0.7152, 0.0722])

# Candidate pool, in *preference* order — used only to break ties between
# glyphs whose measured coverage is within tol of the same target, so the ramp
# reads as ASCII art rather than as "whichever glyph measured closest".
# Bars are deliberately absent: at 4-8 px cell widths `| l I 1` align
# column-to-column and read as scan bars, not as texture.
PREF = list(" .,:;-~=+*ox<>c\"'v^snuazerwt?/\\!ijfyCLOQUJXY0ZSGEAPDNHKRVMW&%#B8@$")

# Stroke weights searched jointly with the glyph (see module docstring).
WEIGHTS = (1.0, 1.6, 2.2, 3.0)


def weights_for(n: int, weights=WEIGHTS):
    """The stroke weights a ramp of *n* steps is allowed to search.

    A heavy stroke buys REACH — it is what takes the densest glyph from ~0.38
    to ~0.70 coverage and moves the highlight clip from sRGB 0.65 to 0.87 — and
    reach is only worth having when there are enough steps to travel there
    gradually. Spend it on a short ramp and the top steps are all cell-filling
    blobs: at n=2 the whole ramp was `' s'` and rendered as a coarse dot
    halftone with no glyph structure at all; at n=4 (`' :*#'`) the top two
    steps drew at 3 px stroke in an 8 px cell and merged into solid white rows.
    Both were looked at, beside the capped versions, on a lit subject.

    So the weight set scales with the ramp: 1 weight below 8 steps, 2 below 12,
    the full search at 16. Capped, the same settings measure `' R'`, `';e$'`
    and `'.;=sQ#$'` — recognisably letters, at every Bits setting the panel can
    reach, which is what the feature is for. The cost is a lower ceiling on a
    short ramp (0.25 / 0.38 / 0.48 coverage), and a short ramp is not where
    highlight range was ever going to come from.
    """
    return tuple(weights[:max(1, min(len(weights), int(n) // 4))])


# ---------------------------------------------------------------------------
# Per-cell-size cache
# ---------------------------------------------------------------------------
#
# A coverage scan is 272 rasters and costs 3.9 ms at 8x16 / 7.1 ms at 24x48,
# so it must be cached — but it must not be cached *forever*: dragging Scale at
# 4K walks ~40 distinct cell heights, and the alpha rasters alone would reach
# tens of MB. Keep the last few cell sizes, evict oldest.

_CELL_CACHE_MAX = 6
_CELLS: "OrderedDict[tuple, dict]" = OrderedDict()


def _cell_entry(cell_w: int, cell_h: int, font: int) -> dict:
    key = (cell_w, cell_h, font)
    entry = _CELLS.get(key)
    if entry is None:
        entry = {"alpha": {}, "coverage": None, "ramps": {}}
        _CELLS[key] = entry
        while len(_CELLS) > _CELL_CACHE_MAX:
            _CELLS.popitem(last=False)
    else:
        _CELLS.move_to_end(key)
    return entry


def clear_caches() -> None:
    """Drop every cached raster/coverage/ramp (tests, and cell-size churn)."""
    _CELLS.clear()


# ---------------------------------------------------------------------------
# Rasterisation + coverage measurement
# ---------------------------------------------------------------------------

def render_alpha(ch: str, cell_w: int, cell_h: int, weight: float = 1.5,
                 font: int = FONT, fill: float = FILL) -> np.ndarray:
    """Rasterise one glyph at one stroke weight into a (cell_h, cell_w) alpha
    plane, float32 in [0, 1].

    Drawn into an SSx supersampled cell with LINE_AA and box-filtered down, so
    the alpha carries the glyph's real partial coverage rather than a jagged
    1-bit stamp. The glyph box is scaled to *fill* of the cell height, clamped
    to 0.95 of the cell width, and centred. Cached per cell size.
    """
    entry = _cell_entry(cell_w, cell_h, font)
    cache = entry["alpha"]
    key = (ch, weight, fill)
    got = cache.get(key)
    if got is not None:
        return got

    W, H = cell_w * SS, cell_h * SS
    big = np.zeros((H, W), np.uint8)
    if ch != " ":
        (tw, th), _ = cv2.getTextSize(ch, font, 1.0, 1)
        scale = (H * fill) / float(max(th, 1))
        (tw, th), _ = cv2.getTextSize(ch, font, scale, 1)
        if tw > W * 0.95 and tw > 0:
            scale *= (W * 0.95) / tw
            (tw, th), _ = cv2.getTextSize(ch, font, scale, 1)
        # thickness scales with the cell so a ramp authored at 8x16 keeps its
        # relative weight at 24x48 (4K) — "authored once, legible 720p->4K"
        thick = max(1, int(round(weight * SS * cell_h / 16.0)))
        cv2.putText(big, ch, ((W - tw) // 2, (H + th) // 2), font, scale,
                    255, thick, cv2.LINE_AA)
    alpha = (cv2.resize(big, (cell_w, cell_h), interpolation=cv2.INTER_AREA)
             .astype(np.float32) / 255.0)
    cache[key] = alpha
    return alpha


def coverage_table(cell_w: int, cell_h: int, pool=PREF, weights=WEIGHTS,
                   font: int = FONT) -> dict:
    """{(glyph, weight): mean alpha} for the whole pool at this cell size.

    Coverage — the mean alpha of the rendered cell — is the number the eye
    integrates over a cell, and it is a property of (glyph, weight, cell size,
    font), never of the character's code point. Cached per cell size; the
    table is palette- and ramp-length-independent.
    """
    entry = _cell_entry(cell_w, cell_h, font)
    table = entry["coverage"]
    if table is None:
        table = {
            (c, w): float(render_alpha(c, cell_w, cell_h, w, font).mean())
            # the blank cell is the same blank cell at every weight
            for c in pool for w in (weights if c != " " else weights[:1])
        }
        entry["coverage"] = table
    return table


def build_ramp(n: int, cell_w: int, cell_h: int, pool=PREF, weights=WEIGHTS,
               font: int = FONT, tol: float = 0.35):
    """Pick *n* (glyph, weight) pairs spanning the achievable coverage range.

    Walk n evenly spaced coverage targets and take, for each, the candidate
    nearest the target — ties inside *tol* of one step broken by the pool's
    preference order. One glyph may appear at most once (at any weight), so
    every step of the ramp is a visibly different character. Returned sorted by
    measured coverage, i.e. monotonic by construction.

    The searched weight set is narrowed for a short ramp (:func:`weights_for`),
    because a heavy stroke on a short ramp is a cell-filling blob rather than a
    glyph.

    Returns [((glyph, weight), coverage), ...], length n.
    """
    n = max(int(n), 1)
    entry = _cell_entry(cell_w, cell_h, font)
    cached = entry["ramps"].get((n, tol))
    if cached is not None:
        return cached

    table = coverage_table(cell_w, cell_h, pool, weights, font)
    keep = set(weights_for(n, weights))
    items = sorted(((k, v) for k, v in table.items() if k[1] in keep),
                   key=lambda kv: kv[1])
    lo, hi = items[0][1], items[-1][1]
    step = (hi - lo) / max(n - 1, 1)
    rank = {c: i for i, c in enumerate(pool)}

    chosen: dict = {}
    for k in range(n):
        target = lo + step * k
        used = {c for c, _ in chosen}
        rest = [kv for kv in items if kv[0][0] not in used]
        if not rest:
            break
        best = min(abs(v - target) for _, v in rest)
        band = [kv for kv in rest if abs(kv[1] - target) <= max(best, step * tol)]
        key, val = min(band, key=lambda kv: (rank[kv[0][0]], kv[0][1]))
        chosen[key] = val

    ramp = sorted(chosen.items(), key=lambda kv: kv[1])
    entry["ramps"][(n, tol)] = ramp
    return ramp


# ---------------------------------------------------------------------------
# Atlas + tone LUT
# ---------------------------------------------------------------------------

def build_atlas(ramp, cell_w: int, cell_h: int, off_rgb, on_rgb,
                font: int = FONT):
    """Pre-colour every ramp step into a (N, cell_h, cell_w, 3) uint8 atlas.

    Also returns each tile's mean **linear** luminance as actually composited
    under this palette — including the anti-aliased edge pixels, honestly. That
    curve, not the raw coverage, is what :func:`build_pos_lut` inverts.
    """
    off, on = np.float32(off_rgb), np.float32(on_rgb)
    tiles = np.empty((len(ramp), cell_h, cell_w, 3), np.uint8)
    lum = np.empty(len(ramp), np.float32)
    for i, ((ch, weight), _cov) in enumerate(ramp):
        a = render_alpha(ch, cell_w, cell_h, weight, font)[:, :, None]
        rgb = off[None, None, :] + a * (on - off)[None, None, :]
        tiles[i] = np.clip(rgb, 0, 255).astype(np.uint8)
        lum[i] = float((srgb_to_linear(rgb / 255.0) * _LUMA).sum(-1).mean())
    return tiles, lum


def build_pos_lut(lum: np.ndarray, off_rgb, on_rgb,
                  gamma: bool = True) -> np.ndarray:
    """256-entry uint8 LUT: source code value -> ramp POSITION (0..255 spans
    glyph 0..N-1).

    Built by inverting the ramp's measured luminance curve against a tone
    target interpolated between the palette's two luminances — so the target
    is normalised to the INK colour, not to the ramp's achievable range (the
    difference is a whole stop of apparent brightness; see the module
    docstring). A non-uniform ramp therefore still reproduces tone correctly,
    and the quantiser downstream stays a plain uniform ordered dither.
    """
    v = np.arange(256, dtype=np.float32) / 255.0
    t = srgb_to_linear(v) if gamma else v
    l_bg = float((srgb_to_linear(np.float32(off_rgb) / 255.0) * _LUMA).sum())
    l_ink = float((srgb_to_linear(np.float32(on_rgb) / 255.0) * _LUMA).sum())
    target = l_bg + t * (l_ink - l_bg)
    idx = np.arange(len(lum), dtype=np.float32)
    ascending = lum[0] <= lum[-1]
    # np.interp needs an increasing x; a dark-ink palette measures downwards,
    # so feed it reversed and the position curve inverts itself.
    pos = np.interp(target, lum if ascending else lum[::-1],
                    idx if ascending else idx[::-1])
    span = max(len(lum) - 1, 1)
    return np.clip(pos / span * 255.0 + 0.5, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Grid geometry
# ---------------------------------------------------------------------------

def grid_for(frame_w: int, frame_h: int, rows_req: float):
    """(cell_w, cell_h, cols, rows, clamped) for a requested row count.

    ``cell_w = cell_h / 2`` and ``cols`` derives from ``cell_w``, so the
    character grid carries the frame's aspect ratio and the subject is not
    stretched 2x vertically (sampling on a square grid and drawing 2:1 cells
    is the classic ASCII-art mistake). At 45 rows that is a 160x45 grid at
    720p, 1080p, 1440p and 4K alike — one authored look, every output res.

    ``clamped`` is True when the requested row count hit ``MIN_CELL_H`` and the
    grid stopped getting finer; the panel says so rather than pretending the
    slider still does something.
    """
    frame_w, frame_h = max(int(frame_w), 1), max(int(frame_h), 1)
    rows_req = max(float(rows_req), 1.0)
    raw = frame_h / rows_req
    # the floor first, the frame second: a cell can never exceed the frame it
    # is tiled into (the panel's 16 px swatch strip renders through this too)
    cell_h = min(max(int(round(raw)), MIN_CELL_H), frame_h)
    cell_w = min(max(3, int(round(cell_h * ASPECT))), frame_w)
    rows = max(1, frame_h // cell_h)
    cols = max(1, frame_w // cell_w)
    return cell_w, cell_h, cols, rows, raw < MIN_CELL_H - 0.5


# ---------------------------------------------------------------------------
# The renderer
# ---------------------------------------------------------------------------

class AsciiRenderer:
    """A configured ASCII quantiser: grid + ramp + atlas + tone LUT.

    Setup does all the expensive work (coverage scan on a cold cell size, ramp
    search, atlas raster, LUT build); :meth:`render` is a downsample, a LUT, an
    integer ordered dither and one gather-blit. :meth:`key` is the identity to
    compare against.

    **Retune with :meth:`configure`, do not construct a second renderer**, for
    anything short of an output-resolution change. The frame buffer dominates
    setup — at 4K a warm construction measured 35.0 ms of which 31.9 ms was the
    24 MB allocate-and-clear — so building a whole renderer for a palette nudge
    turned a Hue drag into 46 ms/frame (22 fps) against 7 ms idle. `configure`
    keeps the buffer, and rebuilds only the parts a change actually
    invalidates: the ramp needs the grid and n, the atlas needs the ramp and
    the palette, the LUT needs the atlas and gamma, the strided view and the
    noise tile need the grid. A palette-only change is atlas + LUT alone.
    """

    def __init__(self, frame_w: int, frame_h: int, rows_req: float = 45.0,
                 n: int = 16, palette=((0, 0, 0), (255, 255, 255)),
                 gamma: bool = True, bias: str = "auto", grain: bool = True,
                 pool=PREF, weights=WEIGHTS, font: int = FONT):
        self.frame_w, self.frame_h = int(frame_w), int(frame_h)
        self._pool, self._weights, self._font = pool, weights, font
        self._ramp_key = self._atlas_key = self._lut_key = self._grid = None

        # Allocated once and reused by every later configure(). Cleared to the
        # background here so a caller that reads `out` before the first render
        # sees a frame rather than uninitialised memory; render() rewrites the
        # whole buffer (grid + border) so the clear is never needed again.
        self.out = np.empty((self.frame_h, self.frame_w, 3), np.uint8)
        self.out[:] = np.uint8(palette[0])
        self.configure(rows_req, n, palette, gamma, bias, grain)

    def configure(self, rows_req: float, n: int, palette, gamma: bool,
                  bias: str, grain: bool = True) -> None:
        """Re-point this renderer at a new setting, in place.

        Same frame size only — the buffer is what this exists to keep. Every
        stage is guarded by the inputs it actually depends on, so the cost
        scales with what changed rather than with the output resolution.
        """
        n = max(int(n), 1)
        palette = (tuple(palette[0]), tuple(palette[1]))
        gamma, grain, bias = bool(gamma), bool(grain), str(bias)
        cw, ch, cols, rows, clamped = grid_for(self.frame_w, self.frame_h,
                                               rows_req)
        self.rows_req, self.clamped = float(rows_req), clamped
        self.cell_w, self.cell_h, self.cols, self.rows = cw, ch, cols, rows
        self.palette, self.gamma = palette, gamma
        self.bias, self.grain = bias, grain
        grid = (cw, ch, cols, rows)

        if (grid, n) != self._ramp_key:
            self.ramp = build_ramp(n, cw, ch, self._pool, self._weights,
                                   self._font)
            self.n = len(self.ramp)
            self._ramp_key = (grid, n)
            self._atlas_key = self._lut_key = None
        if (self._ramp_key, palette) != self._atlas_key:
            self.atlas, self.lum = build_atlas(self.ramp, cw, ch, palette[0],
                                               palette[1], self._font)
            self._atlas_key = (self._ramp_key, palette)
            self._lut_key = None
        if (self._atlas_key, gamma) != self._lut_key:
            self.pos_lut = build_pos_lut(self.lum, palette[0], palette[1],
                                         gamma)
            self._lut_key = (self._atlas_key, gamma)

        if grid != self._grid:
            # ramp-position dither: the project's own 64x64 blue-noise texture,
            # tiled over the CHARACTER grid (not the pixel grid) — same asset,
            # same code shape and same bias semantics as _ordered_dither, one
            # level up.
            bn = _blue_noise_matrix()
            ty = int(np.ceil(rows / bn.shape[0]))
            tx = int(np.ceil(cols / bn.shape[1]))
            self.noise = (np.tile(bn, (ty, tx))[:rows, :cols]
                          * 255.0).astype(np.uint16)

            # A strided (rows, cols, cell_h, cell_w, 3) view of the centred
            # character grid inside the frame buffer, so the blit is one
            # contiguous `view[:] = atlas[idx]` (0.44 ms at 720p — the fastest
            # of five strategies measured, and 30x faster than per-cell
            # putText). A view, so it costs nothing to re-take.
            gh, gw = rows * ch, cols * cw
            self.grid_h, self.grid_w = gh, gw
            self.x0 = (self.frame_w - gw) // 2
            self.y0 = (self.frame_h - gh) // 2
            self.view = (self.out[self.y0:self.y0 + gh, self.x0:self.x0 + gw]
                         .reshape(rows, ch, cols, cw, 3)
                         .transpose(0, 2, 1, 3, 4))
            self._grid = grid

    # ----- identity -----
    @staticmethod
    def make_key(frame_w, frame_h, rows_req, n, palette, gamma, bias, grain):
        cw, ch, cols, rows, _ = grid_for(frame_w, frame_h, rows_req)
        return (frame_w, frame_h, cw, ch, cols, rows, int(n),
                (tuple(palette[0]), tuple(palette[1])), bool(gamma), str(bias),
                bool(grain))

    def key(self):
        return (self.frame_w, self.frame_h, self.cell_w, self.cell_h,
                self.cols, self.rows, self.n, self.palette, self.gamma,
                self.bias, self.grain)

    def set_rows_req(self, rows_req):
        """Re-read the requested row count without rebuilding.

        Past the legibility floor the grid stops changing but the REQUEST keeps
        going, and `clamped` is a property of the request, not of the geometry.
        Rebuilding the atlas just to flip a boolean would put a 7-30 ms hitch
        on the far half of the Scale slider, where by definition nothing about
        the picture is changing."""
        self.rows_req = float(rows_req)
        self.clamped = grid_for(self.frame_w, self.frame_h, rows_req)[4]

    # ----- per-frame -----
    def render(self, gray_u8: np.ndarray, contrast: float = 1.0) -> np.ndarray:
        """One grayscale uint8 frame (any size) -> the full RGB output frame.

        The returned array is this renderer's own buffer, and every pixel of it
        is rewritten on every call (grid *and* the sub-cell border) — so a
        caller that blacks it out or glitches it in place cannot leave a stale
        edge behind.
        """
        small = self._sample(gray_u8)
        if contrast != 1.0:
            small = cv2.convertScaleAbs(small, alpha=contrast,
                                        beta=127.5 * (1.0 - contrast))
        p = cv2.LUT(small, self.pos_lut).astype(np.uint16)   # position 0..255
        m = self.n - 1
        # floor(p/255 * m + t), t in [0, 1) — the same quantiser shape as
        # _ordered_dither, and the divisor is 255 rather than a >> 8 shift on
        # purpose: dividing by 256 pulls every level fractionally down, which
        # costs the ENDPOINTS. Measured on a letterboxed frame, >> 8 sprinkled
        # glyph 1 across pure black (p=0 is safe, but p=255 lands on m-1 for
        # most of the noise texture, and under `bias auto` that inverts into
        # the blacks). Exact division keeps 0 -> sparsest and 255 -> densest
        # for every threshold, which is the property bayer_dither's docstring
        # promises and the reason its blacks stay black.
        if self.grain and m > 0:
            invert = self.bias == "dark" or (self.bias == "auto"
                                             and self._is_dark(small))
            if invert:
                # dither the negative and invert back: mirrored pattern,
                # upward-rounding density, dim detail preserved (dither.py's
                # _ordered_dither docstring, applied to the glyph index)
                np.subtract(np.uint16(255), p, out=p)
            idx = ((p * m + self.noise) // 255).astype(np.uint8)
            if invert:
                np.subtract(np.uint8(m), idx, out=idx)
        else:
            idx = ((p * m + 127) // 255).astype(np.uint8)
        self.view[:] = self.atlas[idx]
        self._paint_border()
        return self.out

    def _sample(self, gray_u8: np.ndarray) -> np.ndarray:
        """Area-average the source down to (cols, rows).

        Sampled from a centred ROI carrying the character grid's aspect ratio,
        with the ROI trimmed to an exact integer multiple of the grid whenever
        that costs under 2% of the picture: cv2's INTER_AREA has a fast integer
        path and a 17x slower general one (measured 0.19 ms vs 3.36 ms at
        720p), and this is the whole ballgame for the frame budget.
        """
        sh, sw = gray_u8.shape[:2]
        want = self.grid_w / float(self.grid_h)
        if sw / float(sh) > want:
            rw_roi, rh_roi = int(round(sh * want)), sh
        else:
            rw_roi, rh_roi = sw, int(round(sw / want))
        rw_roi = max(min(rw_roi, sw), self.cols)
        rh_roi = max(min(rh_roi, sh), self.rows)
        trim_w = rw_roi - rw_roi % self.cols
        if trim_w >= rw_roi * 0.98:
            rw_roi = trim_w
        trim_h = rh_roi - rh_roi % self.rows
        if trim_h >= rh_roi * 0.98:
            rh_roi = trim_h
        x0, y0 = (sw - rw_roi) // 2, (sh - rh_roi) // 2
        roi = gray_u8[y0:y0 + rh_roi, x0:x0 + rw_roi]
        return cv2.resize(roi, (self.cols, self.rows),
                          interpolation=cv2.INTER_AREA)

    def _paint_border(self):
        """Repaint the sub-cell margin around the centred grid. Cheap (it is
        under one cell wide) and unconditional, so the buffer is wholly
        rewritten every frame."""
        bg = np.uint8(self.palette[0])
        y1, x1 = self.y0 + self.grid_h, self.x0 + self.grid_w
        if self.y0:
            self.out[:self.y0] = bg
        if y1 < self.frame_h:
            self.out[y1:] = bg
        if self.x0:
            self.out[self.y0:y1, :self.x0] = bg
        if x1 < self.frame_w:
            self.out[self.y0:y1, x1:] = bg

    def _is_dark(self, small: np.ndarray) -> bool:
        mean = float(small.mean()) / 255.0
        if self.gamma:
            return float(srgb_to_linear(np.float32(mean))) < _MID_GREY_LINEAR
        return mean < 0.5

    # ----- introspection (panel copy, tests) -----
    def ramp_str(self) -> str:
        """The ramp as a string, sparse -> dense."""
        return "".join(ch for (ch, _w), _c in self.ramp)

    def grid_note(self) -> str:
        """The TONE readout line: 'grid 160 x 45 chars - cell 8x16'."""
        return (f"grid {self.cols} x {self.rows} chars - "
                f"cell {self.cell_w}x{self.cell_h}")
