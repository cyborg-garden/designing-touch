"""ASCII renderer — dtouch/ascii_art.py (DESIGN.md §4.2).

The claims worth pinning here are the ones that were arrived at by rendering
the alternative and looking at it, because a picture is exactly what a future
change will not re-check: the ramp is monotonic and every glyph in it is
reachable; the grid carries the frame's aspect at every output resolution; the
tone target is normalised to the ink colour so mid-grey survives; the
endpoints are exact; and the whole step stays in the ordered-dither cost class.
"""
import time

import numpy as np
import pytest

from dtouch.ascii_art import (MIN_CELL_H, WEIGHTS, AsciiRenderer, build_atlas,
                              build_pos_lut, build_ramp, clear_caches,
                              coverage_table, grid_for, render_alpha,
                              weights_for)
from dtouch.dither import srgb_to_linear

WOB = ((0, 0, 0), (255, 255, 255))
BOW = ((245, 245, 245), (16, 16, 16))
LUMA = np.float32([0.2126, 0.7152, 0.0722])


def _lin_luma(rgb_u8):
    return srgb_to_linear(np.asarray(rgb_u8, np.float32) / 255.0) @ LUMA


def _cell_indices(rend, out):
    """Which atlas tile each cell of *out* is — matched by mean colour, which
    is unique per tile because the ramp is monotonic in coverage."""
    gh, gw = rend.grid_h, rend.grid_w
    cells = (out[rend.y0:rend.y0 + gh, rend.x0:rend.x0 + gw]
             .reshape(rend.rows, rend.cell_h, rend.cols, rend.cell_w, 3)
             .transpose(0, 2, 1, 3, 4)
             .mean(axis=(2, 3, 4)))
    tiles = rend.atlas.mean(axis=(1, 2, 3))
    return np.abs(cells[:, :, None] - tiles[None, None, :]).argmin(axis=2)


# ---------- the ramp (measured, not folklore) ----------

@pytest.mark.parametrize("n", [2, 4, 8, 16])
def test_ramp_is_monotonic_and_the_right_length(n):
    ramp = build_ramp(n, 8, 16)
    assert len(ramp) == n
    covs = [c for _k, c in ramp]
    assert covs == sorted(covs)
    assert covs[0] == pytest.approx(0.0)          # the blank cell anchors it


def test_ramp_uses_each_glyph_at_most_once_and_stays_ascii():
    ramp = build_ramp(16, 8, 16)
    glyphs = [ch for (ch, _w), _c in ramp]
    assert len(set(glyphs)) == len(glyphs)
    joined = "".join(glyphs)
    assert joined == joined.encode("ascii", "replace").decode()


def test_no_bar_glyphs_in_the_pool():
    """`| l I 1` align column-to-column at 4-8 px cell widths and read as scan
    bars rather than as texture — they are excluded from the candidate pool,
    not merely unlikely to be chosen."""
    for cell in ((4, 8), (8, 16), (12, 24)):
        chosen = "".join(ch for (ch, _w), _c in build_ramp(16, *cell))
        assert not (set(chosen) & set("|lI1"))


def test_stroke_weight_is_part_of_the_ramp():
    """Hershey is a STROKE font: at one fixed thickness the densest glyph
    covers ~0.38 of the cell, which clips every highlight above sRGB ~0.65.
    Searching (glyph x weight) jointly must reach materially further."""
    table = coverage_table(8, 16)
    thin = max(v for (_c, w), v in table.items() if w == 1.0)
    joint = max(table.values())
    assert thin < 0.45
    assert joint > 0.6
    # and the ramp must actually spend the extra range
    assert build_ramp(16, 8, 16)[-1][1] > 0.6


@pytest.mark.parametrize("n,searched", [(2, 1), (4, 1), (8, 2), (12, 3),
                                        (16, 4)])
def test_a_short_ramp_does_not_spend_the_heavy_strokes(n, searched):
    """A heavy stroke buys REACH, and reach is only worth having when there
    are enough steps to travel there gradually. Spent on a short ramp it is a
    cell-filling blob: ASCII's first impression is Bits 1, which shipped as
    `' s'` and rendered as a coarse dot halftone with no glyph structure at
    all, and Bits 2 (`' :*#'`) drew its top steps at 3 px stroke in an 8 px
    cell and merged them into solid white rows. Both were rendered and looked
    at beside the capped versions."""
    assert weights_for(n) == WEIGHTS[:searched]
    for cell in ((8, 16), (12, 24), (24, 48)):
        ramp = build_ramp(n, *cell)
        used = {w for (_c, w), _v in ramp}
        assert used <= set(WEIGHTS[:searched])


@pytest.mark.parametrize("n", [2, 4, 8])
def test_every_bits_setting_draws_glyphs_rather_than_blobs(n):
    """The claim behind the cap, measured on the mark it actually changes: the
    densest step of a short ramp must still leave most of its cell unlit — a
    letter you can read, not a filled block. Uncapped, Bits 1 topped out at
    's' at 3.0 (52% unlit) and Bits 2 at '#' at 3.0 (34%)."""
    for cell in ((8, 16), (12, 24), (24, 48)):
        (ch, weight), _cov = build_ramp(n, *cell)[-1]
        alpha = render_alpha(ch, cell[0], cell[1], weight)
        assert float((alpha < 0.5).mean()) >= 0.50, (
            "n=%d at %s tops out on %r at %s, a blob" % (n, cell, ch, weight))


def test_the_folklore_ramp_is_not_monotonic_but_ours_is():
    """The measured justification for not shipping `" .:-=+*#%@"`: four of its
    nine steps go backwards at the cell size it would be drawn at."""
    table = coverage_table(8, 16)
    folklore = [table[(c, 2.2)] if c != " " else 0.0 for c in " .:-=+*#%@"]
    backwards = sum(1 for a, b in zip(folklore, folklore[1:]) if b < a)
    assert backwards >= 3, "if Hershey ever changes, re-derive the ramp"
    ours = [c for _k, c in build_ramp(10, 8, 16)]
    assert ours == sorted(ours)


# ---------- the grid ----------

@pytest.mark.parametrize("w,h", [(1280, 720), (1920, 1080), (2560, 1440),
                                 (3840, 2160)])
def test_45_rows_is_the_same_character_grid_at_every_output_res(w, h):
    """DESIGN.md §5 'authored once, legible 720p->4K': the character grid, not
    the cell size, is what a look actually specifies."""
    cw, ch, cols, rows, clamped = grid_for(w, h, 45)
    assert rows == 45 and cols == 160
    assert not clamped
    assert ch == h // 45 and cw == ch // 2


@pytest.mark.parametrize("w,h", [(1280, 720), (1920, 1080), (3840, 2160)])
@pytest.mark.parametrize("req", [30, 45, 60, 72, 90])
def test_grid_aspect_tracks_the_frame_aspect(w, h, req):
    """Sampling on a square grid and drawing 2:1 cells stretches the subject
    2x vertically. cols derives from cell_w = cell_h/2, so it must not."""
    cw, ch, cols, rows, _ = grid_for(w, h, req)
    assert abs((cols * cw) / (rows * ch) - w / h) < 0.06
    assert abs(cw / ch - 0.5) < 0.09          # 8x15 at 1080p/72 is the worst


def test_cell_floor_clamps_and_says_so():
    """`clamped` answers "does pushing this slider further do anything", not
    "did max() fire". The cell rounds to an integer, so at 720p every request
    from ~85 rows up already delivers the same 8 px cell — the old predicate
    (`raw < MIN_CELL_H - 0.5`) did not report until 97, leaving a twelve-row
    dead span that the panel called live travel."""
    live = grid_for(1280, 720, 80)                  # 9 px cells, still moving
    edge = grid_for(1280, 720, 85)                  # rounds onto the floor
    floor = grid_for(1280, 720, 90)                 # exactly on it
    past = grid_for(1280, 720, 720)
    assert live[1] == 9 and not live[4]
    assert edge[1] == MIN_CELL_H and edge[4]
    assert floor[1] == MIN_CELL_H and floor[4]
    assert past[:4] == floor[:4] and past[4]        # and nothing moved
    assert edge[:4] == floor[:4]
    assert not grid_for(1280, 720, 45)[4]


def test_a_cell_never_exceeds_the_frame_it_tiles():
    """The panel's 16 px swatch strip renders through this too."""
    for h in (4, 8, 11, 16):
        cw, ch, cols, rows, _ = grid_for(200, h, 1)
        assert ch <= h and rows * ch <= h and cols * cw <= 200


# ---------- tone (gamma-correct, ink-normalised) ----------

def test_mid_grey_survives_the_round_trip():
    """The trap the design names: normalising the tone target to the ramp's
    achievable RANGE instead of to the ink colour maps sRGB mid-grey to ~0.42
    apparent instead of 0.50 and halves the picture's brightness."""
    rend = AsciiRenderer(640, 360, 45, 16, WOB, gamma=True, bias="light")
    out = rend.render(np.full((360, 640), 128, np.uint8))
    got = float(_lin_luma(out[rend.y0:rend.y0 + rend.grid_h,
                              rend.x0:rend.x0 + rend.grid_w]).mean())
    want = float(srgb_to_linear(np.float32(128 / 255.0)))
    assert abs(got - want) < 0.02


def test_tone_tracks_the_source_until_the_ramp_clips():
    """A ramp topping out near 0.70 coverage clips above sRGB ~0.83; below
    that the output's mean linear luminance must follow the input's."""
    rend = AsciiRenderer(1280, 720, 45, 16, WOB, gamma=True, bias="light")
    grad = np.tile(np.linspace(0, 255, 1280).astype(np.uint8), (720, 1))
    out = rend.render(grad)
    for frac in (0.25, 0.5, 0.75):
        x = rend.x0 + int(frac * (rend.cols - 1)) * rend.cell_w
        got = float(_lin_luma(out[:, x:x + rend.cell_w]).mean())
        src = float(srgb_to_linear(
            grad[:, x:x + rend.cell_w].astype(np.float32) / 255.0).mean())
        assert abs(got - src) < 0.03, f"tone broke at {frac}"


def test_a_dark_ink_palette_inverts_the_lut_with_no_special_case():
    """Building the LUT from measured composited tile luminance (not from
    coverage) is what makes black-on-white work: the luminances run downwards
    and the interpolation follows. Ink density tracks input brightness for
    both palettes — which is the shipped `_palette_map` semantic, where a
    white input lands on the palette's `on` colour whichever end that is."""
    for palette in (WOB, BOW):
        rend = AsciiRenderer(320, 180, 45, 16, palette, gamma=True,
                             bias="light")
        lut = build_pos_lut(rend.lum, palette[0], palette[1], True)
        assert lut[0] == 0 and lut[255] == 255
        assert (np.diff(lut.astype(int)) >= 0).all()
    # ...and the two palettes disagree about which is darker, as they must
    assert _lin_luma(WOB[1]) > _lin_luma(WOB[0])
    assert _lin_luma(BOW[1]) < _lin_luma(BOW[0])


def test_the_atlas_luminance_curve_cannot_go_backwards():
    """`build_pos_lut` feeds this curve to np.interp, which needs a monotone
    x. The ramp is monotone in COVERAGE by construction, but composited
    luminance is a different number (sRGB encode/decode on 8-bit tiles) and it
    goes backwards by a step in some (res x rows x n x palette) combos — nine
    of the eleven shipped palettes at 4K and the default 45 rows. The
    magnitude is small and pos_lut still came out monotone, but a silent
    violation costs a tone, so build_atlas sorts instead of hoping."""
    from dtouch.modes.dithergirl import PALETTES

    reordered = 0
    for res in ((1280, 720), (1920, 1080), (3840, 2160)):
        for rows in (30, 45, 72):
            cw, ch, _cols, _rows, _c = grid_for(res[0], res[1], rows)
            for n in (2, 4, 8, 16):
                ramp = build_ramp(n, cw, ch)
                for name, (off, on) in PALETTES.items():
                    out, tiles, lum = build_atlas(list(ramp), cw, ch, off, on)
                    d = np.diff(lum.astype(np.float64))
                    assert (d >= 0).all() or (d <= 0).all(), (
                        "%s at %s / %s rows / n=%d" % (name, res, rows, n))
                    assert len(out) == len(ramp) == len(tiles) == len(lum)
                    assert sorted(k for k, _v in out) == sorted(
                        k for k, _v in ramp)          # same steps, reordered
                    if [k for k, _v in out] != [k for k, _v in ramp]:
                        reordered += 1
    assert reordered, "if this stops firing the sort has become dead code"


def test_bias_auto_means_on_the_same_domain_as_the_pixel_dithers():
    """`_ordered_dither` means the LINEARIZED plane; ASCII used to linearize
    the MEAN. By Jensen those differ on any frame that is not flat — most on a
    high-contrast one, which is this app's own use case. And because
    srgb_to_linear is monotone and _MID_GREY_LINEAR is srgb_to_linear(0.5),
    linearizing the mean made the gamma-on and gamma-off branches the SAME
    predicate: the Gamma toggle provably could not move the decision."""
    from dtouch.dither import _MID_GREY_LINEAR

    small = np.zeros((10, 10), np.uint8)
    small[:3] = 255                       # 30% of a dark frame is lit
    plane = small.astype(np.float32) / 255.0

    rend = AsciiRenderer(640, 360, 45, 16, WOB, gamma=True)
    want = float(srgb_to_linear(plane).mean()) < _MID_GREY_LINEAR
    assert rend._is_dark(small) is want
    assert want is False                  # in linear light this frame is LIT
    assert float(plane.mean()) < 0.5      # ...and the old rule called it dark

    # so Gamma now moves it, which is the whole point of the toggle
    plain = AsciiRenderer(640, 360, 45, 16, WOB, gamma=False)
    assert plain._is_dark(small) is True


@pytest.mark.parametrize("palette", [WOB, BOW])
def test_the_endpoints_are_exact(palette):
    """0 -> the sparsest glyph and 255 -> the densest, for EVERY threshold in
    the noise texture. `>> 8` instead of `// 255` pulls every level
    fractionally down and sprinkles glyph 1 across pure black."""
    rend = AsciiRenderer(640, 360, 45, 16, palette, gamma=True, bias="light")
    black = _cell_indices(rend, rend.render(np.zeros((360, 640), np.uint8)))
    assert (black == 0).all()
    white = _cell_indices(rend, rend.render(np.full((360, 640), 255, np.uint8)))
    assert (white == rend.n - 1).all()


def test_every_glyph_in_the_ramp_is_reachable():
    """A ramp with dead steps is a shorter ramp that costs the same."""
    rend = AsciiRenderer(1280, 720, 45, 16, WOB, gamma=True, bias="light")
    grad = np.tile(np.linspace(0, 255, 1280).astype(np.uint8), (720, 1))
    used = set(np.unique(_cell_indices(rend, rend.render(grad))).tolist())
    assert used == set(range(rend.n))


# ---------- grain (the ramp-position dither) ----------

def test_grain_breaks_gradient_banding():
    """Without it a smooth gradient posterises into hard bands of one glyph."""
    grad = np.tile(np.linspace(40, 210, 640).astype(np.uint8), (360, 1))
    plain = AsciiRenderer(640, 360, 45, 4, WOB, bias="light", grain=False)
    noisy = AsciiRenderer(640, 360, 45, 4, WOB, bias="light", grain=True)
    a = _cell_indices(plain, plain.render(grad))
    b = _cell_indices(noisy, noisy.render(grad))
    # a banded render has the same glyph down every column; a dithered one
    # mixes two neighbouring glyphs within a column
    assert (a.std(axis=0) == 0).all()
    assert (b.std(axis=0) > 0).mean() > 0.4


def test_bias_mirrors_the_pattern_and_auto_follows_the_frame():
    """Same semantics as `_ordered_dither`, one level up: the two directions
    light the same NUMBER of cells (the mean is preserved either way) but a
    mirrored SET of them, and `auto` picks per frame off the working-domain
    mean against mid-grey."""
    def idx(bias, level):
        r = AsciiRenderer(640, 360, 45, 16, WOB, bias=bias)
        return _cell_indices(r, r.render(np.full((360, 640), level, np.uint8)))

    light, dark = idx("light", 20), idx("dark", 20)
    assert not np.array_equal(light, dark)
    assert abs(light.mean() - dark.mean()) < 0.02
    assert np.array_equal(idx("auto", 20), dark)                    # dark
    assert np.array_equal(idx("auto", 200), idx("light", 200))      # bright


def test_a_static_frame_gives_a_static_pattern():
    """The threshold texture is static, so nothing shimmers between frames."""
    rend = AsciiRenderer(640, 360, 45, 16, WOB)
    frame = np.tile(np.linspace(0, 255, 640).astype(np.uint8), (360, 1))
    first = rend.render(frame).copy()
    assert np.array_equal(first, rend.render(frame))


# ---------- the output buffer ----------

def test_the_whole_buffer_is_rewritten_every_frame():
    """The shell blacks `out` in place for blackout and the glitch rack may
    write through it. A reused buffer that only repaints its character grid
    would keep a black margin forever after the first blackout."""
    rend = AsciiRenderer(1000, 333, 45, 16, BOW)   # 333 % 8 leaves a margin
    frame = np.full((333, 1000), 90, np.uint8)
    out = rend.render(frame)
    assert rend.grid_h < rend.frame_h or rend.grid_w < rend.frame_w
    out[:] = 0                                     # what blackout does
    again = rend.render(frame)
    assert not (again == 0).all()
    assert tuple(again[0, 0]) == tuple(np.uint8(BOW[0]))   # margin repainted


def test_the_grid_is_centred():
    rend = AsciiRenderer(1000, 333, 45, 16, WOB)
    assert rend.x0 == (1000 - rend.grid_w) // 2
    assert rend.y0 == (333 - rend.grid_h) // 2
    assert 0 <= rend.x0 < rend.cell_w and 0 <= rend.y0 < rend.cell_h


def test_any_source_size_renders_without_stretching():
    """The camera is not the output: the ROI is cropped to the grid's aspect
    before it is area-averaged down."""
    rend = AsciiRenderer(1280, 720, 45, 16, WOB)
    for src in ((480, 640), (720, 1280), (1080, 1920), (240, 320)):
        out = rend.render(np.full(src, 128, np.uint8))
        assert out.shape == (720, 1280, 3)


# ---------- identity + caching ----------

def test_key_identifies_a_rebuild_exactly():
    args = dict(rows_req=45, n=16, palette=WOB, gamma=True, bias="auto",
                grain=True)
    rend = AsciiRenderer(640, 360, **args)
    assert rend.key() == AsciiRenderer.make_key(640, 360, **args)
    for change in (dict(rows_req=30), dict(n=8), dict(palette=BOW),
                   dict(gamma=False), dict(bias="light")):
        assert rend.key() != AsciiRenderer.make_key(640, 360,
                                                    **{**args, **change})
    # a Scale nudge too small to move the cell size must NOT force a rebuild
    assert rend.key() == AsciiRenderer.make_key(640, 360,
                                                **{**args, "rows_req": 45.4})


def test_the_request_is_re_read_without_a_rebuild():
    """Past the floor the grid stops changing, so nothing rebuilds — but
    `clamped` is a property of the REQUEST, and the panel has to be able to
    say the slider has run out of room."""
    rend = AsciiRenderer(1280, 720, 45, 16, WOB)
    assert not rend.clamped
    before = rend.atlas
    rend.set_rows_req(720)
    assert rend.clamped and rend.atlas is before
    rend.set_rows_req(45)
    assert not rend.clamped


def test_the_cell_cache_is_bounded():
    """Dragging Scale at 4K walks ~40 distinct cell heights; caching every
    one's 272 rasters forever is tens of MB."""
    clear_caches()
    from dtouch import ascii_art

    # against the LITERAL, not against the constant: `len(cache) <= the
    # constant that sets len(cache)` passes for any value of the constant,
    # including one that puts tens of MB of alpha rasters back
    assert ascii_art._CELL_CACHE_MAX == 6
    for h in range(9, 40):
        build_ramp(4, max(3, h // 2), h)
    assert len(ascii_art._CELLS) <= 6


# ---------- cost ----------

@pytest.mark.parametrize("res,rows,budget", [((1280, 720), 45, 2.5),
                                             ((1280, 720), 90, 3.0),
                                             ((1920, 1080), 45, 4.0)])
def test_the_step_stays_in_the_ordered_dither_cost_class(res, rows, budget):
    """Measured 0.6-1.4 ms on the development machine; the budgets here are
    generous multiples so this fails on an algorithmic regression (a per-cell
    putText path measures 20.7 ms at 720p) rather than on a busy CI box."""
    rend = AsciiRenderer(res[0], res[1], rows, 16, WOB)
    frame = np.tile(np.linspace(0, 255, res[0]).astype(np.uint8), (res[1], 1))
    for _ in range(3):
        rend.render(frame)
    t0 = time.perf_counter()
    for _ in range(10):
        rend.render(frame)
    assert (time.perf_counter() - t0) / 10 * 1000 < budget


def test_the_grid_note_reads_like_a_grid():
    rend = AsciiRenderer(1280, 720, 45, 16, WOB)
    note = rend.grid_note()
    assert note == "grid 160 x 45 chars - cell 8x16"
    assert note == note.encode("ascii", "replace").decode()
    assert len(rend.ramp_str()) == 16
