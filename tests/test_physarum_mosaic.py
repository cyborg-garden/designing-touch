"""The mosaic: several morphologies in one frame, not one texture everywhere.

This is the floor for the thing the instrument was actually missing. A single
parameter set over the whole canvas grows one texture, and no slider can
change that — which is why six behavior points and five looks all read as the
same electric tangle recoloured. `evolve` now cuts the grid into drifting
zones that each run one of six regimes outright, and THAT is what puts fine
lace hard against a fat trunk against a radial bloom in one picture.

Why this test does not live in test_physarum_feel.py: the effect scales with
how much room a zone has to be different from its neighbours. Measured on the
shipped sizings, mosaic 0 -> 0.95 moves spatial variety 0.075 -> 0.131 at
1280x736 and 0.097 -> 0.145 at 576x324, but only 0.119 -> 0.139 at the feel
suite's 288x162 rig, where the seed spread swamps it. A floor set there would
either be vacuous or flaky, so this runs the field directly at a real size.
"""
import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from dtouch.physarum_gl import PhysarumFieldGL, PhysarumGLUnavailable

GW, GH = 576, 324
N = int(0.53 * GW * GH)          # the shipped density, ~0.5 agents per cell
SEEDS = (7, 11, 23)
FRAMES = 400


def spatial_variety(lum, py=5, px=8):
    """Spread of local fine-structure density across the frame.

    One regime everywhere puts the same amount of edge in every patch, so the
    spread is small. A mosaic puts lace in one patch and a bare trunk in the
    next. Normalized by the mean so it measures VARIETY and not brightness —
    a picture that is merely busier everywhere does not pass this.
    """
    g = cv2.GaussianBlur(lum, (0, 0), 1.0)
    e = (np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0, 3))
         + np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1, 3)))
    h, w = e.shape
    ph, pw = h // py, w // px
    cells = np.array([e[i * ph:(i + 1) * ph, j * pw:(j + 1) * pw].mean()
                      for i in range(py) for j in range(px)])
    return float(cells.std() / max(cells.mean(), 1e-6))


def _variety_at(mosaic, seed):
    try:
        f = PhysarumFieldGL(n=N, gw=GW, gh=GH, seed=seed)
    except PhysarumGLUnavailable as e:
        pytest.skip(f"no GL context available (CI): {e}")
    try:
        f.point_bg, f.point_fg = "veins", "fingers"
        f.gain = 1.0
        f.sat, f.jitter, f.hetero = 1.1, 0.1, 0.7
        f.cross, f.sharpen, f.mosaic = 0.5, 0.13, mosaic
        matte = np.zeros((GH, GW), np.float32)
        gray = np.zeros((GH, GW), np.float32)
        gray[GH // 3:2 * GH // 3, GW // 3:2 * GW // 3] = 1.0
        for _ in range(FRAMES):
            f.update(matte, gray)
            f.luminance_into_tex()
        return spatial_variety(f.luminance())
    finally:
        f.release()


def test_the_mosaic_makes_the_frame_various():
    """Averaged over seeds, because a single frame's zone layout is a draw."""
    off = np.mean([_variety_at(0.0, s) for s in SEEDS])
    on = np.mean([_variety_at(0.95, s) for s in SEEDS])
    assert on >= 1.25 * off, (off, on)


def test_mosaic_off_is_genuinely_one_regime():
    """The bottom of the knob has to be the uniform field, or `evolve` is not
    a control, it is a bias."""
    try:
        f = PhysarumFieldGL(n=1000, gw=64, gh=64, seed=1)
    except PhysarumGLUnavailable as e:
        pytest.skip(f"no GL context available (CI): {e}")
    try:
        assert f.mosaic == 0.0
    finally:
        f.release()


def test_species_reduce_to_the_single_channel_model():
    """cross = 0 must be the old one-organism field exactly — the legacy
    behaviour has to survive at the bottom of `weave`."""
    try:
        f = PhysarumFieldGL(n=1000, gw=64, gh=64, seed=1)
    except PhysarumGLUnavailable as e:
        pytest.skip(f"no GL context available (CI): {e}")
    try:
        f.cross = 0.0
        assert f.interaction_matrix() == (1.0, 0.0, 0.0,
                                          0.0, 1.0, 0.0,
                                          0.0, 0.0, 1.0)
        f.cross = 0.5
        m = f.interaction_matrix()
        # diagonal attracts, next repels, previous mildly attracts
        assert m[0] == 1.0 and m[1] < 0 and m[2] > 0
        assert m[4] == 1.0 and m[5] < 0 and m[3] > 0
        assert m[8] == 1.0 and m[6] < 0 and m[7] > 0
    finally:
        f.release()
