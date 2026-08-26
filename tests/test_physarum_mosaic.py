"""The mechanisms, tested through the controls that are supposed to drive them.

An earlier version of this file poked `field.mosaic` and `field.cross`
directly and asserted the picture changed. That proved the shader worked and
nothing else: an auditor zeroed `pf.mosaic = 0.95 * e` and `pf.cross = w` —
the two lines that wire `evolve` and `weave` to the new physics — and the
whole suite stayed green. A test that cannot tell a disconnected knob from a
connected one is decoration. These drive the sliders, and every bar in them
was set by cutting the wire and measuring, not by taste.

They also refuse to skip on a broken shader. `PhysarumFieldGL` raises
`PhysarumGLUnavailable` for ANY construction failure, including a uniform that
was optimised out because the branch reading it went dead, so a blanket
`except -> skip` reported a compile error as "no GPU". `_require_gl` proves a
context can exist FIRST; after that, a failure is a failure.
"""
import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from dtouch.physarum import BALLISTIC_DECAY, species_matrix
from dtouch.physarum_gl import PhysarumFieldGL, PhysarumGLUnavailable

GW, GH = 576, 324
N = int(0.53 * GW * GH)          # the shipped density, ~0.5 agents per cell
SEEDS = (7, 11, 23)
FRAMES = 400


def _require_gl():
    """Skip only when this machine genuinely has no GL context.

    Anything that goes wrong AFTER a context exists — a shader that will not
    compile, a uniform that is gone because its branch died — is a defect and
    must fail, not vanish into a skip. That distinction is the whole reason
    this helper exists rather than a bare try/except around the field.
    """
    moderngl = pytest.importorskip("moderngl")
    try:
        ctx = moderngl.create_standalone_context()
    except Exception as e:                                   # noqa: BLE001
        pytest.skip(f"no GL context on this machine: {e}")
    ctx.release()


def _field(**kw):
    _require_gl()
    try:
        return PhysarumFieldGL(**kw)
    except PhysarumGLUnavailable as e:
        pytest.fail(f"GL is present but the physarum field would not build: {e}")


def _tuned(f, **over):
    f.point_bg, f.point_fg = "veins", "fingers"
    f.gain = 1.0
    f.sat, f.jitter, f.hetero = 1.1, 0.1, 0.7
    f.cross, f.sharpen, f.mosaic = 0.5, 0.13, 0.0
    for k, v in over.items():
        setattr(f, k, v)
    return f


def _scene(gw=GW, gh=GH):
    matte = np.zeros((gh, gw), np.float32)
    gray = np.zeros((gh, gw), np.float32)
    gray[gh // 3:2 * gh // 3, gw // 3:2 * gw // 3] = 1.0
    return matte, gray


def _frame(w, h):
    """A deliberately FLAT scene.

    A gaussian subject gives the matte its own spatial structure, and with
    matte=luma the field/body blend then varies across the frame all by
    itself — which swamps what the mosaic contributes. Flat light means every
    bit of variety in the picture was grown, not lit.
    """
    return np.full((h, w, 3), 128, np.uint8)


# ---------- metrics ----------

def dominance(trail3):
    """Mean share of the leading species where there IS mold, trail-weighted.

    1/3 is perfectly mixed — three populations standing on top of each other,
    which is what "one organism" looks like in three channels. Up toward 1 is
    territory: each cell owned outright by one species, with the exclusion
    membranes between them that only signed cross-terms can produce.
    """
    tot = trail3.sum(axis=2)
    m = tot > np.percentile(tot, 60)
    if m.sum() == 0:
        return 0.0
    return float((trail3.max(axis=2)[m] / np.maximum(tot[m], 1e-6)).mean())


def patchiness(trail3, py=5, px=8):
    """Spatial spread of local dominance — the mosaic's signature.

    The mosaic's whole job is to make the physics DIFFER from place to place:
    high-repulsion zones segregate, low-repulsion zones stay mixed. Everything
    else `evolve` drives — regime hops, the phantom, breathing, melt — is
    global, moving the whole frame together and leaving this flat. So this
    isolates the spatial axis from the temporal one, which matters because
    plain dominance does NOT: cutting the mosaic moves dominance only
    0.539 -> 0.511, while patchiness collapses 0.177 -> 0.104.
    """
    tot = trail3.sum(axis=2)
    dom = trail3.max(axis=2) / np.maximum(tot, 1e-6)
    h, w = dom.shape
    ph, pw = h // py, w // px
    floor = np.percentile(tot, 60)
    cells = []
    for i in range(py):
        for j in range(px):
            d = dom[i * ph:(i + 1) * ph, j * pw:(j + 1) * pw]
            t = tot[i * ph:(i + 1) * ph, j * pw:(j + 1) * pw]
            m = t > floor
            if m.sum() > 20:
                cells.append(float(d[m].mean()))
    if len(cells) < 8:
        return 0.0
    c = np.array(cells)
    return float(c.std() / max(c.mean(), 1e-6))


def _measure(weave, evolve, seed, metric):
    """Run PhysarumMode through its SLIDERS and measure."""
    from dtouch.modes.physarum import PhysarumMode
    from tests.test_physarum_feel import _Host

    _require_gl()
    host = _Host()
    mode = PhysarumMode(matte="luma", grid=(GW, GH), n=N, seed=seed, engine="gl")
    mode.start(host)
    if mode.engine != "gl":
        pytest.fail("GL is present but PhysarumMode fell back to the CPU field")
    mode.configure_ui(host.ui)
    host.ui.ph_matte_idx = 5
    host.ui.ph_weave, host.ui.ph_evolve, host.ui.ph_react = weave, evolve, 0.0
    frame = _frame(GW, GH)
    try:
        for _ in range(FRAMES):
            mode.step(frame, None, 1 / 30)
        return metric(mode.pf.trail_species)
    finally:
        mode.stop()


# ---------- the mosaic, driven by evolve ----------

def test_evolve_drives_the_mosaic():
    """`evolve` must reach the spatial regimes.

    Bar set by mutation. Zeroing `pf.mosaic = 0.95 * e` in modes/physarum.py
    takes patchiness from 1.79x down to 1.06x, because evolve's other
    mechanisms are all global and leave this flat. 1.4x sits between the two
    with room on both sides.
    """
    off = np.mean([_measure(0.5, 0.0, s, patchiness) for s in SEEDS])
    on = np.mean([_measure(0.5, 1.0, s, patchiness) for s in SEEDS])
    assert on >= 1.4 * off, (off, on)


def test_mosaic_off_is_genuinely_one_regime():
    """The bottom of the knob has to be the uniform field, or `evolve` is not
    a control, it is a bias."""
    f = _field(n=1000, gw=64, gh=64, seed=1)
    try:
        assert f.mosaic == 0.0
    finally:
        f.release()


# ---------- the species, driven by weave ----------

def test_weave_drives_the_species_apart():
    """At weave 0 the three populations are one organism standing on top of
    itself; up, they carve each other into territories.

    Bar set by mutation. Zeroing `pf.cross = w` pins weave 1 at 0.363 — the
    perfectly-mixed floor, indistinguishable from weave 0 — against 0.693
    when it is wired.
    """
    together = np.mean([_measure(0.0, 0.0, s, dominance) for s in SEEDS])
    apart = np.mean([_measure(1.0, 0.0, s, dominance) for s in SEEDS])
    assert together < 0.40, (
        f"weave 0 must leave the species mixed (1/3 is perfectly mixed), "
        f"got {together:.3f}")
    assert apart >= together + 0.15, (together, apart)


def test_species_reduce_to_the_single_channel_model():
    """cross = 0 must be the legacy one-organism field, and ALL-ONES is what
    that means — not the identity.

    Three populations depositing into three channels and sensing their SUM is
    arithmetically the same field as one population depositing into one. An
    identity row instead gives three mutually invisible organisms at a third
    of the density each: measured, that left the channels uncorrelated at
    0.04 where the legacy field has them locked together at 0.999.
    """
    assert species_matrix(3, 0.0) == (1.0, 1.0, 1.0,
                                      1.0, 1.0, 1.0,
                                      1.0, 1.0, 1.0)
    m = species_matrix(3, 1.0)
    # diagonal attracts, next repels outright, previous mildly attracts
    assert m[0] == 1.0 and m[1] == pytest.approx(-1.0) and m[2] > 0
    assert m[4] == 1.0 and m[5] == pytest.approx(-1.0) and m[3] > 0
    assert m[8] == 1.0 and m[6] == pytest.approx(-1.0) and m[7] > 0
    # one species is the single-channel model outright
    assert species_matrix(1, 0.5) == (1.0, 0.0, 0.0,
                                      0.0, 1.0, 0.0,
                                      0.0, 0.0, 1.0)


def test_both_engines_agree_on_the_species_matrix():
    """The GL field and the CPU field disagreed about what cross 0 meant for
    exactly as long as each carried its own copy of this table."""
    from dtouch.physarum import PhysarumField

    cpu = PhysarumField(n=10, gw=32, gh=32, seed=0, species=3, cross=0.4)
    gl = _field(n=1000, gw=64, gh=64, seed=1)
    try:
        gl.cross = 0.4
        assert cpu.interaction_matrix() == gl.interaction_matrix()
        assert cpu.interaction_matrix() == species_matrix(3, 0.4)
    finally:
        gl.release()


# ---------- the wave's ballistic phase ----------

def _radial_coherence(f, cx, cy):
    """How outward the organism is actually pointing, -1..1."""
    px, py, h = f.agents()
    ang = np.arctan2(py - cy, px - cx)
    return float(np.cos(h - ang).mean())


def test_the_ballistic_phase_actually_holds_the_front_open():
    """The claim is that steering is SUPPRESSED after a wave so the front
    travels. Assert the suppression, not the bookkeeping.

    The earlier version only watched the `ballistic` float count down;
    deleting `turn *= 1.0 - u_ballistic` from the shader — the entire
    mechanism — left it green. This runs the same wave twice on identical
    fields and differs only in whether the phase is allowed to hold, so it
    cannot pass unless the uniform reaches the steering.
    """
    cx, cy = GW * 0.5, GH * 0.5
    got = {}
    for hold in (True, False):
        f = _tuned(_field(n=N, gw=GW, gh=GH, seed=5))
        try:
            matte, gray = _scene()
            for _ in range(200):
                f.update(matte, gray)
                f.luminance_into_tex()
            f.wave(cx, cy)
            if not hold:
                f.ballistic = 0.0          # the only difference
            for _ in range(8):
                f.update(matte, gray)
            got[hold] = _radial_coherence(f, cx, cy)
        finally:
            f.release()
    assert got[True] > got[False] + 0.05, got
    assert got[True] > 0.5, got


def test_the_wave_arms_the_phase_and_lets_go():
    f = _field(n=1000, gw=64, gh=64, seed=1)
    try:
        assert f.ballistic == 0.0, "steering is suppressed before any wave"
        f.wave(32.0, 32.0)
        assert f.ballistic == 1.0
        matte = np.zeros((64, 64), np.float32)
        gray = np.zeros((64, 64), np.float32)
        f.update(matte, gray)
        assert f.ballistic == pytest.approx(BALLISTIC_DECAY)
        for _ in range(60):
            f.update(matte, gray)
        assert f.ballistic < 0.01, "the mold never got its steering back"
    finally:
        f.release()


def test_a_burst_does_not_arm_the_ballistic_phase():
    """Burst teleports agents; it is not a shockwave and must not freeze
    everyone's steering on its way through."""
    f = _field(n=1000, gw=64, gh=64, seed=1)
    try:
        f.spawn_burst(32.0, 32.0)
        assert f.ballistic == 0.0
    finally:
        f.release()
