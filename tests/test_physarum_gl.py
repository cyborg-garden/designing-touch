"""PhysarumFieldGL — the GPU field behind PhysarumMode, plus engine selection.

The GPU tests skip (not fail) wherever a standalone GL context cannot be
created — CI has no GPU — through the `gl_field` fixture. The mode-level
fallback test needs no GL at all: it mocks the GPU field's constructor into
raising and asserts the CPU field boots with an amber toast, which is the
path CI actually exercises for every other physarum mode test.

The engine tests mirror tests/test_physarum.py's CPU assertions so the two
fields are held to the same contract: deposit + bounded luminance, the matte
as the pen, luminance as food, determinism, burst, wave.
"""
import numpy as np
import pytest

from dtouch.modes.physarum import PhysarumMode, _colorize, _fmt_agents, _palette_lut
from dtouch.physarum import POINTS, PhysarumField
from dtouch.physarum_gl import PhysarumFieldGL, PhysarumGLUnavailable
from dtouch.shell import Host

from test_shell import SyntheticSource

RES = (192, 108)


def _paths(tmp_path):
    return dict(presets_path=str(tmp_path / "presets.json"),
                state_path=str(tmp_path / "state.json"))


def _host(tmp_path, mode, **kw):
    kw.setdefault("res", RES)
    kw.setdefault("show", False)
    kw.setdefault("preset", None)
    return Host(mode, source=SyntheticSource(), **_paths(tmp_path), **kw)


@pytest.fixture
def gl_field():
    """Factory for small GPU fields; skips the test when GL is absent and
    releases every field it built."""
    built = []

    def make(**kw):
        kw.setdefault("n", 4000)
        kw.setdefault("gw", 96)
        kw.setdefault("gh", 54)
        kw.setdefault("seed", 7)
        try:
            f = PhysarumFieldGL(**kw)
        except PhysarumGLUnavailable as e:
            pytest.skip(f"no GL context available (CI): {e}")
        built.append(f)
        return f
    yield make
    for f in built:
        f.release()


def _flat(f, v=0.0):
    return np.full((f.gh, f.gw), v, np.float32)


# ---------- engine ----------

def test_update_deposits_trail_and_luminance_is_a_picture(gl_field):
    f = gl_field()
    for _ in range(5):
        f.update(_flat(f), _flat(f))
    trail = f.trail
    assert trail.shape == (f.gh, f.gw) and trail.sum() > 0
    assert np.isfinite(trail).all()
    lum = f.luminance()
    assert lum.shape == (f.gh, f.gw) and lum.dtype == np.float32
    assert float(lum.min()) >= 0.0 and float(lum.max()) <= 1.0
    # non-black with structure, not a correctly shaped nothing
    assert float(lum.mean()) > 0.05
    assert float(lum.std()) > 0.01


def test_matte_is_the_pen_blends_toward_body_point(gl_field):
    """Same assertion as the CPU field: matte=0 runs the field point, matte=1
    the body point, and mean per-frame displacement follows the step."""
    def mean_step(matte_value):
        f = gl_field(point_bg="haze", point_fg="fingers")
        m, g = _flat(f, matte_value), _flat(f)
        f.update(m, g)
        px, py, _ = f.agents()
        f.update(m, g)
        qx, qy, _ = f.agents()
        dx = np.minimum(np.abs(qx - px), f.gw - np.abs(qx - px))
        dy = np.minimum(np.abs(qy - py), f.gh - np.abs(qy - py))
        return float(np.hypot(dx, dy).mean())

    slow, fast = mean_step(0.0), mean_step(1.0)
    assert slow == pytest.approx(POINTS["haze"]["step"], rel=0.15)
    assert fast == pytest.approx(POINTS["fingers"]["step"], rel=0.15)


def test_luminance_is_food_draws_the_mold(gl_field):
    f = gl_field(food=1.0, reseed_frac=0.05)
    gray = _flat(f)
    band = slice(f.gw // 3, 2 * f.gw // 3)
    gray[:, band] = 1.0
    for _ in range(40):
        f.update(_flat(f), gray)
    trail = f.trail
    inside = float(trail[:, band].mean())
    outside = float(np.delete(trail, np.s_[band], axis=1).mean())
    assert inside > outside * 1.3


def test_reseed_lands_on_the_lit_subject(gl_field):
    """With every agent recycled each frame and food off, positions must
    follow matte*luma — the rejection sampler's whole job."""
    f = gl_field(reseed_frac=1.0, food=0.0)
    matte, gray = _flat(f), _flat(f, 1.0)
    matte[:, :f.gw // 2] = 1.0
    f.update(matte, gray)
    px, _, _ = f.agents()
    assert (px < f.gw // 2).mean() > 0.98


def test_trail_rows_and_columns_are_grid_coordinates(gl_field):
    """The readback must not be flipped: park the pool on one cell (burst
    with a hair of radius, tempo ~0 so nobody walks off it, no blur) and the
    trail's peak must sit at that (row=y, col=x)."""
    f = gl_field(diffuse=0, gain=1e-4, reseed_frac=0.0)
    f.spawn_burst(70.0, 12.0, frac=1.0, radius=0.05)
    f.update(_flat(f), _flat(f))
    y, x = np.unravel_index(int(f.trail.argmax()), (f.gh, f.gw))
    assert (y, x) == (12, 70)


def test_same_seed_same_trail(gl_field):
    a, b = gl_field(), gl_field()
    for _ in range(3):
        a.update(_flat(a), _flat(a))
        b.update(_flat(b), _flat(b))
    assert np.array_equal(a.trail, b.trail)


def test_spawn_burst_concentrates_agents(gl_field):
    f = gl_field()
    f.spawn_burst(20.0, 20.0, frac=0.5, radius=2.0)
    px, py, _ = f.agents()
    d = np.hypot(px - 20.0, py - 20.0)
    assert (d < 8.0).mean() > 0.4


def test_wave_points_everyone_outward(gl_field):
    f = gl_field()
    f.wave(f.gw / 2, f.gh / 2)
    px, py, h = f.agents()
    dx, dy = px - f.gw / 2, py - f.gh / 2
    outward = np.cos(h) * dx + np.sin(h) * dy
    assert (outward >= 0).mean() > 0.99


def test_gpu_and_cpu_settle_to_the_same_trail_mass(gl_field):
    """Same agents, same deposit, same decay: the two engines' total trail
    must converge to the same equilibrium (n * deposit * decay / (1 - decay))
    — the check that this is the same model, not a look-alike."""
    g = gl_field(n=8000, diffuse=1, decay=0.9)
    c = PhysarumField(n=8000, gw=g.gw, gh=g.gh, seed=7, diffuse=1, decay=0.9)
    m, y = _flat(g), _flat(g)
    for _ in range(80):
        g.update(m, y)
        c.update(m, y)
    assert float(g.trail.sum()) == pytest.approx(float(c.trail.sum()), rel=0.1)


def test_rejects_bad_sizes_before_touching_gl():
    with pytest.raises(ValueError):
        PhysarumFieldGL(n=0)
    with pytest.raises(ValueError):
        PhysarumFieldGL(n=10, gw=0, gh=5)
    with pytest.raises(ValueError):
        PhysarumFieldGL(n=10, decay=1.5)


def test_update_rejects_wrong_shaped_input(gl_field):
    f = gl_field()
    with pytest.raises(ValueError):
        f.update(np.zeros((3, 3), np.float32), _flat(f))


def test_release_is_idempotent(gl_field):
    f = gl_field()
    f.release()
    f.release()
    assert f.ctx is None


# ---------- mode: engine selection + fallback ----------

def test_mode_boots_the_gpu_field_when_gl_is_there(tmp_path):
    try:
        probe = PhysarumFieldGL(n=16, gw=8, gh=8)
    except PhysarumGLUnavailable as e:
        pytest.skip(f"no GL context available (CI): {e}")
    probe.release()
    m = PhysarumMode()
    host = _host(tmp_path, m)
    m.start(host)
    try:
        assert m.engine == "gl" and isinstance(m.pf, PhysarumFieldGL)
        assert (m.grid, m.n) == (PhysarumMode.GL_GRID, PhysarumMode.GL_N)
        out = m.step(np.full((36, 64, 3), 255, np.uint8), None, 1 / 30)
        assert out.shape == (RES[1], RES[0], 3) and out.dtype == np.uint8
        assert m.status_tail("synthetic").startswith("gl 2.0M")
    finally:
        m.stop()
    assert m.pf is None
    m.stop()                                  # idempotent after release


def test_mode_engine_cpu_is_an_explicit_opt_out(tmp_path):
    m = PhysarumMode(engine="cpu")
    host = _host(tmp_path, m)
    m.start(host)
    try:
        assert m.engine == "cpu" and isinstance(m.pf, PhysarumField)
        assert (m.grid, m.n) == (PhysarumMode.CPU_GRID, PhysarumMode.CPU_N)
        assert m.status_tail("synthetic") == "cpu 400k  cam synthetic"
    finally:
        m.stop()


def test_mode_falls_back_to_cpu_with_a_toast_when_gl_init_fails(tmp_path, monkeypatch):
    """No GL needed: the GPU constructor is mocked into failing the way a
    missing driver does, and the show must still start — on the CPU field,
    at the CPU sizing, with the reason toasted in amber (DESIGN.md §6.4)."""
    def boom(*a, **k):
        raise PhysarumGLUnavailable("GPU physarum unavailable: no context")
    monkeypatch.setattr("dtouch.modes.physarum.PhysarumFieldGL", boom)
    m = PhysarumMode()                        # engine="auto"
    host = _host(tmp_path, m)
    m.start(host)
    try:
        assert m.engine == "cpu" and isinstance(m.pf, PhysarumField)
        assert (m.grid, m.n) == (PhysarumMode.CPU_GRID, PhysarumMode.CPU_N)
        toast = host.hud.toasts._center
        assert toast is not None and "CPU" in toast.text
        out = m.step(np.full((36, 64, 3), 255, np.uint8), None, 1 / 30)
        assert out.shape == (RES[1], RES[0], 3)
    finally:
        m.stop()


def test_explicit_grid_and_agents_win_on_either_engine(tmp_path, monkeypatch):
    monkeypatch.setattr("dtouch.modes.physarum.PhysarumFieldGL",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no")))
    m = PhysarumMode(grid=(64, 36), n=1234)
    host = _host(tmp_path, m)
    m.start(host)
    try:
        assert (m.grid, m.n) == ((64, 36), 1234)
    finally:
        m.stop()


def test_unknown_engine_is_refused():
    with pytest.raises(ValueError):
        PhysarumMode(engine="metal")


def test_fmt_agents():
    assert _fmt_agents(400_000) == "400k"
    assert _fmt_agents(2_000_000) == "2.0M"
    assert _fmt_agents(1_234) == "1k"


def test_colorize_matches_the_lut_index():
    """applyColorMap replaced numpy fancy indexing for speed; it must stay
    pixel-identical, RGB order intact."""
    lum = np.random.default_rng(3).random((20, 30)).astype(np.float32)
    idx = (lum * 255.0).astype(np.uint8)
    for pal in ("arctic", "fire", "mono"):
        assert np.array_equal(_colorize(lum, pal), _palette_lut(pal)[idx])
