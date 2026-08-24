"""SIGNAL rack on the GPU (dtouch.rack_gl) — parity against the CPU rack.

The contract under test: CircuitBent.plan() owns every random draw, and the
two pixel backends (apply_plan on numpy, SignalRackGL as fragment passes)
bend the picture identically for the same seed and settings.

Parity metrics, stated per case:

- pixel-move stages (drift, chroma, glitch), crush, scanlines, ordered
  dither, error diffusion, and the shipped default chain: **bit-exact**
  (assert_array_equal on the uint8 frames). The GPU path uploads the CPU's
  own float32 LUTs and mirrors cv2's resize conventions, so the arithmetic
  is the same IEEE op sequence.
- chains that combine crush with dither, and the mode-level composite
  (colorize/upscale ride cv2's fixed-point u8 bilinear): sparse
  single-dither-level flips where a last-ulp difference crosses a
  threshold. Asserted as a bounded fraction of differing pixels plus a
  blurred (sigma=4) mean-absolute-error bound — the dither pattern's
  statistics, i.e. the LOOK, must be identical.

Every GPU test skips (not fails) where no GL context can be created; the
plan-refactor tests and the CPU-engine gating test run everywhere.
"""
import cv2
import numpy as np
import pytest

from dtouch.circuit_bent import CircuitBent
from dtouch.modes.physarum import PhysarumMode, _colorize, _palette_lut
from dtouch.modes.particles import composite_video_bg
from dtouch.overlay_ui import sync_signal

H, W = 288, 512          # rack-level test frames
OUT_RES = (640, 368)     # mode-level output


def _frames(n=8, seed=3):
    """Smooth random frames — busy enough to exercise every stage."""
    rng = np.random.default_rng(seed)
    return [cv2.GaussianBlur((rng.random((H, W, 3)) * 255).astype(np.uint8),
                             (0, 0), 5) for _ in range(n)]


@pytest.fixture
def gl_ctx():
    moderngl = pytest.importorskip("moderngl")
    try:
        ctx = moderngl.create_standalone_context()
    except Exception as e:                       # noqa: BLE001
        pytest.skip(f"no GL context available (CI): {e}")
    yield ctx
    ctx.release()


def _gpu_process(ctx, rack, cb, frame):
    """One frame through the GPU rack — the shape of the mode's call."""
    h, w = frame.shape[:2]
    plan = cb.plan(h, w)
    src = ctx.texture((w, h), 4, dtype="f1")
    src.write(np.ascontiguousarray(
        np.dstack([frame, np.full((h, w), 255, np.uint8)])))
    rack.run(cb, plan, src)
    out = rack.read()
    src.release()
    return out


def _run_both(ctx, frames, **kw):
    """(cpu_outs, gpu_outs) for the same seed and settings."""
    from dtouch.rack_gl import SignalRackGL
    cbc = CircuitBent(seed=5, **kw)
    cbg = CircuitBent(seed=5, **kw)
    rack = SignalRackGL(ctx, W, H)
    try:
        cpu = [cbc.process(f) for f in frames]
        gpu = [_gpu_process(ctx, rack, cbg, f) for f in frames]
    finally:
        rack.release()
    return cpu, gpu


def _assert_exact(cpu, gpu):
    for i, (a, b) in enumerate(zip(cpu, gpu)):
        np.testing.assert_array_equal(a, b, err_msg=f"frame {i}")


def _blurred_mae(a, b, sigma=4):
    return float(np.abs(cv2.GaussianBlur(a.astype(np.float32), (0, 0), sigma)
                        - cv2.GaussianBlur(b.astype(np.float32), (0, 0), sigma)
                        ).mean())


OFF = dict(chroma_shift=0, scan_drift=0, glitch_prob=0.0, bit_crush=0,
           scanlines=False, dither_mode=None)


# ---------- plan refactor (no GL needed) ----------

def test_process_is_plan_plus_apply():
    """The public process() must be exactly plan() + apply_plan() — the
    split the GPU backend rides on."""
    frames = _frames()
    a = CircuitBent(seed=9)
    b = CircuitBent(seed=9)
    for f in frames:
        ref = a.process(f)
        h, w = f.shape[:2]
        plan = b.plan(h, w)
        out = b.apply_plan(f.astype(np.float32) / 255.0, plan)
        np.testing.assert_array_equal(ref, (out * 255.0).astype(np.uint8))


def test_plan_gates_follow_the_settings():
    cb = CircuitBent(seed=1, chroma_shift=0, scan_drift=0, glitch_prob=0.0)
    p = cb.plan(H, W)
    assert p.drift_px is None
    assert p.shift_r == 0 and p.shift_b == 0
    assert p.glitch_capture is None and p.glitch_rect is None
    cb2 = CircuitBent(seed=1, scan_drift=8)
    p2 = cb2.plan(H, W)
    assert p2.drift_px is not None and p2.drift_px.shape == (H,)


def test_glitch_seq_counts_fires():
    """Each glitch event gets a fresh sequence number, so a backend never
    replays a tile it captured for an earlier event (engine switches)."""
    cb = CircuitBent(seed=1, glitch_prob=1.0, glitch_hold=1)
    seqs = set()
    for _ in range(8):
        p = cb.plan(H, W)
        if p.glitch_capture is not None:
            seqs.add(p.glitch_seq)
    assert len(seqs) >= 2      # hold=1 -> several distinct fires in 8 frames


# ---------- per-stage parity (bit-exact) ----------

def test_passthrough_is_exact(gl_ctx):
    _assert_exact(*_run_both(gl_ctx, _frames(), **OFF))


def test_scan_drift_is_exact(gl_ctx):
    _assert_exact(*_run_both(gl_ctx, _frames(), **{**OFF, "scan_drift": 8}))


def test_chroma_shift_is_exact(gl_ctx):
    _assert_exact(*_run_both(gl_ctx, _frames(), **{**OFF, "chroma_shift": 10}))


def test_glitch_capture_and_hold_are_exact(gl_ctx):
    _assert_exact(*_run_both(gl_ctx, _frames(),
                             **{**OFF, "glitch_prob": 1.0, "glitch_hold": 2}))


def test_bit_crush_is_exact(gl_ctx):
    _assert_exact(*_run_both(gl_ctx, _frames(), **{**OFF, "bit_crush": 3}))


def test_scanlines_are_exact(gl_ctx):
    _assert_exact(*_run_both(gl_ctx, _frames(), **{**OFF, "scanlines": True}))


def test_bayer_dither_at_working_res_is_exact(gl_ctx):
    _assert_exact(*_run_both(gl_ctx, _frames(),
                             **{**OFF, "dither_mode": "bayer",
                                "dither_size": 72}))


def test_bayer_dither_full_res_is_exact(gl_ctx):
    _assert_exact(*_run_both(gl_ctx, _frames(),
                             **{**OFF, "dither_mode": "bayer",
                                "dither_size": None}))


def test_blue_noise_dither_is_exact(gl_ctx):
    _assert_exact(*_run_both(gl_ctx, _frames(),
                             **{**OFF, "dither_mode": "blue",
                                "dither_size": 72}))


def test_dither_bias_and_gamma_paths_are_exact(gl_ctx):
    for invert, gamma in ((True, False), (False, True), ("auto", True),
                          ("auto", False)):
        _assert_exact(*_run_both(gl_ctx, _frames(4),
                                 **{**OFF, "dither_mode": "bayer",
                                    "dither_size": 72,
                                    "dither_invert": invert,
                                    "dither_gamma": gamma}))


def test_auto_bias_flips_on_dark_frames_identically(gl_ctx):
    dark = [(f * 0.15).astype(np.uint8) for f in _frames(4)]
    _assert_exact(*_run_both(gl_ctx, dark,
                             **{**OFF, "dither_mode": "bayer",
                                "dither_size": 72,
                                "dither_invert": "auto"}))


def test_error_diffusion_modes_are_exact(gl_ctx):
    """fs / riemersma run the same dtouch.dither code on the CPU at the
    working res in both backends — with the GPU downsample matching cv2's
    float bilinear exactly, the output is bit-identical."""
    for mode in ("fs", "riemersma"):
        _assert_exact(*_run_both(gl_ctx, _frames(3),
                                 **{**OFF, "dither_mode": mode,
                                    "dither_size": 72}))


def test_default_chain_is_exact(gl_ctx):
    """The shipped rack defaults (chroma 10, drift 8, glitch 0.10, bayer @
    72 rows, scanlines, no crush) — the look the owner actually performs
    with — must be bit-exact between the two backends."""
    _assert_exact(*_run_both(gl_ctx, _frames(),
                             chroma_shift=10, scan_drift=8, glitch_prob=0.10,
                             glitch_hold=4, bit_crush=0, scanlines=True,
                             dither_mode="bayer", dither_size=72))


# ---------- crush + dither: bounded sparse flips ----------

def test_crush_plus_dither_chain_within_tolerance(gl_ctx):
    """Crush puts values off the u8 lattice; bilinear last-ulp differences
    then flip rare dither thresholds. Bounded: <0.2% of pixels differ, and
    the blurred picture (the LOOK) is unchanged."""
    cpu, gpu = _run_both(gl_ctx, _frames(),
                         chroma_shift=10, scan_drift=8, glitch_prob=0.5,
                         glitch_hold=2, bit_crush=3, scanlines=True,
                         dither_mode="bayer", dither_size=96)
    for a, b in zip(cpu, gpu):
        d = np.abs(a.astype(np.int16) - b.astype(np.int16))
        assert float((d > 0).mean()) < 0.002
        assert _blurred_mae(a, b) < 0.5


def test_stale_tile_is_never_replayed(gl_ctx):
    """A glitch that fired while the CPU backend was active must not paint
    the GPU backend's old tile mid-hold: the GPU skips the overlay (its
    tile belongs to another glitch_seq), exactly like the numpy backend
    skips when it holds no tile."""
    from dtouch.rack_gl import SignalRackGL
    frames = _frames(6)
    kw = dict(chroma_shift=0, scan_drift=0, glitch_prob=1.0, glitch_hold=3,
              bit_crush=0, scanlines=False, dither_mode=None)
    cb = CircuitBent(seed=5, **kw)          # drives the GPU mid-stream
    ref = CircuitBent(seed=5, **kw)         # reference: tile-less numpy
    rack = SignalRackGL(gl_ctx, W, H)
    try:
        # frame 0 fires on the CPU path for both
        cb.process(frames[0])
        ref.process(frames[0])
        ref._glitch_tile = None             # simulate the backend switch
        for f in frames[1:3]:               # still inside the hold window
            got = _gpu_process(gl_ctx, rack, cb, f)
            h, w = f.shape[:2]
            plan = ref.plan(h, w)
            want = (ref.apply_plan(f.astype(np.float32) / 255.0, plan)
                    * 255.0).astype(np.uint8)
            np.testing.assert_array_equal(want, got)
    finally:
        rack.release()


# ---------- mode-level integration ----------

class _StubToasts:
    def __init__(self):
        self.texts = []

    def flash(self, *a, **k):
        self.texts.append(a[0] if a else "")


class _StubHost:
    def __init__(self, res):
        self.res = res
        self.ui = type("Ui", (), {})()
        self.hud = type("Hud", (), {})()
        self.hud.toasts = _StubToasts()


def _signal_ui(ui, glitch=True, palette_idx=0, video_bg=False):
    ui.glitch = glitch
    ui.chroma, ui.drift, ui.crush = 10.0, 8.0, 0.0
    ui.sig_bits, ui.sig_gamma, ui.sig_bias_idx = 3.0, True, 0
    ui.scanlines = True
    ui.dither_name = "bayer"
    ui.ph_matte_idx = 5                     # luma: deterministic headless
    ui.ph_palette_idx = palette_idx
    ui.ph_video_bg = video_bg
    ui.ph_video_mix = 0.6


def _blob_frames(n, w=320, h=180):
    base = np.zeros((h, w, 3), np.uint8)
    base[:] = np.linspace(20, 50, w, dtype=np.uint8)[None, :, None]
    out = []
    for i in range(n):
        f = base.copy()
        cx = int(w * (0.2 + 0.6 * i / n))
        cy = int(h * (0.5 + 0.25 * np.sin(i * 0.15)))
        cv2.circle(f, (cx, cy), 28, (235, 235, 235), -1)
        out.append(cv2.GaussianBlur(f, (15, 15), 0))
    return out


def _boot_gl_mode(res=OUT_RES):
    m = PhysarumMode(matte="luma", seed=1, engine="gl")
    host = _StubHost(res)
    m.start(host)
    if m.engine != "gl":
        m.stop()
        pytest.skip("no GL context available (CI)")
    return m, host


def _mode_pair_run(palette_idx=0, video_bg=False, nframes=24):
    """The same sim (same seed) through both rack backends.

    GPU: ui.glitch on, host.cb seeded 42. CPU reference: rack applied the
    way the shell does — sync_signal + cb.process on the mode's output."""
    frames = _blob_frames(nframes)

    m, host = _boot_gl_mode()
    _signal_ui(host.ui, glitch=True, palette_idx=palette_idx,
               video_bg=video_bg)
    host.cb = CircuitBent(seed=42)
    try:
        gpu = []
        for f in frames:
            out = m.step(f, None, 1 / 30)
            assert m.signal_done, "GPU rack path did not engage"
            gpu.append(out)
    finally:
        m.stop()

    m, host = _boot_gl_mode()
    _signal_ui(host.ui, glitch=False, palette_idx=palette_idx,
               video_bg=video_bg)
    cb = CircuitBent(seed=42)
    try:
        cpu = []
        for f in frames:
            out = m.step(f, None, 1 / 30)
            assert not m.signal_done
            sync_signal(cb, host.ui, m, OUT_RES[1])
            cpu.append(cb.process(out))
    finally:
        m.stop()
    return cpu, gpu


def test_mode_gpu_rack_matches_cpu_rack(gl_ctx):
    """Full pipeline, arctic palette: the ported path must render the CPU
    path's picture. Colorize/upscale ride cv2's fixed-point u8 bilinear on
    the CPU, so +-1-code inputs flip rare dither thresholds — bounded as
    sparse flips + an unchanged blurred picture."""
    del gl_ctx                              # only used as the GL gate
    cpu, gpu = _mode_pair_run(palette_idx=0)
    for a, b in zip(cpu, gpu):
        assert a.shape == b.shape == (OUT_RES[1], OUT_RES[0], 3)
        d = np.abs(a.astype(np.int16) - b.astype(np.int16))
        assert float((d > 1).mean()) < 0.01
        assert _blurred_mae(a, b) < 1.0


def test_mode_gpu_rack_matches_with_video_palette_and_bg(gl_ctx):
    del gl_ctx
    cpu, gpu = _mode_pair_run(palette_idx=7, video_bg=True, nframes=12)
    for a, b in zip(cpu, gpu):
        d = np.abs(a.astype(np.int16) - b.astype(np.int16))
        assert float((d > 1).mean()) < 0.06
        assert _blurred_mae(a, b) < 1.5


def test_mode_colorize_upscale_parity_no_rack(gl_ctx):
    """The composer alone (PhysarumOutGL.compose) against the CPU tail —
    _colorize + cv2.resize (+ composite_video_bg): within cv2's fixed-point
    bilinear (+-1), plus the same tolerance through the screen blend."""
    from dtouch.rack_gl import PhysarumOutGL
    from dtouch.physarum_gl import PhysarumFieldGL, PhysarumGLUnavailable
    try:
        pf = PhysarumFieldGL(n=4000, gw=96, gh=54, seed=7)
    except PhysarumGLUnavailable as e:
        pytest.skip(f"no GL context available (CI): {e}")
    try:
        rng = np.random.default_rng(0)
        lum8 = rng.integers(0, 256, (54, 96), dtype=np.uint8)
        lum = lum8.astype(np.float32) / 255.0
        cam = rng.integers(0, 256, (72, 128, 3), dtype=np.uint8)
        rw, rh = 320, 184
        with pf.ctx:
            pf.tex_lum.write(np.ascontiguousarray(lum8))
            glout = PhysarumOutGL(pf.ctx, 96, 54, rw, rh)
            # palette path
            glout.compose(pf.tex_lum, _palette_lut("fire"), None, None, 0.0)
            raw = glout.fbo_src.read(components=3)
            got = np.frombuffer(raw, np.uint8).reshape(rh, rw, 3)
            want = cv2.resize(_colorize(lum, "fire"), (rw, rh),
                              interpolation=cv2.INTER_LINEAR)
            d = np.abs(want.astype(np.int16) - got.astype(np.int16))
            assert int(d.max()) <= 2
            # + video background (screen blend)
            glout.compose(pf.tex_lum, _palette_lut("fire"), None,
                          cv2.cvtColor(cam, cv2.COLOR_BGR2RGB), 0.6)
            raw = glout.fbo_src.read(components=3)
            got = np.frombuffer(raw, np.uint8).reshape(rh, rw, 3)
            want2 = composite_video_bg(want, cam, 0.6)
            d = np.abs(want2.astype(np.int16) - got.astype(np.int16))
            assert int(d.max()) <= 3
            glout.release()
    finally:
        pf.release()


def test_gpu_failure_falls_back_to_cpu_for_good(gl_ctx, monkeypatch):
    """One GL failure inside the ported path must not lose the show: the
    frame comes from the CPU tail (unracked here — the shell's rack runs
    when signal_done is False), an amber toast says so, and the path stays
    off (no retry storm)."""
    del gl_ctx
    calls = {"n": 0}

    def boom(*a, **k):
        calls["n"] += 1
        raise RuntimeError("synthetic GL loss")
    monkeypatch.setattr("dtouch.modes.physarum.PhysarumOutGL", boom)
    m, host = _boot_gl_mode()
    _signal_ui(host.ui, glitch=True)
    try:
        f = _blob_frames(1)[0]
        out = m.step(f, None, 1 / 30)
        assert out.shape == (OUT_RES[1], OUT_RES[0], 3)
        assert m.signal_done is False
        assert m._rack_gl_ok is False
        assert any("CPU" in t for t in host.hud.toasts.texts)
        m.step(f, None, 1 / 30)
        assert calls["n"] == 1              # disabled after the first failure
    finally:
        m.stop()


def test_gpu_rack_only_with_rack_on(gl_ctx):
    del gl_ctx
    m, host = _boot_gl_mode()
    _signal_ui(host.ui, glitch=False)
    try:
        m.step(_blob_frames(1)[0], None, 1 / 30)
        assert m.signal_done is False
        assert m._glout is None             # never built
    finally:
        m.stop()


def test_cpu_engine_never_takes_the_gpu_rack_path(tmp_path):
    """Runs without GL: the CPU engine with the rack armed keeps the plain
    CPU output (the shell applies the CPU rack afterwards)."""
    m = PhysarumMode(matte="luma", seed=1, engine="cpu")
    host = _StubHost(OUT_RES)
    m.start(host)
    _signal_ui(host.ui, glitch=True)
    try:
        out = m.step(_blob_frames(1)[0], None, 1 / 30)
        assert out.shape == (OUT_RES[1], OUT_RES[0], 3)
        assert m.signal_done is False
        assert m._glout is None
    finally:
        m.stop()


def test_quality_switch_rebuilds_the_composer(gl_ctx):
    """A quality-tier hop rebuilds the field on a new context; the composer
    must follow instead of holding textures of a released context."""
    del gl_ctx
    m, host = _boot_gl_mode()
    _signal_ui(host.ui, glitch=True)
    host.ui.ph_quality_idx = 0
    frames = _blob_frames(4)
    try:
        m.step(frames[0], None, 1 / 30)
        assert m.signal_done
        first = m._glout
        assert first is not None
        host.ui.ph_quality_idx = 1          # balance: rebuild live
        m.step(frames[1], None, 1 / 30)
        assert m.signal_done
        assert m._glout is not None and m._glout is not first
        assert m.grid == PhysarumMode.QUALITY["balance"][0]
    finally:
        m.stop()
