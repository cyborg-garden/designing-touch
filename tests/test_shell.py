"""Host/Mode split — DESIGN.md §2, migration step 6.

The shell runs the full loop headless against an injectable frame source
(synthetic frames, show=False, max_frames — live_flow's semantics kept), modes
never see None frames, mode lifecycle is idempotent-stop/reinstate-on-failure,
and the GL soak pins the context lifecycle (DESIGN.md §9: GL context lifecycle
across swaps — soak test before the menu ships).
"""
import numpy as np
import pytest

from dtouch import presets
from dtouch.live import live_flow
from dtouch.modes import REGISTRY, mode_by_id
from dtouch.modes.particles import ParticlesMode
from dtouch.shell import Host

RES = (192, 108)
GRID = (48, 27)
N = 400


def _gl_available():
    try:
        import moderngl
        ctx = moderngl.create_standalone_context()
        ctx.release()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _gl_available(),
                                reason="no GL context available (CI)")


class SyntheticSource:
    """cv2-style frame source: read() -> (ok, frame_bgr). `fail_after` makes
    reads fail from that frame on (camera-loss simulation)."""

    def __init__(self, frames=None, w=64, h=36, fail_after=None):
        self.w, self.h = w, h
        self.frames = frames
        self.fail_after = fail_after
        self.reads = 0
        self.released = False
        self.name = "synthetic"

    def read(self):
        self.reads += 1
        if self.fail_after is not None and self.reads > self.fail_after:
            return False, None
        rng = np.random.default_rng(self.reads)
        frame = rng.integers(0, 256, (self.h, self.w, 3), np.uint8)
        return True, frame

    def release(self):
        self.released = True


def _mode(**kw):
    kw.setdefault("grid", GRID)
    kw.setdefault("n", N)
    kw.setdefault("seed", 0)
    return ParticlesMode(**kw)


def _paths(tmp_path):
    """Isolated preset/state files so tests never touch the launch dir's."""
    return dict(presets_path=str(tmp_path / "presets.json"),
                state_path=str(tmp_path / "state.json"))


def _host(tmp_path, mode=None, src=None, **kw):
    kw.setdefault("res", RES)
    kw.setdefault("show", False)
    return Host(mode or _mode(), source=src or SyntheticSource(),
                **_paths(tmp_path), **kw)


# ---------- the headless loop ----------

def test_headless_loop_runs_max_frames_and_releases(tmp_path):
    src = SyntheticSource()
    count, out = live_flow(source=src, res=RES, grid=GRID, n=N,
                           show=False, max_frames=5, **_paths(tmp_path))
    assert count == 5
    assert out.shape == (RES[1], RES[0], 3) and out.dtype == np.uint8
    assert src.released, "the shell owns the source's lifecycle"


def test_modes_never_see_none_frames_on_camera_loss(tmp_path):
    """DESIGN.md §2.2: on camera loss the host passes the last good frame."""
    seen = []

    class SpyMode(ParticlesMode):
        def step(self, frame_bgr, audio_levels, dt):
            assert frame_bgr is not None
            seen.append(frame_bgr.copy())
            return super().step(frame_bgr, audio_levels, dt)

    src = SyntheticSource(fail_after=2)
    mode = SpyMode(grid=GRID, n=N, seed=0)
    host = _host(tmp_path, mode=mode, src=src, max_frames=6)
    count, _ = host.run()
    assert count == 6 and len(seen) == 6
    # frames 3..6 are the held (mirrored) copy of frame 2
    assert np.array_equal(seen[2], seen[1])
    assert np.array_equal(seen[5], seen[1])


def test_loss_with_no_good_frame_ends_bounded_run(tmp_path):
    src = SyntheticSource(fail_after=0)     # never yields a frame
    count, out = live_flow(source=src, res=RES, grid=GRID, n=N,
                           show=False, max_frames=4, **_paths(tmp_path))
    assert count == 0 and out is None


# ---------- mode lifecycle ----------

def test_stop_is_idempotent(tmp_path):
    host = _host(tmp_path)
    m = _mode()
    m.start(host)
    m.stop()
    m.stop()                                # second stop must be a no-op
    assert m.glow is None and m.pf is None


def test_set_mode_start_failure_reinstates_previous(tmp_path):
    """DESIGN.md §2.2: start() may raise — the shell catches, toasts, and
    reinstates the previous mode."""

    class BrokenMode(ParticlesMode):
        id = "broken"
        title = "Broken"

        def start(self, host):
            raise RuntimeError("no engine for you")

    host = _host(tmp_path)
    good = _mode()
    assert host.set_mode(good) is True
    bad = BrokenMode(grid=GRID, n=N)
    assert host.set_mode(bad) is False
    assert host.mode is good
    assert good.glow is not None, "the previous mode must be running again"
    assert host.hud.toasts.active(), "silence-on-failure is a bug"
    good.stop()


def test_boot_mode_start_failure_is_fatal(tmp_path):
    class Doomed(ParticlesMode):
        def start(self, host):
            raise RuntimeError("boom")

    host = _host(tmp_path, mode=Doomed(grid=GRID, n=N), max_frames=1)
    with pytest.raises(RuntimeError):
        host.run()


def test_registry_lists_particles():
    assert ParticlesMode in REGISTRY
    assert mode_by_id("particles") is ParticlesMode
    assert mode_by_id("nope") is None


# ---------- bank seeding + state autosave/resume (DESIGN.md §6.4/§7) ----------

def test_unstored_bank_seeds_builtins_on_slots_1_to_9(tmp_path):
    host = _host(tmp_path, max_frames=1)
    host.run()
    builtins = list(ParticlesMode.BUILTIN)
    assert host.ui.bank == {str(i + 1): n for i, n in enumerate(builtins[:9])}
    assert host.ui.setlist == []            # empty = all looks, load order


def test_stored_bank_and_setlist_win_over_seeding(tmp_path):
    p = _paths(tmp_path)
    presets.set_bank({"7": "sigil"}, path=p["presets_path"])
    presets.set_setlist(["embers", "abstract"], path=p["presets_path"])
    host = _host(tmp_path, max_frames=1)
    host.run()
    assert host.ui.bank == {"7": "sigil"}
    assert host.ui.setlist == ["embers", "abstract"]


def test_explicitly_empty_bank_stays_empty(tmp_path):
    p = _paths(tmp_path)
    presets.set_bank({}, path=p["presets_path"])
    host = _host(tmp_path, max_frames=1)
    host.run()
    assert host.ui.bank == {}


def test_state_autosaved_on_boot_and_resumed_when_no_preset_given(tmp_path):
    p = _paths(tmp_path)
    host = _host(tmp_path, max_frames=1, preset="embers")
    host.run()
    st = presets.load_state(p["state_path"])
    assert st["mode"] == "particles" and st["preset"] == "embers"
    assert st["bank"]["particles"]["1"] == "abstract"

    # crash restart: preset=None resumes the autosaved look
    host2 = _host(tmp_path, max_frames=1, preset=None)
    host2.run()
    assert host2.ui.preset_name == "embers"
    assert host2.ui.palette_name == "fire"      # embers actually applied


def test_no_state_file_boots_the_safe_look(tmp_path):
    host = _host(tmp_path, max_frames=1, preset=None)
    host.run()
    assert host.ui.preset_name == "abstract"


# ---------- GL lifecycle soak (DESIGN.md §8 step 6 / §9) ----------

def test_gl_soak_start_stop_25_cycles(tmp_path):
    """Start/stop the mode 30 times at a tiny res: no leak of usable state, no
    crash, and the engines still render after the churn."""
    host = _host(tmp_path)
    frame = np.full((36, 64, 3), 128, np.uint8)
    m = _mode()
    for _ in range(30):
        m.start(host)
        out = m.step(frame, None, 1 / 30)
        assert out.shape == (RES[1], RES[0], 3)
        m.stop()
    m.stop()                                # idempotent after the soak too
    # one more full cycle proves the GL stack is still healthy
    m.start(host)
    assert m.step(frame, None, 1 / 30).dtype == np.uint8
    m.stop()
