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


def test_slot_badge_click_assigns_next_free_then_clears(tmp_path):
    """DESIGN.md §6.3: clicking a preset row's slot badge assigns the next
    free bank number; clicking an assigned badge clears it. Both persist."""
    p = _paths(tmp_path)
    presets.set_bank({"1": "abstract"}, path=p["presets_path"])
    host = _host(tmp_path, max_frames=1)
    host.run()

    host.ui.pending_slot = "embers"
    host._pump_preset_mailboxes()
    assert host.ui.bank == {"1": "abstract", "2": "embers"}
    assert presets.bank(p["presets_path"]) == {"1": "abstract", "2": "embers"}
    assert host.ui.pending_slot is None

    host.ui.pending_slot = "embers"                 # assigned → clears
    host._pump_preset_mailboxes()
    assert host.ui.bank == {"1": "abstract"}
    assert presets.bank(p["presets_path"]) == {"1": "abstract"}


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


def test_gl_soak_alternating_particles_dithergirl_25_cycles(tmp_path):
    """DESIGN.md §8 step 8: the GL context lifecycle survives alternating
    particles <-> dithergirl swaps (26 cycles), and both still render after."""
    from dtouch.modes.dithergirl import DitherGirlMode

    host = _host(tmp_path)
    frame = np.full((36, 64, 3), 128, np.uint8)
    p, d = _mode(), DitherGirlMode()
    for _ in range(26):
        p.start(host)
        assert p.step(frame, None, 1 / 30).shape == (RES[1], RES[0], 3)
        p.stop()
        d.start(host)
        assert d.step(frame, None, 1 / 30).shape == (RES[1], RES[0], 3)
        d.stop()
    p.stop(); d.stop()                      # idempotent after the soak too
    p.start(host)
    assert p.step(frame, None, 1 / 30).dtype == np.uint8
    p.stop()
    d.start(host)
    assert d.step(frame, None, 1 / 30).dtype == np.uint8
    d.stop()


# ---------- live mode switching (DESIGN.md §3 / §8 step 8) ----------

class FakeWriter:
    def __init__(self):
        self.frames = []
        self.closed = False

    def append_data(self, f):
        self.frames.append(np.asarray(f))

    def close(self):
        self.closed = True


class DummyMic:
    available = False

    def stop(self):
        pass


def _patch_on_read(src, on_read):
    """Fire a hook after every source read — drives mid-loop events."""
    orig = src.read

    def read():
        r = orig()
        on_read(src.reads)
        return r
    src.read = read


def test_switch_to_dithergirl_mid_loop_preserves_perform_state(tmp_path):
    """Switching draws a recorded boot card, lands on the new mode's safe
    look, recolors the chrome, and preserves overlay state, blackout,
    recording, and the audio toggle (DESIGN.md §3)."""
    from dtouch.hud import OverlayState
    from dtouch.modes.dithergirl import ACCENT, DitherGirlMode

    host = _host(tmp_path, max_frames=8)
    writer = FakeWriter()

    def on_read(n):
        if n == 2:
            host.overlay = OverlayState.PANEL
            host.ps.blackout = True
            host.ui.audio = True
            host.mic = DummyMic()           # keep _sync from opening a real mic
            host.ui.record = True
            host.writer, host.rec_path = writer, "fake"
            host.ui.glitch = True
        if n == 3:
            host.request_mode("dithergirl")

    _patch_on_read(host._source, on_read)
    count, out = host.run()

    assert host.mode.id == "dithergirl"
    assert host.ui.accent == ACCENT
    assert host.ui.preset_name == "classic"              # landed on the safe look
    titles = [s.title for s in host.ui.spec
              if hasattr(s, "title")]
    assert "ALGORITHM" in titles and "LOOK" not in titles
    # preserved across the switch:
    assert host.overlay is OverlayState.PANEL
    assert host.ps.blackout is True
    assert host.ui.audio is True
    assert host.ui.record is True and writer.closed      # closed by shutdown only
    assert host.ui.glitch is True
    # blackout was armed during the switch (amended DESIGN.md §3): the
    # recorder captured a black frame with the amber tick — never the bright
    # boot card — and kept recording mode frames after the switch
    for f in writer.frames:
        assert float(f.mean()) < 1.0                     # switch stayed under black
    assert any(f[:8, -8:].max() > 0 for f in writer.frames)   # the tick frame
    assert len(writer.frames) > 1                        # recording never stopped


def test_switch_back_to_particles_restores_its_panel(tmp_path):
    from dtouch.modes.dithergirl import DitherGirlMode

    host = _host(tmp_path, mode=DitherGirlMode(), max_frames=6)

    def on_read(n):
        if n == 2:
            host.request_mode("particles")
    _patch_on_read(host._source, on_read)
    host.run()
    assert host.mode.id == "particles"
    titles = [s.title for s in host.ui.spec if hasattr(s, "title")]
    assert "LOOK" in titles and "ALGORITHM" not in titles
    assert host.ui.accent == ParticlesMode.accent
    assert host.ui.preset_name == "abstract"             # particles' safe look


def test_switch_failure_reinstates_running_mode(tmp_path, monkeypatch):
    """A mode whose start() raises mid-switch toasts and leaves the previous
    mode running (DESIGN.md §6.4), even through the pending-mode path."""
    import dtouch.modes as modes

    class BrokenMode(ParticlesMode):
        id = "broken"
        title = "Broken"

        def start(self, host):
            raise RuntimeError("no engine for you")

    monkeypatch.setattr(modes, "REGISTRY", modes.REGISTRY + [BrokenMode])
    host = _host(tmp_path, max_frames=5)

    def on_read(n):
        if n == 2:
            host.request_mode("broken")
    _patch_on_read(host._source, on_read)
    count, out = host.run()
    assert count == 5                                    # the show went on
    assert host.mode.id == "particles"
    assert out.shape == (RES[1], RES[0], 3)


def test_request_same_mode_is_a_hint_not_a_switch(tmp_path):
    host = _host(tmp_path, max_frames=1)
    host.run()
    host.request_mode("particles")
    assert host.pending_mode is None
    assert host.hud.toasts.active()
