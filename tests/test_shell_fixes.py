"""Shell-level regression pins for the verified UI-overhaul bug list.

Everything here drives the real Host loop headless with DitherGirlMode (pure
numpy/cv2 — no GL, so these run everywhere test_dithergirl.py does). GUI-path
tests monkeypatch cv2's window calls so `show=True` code paths run headless.
"""
import numpy as np
import pytest

import cv2

from dtouch import presets
from dtouch.hud import OverlayState
from dtouch.modes.dithergirl import DitherGirlMode
from dtouch.shell import Host

RES = (192, 108)


class SyntheticSource:
    """Deterministic frame source (seeded per read) with an optional per-read
    hook and a fail-from mode (fail_after=0 = never yields a frame)."""

    def __init__(self, w=64, h=36, on_read=None, fail_after=None):
        self.w, self.h = w, h
        self.reads = 0
        self.on_read = on_read
        self.fail_after = fail_after
        self.released = False
        self.name = "synthetic"

    def read(self):
        self.reads += 1
        if self.on_read:
            self.on_read(self.reads)
        if self.fail_after is not None and self.reads > self.fail_after:
            return False, None
        rng = np.random.default_rng(self.reads)
        return True, rng.integers(0, 256, (self.h, self.w, 3), np.uint8)

    def release(self):
        self.released = True


class FakeWriter:
    def __init__(self):
        self.frames = []
        self.closed = False

    def append_data(self, f):
        assert not self.closed
        self.frames.append(np.asarray(f).copy())

    def close(self):
        self.closed = True


def _paths(tmp_path):
    return dict(presets_path=str(tmp_path / "presets.json"),
                state_path=str(tmp_path / "state.json"))


def _host(tmp_path, mode=None, src=None, **kw):
    kw.setdefault("res", RES)
    kw.setdefault("show", False)
    kw.setdefault("preset", None)
    return Host(mode or DitherGirlMode(), source=src or SyntheticSource(),
                **_paths(tmp_path), **kw)


def _booted(tmp_path, **kw):
    host = _host(tmp_path, max_frames=1, **kw)
    host.run()
    return host


def _hints(host):
    return [t.text for t in host.hud.toasts._hints]


def _patch_gui(monkeypatch, keys=(), shown=None):
    """Run show=True paths headless: no-op the cv2 window calls; waitKey pops
    from `keys` (then 255); imshow appends to `shown` when given."""
    seq = list(keys)
    monkeypatch.setattr(cv2, "namedWindow", lambda *a, **k: None)
    monkeypatch.setattr(cv2, "setMouseCallback", lambda *a, **k: None)
    monkeypatch.setattr(cv2, "imshow",
                        (lambda win, img: shown.append(img.copy()))
                        if shown is not None else (lambda *a: None))
    monkeypatch.setattr(cv2, "waitKey",
                        lambda ms=0: seq.pop(0) if seq else 255)
    monkeypatch.setattr(cv2, "getWindowProperty", lambda *a: 1.0)
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)


# ---------- res change while recording: stop cleanly, never crash ----------

def test_res_change_stops_recording_before_resize(tmp_path):
    """imageio's ffmpeg writer needs a constant frame size — a res cycle while
    recording must close the writer first (same path as 'r' off), toast why,
    then resize."""
    resized = []

    class Spy(DitherGirlMode):
        def on_resize(self, w, h):
            resized.append((w, h, self.host.writer is None))

    writer = FakeWriter()
    host = _host(tmp_path, mode=Spy(), max_frames=6)

    def on_read(n):
        if n == 2:
            host.ui.record = True
            host.writer, host.rec_path = writer, "fake.mp4"
        if n == 4:
            host.ui.res_idx = 1              # cycle output res mid-recording

    host._source.on_read = on_read
    host.run()
    assert resized and resized[0][2] is True     # writer closed BEFORE resize
    assert host.ui.record is False               # recording stopped cleanly
    hints = _hints(host)
    assert any("recording stopped - resolution changed" in t for t in hints)
    assert any(t.startswith("saved") for t in hints)


# ---------- blackout defeats the bright boot card ----------

def test_blackout_switch_records_black_not_the_bright_card(tmp_path,
                                                           monkeypatch):
    """DESIGN.md §3: while blackout is armed the switch happens under black —
    a black frame with the amber tick is shown/recorded, never the bright
    card."""
    import dtouch.modes as modes

    class OtherMode(DitherGirlMode):
        id = "other"
        title = "Other"

    monkeypatch.setattr(modes, "REGISTRY", modes.REGISTRY + [OtherMode])
    writer = FakeWriter()
    host = _host(tmp_path, max_frames=6)

    def on_read(n):
        if n == 2:
            host.ps.blackout = True
            host.ui.record = True
            host.writer, host.rec_path = writer, "fake.mp4"
        if n == 3:
            host.request_mode("other")

    host._source.on_read = on_read
    host.run()
    assert host.mode.id == "other"
    assert writer.frames
    for f in writer.frames:
        assert float(f.mean()) < 1.0     # no bright frame ever hit the recorder
    # the switch frame still carries the amber corner tick on black (§3/§5)
    assert any(f[:8, -8:].max() > 0 for f in writer.frames)


# ---------- recorder content: never panel/HUD/toast pixels ----------

def _record_run(tmp_path, decorate, monkeypatch):
    writer = FakeWriter()
    src = SyntheticSource()
    host = _host(tmp_path, src=src, max_frames=8, show=decorate, panel=True)

    def on_read(n):
        if n == 1:
            host.ui.record = True
            host.writer, host.rec_path = writer, "fake.mp4"
        if n >= 6:
            host.ps.blackout = True
        if decorate:                         # panel open + toasts firing
            host.overlay = OverlayState.PANEL
            host.hud.toasts.flash("TESTING")
            host.hud.toasts.hint("hint hint")

    src.on_read = on_read
    host.run()
    return writer.frames


def test_recordings_contain_no_panel_hud_or_toast_pixels(tmp_path,
                                                         monkeypatch):
    """Recording captures what the audience sees, minus panel/HUD (DESIGN.md
    principle 2): a run with the panel open and toasts firing records frames
    identical to a UI-free run of the same source; blackout frames record
    black."""
    _patch_gui(monkeypatch)
    clean = _record_run(tmp_path, decorate=False, monkeypatch=monkeypatch)
    ui_run = _record_run(tmp_path, decorate=True, monkeypatch=monkeypatch)
    assert len(clean) == len(ui_run) == 8
    for a, b in zip(clean, ui_run):
        assert np.array_equal(a, b), "UI pixels leaked into the recording"
    assert any(float(f.mean()) > 1.0 for f in ui_run[:5])    # really rendered
    for f in ui_run[5:]:                                     # blackout frames
        assert not f.any()


# ---------- help modal swallows the mouse ----------

def test_open_help_swallows_mouse_and_click_closes_it(tmp_path):
    host = _booted(tmp_path)
    host.ps.help_open = True
    host.ui._hot = [((0, 0, 200, 200), "quit", None)]
    host._on_mouse(cv2.EVENT_LBUTTONDOWN, 50, 50, 0)
    assert host.ui.quit is False             # the click never reached the panel
    assert host.ps.help_open is False        # ...and it closed the help
    host._on_mouse(cv2.EVENT_LBUTTONDOWN, 50, 50, 0)
    assert host.ui.quit is True              # normal routing resumed


# ---------- menu key routing through the shell ----------

def test_menu_q_closes_and_arms_quit_confirm(tmp_path):
    host = _booted(tmp_path)
    host._wire_keys()
    host.menu.show("dithergirl")
    host._route_key(ord("q"))
    assert host.menu.open is False
    assert any("q again to quit" in t for t in _hints(host))
    assert host.ps.quit is False
    host._route_key(ord("q"))                # second press, menu closed
    assert host.ps.quit is True


def test_menu_unknown_key_hints(tmp_path):
    host = _booted(tmp_path)
    host._wire_keys()
    host.menu.show("dithergirl")
    host._route_key(ord("z"))
    assert host.menu.open is True
    assert any("? for keys" in t for t in _hints(host))


# ---------- camera that never yields: responsive, quittable ----------

def test_camera_that_never_yields_keeps_keys_alive_and_quits(tmp_path,
                                                             monkeypatch):
    """DESIGN.md §6.4: no keyboard-reachable state requires a restart — a
    camera that never yields shows an intentional black frame with the human
    fix and keeps pumping keys, so q/quit works."""
    shown = []
    _patch_gui(monkeypatch, keys=[255, ord("q"), ord("q")], shown=shown)
    host = _host(tmp_path, src=SyntheticSource(fail_after=0), show=True)
    count, out = host.run()
    assert count == 0
    assert host.ps.quit is True              # q reached the quit confirm
    assert any("q again to quit" in t for t in _hints(host))
    assert shown                             # an intentional image was shown
    f = shown[0]
    assert f.shape == (RES[1], RES[0], 3)
    assert float(f.mean()) < 20 and f.max() > 0   # black + on-canvas note


def test_camera_that_never_yields_bounded_run_exits(tmp_path, monkeypatch):
    _patch_gui(monkeypatch)
    host = _host(tmp_path, src=SyntheticSource(fail_after=0), show=True,
                 max_frames=3)
    count, out = host.run()
    assert count == 0 and out is None


# ---------- panic through the shell wiring ----------

def test_panic_key_via_shell_wiring_lands_on_safe_look(tmp_path):
    host = _booted(tmp_path)
    host._wire_keys()
    host.ui.preset_idx = host.ui.presets.index("phosphor")
    host.ui.glitch = True
    host.ps.blackout = True
    host.reg.dispatch(ord("0"))
    assert host.ps.blackout is False and host.ui.glitch is False
    assert host.ui.pending_preset == "classic"       # the mode's safe look


# ---------- switch-away-and-back contract (amended DESIGN.md §6.2) ----------

def test_first_entry_applies_safe_look_reentry_preserves_settings(
        tmp_path, monkeypatch):
    """FIRST entry to a mode applies safe_look(); RE-entry to an
    already-visited mode preserves its current settings (rehearsal parity —
    presets are an instrument, not a reset trap)."""
    import dtouch.modes as modes
    from dtouch.modes.dithergirl import ALGOS

    class OtherMode(DitherGirlMode):
        id = "other"
        title = "Other"

    monkeypatch.setattr(modes, "REGISTRY", modes.REGISTRY + [OtherMode])
    host = _host(tmp_path, max_frames=10)
    seen = {}

    def on_read(n):
        ui = host.ui
        if n == 2:
            host.request_mode("other")       # FIRST entry
        if n == 4:
            seen["after_first_entry"] = (ui.preset_name, ui.dg_algo_idx)
            ui.dg_algo_idx = ALGOS.index("Bayer")    # operator reshapes look
            ui.preset_idx = ui.presets.index("phosphor")
        if n == 6:
            host.request_mode("dithergirl")  # back to boot mode (re-entry)
        if n == 8:
            host.request_mode("other")       # re-entry to 'other'

    host._source.on_read = on_read
    host.run()
    # first entry landed on the safe look
    assert seen["after_first_entry"] == ("classic",
                                         ALGOS.index("Floyd-Steinberg"))
    # re-entry preserved the operator's settings — no safe_look re-post
    assert host.mode.id == "other"
    assert host.ui.dg_algo_idx == ALGOS.index("Bayer")
    assert host.ui.preset_name == "phosphor"
    assert host.ui.pending_preset is None


# ---------- save flow (amended DESIGN.md §3: naming is one flow) ----------

def test_s_in_panel_saves_and_opens_rename_with_toast(tmp_path):
    host = _booted(tmp_path)
    host._wire_keys()
    host.overlay = OverlayState.PANEL
    host.reg.dispatch(ord("s"))
    assert host.ui.pending_save is True
    host._pump_preset_mailboxes()
    assert host.ui.user_presets                      # the look landed
    name = next(iter(host.ui.user_presets))
    assert host.ui.renaming == name                  # rename box opened
    assert host.ui.rename_buf == name
    assert host.ui.preset_name == name
    assert any("saved" in t for t in _hints(host))   # toast, not just stdout


def test_s_outside_panel_does_not_save_but_hints(tmp_path):
    host = _booted(tmp_path)
    host._wire_keys()
    host.overlay = OverlayState.HUD
    host.reg.dispatch(ord("s"))
    assert host.ui.pending_save is False
    assert host.hud.toasts.active()                  # gated, with feedback


def test_two_saves_in_the_same_second_get_distinct_names(tmp_path,
                                                         monkeypatch):
    host = _booted(tmp_path)
    import dtouch.shell as shell_mod
    monkeypatch.setattr(shell_mod.time, "strftime", lambda fmt: "120000")
    host.ui.pending_save = True
    host._pump_preset_mailboxes()
    host.ui.renaming = None
    host.ui.pending_save = True
    host._pump_preset_mailboxes()
    assert host.ui.user_presets == {"mine_120000", "mine_120000_2"}


def test_rename_and_delete_confirm_with_toasts(tmp_path):
    host = _booted(tmp_path)
    host.ui.pending_save = True
    host._pump_preset_mailboxes()
    name = host.ui.renaming
    host.ui.renaming = None
    host.ui.pending_rename = (name, "neon")
    host._pump_preset_mailboxes()
    assert any("renamed" in t for t in _hints(host))
    host.ui.pending_delete = "neon"
    host._pump_preset_mailboxes()
    assert any("deleted" in t for t in _hints(host))
    assert "neon" not in host.ui.presets


# ---------- rename mailbox pump ----------

def test_rename_mailbox_reloads_names_follows_selection_and_bank(tmp_path):
    host = _booted(tmp_path)
    ui = host.ui
    ui.pending_save = True
    host._pump_preset_mailboxes()
    name = next(iter(ui.user_presets))
    ui.renaming = None                       # close any auto-opened rename box
    ui.preset_idx = ui.presets.index(name)
    ui.pending_slot = name                   # assign a bank slot
    host._pump_preset_mailboxes()
    slot = next(s for s, n in ui.bank.items() if n == name)
    ui.pending_rename = (name, "neon dancer")
    host._pump_preset_mailboxes()
    assert "neon dancer" in ui.presets and name not in ui.presets
    assert ui.preset_name == "neon dancer"           # selection followed
    assert ui.bank[slot] == "neon dancer"            # bank followed
    assert presets.bank(host.presets_path,
                        mode="dithergirl")[slot] == "neon dancer"


# ---------- boot: hint + last-used-mode resume (DESIGN.md §3) ----------

def test_boot_posts_the_4s_hud_hint(tmp_path):
    """DESIGN.md §3: a HUD hint fades in for 4 s on boot — the three doors."""
    host = _booted(tmp_path)
    hint = next((t for t in host.hud.toasts._hints
                 if t.text == "m menu - TAB panel - ? keys"), None)
    assert hint is not None
    assert hint.ttl == 4.0


def test_no_explicit_mode_resumes_last_used_from_state(tmp_path):
    """DESIGN.md §3: launch goes straight into the last-used mode — a Host
    built with mode=None reads state.json and boots into it."""
    p = _paths(tmp_path)
    presets.save_state({"mode": "dithergirl", "preset": None, "bank": {}},
                       path=p["state_path"])
    host = Host(None, source=SyntheticSource(), res=RES, show=False,
                preset=None, max_frames=1, **p)
    host.run()
    assert host.mode.id == "dithergirl"


def test_no_state_file_resolves_boot_mode_to_particles(tmp_path):
    from dtouch.modes.particles import ParticlesMode
    host = _host(tmp_path)
    assert isinstance(host._resolve_boot_mode(), ParticlesMode)


def test_unknown_state_mode_resolves_to_particles(tmp_path):
    from dtouch.modes.particles import ParticlesMode
    p = _paths(tmp_path)
    presets.save_state({"mode": "gone-forever"}, path=p["state_path"])
    host = _host(tmp_path)
    assert isinstance(host._resolve_boot_mode(), ParticlesMode)


# ---------- corrupt presets file: the note reaches the toasts ----------

def test_corrupt_presets_note_reaches_the_toasts(tmp_path):
    p = _paths(tmp_path)
    with open(p["presets_path"], "w") as f:
        f.write("{broken")
    host = _booted(tmp_path)
    assert any("backup" in t for t in _hints(host))
    assert list(tmp_path.glob("presets.corrupt.*.bak.json"))
