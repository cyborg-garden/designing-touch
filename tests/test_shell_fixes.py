"""Shell-level regression pins for the verified UI-overhaul bug list.

Everything here drives the real Host loop headless with DitherGirlMode (pure
numpy/cv2 — no GL, so these run everywhere test_dithergirl.py does). GUI-path
tests monkeypatch cv2's window calls so `show=True` code paths run headless.
"""
import os

import numpy as np
import pytest

import cv2

from dtouch import matte as matte_mod
from dtouch import presets
from dtouch import shell as shell_mod
from dtouch.hud import OverlayState
from dtouch.modes.dithergirl import MATTES_DG, DitherGirlMode
from dtouch.shell import CameraSource, Host

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


def test_open_menu_consumes_comma_period_before_param_nudge(tmp_path):
    """','/'.' inside the open menu move card selection — the menu keeps
    priority over the param-nudge selection (DESIGN.md §3 vs §6.2)."""
    host = _booted(tmp_path)
    host._wire_keys()
    host.menu.show("dithergirl")
    sel = host.menu.sel
    host._route_key(ord(","))
    assert host.menu.sel != sel              # the menu moved...
    assert host.ui.nudge_idx == 0            # ...the nudge selection did not
    assert host.hud.osd._show is None


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


# ---------- SIGNAL rack dither-quality controls (DESIGN.md §2.4/§4.1) ----------

class NoClaimsMode(DitherGirlMode):
    """A GL-free mode that claims nothing — the full rack stays visible."""
    id = "noclaims"
    title = "NoClaims"
    claims = frozenset()


def test_rack_quality_controls_drive_circuitbent(tmp_path):
    """Bits/Gamma/Bias from the rack reach the CircuitBent dither call."""
    host = _host(tmp_path, mode=NoClaimsMode(), max_frames=3)

    def on_read(n):
        if n == 2:
            ui = host.ui
            ui.glitch = True
            ui.dither_idx = 0                # bayer
            ui.sig_bits = 1.6                # int-snaps to 2
            ui.sig_gamma = False
            ui.sig_bias_idx = 2              # dark -> invert True

    host._source.on_read = on_read
    host.run()
    assert host.cb is not None
    assert host.cb.dither_bits == 2
    assert host.cb.dither_gamma is False
    assert host.cb.dither_invert is True


def test_rack_quality_controls_round_trip_inside_signal(tmp_path):
    """The rack's Bits/Gamma/Bias persist under the look's "signal" block and
    apply back (DESIGN.md §7)."""
    host = _booted(tmp_path, mode=NoClaimsMode())
    ui = host.ui
    ui.sig_bits, ui.sig_gamma, ui.sig_bias_idx = 2.0, False, 2
    cfg = host._capture_cfg()
    assert cfg["signal"]["bits"] == 2.0
    assert cfg["signal"]["gamma"] is False
    assert cfg["signal"]["bias"] == "dark"
    presets.save("bent", cfg, path=host.presets_path, mode="noclaims")
    host._reload_presets()
    ui.sig_bits, ui.sig_gamma, ui.sig_bias_idx = 3.0, True, 0   # scramble
    ui.pending_preset = "bent"
    host._apply_pending_preset()
    assert (ui.sig_bits, ui.sig_gamma, ui.sig_bias_idx) == (2.0, False, 2)


# ---------- global Menu (M) row + panel title + 'i' in HIDDEN ----------

def test_menu_row_action_opens_the_menu(tmp_path):
    """DESIGN.md §4.1: the global 'Menu (M)' Action posts menu.open through
    pending_commands; the shell routes it to the same command 'm' runs."""
    host = _booted(tmp_path)
    host._wire_keys()
    host.ui.pending_commands.append("menu.open")
    host._pump_preset_mailboxes()
    assert host.menu.open is True
    assert host.ui.pending_commands == []


def test_panel_title_carries_the_mode(tmp_path):
    """DESIGN.md §4.1/§4.2 sketch: panel title is 'dtouch - MODE TITLE'."""
    host = _booted(tmp_path)
    assert host.ui.panel_title == "dtouch - DITHER GIRL"


def test_i_in_hidden_posts_the_debug_line_as_a_toast(tmp_path):
    host = _booted(tmp_path)
    host._wire_keys()
    host.overlay = OverlayState.HIDDEN
    host.reg.dispatch(ord("i"))
    assert any("fps" in t for t in _hints(host))
    n = sum("fps" in t for t in _hints(host))
    host.overlay = OverlayState.HUD          # HUD: the status line IS feedback
    host.reg.dispatch(ord("i"))
    assert sum("fps" in t for t in _hints(host)) == n


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


# ---------- a failed store write never takes the show down (§6.4) ----------

def _full_disk(monkeypatch, *names):
    def boom(*a, **kw):
        raise OSError(28, "No space left on device")
    for name in names:
        monkeypatch.setattr(presets, name, boom)


def test_save_click_on_a_full_disk_toasts_and_keeps_running(tmp_path,
                                                            monkeypatch):
    """json.dump's OSError used to propagate out of the mailbox pump and exit
    run() mid-performance — a save click could kill the show."""
    host = _host(tmp_path, max_frames=4,
                 src=SyntheticSource(
                     on_read=lambda n: setattr(host.ui, "pending_save", True)
                     if n == 2 and host.ui else None))
    _full_disk(monkeypatch, "save")
    count, out = host.run()
    assert count == 4                                # the loop survived
    assert host.hud.toasts._center.text == "save failed - disk?"
    assert not host.ui.user_presets                  # nothing pretends to exist
    assert host.ui.renaming is None                  # no rename box for a ghost


@pytest.mark.parametrize("mailbox,call,value", [
    ("pending_delete", "delete", lambda name: name),
    ("pending_rename", "rename", lambda name: (name, "neon")),
    ("pending_slot", "set_bank", lambda name: name),
])
def test_every_store_mailbox_survives_a_full_disk(tmp_path, monkeypatch,
                                                  mailbox, call, value):
    host = _booted(tmp_path)
    host.ui.pending_save = True
    host._pump_preset_mailboxes()                    # a real look to act on
    host.ui.renaming = None
    name = next(iter(host.ui.user_presets))
    _full_disk(monkeypatch, call)
    setattr(host.ui, mailbox, value(name))
    host._pump_preset_mailboxes()                    # must not raise
    assert host.hud.toasts._center.text == "save failed - disk?"


# ---------- a REFUSED write is not a successful one (DESIGN.md §9) ----------

def test_refused_save_never_says_saved_and_never_opens_the_rename_box(
        tmp_path, monkeypatch):
    """The store refuses to overwrite a file it could not back up. That
    refusal raises nothing, so the shell used to report 'saved - <name>',
    print 'saved preset', and drop the operator into a rename box on a preset
    that does not exist — a text field that then eats every keypress (q, m,
    panic included) with nothing on screen to explain it."""
    host = _booted(tmp_path)
    monkeypatch.setattr(presets, "write_file", lambda data, path=None: False)
    host.ui.pending_save = True
    host._pump_preset_mailboxes()

    assert host.ui.renaming is None                  # no text field on a ghost
    assert not host.ui.user_presets                  # nothing pretends to exist
    assert not any("saved -" in t for t in _hints(host))
    assert host.hud.toasts._center.text == "save refused"


def test_refused_store_write_toasts_the_stores_own_reason(tmp_path,
                                                          monkeypatch):
    """`_store_write` drains the store's queued notes onto the toasts, so the
    operator gets the REASON, not just 'refused' (silence on a data-loss event
    is a bug — DESIGN.md §9)."""
    host = _booted(tmp_path)

    def refuse(data, path=None):
        presets._note("presets.json not saved - unreadable and could not be "
                      "backed up")
        return False
    monkeypatch.setattr(presets, "write_file", refuse)
    host.ui.pending_save = True
    host._pump_preset_mailboxes()
    assert any("not saved" in t for t in _hints(host))
    presets.take_notes()


# ---------- the frame loop is contained: no traceback ends the show ----------
# DESIGN.md §6.4 + §8 step 10 ("any traceback = release blocker"). Nothing in
# the per-frame loop used to be wrapped, so ONE exception out of source.read,
# cv2.flip, mode.step or writer.append_data ended the performance with a stack
# trace on the projector.


class BoomMode(DitherGirlMode):
    """A mode whose step() raises from frame `fail_from` onward."""

    def __init__(self, fail_from=2, exc=None, **kw):
        super().__init__(**kw)
        self.fail_from = fail_from
        self.exc = exc or ModuleNotFoundError("No module named 'mediapipe'")
        self.steps = 0

    def step(self, frame_bgr, audio_levels, dt):
        self.steps += 1
        if self.steps >= self.fail_from:
            raise self.exc
        return super().step(frame_bgr, audio_levels, dt)


def test_mode_step_exception_holds_the_last_frame_instead_of_ending_the_show(
        tmp_path):
    """The most reachable path: a start.command install with no mediapipe,
    one click on the shipped `portrait` template, ModuleNotFoundError out of
    mode.step(). The loop must survive it and keep holding the last good
    picture."""
    mode = BoomMode(fail_from=3)
    host = _host(tmp_path, mode=mode, max_frames=8)
    count, out = host.run()

    assert count == 8                                # the loop never exited
    assert mode.steps == 8                           # ...and kept stepping
    assert out is not None and out.any()             # last good picture held
    assert host.hud.toasts._center.text == "something went wrong - show continues"
    assert any("ModuleNotFoundError" in t for t in _hints(host))


def test_a_frame_error_on_the_very_first_frame_is_survivable(tmp_path):
    """Nothing good has been rendered yet, so there is no picture to hold —
    an intentional black frame, not an unbound-variable crash."""
    host = _host(tmp_path, mode=BoomMode(fail_from=1), max_frames=3)
    count, out = host.run()
    assert count == 3
    assert out is not None and not out.any()


def test_repeating_frame_error_does_not_spam_the_toasts(tmp_path):
    """A per-frame exception at 60 fps would refill the toast stack sixty
    times a second and bury the status line under its own error."""
    host = _host(tmp_path, mode=BoomMode(fail_from=1), max_frames=40)
    flashes = []
    host.hud.toasts.flash = lambda text, color=None: flashes.append(text)
    host.run()
    assert flashes.count("something went wrong - show continues") == 1


def test_a_different_frame_error_toasts_immediately(tmp_path):
    """Rate-limiting is per (type, message) — a NEW failure is news."""
    host = _host(tmp_path, max_frames=1)
    host.hud.toasts.flash = lambda *a, **k: None
    host._frame_error(ValueError("first"))
    first = host._err_at
    host._frame_error(TypeError("second"))
    assert host._err_at != first
    assert host._err_key == ("TypeError", "second")


def test_keys_still_work_while_every_single_frame_fails(tmp_path, monkeypatch):
    """The contract that matters (DESIGN.md §6.4: no keyboard-reachable state
    requires a restart) — the produce half is contained, so the present half
    still draws and still pumps `q`."""
    _patch_gui(monkeypatch, keys=[255, ord("q"), ord("q")])
    host = _host(tmp_path, mode=BoomMode(fail_from=1), show=True)
    host.run()
    assert host.ps.quit is True


def test_keyboard_interrupt_still_propagates(tmp_path):
    """Ctrl-C means stop. The guard is for bugs, not for the operator."""
    host = _host(tmp_path, mode=BoomMode(fail_from=1, exc=KeyboardInterrupt()),
                 max_frames=4)
    with pytest.raises(KeyboardInterrupt):
        host.run()


def test_source_read_exception_is_contained(tmp_path):
    """A camera that raises rather than returning (False, None) — a USB yank
    mid-read — is the same event to the operator."""
    src = SyntheticSource()

    def boom(n):
        if n >= 2:
            raise OSError(5, "Input/output error")
    src.on_read = boom
    host = _host(tmp_path, src=src, max_frames=5)
    count, _ = host.run()
    assert count == 5
    assert any("OSError" in t for t in _hints(host))


# ---------- a failing recorder stops recording, not the show ----------

def test_recorder_write_failure_stops_the_take_and_keeps_running(tmp_path):
    class BadWriter(FakeWriter):
        def append_data(self, f):
            raise OSError(28, "No space left on device")

    host = _host(tmp_path, max_frames=6)

    def on_read(n):
        if n == 2:
            host.ui.record = True
            host.writer, host.rec_path = BadWriter(), "fake.mp4"

    host._source.on_read = on_read
    count, out = host.run()

    assert count == 6                                # the show carried on
    assert host.writer is None                       # closed cleanly
    assert host.ui.record is False                   # and disarmed, not retried
    assert host.hud.toasts._center.text == "recording stopped - write failed"


# ---------- recordings: out/ on first take, and never a silent overwrite ----------

def test_out_dir_is_not_created_at_boot(tmp_path, monkeypatch):
    """`os.makedirs("out")` at boot raised PermissionError on a read-only
    launch dir (a run from a mounted DMG) — a traceback before any window
    existed, for a directory most sessions never use."""
    monkeypatch.chdir(tmp_path)
    _booted(tmp_path)
    assert not (tmp_path / "out").exists()


def test_first_record_creates_out_and_names_the_take(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    host = _booted(tmp_path)
    made = []
    monkeypatch.setattr(shell_mod.imageio, "get_writer",
                        lambda p, **kw: made.append(p) or FakeWriter())
    host.ui.record = True
    host._sync_host_state()
    assert (tmp_path / "out").is_dir()
    assert made and made[0] == host.rec_path
    assert os.path.basename(host.rec_path).startswith("rec_")


def test_a_launch_dir_that_cannot_take_recordings_toasts_and_disarms(
        tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "out").write_text("not a directory")   # makedirs raises here
    host = _booted(tmp_path)
    host.ui.record = True
    host._sync_host_state()                            # must not raise
    assert host.writer is None
    assert host.ui.record is False                     # disarmed, not retried
    assert host.hud.toasts._center.text == "cannot write recordings here"


def test_two_takes_in_the_same_second_get_distinct_filenames(tmp_path,
                                                             monkeypatch):
    """Names are second-resolution. Stopping and restarting a take inside one
    second reused the name and silently truncated the first file — the take
    you just made, gone."""
    monkeypatch.chdir(tmp_path)
    host = _booted(tmp_path)
    monkeypatch.setattr(shell_mod.time, "strftime",
                        lambda fmt: "20260101_120000")
    os.makedirs("out", exist_ok=True)

    first = host._rec_path()
    open(first, "wb").close()                          # the take now on disk
    second = host._rec_path()
    open(second, "wb").close()
    third = host._rec_path()

    assert os.path.basename(first) == "rec_20260101_120000.mp4"
    assert os.path.basename(second) == "rec_20260101_120000_2.mp4"
    assert os.path.basename(third) == "rec_20260101_120000_3.mp4"


# ---------- a camera that will not open never reaches a traceback ----------

def test_camera_that_cannot_be_opened_still_opens_the_window(tmp_path,
                                                             monkeypatch):
    """`open_camera` raises whenever isOpened() is False — no camera, TCC
    denied, or the device held by Zoom/OBS. That traceback landed before any
    window existed, so the plain-language permission card DESIGN.md §6.4
    promises could never be shown."""
    def denied(device):
        raise RuntimeError("could not open camera 'builtin'")
    monkeypatch.setattr(shell_mod, "open_camera", denied)
    shown = []
    _patch_gui(monkeypatch, keys=[255, ord("q"), ord("q")], shown=shown)

    host = Host(DitherGirlMode(), source=None, res=RES, show=True, preset=None,
                **_paths(tmp_path))
    count, out = host.run()                            # must not raise

    assert count == 0
    assert host.ps.quit is True                        # still quittable
    assert shown and float(shown[0].mean()) < 20 and shown[0].max() > 0
    assert any("could not open camera" in t for t in _hints(host))


def test_camera_source_recovers_when_the_camera_appears(monkeypatch):
    """Recovery is automatic: the open is retried, so the show starts by
    itself when the device is freed."""
    class FakeCap:
        def read(self):
            return True, np.zeros((4, 4, 3), np.uint8)

        def release(self):
            pass

    tries = []
    clock = [0.0]

    def flaky(device):
        tries.append(device)
        if len(tries) < 3:
            raise RuntimeError("busy")
        return FakeCap(), "FaceTime HD"
    monkeypatch.setattr(shell_mod, "open_camera", flaky)

    src = CameraSource("builtin", now=lambda: clock[0])
    assert src.read() == (False, None)
    assert src.name == "no camera" and src.error is not None

    clock[0] += CameraSource.RETRY_S
    assert src.read() == (False, None)                 # retried, still busy
    clock[0] += CameraSource.RETRY_S
    ok, frame = src.read()
    assert ok and frame is not None
    assert src.name == "FaceTime HD" and src.error is None
    src.release()


def test_camera_source_does_not_retry_faster_than_the_backoff(monkeypatch):
    calls = []

    def busy(device):
        calls.append(device)
        raise RuntimeError("nope")
    monkeypatch.setattr(shell_mod, "open_camera", busy)
    src = CameraSource("builtin", now=lambda: 0.0)
    for _ in range(20):
        src.read()
    assert len(calls) == 1                             # one open, not twenty


def test_a_late_camera_gets_its_name_onto_the_status_line(tmp_path):
    """A CameraSource that only opened on its third retry boots named 'no
    camera'; the status line has to pick up the real name when frames start."""
    src = SyntheticSource()
    src.name = "no camera"
    host = _host(tmp_path, src=src, max_frames=4)

    def on_read(n):
        if n >= 3:
            src.name = "FaceTime HD"                   # the camera turns up

    src.on_read = on_read
    host.run()
    assert host.cam_name == "FaceTime HD"


# ---------- a missing optional matte reverts the cycle, it never raises ----------

def _no_mediapipe(monkeypatch):
    def missing():
        raise ModuleNotFoundError("No module named 'mediapipe'")
    monkeypatch.setitem(matte_mod.KINDS, "person", missing)


def test_missing_person_matte_reverts_the_cycle_and_toasts_the_extra(
        tmp_path, monkeypatch):
    """The shipped `portrait` and `sigil` templates both select `person`, so
    on any install without the extra this is one click from the frame loop."""
    _no_mediapipe(monkeypatch)
    host = _host(tmp_path, max_frames=6)

    def on_read(n):
        if n == 3:
            host.ui.dg_matte_idx = MATTES_DG.index("person")

    host._source.on_read = on_read
    count, out = host.run()

    assert count == 6
    assert host.ui.dg_matte_idx == MATTES_DG.index("off")   # reverted
    # a graceful step-back with the fix in it, NOT the generic frame-error path
    assert host.hud.toasts._center.text == "person matte needs the [person] extra"


def test_select_matte_keeps_the_callers_operator_on_refusal(monkeypatch):
    _no_mediapipe(monkeypatch)

    class UI:
        idx = 1
    ui = UI()
    mat, kind = matte_mod.select_matte(ui, "idx", ["luma", "person"], "luma")
    assert mat is None and kind == "luma"
    assert ui.idx == 0                                 # the cycle stepped back


def test_make_matte_raises_matte_unavailable_not_the_raw_import_error(
        monkeypatch):
    _no_mediapipe(monkeypatch)
    with pytest.raises(matte_mod.MatteUnavailable) as e:
        matte_mod.make_matte("person")
    assert "[person] extra" in str(e.value)
    with pytest.raises(KeyError):                      # a bad kind is still a bug
        matte_mod.make_matte("nonsense")
