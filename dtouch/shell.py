"""The shell (Host) — owns the show; modes are plugins racked into it.

DESIGN.md §2.1 split: the shell owns the window (WINDOW_AUTOSIZE, refit on res
change), camera capture + loss degradation (hold last good frame + amber HUD
note — modes never see None), mouse routing, key routing through the command
registry, audio (LiveMic, lazy), the recorder (post-FX, pre-panel — the shipped
invariant), preset store CRUD, the overlay state machine + HUD, blackout and
panic, the shared SIGNAL post-FX rack (CircuitBent applied after mode output +
video composite — §2.4: Glitch is not a mode), fps/status, mode lifecycle
(idempotent stop → start; a start() exception toasts and reinstates the
previous mode), and quit (double-q confirm).

Testability: Host accepts an injectable frame source (any object with
cv2-style ``read() -> (ok, frame_bgr)`` and optional ``release()``/``name``),
so tests drive the full loop headless with synthetic frames; ``show=False`` /
``max_frames`` semantics are live_flow's, kept.
"""
from __future__ import annotations

import os
import time

import cv2
import imageio.v2 as imageio

from .audio import LiveMic
from .camera import open_camera
from .circuit_bent import CircuitBent
from .commands import Command, CommandRegistry
from .hud import (AMBER, RED, Hud, OverlayState, cycle_overlay, draw_help,
                  esc_overlay)
from .overlay_ui import (OverlayUI, build_signal_section, build_global_rows)
from . import presets as _presets


class PerformState:
    """Mutable perform-layer flags (DESIGN.md §6.2), owned by the shell."""

    def __init__(self, now=time.monotonic):
        self._now = now
        self.blackout = False     # output.blackout — hard black, recorded
        self.help_open = False    # '?' overlay; any key closes it
        self.quit = False
        self._q_at = None         # first-press time of the quit confirm

    def q_pressed(self):
        """Quit confirm: 'q' twice within 2 s quits. True when this press quits."""
        t = self._now()
        if self._q_at is not None and t - self._q_at <= 2.0:
            self.quit = True
            return True
        self._q_at = t
        return False


def ui_toggle_command(ui, toasts, name, label, key, attr, flash_label):
    """A named command that flips a UI-state attr with momentary feedback.
    Shared by the shell's global toggles and the modes' local ones, so both
    paths stay one implementation."""
    def run():
        v = not getattr(ui, attr)
        setattr(ui, attr, v)
        toasts.flash(f"{flash_label} ON" if v else f"{flash_label.lower()} off")
    return Command(name, label, key, run)


def _register_quit(reg, ps, toasts):
    def run():
        if not ps.q_pressed():
            toasts.hint("q again to quit")
    reg.add("app.quit", "Quit (press twice)", "q", run)


def _wire_perform_keys(reg, ui, hud, ps, recall, mode_commands=None):
    """Register the perform layer (DESIGN.md §6.2) on `reg`.

    `recall(name)` must route a preset apply through the same path a panel click
    takes (the shell's pending_preset mailbox). Commands read `ui.presets` at
    press time, so saves/deletes are picked up live. The digit bank is the flat
    preset list (built-ins + user, load order) — a temporary global bank until
    presets go per-mode (DESIGN.md §8 step 7). The panel's own toggles remain
    and stay in sync: both paths set the same ui attributes.

    `mode_commands` is the active mode's `commands()` dict; when None the
    Particles-local toggles (F flock, V video bg) are registered directly —
    same commands, same keys, so direct callers see the full shipped key set.
    """
    toasts = hud.toasts

    def apply_idx(i):
        name = ui.presets[i]
        ui.preset_idx = i
        recall(name)
        toasts.flash(f"{i + 1} - {name}")

    def blackout():
        ps.blackout = not ps.blackout
        if ps.blackout:
            toasts.flash("BLACKOUT", AMBER)
        else:
            toasts.hint("blackout off")

    def panic():
        ps.blackout = False           # panic disarms blackout (DESIGN.md §6.2 '0')
        recall(ui.preset_name)        # re-apply the current preset's defaults
        toasts.flash("RESET", AMBER)

    def record():
        ui.record = not ui.record
        if ui.record:
            toasts.flash("REC", RED)
        # the stop toast (filename) comes from the loop when the file closes

    reg.add("output.blackout", "Blackout", " ", blackout)
    reg.add("preset.panic", "Panic reset", "0", panic)
    for i in range(9):
        reg.add(f"preset.recall.{i + 1}", f"Recall preset {i + 1}", str(i + 1),
                lambda i=i: (apply_idx(i) if i < len(ui.presets)
                             else toasts.hint(f"no preset {i + 1}")))
    reg.add("preset.prev", "Previous preset", "[",
            lambda: apply_idx((ui.preset_idx - 1) % len(ui.presets)))
    reg.add("preset.next", "Next preset", "]",
            lambda: apply_idx((ui.preset_idx + 1) % len(ui.presets)))
    if mode_commands is None:
        mode_commands = {
            "layer.flock": ui_toggle_command(ui, toasts, "layer.flock", "Flock",
                                             "f", "flock", "FLOCK"),
            "video_bg.toggle": ui_toggle_command(ui, toasts, "video_bg.toggle",
                                                 "Video background", "v",
                                                 "video_bg", "VIDEO BG"),
        }
    for cmd in mode_commands.values():
        reg.register(cmd)
    reg.register(ui_toggle_command(ui, toasts, "layer.glitch", "Glitch",
                                   "g", "glitch", "GLITCH"))
    reg.register(ui_toggle_command(ui, toasts, "audio.toggle", "Sound react",
                                   "a", "audio", "SOUND"))
    reg.add("record.toggle", "Record", "r", record)
    reg.add("debug.toggle", "Debug readout", "i",
            lambda: setattr(hud, "debug", not hud.debug))   # feedback: the HUD line
    reg.add("help.overlay", "Key map", "?",
            lambda: setattr(ps, "help_open", True))
    _register_quit(reg, ps, toasts)
    reg.on_unknown = lambda code: toasts.hint("? for keys")


def _perform_key(key, overlay, ps, reg, hud):
    """One unconsumed keypress through the perform layer. Returns the overlay state.

    Order matters: an open help overlay eats the key (any key closes it), then
    TAB/Esc step the overlay state, then the command registry dispatches —
    unknown printable keys fall through to the gentle '? for keys' toast.
    """
    if ps.help_open:
        ps.help_open = False
        return overlay
    overlay, handled = _overlay_key(key, overlay, hud.toasts)
    if not handled:
        reg.dispatch(key)
    return overlay


def _overlay_key(key, state, toasts):
    """TAB/Esc overlay-state stepping (DESIGN.md §6.1). Returns (state, handled).

    TAB cycles HIDDEN -> HUD -> PANEL -> HIDDEN; Esc steps one toward HIDDEN and
    in HIDDEN does nothing — Esc never quits. Entering HIDDEN emits one final
    toast, then the output is provably clean once it fades.
    """
    if key == 9:       # TAB
        new = cycle_overlay(state)
    elif key == 27:    # Esc
        new = esc_overlay(state)
    else:
        return state, False
    if new is OverlayState.HIDDEN and state is not OverlayState.HIDDEN:
        toasts.hint("overlay hidden - TAB to show")
    return new, True


def _apply_fx(ui, raw):
    """Restore MOTION + SIGNAL state from a saved look, if it recorded any.

    Every key is optional: the built-in presets predate these effects and saved looks
    from before this change have none of them. A missing key must leave the live toggle
    alone rather than resetting it — otherwise hopping between built-in templates would
    silently switch your glitch off.
    """
    from .overlay_ui import DITHERS
    if "flock" in raw: ui.flock = bool(raw["flock"])
    if "cohere" in raw: ui.cohere = float(raw["cohere"])
    if "align" in raw: ui.align = float(raw["align"])
    if "separate" in raw: ui.separate = float(raw["separate"])
    if "glitch" in raw: ui.glitch = bool(raw["glitch"])
    if "chroma" in raw: ui.chroma = float(raw["chroma"])
    if "drift" in raw: ui.drift = float(raw["drift"])
    if "crush" in raw: ui.crush = float(raw["crush"])
    if "scanlines" in raw: ui.scanlines = bool(raw["scanlines"])
    if raw.get("dither") in DITHERS: ui.dither_idx = DITHERS.index(raw["dither"])


class CameraSource:
    """The default frame source — wraps camera selection (dtouch.camera)."""

    def __init__(self, device="builtin"):
        self.cap, self.name = open_camera(device)

    def read(self):
        return self.cap.read()

    def release(self):
        self.cap.release()


class Host:
    """The shell. Construct with a mode (and optionally an injected frame
    source), then ``run()`` — returns ``(frame_count, last_rgb_frame)`` exactly
    like live_flow did."""

    WIN = "dtouch - flow"

    def __init__(self, mode, source=None, device="builtin", res=(1920, 1080),
                 mirror=True, seed=1, preset="abstract", audio=False,
                 panel=True, show=True, max_frames=None,
                 presets_path="presets.json"):
        self.mode = None
        self._boot_mode = mode
        self._source = source
        self._device = device
        self.res = tuple(res)
        self.mirror = mirror
        self.seed = seed
        self._boot_preset = preset
        self._boot_audio = audio
        self.panel = panel
        self.show = show
        self.max_frames = max_frames
        self.presets_path = presets_path

        self.ui = None
        self.hud = Hud()
        self.overlay = OverlayState.HUD   # boot state: HUD (DESIGN.md §6.1)
        self.ps = PerformState()
        self.reg = CommandRegistry()
        self.cb = None                    # SIGNAL rack post-FX, built on first use
        self.mic = None
        self.writer = None
        self.rec_path = None
        self.all_presets = {}
        self.cam_name = "?"
        self.fps = 0.0

    # ----- mode lifecycle (DESIGN.md §2.1) -----
    def set_mode(self, mode):
        """Idempotent stop of the current mode, then start the new one. A
        start() exception toasts the human summary and reinstates the previous
        mode; with no previous mode it propagates (boot failure is fatal)."""
        prev = self.mode
        if prev is not None:
            prev.stop()
        try:
            mode.start(self)
        except Exception as e:                       # noqa: BLE001 — §6.4
            if prev is None:
                raise
            self.hud.toasts.flash(f"{mode.title} failed to start", AMBER)
            self.hud.toasts.hint(str(e)[:80])
            prev.start(self)
            self.mode = prev
            return False
        self.mode = mode
        return True

    def compose_spec(self, mode):
        """The mode's declared sections + the shell's SIGNAL rack + global rows
        (DESIGN.md §2.4: the rack is a shell-owned section on every panel)."""
        return mode.panel_spec() + [build_signal_section()] + build_global_rows()

    # ----- preset plumbing -----
    def _reload_presets(self):
        self.all_presets = _presets.load(self.presets_path)
        names = list(self.all_presets.keys())
        if self.ui is not None:
            self.ui.presets = names
            self.ui.user_presets = _presets.user_names(self.presets_path)
        return names

    def _capture_cfg(self):
        """Today's hand-written save dict (spec-derived capture lands at
        DESIGN.md §8 step 7)."""
        ui, mode = self.ui, self.mode
        return dict(matte=mode.matte_kind, palette=mode.pf.palette,
                    fade=mode.glow.fade, exposure=ui.exposure, spark=ui.spark,
                    curl_amp=ui.curl, reseed_frac=ui.reseed, base_size=ui.dot,
                    damp=ui.damp, pull_falloff=ui.pull,
                    attract_speed=mode.pf.attract_speed,
                    video_bg=ui.video_bg, video_mix=ui.video_mix,
                    audio=ui.audio, sens=ui.sens,
                    # MOTION + SIGNAL travel with the look; without these a
                    # saved glitch preset would come back clean.
                    flock=ui.flock, cohere=ui.cohere, align=ui.align,
                    separate=ui.separate,
                    glitch=ui.glitch, chroma=ui.chroma, drift=ui.drift,
                    crush=ui.crush, dither=ui.dither_name,
                    scanlines=ui.scanlines)

    def _apply_pending_preset(self):
        ui = self.ui
        raw = self.mode.apply_look(ui.pending_preset, self.all_presets)
        ui.sync_from(self.mode.pf, self.mode.glow, self.mode.matte_kind)
        if "video_bg" in raw: ui.video_bg = bool(raw["video_bg"])
        if "video_mix" in raw: ui.video_mix = float(raw["video_mix"])
        if "audio" in raw: ui.audio = bool(raw["audio"])
        if "sens" in raw: ui.sens = float(raw["sens"])
        _apply_fx(ui, raw)
        ui.pending_preset = None

    def _pump_preset_mailboxes(self):
        """The pending_* mailboxes a panel click posts to (shipped semantics)."""
        ui = self.ui
        if ui.pending_preset:
            self._apply_pending_preset()
        if ui.pending_save:
            name = "mine_%s" % time.strftime("%H%M%S")
            _presets.save(name, self._capture_cfg(), path=self.presets_path)
            names = self._reload_presets()
            if name in names:
                ui.preset_idx = names.index(name)
            print("saved preset", name)
            ui.pending_save = False
        if ui.pending_delete:
            sel = ui.preset_name if ui.preset_idx < len(ui.presets) else None
            if _presets.delete(ui.pending_delete, path=self.presets_path):
                names = self._reload_presets()
                ui.preset_idx = names.index(sel) if sel in names else 0
                print("deleted preset", ui.pending_delete)
            ui.pending_delete = None
        if ui.pending_rename:
            old, new = ui.pending_rename
            sel = ui.preset_name if ui.preset_idx < len(ui.presets) else None
            if _presets.rename(old, new, path=self.presets_path):
                names = self._reload_presets()
                target = new if sel == old else sel
                ui.preset_idx = names.index(target) if target in names else 0
                print("renamed preset", old, "->", new)
            else:
                print("rename refused (name taken or invalid):", old, "->", new)
            ui.pending_rename = None

    # ----- host-owned per-frame sync (mirror / res / mic / recorder) -----
    def _sync_host_state(self):
        ui = self.ui
        self.mirror = ui.mirror
        nw, nh = ui.res_wh
        if (nw, nh) != self.res:
            self.res = (nw, nh)
            self.mode.on_resize(nw, nh)
            ui.w, ui.h = nw, nh   # AUTOSIZE window refits on next imshow
        if ui.audio and self.mic is None:
            self.mic = LiveMic(); self.mic.start()
        elif not ui.audio and self.mic is not None:
            self.mic.stop(); self.mic = None
        if ui.record and self.writer is None:
            self.rec_path = os.path.join(
                "out", "rec_%s.mp4" % time.strftime("%Y%m%d_%H%M%S"))
            self.writer = imageio.get_writer(self.rec_path, fps=24, macro_block_size=8)
        elif not ui.record and self.writer is not None:
            self.writer.close(); print("saved", self.rec_path)
            self.hud.toasts.hint("saved " + self.rec_path)   # filename toast on stop
            self.writer = None

    # ----- the loop -----
    def run(self):
        rw, rh = self.res
        if self._source is None:
            self._source = CameraSource(self._device)
        self.cam_name = getattr(self._source, "name", "source")

        if not self.set_mode(self._boot_mode):
            raise RuntimeError("boot mode failed to start")
        mode = self.mode
        self.all_presets = _presets.load(self.presets_path)
        preset = self._boot_preset
        raw0 = (mode.apply_look(preset, self.all_presets)
                if preset in self.all_presets else {})
        video_bg = bool(raw0.get("video_bg", mode.boot_video_bg))
        video_mix = float(raw0.get("video_mix", mode.boot_video_mix))
        audio = bool(raw0.get("audio", self._boot_audio))

        ui = None
        if self.show:
            # AUTOSIZE: the window is fixed at the render resolution so the OS
            # can't maximize/scale it — that scaling was tanking fps (display
            # upscaling) and breaking click mapping. To go bigger, switch the
            # 'output' resolution; the window resizes to match natively.
            cv2.namedWindow(self.WIN, cv2.WINDOW_AUTOSIZE)
            if self.panel:
                ui = self.ui = OverlayUI(rw, rh, list(self.all_presets.keys()),
                                         list(mode.palettes), list(mode.mattes),
                                         preset=preset, matte=mode.matte_kind,
                                         palette=mode.pf.palette)
                ui.set_spec(self.compose_spec(mode))
                ui.sync_from(mode.pf, mode.glow, mode.matte_kind)
                ui.mirror = self.mirror
                ui.audio = audio
                ui.video_bg = video_bg
                ui.video_mix = video_mix
                if "sens" in raw0:
                    ui.sens = float(raw0["sens"])
                _apply_fx(ui, raw0)   # the look loaded at startup, same as a live switch
                mode.configure_ui(ui)
                self.ui.user_presets = _presets.user_names(self.presets_path)
                cv2.setMouseCallback(self.WIN, ui.on_mouse)

        # Key routing goes through the command registry (DESIGN.md principle 7).
        if ui is not None:
            _wire_perform_keys(self.reg, ui, self.hud, self.ps,
                               lambda name: setattr(ui, "pending_preset", name),
                               mode_commands=mode.commands())
        else:
            _register_quit(self.reg, self.ps, self.hud.toasts)
        help_rows = self.reg.table() + [("TAB", "Cycle overlay"),
                                        ("Esc", "Step toward hidden")]

        if audio:
            self.mic = LiveMic(); self.mic.start()
        os.makedirs("out", exist_ok=True)

        t0 = time.time(); fps = 0.0; count = 0
        black_streak = 0
        last_frame = None    # last good camera frame, held across read failures
        camera_lost = False
        last_t = time.monotonic()
        out = None
        try:
            while True:
                ok, frame = self._source.read()
                if not ok:
                    # Camera loss: hold the last good frame and say so on the HUD
                    # (DESIGN.md §6.4); recovery is automatic when reads resume.
                    # Modes never see None (DESIGN.md §2.2).
                    if last_frame is not None:
                        frame, camera_lost = last_frame, True
                    elif self.max_frames is None:
                        continue
                    else:
                        break
                else:
                    last_frame, camera_lost = frame, False
                black_streak = black_streak + 1 if float(frame.mean()) < 3.0 else 0

                if ui is not None:
                    self._pump_preset_mailboxes()
                    self._sync_host_state()

                if self.mirror:
                    frame = cv2.flip(frame, 1)

                now_t = time.monotonic()
                dt, last_t = now_t - last_t, now_t
                levels = (self.mic.levels()
                          if self.mic is not None and self.mic.available else None)
                out = mode.step(frame, levels, dt)

                # SIGNAL rack post-FX (DESIGN.md §2.4). Applied here on purpose:
                # after the mode's render + video composite (so it bends the whole
                # picture), before the recorder (so captures match what you see)
                # and before ui.draw (so the panel never gets glitched into
                # unreadability). CircuitBent is documented for BGR; `out` is RGB,
                # which only swaps which channel drifts left vs right — the offsets
                # are independent symmetric draws, so the look is identical.
                # Constructed lazily so a session that never enables it pays nothing.
                if ui is not None and ui.glitch:
                    if self.cb is None:
                        self.cb = CircuitBent(seed=self.seed)
                    cb = self.cb
                    cb.chroma_shift = ui.chroma
                    cb.scan_drift = ui.drift
                    cb.bit_crush = int(ui.crush)
                    cb.scanlines = ui.scanlines
                    cb.dither_mode = None if ui.dither_name == "off" else ui.dither_name
                    out = cb.process(out)
                if self.ps.blackout:
                    # Hard black AFTER mode render/composite/glitch, BEFORE the
                    # recorder — blackout is part of the show and IS recorded; the
                    # UI/HUD still draw on top per overlay state (DESIGN.md §6.2).
                    out[:] = 0
                if self.writer is not None:
                    self.writer.append_data(out)
                bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

                count += 1
                if count % 10 == 0:
                    now = time.time(); fps = 10.0 / (now - t0); t0 = now
                self.fps = fps

                if self.show:
                    # HUD/panel draw AFTER the recorder write above — recordings
                    # never contain HUD or panel (the shipped invariant, kept).
                    rw, rh = self.res
                    status = mode.status_line(self.cam_name)
                    dbg = f"{fps:4.1f}fps  {1000.0 / fps if fps > 0 else 0.0:5.1f}ms  {rw}x{rh}"
                    if ui is not None and self.overlay is OverlayState.PANEL:
                        ui.draw(bgr, {"status": ""})
                    elif ui is not None:
                        ui._hot = []   # panel hidden: stale hit-rects must not eat clicks
                    self.hud.draw(bgr, self.overlay, status=status, debug_status=dbg,
                                  recording=(self.writer is not None),
                                  blackout=self.ps.blackout,
                                  camera_lost=(camera_lost or black_streak > 15))
                    if self.ps.help_open:
                        draw_help(bgr, help_rows)   # works in every overlay state
                    cv2.imshow(self.WIN, bgr)
                    key = cv2.waitKey(1) & 0xFF   # pump GUI + mouse
                    # rename-typing consumes every key; Esc only cancels the rename —
                    # while renaming, no global keys fire (DESIGN.md §6.2)
                    consumed = ui.on_key(key) if (ui is not None and key != 255) else False
                    if not consumed and key != 255:
                        self.overlay = _perform_key(key, self.overlay, self.ps,
                                                    self.reg, self.hud)
                    if self.ps.quit or (ui is not None and ui.quit):
                        break
                    # quit only when the window is actually destroyed (red X) ->
                    # property is -1. A minimized window reports 0, so this does
                    # NOT quit on minimize.
                    if cv2.getWindowProperty(self.WIN, cv2.WND_PROP_VISIBLE) < 0:
                        break
                if self.max_frames is not None and count >= self.max_frames:
                    break
        finally:
            if self.writer is not None:
                self.writer.close(); print("saved", self.rec_path)
                self.writer = None
            if self.mic is not None:
                self.mic.stop(); self.mic = None
            if hasattr(self._source, "release"):
                self._source.release()
            if self.mode is not None:
                self.mode.stop()
            if self.show:
                cv2.destroyAllWindows(); cv2.waitKey(1)
        return count, out
