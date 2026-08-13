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
import numpy as np

from .audio import LiveMic
from .camera import open_camera
from .circuit_bent import CircuitBent
from .commands import Command, CommandRegistry
from .hud import (AMBER, RED, Hud, OverlayState, cycle_overlay,
                  draw_corner_tick, draw_help, esc_overlay, put_outlined,
                  u as _u)
from .menu import Menu, draw_menu, render_boot_card
from .modes import REGISTRY, mode_by_id
from .overlay_ui import (OverlayUI, build_signal_section, build_global_rows)
from .panelspec import apply_look, capture_look
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


def _wire_perform_keys(reg, ui, hud, ps, recall, mode_commands=None,
                       safe_look=None, get_overlay=None):
    """Register the perform layer (DESIGN.md §6.2) on `reg`.

    `recall(name)` must route a preset apply through the same path a panel click
    takes (the shell's pending_preset mailbox). Commands read `ui.presets`,
    `ui.bank`, and `ui.setlist` at press time, so saves/deletes/assignments are
    picked up live. Digits 1–9 recall from the ACTIVE mode's bank — explicit
    slot assignments, not list positions (DESIGN.md §7); `[`/`]` walk the
    mode's setlist (empty = all looks in load order). The panel's own toggles
    remain and stay in sync: both paths set the same ui attributes.

    `mode_commands` is the active mode's `commands()` dict; when None the
    Particles-local toggles (F flock, V video bg) are registered directly —
    same commands, same keys, so direct callers see the full shipped key set.
    """
    toasts = hud.toasts

    def _slot_of(name):
        return next((s for s, n in (ui.bank or {}).items() if n == name), None)

    def _recall(name):
        ui.preset_idx = ui.presets.index(name)
        recall(name)

    def apply_slot(i):
        name = (ui.bank or {}).get(str(i))
        if not name or name not in ui.presets:
            toasts.hint(f"no preset {i}")
            return
        _recall(name)
        toasts.flash(f"{i} - {name}")

    def setlist_step(d):
        order = [n for n in (ui.setlist or list(ui.presets)) if n in ui.presets]
        if not order:
            toasts.hint("setlist empty")
            return
        cur = ui.preset_name if ui.preset_idx < len(ui.presets) else None
        if cur in order:
            i = (order.index(cur) + d) % len(order)
        else:
            i = 0 if d > 0 else len(order) - 1
        name = order[i]
        _recall(name)
        slot = _slot_of(name)
        toasts.flash(f"{slot} - {name}" if slot else name)

    def blackout():
        ps.blackout = not ps.blackout
        if ps.blackout:
            toasts.flash("BLACKOUT", AMBER)
        else:
            toasts.hint("blackout off")

    def panic():
        # Amended DESIGN.md §6.2 '0': panic restores a known-good PICTURE —
        # the mode's safe_look, blackout disarmed, SIGNAL rack (glitch) off.
        ps.blackout = False
        ui.glitch = False
        name = safe_look() if safe_look is not None else ui.preset_name
        if isinstance(name, str):
            if name in ui.presets:
                _recall(name)
            else:
                recall(name)
        toasts.flash("RESET", AMBER)

    def record():
        ui.record = not ui.record
        if ui.record:
            toasts.flash("REC", RED)
        # the stop toast (filename) comes from the loop when the file closes

    def save_preset():
        # 's' = preset.save (amended DESIGN.md §3/§6.2): PANEL state only —
        # saving is an edit action; elsewhere it hints instead of silence.
        state = (get_overlay() if get_overlay is not None
                 else OverlayState.PANEL)
        if state is not OverlayState.PANEL:
            toasts.hint("save is a panel action - TAB to open the panel")
            return
        ui.pending_save = True

    reg.add("output.blackout", "Blackout", " ", blackout)
    reg.add("preset.panic", "Panic reset", "0", panic)
    reg.add("preset.save", "Save current look", "s", save_preset)
    for i in range(1, 10):
        reg.add(f"preset.recall.{i}", f"Recall bank slot {i}", str(i),
                lambda i=i: apply_slot(i))
    reg.add("preset.prev", "Previous in setlist", "[", lambda: setlist_step(-1))
    reg.add("preset.next", "Next in setlist", "]", lambda: setlist_step(+1))
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


class CameraSource:
    """The default frame source — wraps camera selection (dtouch.camera)."""

    def __init__(self, device="builtin"):
        self.cap, self.name = open_camera(device)

    def read(self):
        return self.cap.read()

    def release(self):
        self.cap.release()


class StillSource:
    """A loaded still image as a frame source (DESIGN.md §2.1: the shell owns
    still sources; modes never special-case stills). read() returns the same
    frame every tick."""

    def __init__(self, path):
        frame = cv2.imread(path, cv2.IMREAD_COLOR)
        if frame is None:
            raise FileNotFoundError(f"could not read still image: {path}")
        self.frame = frame
        self.name = os.path.basename(path)

    def read(self):
        return True, self.frame

    def release(self):
        pass


class Host:
    """The shell. Construct with a mode (and optionally an injected frame
    source), then ``run()`` — returns ``(frame_count, last_rgb_frame)`` exactly
    like live_flow did."""

    WIN = "dtouch - flow"

    def __init__(self, mode, source=None, device="builtin", res=(1920, 1080),
                 mirror=True, seed=1, preset="abstract", audio=False,
                 panel=True, show=True, max_frames=None, still=None,
                 presets_path="presets.json", state_path="state.json"):
        self.mode = None
        self._boot_mode = mode
        self._source = source
        self._device = device
        self._still_path = still if isinstance(still, str) else None
        self.still = still if not isinstance(still, str) else None
        self.res = tuple(res)
        self.mirror = mirror
        self.seed = seed
        self._boot_preset = preset
        self._boot_audio = audio
        self.panel = panel
        self.show = show
        self.max_frames = max_frames
        self.presets_path = presets_path
        self.state_path = state_path

        self.ui = None
        self.hud = Hud()
        self.overlay = OverlayState.HUD   # boot state: HUD (DESIGN.md §6.1)
        self.ps = PerformState()
        self.reg = CommandRegistry()
        self.menu = Menu()                # home menu — a shell overlay state (§3)
        self.pending_mode = None          # mode id posted by a key/menu commit
        self._mode_instances = {}         # id -> constructed Mode (reused on switch)
        self._mode_preset = {}            # id -> last selected look (re-entry)
        self.help_rows = []
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
        if self.ui is not None:
            # live switch (step 8): rebind the option lists + accent, recompose
            # the panel, seed the mode's own UI attrs, and reload the new
            # mode's looks/bank/setlist
            ui = self.ui
            ui.palettes = list(getattr(mode, "palettes", ui.palettes))
            ui.mattes = list(getattr(mode, "mattes", ui.mattes))
            pal = getattr(getattr(mode, "pf", None), "palette", None)
            ui.palette_idx = (ui.palettes.index(pal)
                              if pal in ui.palettes else 0)
            mk = getattr(mode, "matte_kind", None)
            ui.matte_idx = ui.mattes.index(mk) if mk in ui.mattes else 0
            ui.accent = mode.accent
            ui.set_spec(self.compose_spec(mode))
            mode.configure_ui(ui)
            self._reload_presets()
            if ui.preset_idx >= len(ui.presets):
                ui.preset_idx = 0
            self._seed_bank_setlist()
            self._autosave_state()
        return True

    def compose_spec(self, mode):
        """The mode's declared sections + the shell's SIGNAL rack + global rows
        (DESIGN.md §2.4: the rack is a shell-owned section on every panel).

        Suppression rule (DESIGN.md §2.4, judge finding): the rack hides any
        control the active mode claims — a mode declares `claims` (a set of
        store keys, e.g. Dither Girl claims "dither" because it owns dithering
        as the primary image; two visible dither subsystems in one panel is
        the bolted-features incoherence the overhaul exists to kill)."""
        rack = build_signal_section()
        claims = frozenset(getattr(mode, "claims", ()))
        if claims:
            rack.widgets = [w for w in rack.widgets
                            if getattr(w, "store_key", None) not in claims]
        return mode.panel_spec() + [rack] + build_global_rows()

    # ----- mode switching (DESIGN.md §3 / §8 step 8) -----
    def request_mode(self, mode_id):
        """Post a switch — the loop performs it at the top of the next frame
        (boot card, stop → start, toast). Same-mode requests are a hint."""
        if self.mode is not None and mode_id == self.mode.id:
            self.hud.toasts.hint(f"already in {self.mode.title}")
            return
        self.pending_mode = mode_id

    def _switch_mode(self, mode_id):
        """One live mode switch: static boot card (shown + recorded — the
        recorder captures the card, not a gray flash), old.stop() →
        new.start(host) via set_mode (which toasts + reinstates the previous
        mode on failure), then the mode-title center toast in the new mode's
        accent. Overlay state, blackout, recording, and the audio toggle are
        host/UI state and survive untouched."""
        cls = mode_by_id(mode_id)
        if cls is None:
            self.hud.toasts.hint(f"unknown mode {mode_id}")
            return False
        first_entry = mode_id not in self._mode_instances
        # remember the outgoing mode's selected look so re-entry can restore
        # the selection highlight without re-applying anything
        if (self.mode is not None and self.ui is not None
                and self.ui.preset_idx < len(self.ui.presets)):
            self._mode_preset[self.mode.id] = self.ui.preset_name
        new = self._mode_instances.get(mode_id) or cls()
        if self.ps.blackout:
            # DESIGN.md §3: while blackout is armed the switch happens under
            # black — never flash the bright card to screen or recorder; only
            # the amber corner tick stays.
            card = np.zeros((self.res[1], self.res[0], 3), np.uint8)
            draw_corner_tick(card, 2.0 * _u(self.res[1]))
        else:
            card = render_boot_card(self.res, new.title, new.accent)
        if self.writer is not None:
            self.writer.append_data(cv2.cvtColor(card, cv2.COLOR_BGR2RGB))
        if self.show:
            cv2.imshow(self.WIN, card)
            cv2.waitKey(1)
        if not self.set_mode(new):
            return False                       # toasted + previous reinstated
        self._mode_instances[mode_id] = new
        if first_entry:
            # FIRST entry lands on the mode's known-good look (predictable
            # from 2 m away). RE-entry to an already-visited mode preserves
            # its current settings (amended DESIGN.md §6.2 — presets are an
            # instrument; switching away and back must not reset the look).
            safe = new.safe_look()
            if isinstance(safe, str) and safe in self.ui.presets:
                self.ui.preset_idx = self.ui.presets.index(safe)
                self.ui.pending_preset = safe
        else:
            prev = self._mode_preset.get(mode_id)
            if prev in self.ui.presets:
                self.ui.preset_idx = self.ui.presets.index(prev)
        if self.show:
            self._wire_keys()                  # mode-local commands changed
        self.hud.toasts.flash(new.title, new.accent)
        return True

    def _register_shell_commands(self):
        """Menu + direct mode-switch keys (DESIGN.md §6.2: `m` opens the menu
        from any overlay state; each mode's letter switches directly)."""
        self.reg.add("menu.open", "Menu", "m",
                     lambda: self.menu.toggle(self.mode.id if self.mode else None))
        for cls in REGISTRY:
            self.reg.add(f"mode.{cls.id}", f"Switch to {cls.title}", cls.id[:1],
                         lambda mid=cls.id: self.request_mode(mid))

    def _wire_keys(self):
        """(Re)build the command registry for the active mode — called at boot
        and after every switch (mode-local commands differ per mode)."""
        self.reg = CommandRegistry()
        _wire_perform_keys(self.reg, self.ui, self.hud, self.ps,
                           lambda name: setattr(self.ui, "pending_preset", name),
                           mode_commands=self.mode.commands(),
                           safe_look=lambda: self.mode.safe_look(),
                           get_overlay=lambda: self.overlay)
        self._register_shell_commands()
        self.help_rows = self.reg.table() + [("TAB", "Cycle overlay"),
                                             ("Esc", "Step toward hidden")]

    def _on_mouse(self, event, x, y, flags, param=None):
        """Window mouse routing: an open help modal swallows every mouse event
        (a click closes it — help already closes on any key); the open menu
        eats clicks (a card commits a switch); otherwise the panel gets the
        event."""
        if self.ps.help_open:
            if event == cv2.EVENT_LBUTTONDOWN:
                self.ps.help_open = False
            return
        if self.menu.open:
            if event == cv2.EVENT_LBUTTONDOWN:
                mode_id = self.menu.click((x, y))
                if mode_id:
                    self.request_mode(mode_id)
            return
        if self.ui is not None:
            self.ui.on_mouse(event, x, y, flags, param)

    def _route_key(self, key):
        """One waitKey code through the routing contract: the open menu
        consumes every key (DESIGN.md §3 — unknown keys hint, q closes the
        menu AND arms the quit confirm); then rename-typing consumes every
        key (Esc only cancels the rename — §6.2); then the perform layer."""
        if key == 255:
            return
        if self.menu.open:
            action, mode_id = self.menu.key(key)
            if action == "switch":
                self.request_mode(mode_id)
            elif action == "quit":
                self.reg.dispatch(ord("q"))     # first press toasts (§6.2)
            elif action == "unknown":
                self.hud.toasts.hint("? for keys")
            return
        if self.ui is not None and self.ui.on_key(key):
            return
        self.overlay = _perform_key(key, self.overlay, self.ps,
                                    self.reg, self.hud)

    def _draw_waiting_note(self, img):
        """No frame has ever arrived (DESIGN.md §6.4): a plain-language
        on-canvas explanation — never a traceback, never a frozen gray box."""
        h, w = img.shape[:2]
        uu = _u(h)
        ix = int(w * 0.035)                    # title-safe inset (§5)
        put_outlined(img, "waiting for camera...",
                     (ix, h // 2), max(int(1.0 * uu), 10), AMBER)
        put_outlined(img, "check camera permissions in System Settings",
                     (ix, h // 2 + int(1.4 * uu)), max(int(0.75 * uu), 8),
                     AMBER)

    # ----- preset plumbing (per-mode: looks, bank, setlist — DESIGN.md §7) -----
    def _load_presets(self):
        loaded = _presets.load(self.presets_path, mode=self.mode.id,
                               builtin=getattr(self.mode, "BUILTIN", {}))
        for note in _presets.take_notes():
            # store warnings (e.g. corrupt-file backup) surface as toasts —
            # silence on a data-loss event is a bug (DESIGN.md §9)
            self.hud.toasts.hint(note, AMBER)
        return loaded

    def _reload_presets(self):
        self.all_presets = self._load_presets()
        names = list(self.all_presets.keys())
        if self.ui is not None:
            self.ui.presets = names
            self.ui.user_presets = _presets.user_names(self.presets_path,
                                                       mode=self.mode.id)
            # delete/rename keep the stored bank+setlist consistent; mirror that
            stored = _presets.bank(self.presets_path, mode=self.mode.id)
            if stored is not None:
                self.ui.bank = stored
            stored = _presets.setlist(self.presets_path, mode=self.mode.id)
            if stored is not None:
                self.ui.setlist = stored
        return names

    def _seed_bank_setlist(self):
        """Seed the UI's bank + setlist for the active mode. Stored assignments
        win; a mode that never stored a bank gets its built-ins on slots 1..9
        in order (the instrument is playable blind out of the box — DESIGN.md
        principle 6; an explicitly emptied bank stays empty). An empty setlist
        means 'all looks, load order' and tracks saves live."""
        ui = self.ui
        stored = _presets.bank(self.presets_path, mode=self.mode.id)
        if stored is not None:
            ui.bank = stored
        else:
            builtin = list(getattr(self.mode, "BUILTIN", {}))[:9]
            ui.bank = {str(i + 1): n for i, n in enumerate(builtin)}
        ui.setlist = _presets.setlist(self.presets_path, mode=self.mode.id) or []

    def _autosave_state(self):
        """Autosave: active mode + preset + bank → state.json (DESIGN.md §6.4;
        crash restart resumes the same look). The bank's persistent authority
        is presets.json — the copy here is a crash-recovery snapshot."""
        ui = self.ui
        current = (ui.preset_name
                   if ui is not None and ui.preset_idx < len(ui.presets) else None)
        _presets.save_state({"mode": self.mode.id, "preset": current,
                             "bank": {self.mode.id: dict(ui.bank)} if ui else {}},
                            path=self.state_path)

    def _capture_cfg(self):
        """Spec-derived capture (DESIGN.md §2.1/§7): walk the composed panel
        spec's save=True widgets — the single schema authority. The SIGNAL
        rack's block nests under "signal" (its Section declares store)."""
        return capture_look(self.ui, self.ui.spec)

    def _apply_pending_preset(self):
        """Spec-derived apply onto the shared UI state; engines pick the values
        up in the mode's next step() sync. apply="keep" widgets are untouched
        by look-switching; apply="reset" merges over the mode's defaults."""
        ui = self.ui
        name = ui.pending_preset
        if name in self.all_presets:
            apply_look(ui, ui.spec, self.all_presets[name], self.mode.DEFAULTS)
            self._autosave_state()
        ui.pending_preset = None

    def _assign_slot(self, name):
        """Slot-badge click (DESIGN.md §6.3): an unbanked look takes the next
        free slot (1-9); clicking an assigned look's badge clears its slot
        (the cheap, reversible correction path — §6.3 names only assignment,
        clearing is the resolved counterpart). Persists to presets.json (the
        bank's authority) and snapshots state.json."""
        ui, toasts = self.ui, self.hud.toasts
        slot = next((s for s, n in ui.bank.items() if n == name), None)
        if slot is not None:
            del ui.bank[slot]
            toasts.hint(f"slot {slot} cleared - {name}")
        else:
            free = next((str(i) for i in range(1, 10) if str(i) not in ui.bank),
                        None)
            if free is None:
                toasts.hint("bank full (1-9)")
                return
            ui.bank[free] = name
            toasts.flash(f"{free} - {name}")
        _presets.set_bank(ui.bank, path=self.presets_path, mode=self.mode.id)
        self._autosave_state()

    def _pump_preset_mailboxes(self):
        """The pending_* mailboxes a panel click posts to (shipped semantics)."""
        ui = self.ui
        if ui.pending_preset:
            self._apply_pending_preset()
        if ui.pending_slot:
            self._assign_slot(ui.pending_slot)
            ui.pending_slot = None
        if ui.pending_save:
            # amended DESIGN.md §3: auto-name (suffix same-second collisions),
            # save, then immediately open the rename box — naming is one flow
            # — and confirm with a toast, never stdout alone.
            base = "mine_%s" % time.strftime("%H%M%S")
            existing = set(ui.presets) | set(ui.user_presets)
            name, n = base, 2
            while name in existing:
                name, n = "%s_%d" % (base, n), n + 1
            _presets.save(name, self._capture_cfg(), path=self.presets_path,
                          mode=self.mode.id)
            names = self._reload_presets()
            if name in names:
                ui.preset_idx = names.index(name)
            ui.renaming = name
            ui.rename_buf = name
            self.hud.toasts.hint("saved - " + name)
            print("saved preset", name)
            ui.pending_save = False
        if ui.pending_delete:
            sel = ui.preset_name if ui.preset_idx < len(ui.presets) else None
            if _presets.delete(ui.pending_delete, path=self.presets_path,
                               mode=self.mode.id):
                names = self._reload_presets()
                ui.preset_idx = names.index(sel) if sel in names else 0
                self.hud.toasts.hint("deleted - " + ui.pending_delete)
                print("deleted preset", ui.pending_delete)
            ui.pending_delete = None
        if getattr(ui, "pending_still_path", None):
            # still-image mailbox (v1: filled by the CLI --still flag or a
            # future drop event — there is no file dialog)
            path = ui.pending_still_path
            frame = cv2.imread(path, cv2.IMREAD_COLOR)
            if frame is None:
                self.hud.toasts.hint(f"could not read still: {path[-40:]}")
            else:
                self.still = frame
                self.hud.toasts.hint(f"still loaded - {os.path.basename(path)}")
            ui.pending_still_path = None
        if ui.pending_rename:
            old, new = ui.pending_rename
            sel = ui.preset_name if ui.preset_idx < len(ui.presets) else None
            if _presets.rename(old, new, path=self.presets_path,
                               mode=self.mode.id,
                               builtin=getattr(self.mode, "BUILTIN", {})):
                names = self._reload_presets()
                target = new if sel == old else sel
                ui.preset_idx = names.index(target) if target in names else 0
                self.hud.toasts.hint("renamed - " + new)
                print("renamed preset", old, "->", new)
            else:
                self.hud.toasts.hint("rename refused - name taken or invalid")
                print("rename refused (name taken or invalid):", old, "->", new)
            ui.pending_rename = None

    # ----- host-owned per-frame sync (mirror / res / mic / recorder) -----
    def _sync_host_state(self):
        ui = self.ui
        self.mirror = ui.mirror
        nw, nh = ui.res_wh
        if (nw, nh) != self.res:
            if self.writer is not None:
                # imageio's ffmpeg writer needs a constant frame size — stop
                # the recording cleanly first (same path as 'r' off), then
                # apply the res change.
                self.writer.close()
                self.writer = None
                ui.record = False
                self.hud.toasts.hint("recording stopped - resolution changed")
                print("saved", self.rec_path)
                self.hud.toasts.hint("saved " + str(self.rec_path))
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
        if self._still_path is not None and self.still is None:
            self.still = cv2.imread(self._still_path, cv2.IMREAD_COLOR)
            if self.still is None:
                print("could not read still:", self._still_path)

        if not self.set_mode(self._boot_mode):
            raise RuntimeError("boot mode failed to start")
        mode = self.mode
        self._mode_instances[mode.id] = mode
        self.all_presets = self._load_presets()
        preset = self._boot_preset
        if preset is None:
            # crash-restart resume (DESIGN.md §6.4): state.json is the single
            # last-mode/last-preset authority (see dtouch.presets docstring)
            st = _presets.load_state(self.state_path)
            if st.get("mode") in (None, mode.id):
                preset = st.get("preset")
        if preset is None:
            preset = mode.safe_look()

        # The shared UI-state object ALWAYS exists — it is the mode's parameter
        # surface (spec capture/apply target + step()'s per-frame sync source).
        # `show`/`panel` only govern whether it is drawn and clickable.
        palettes = list(getattr(mode, "palettes", [])) or [""]
        mattes = list(getattr(mode, "mattes", [])) or [""]
        boot_palette = (getattr(getattr(mode, "pf", None), "palette", None)
                        or palettes[0])
        ui = self.ui = OverlayUI(rw, rh, list(self.all_presets.keys()),
                                 palettes, mattes,
                                 preset=preset,
                                 matte=getattr(mode, "matte_kind", mattes[0]),
                                 palette=boot_palette)
        ui.accent = mode.accent
        ui.set_spec(self.compose_spec(mode))
        ui.mirror = self.mirror
        ui.audio = self._boot_audio
        ui.video_bg = getattr(mode, "boot_video_bg", False)
        ui.video_mix = getattr(mode, "boot_video_mix", ui.video_mix)
        if preset in self.all_presets:
            # the look loaded at startup, same as a live switch (spec-derived)
            apply_look(ui, ui.spec, self.all_presets[preset], mode.DEFAULTS)
        mode.configure_ui(ui)
        ui.user_presets = _presets.user_names(self.presets_path, mode=mode.id)
        self._seed_bank_setlist()
        self._autosave_state()

        if self.show:
            # AUTOSIZE: the window is fixed at the render resolution so the OS
            # can't maximize/scale it — that scaling was tanking fps (display
            # upscaling) and breaking click mapping. To go bigger, switch the
            # 'output' resolution; the window resizes to match natively.
            cv2.namedWindow(self.WIN, cv2.WINDOW_AUTOSIZE)
            if self.panel:
                cv2.setMouseCallback(self.WIN, self._on_mouse)
            # Key routing goes through the command registry (DESIGN.md
            # principle 7) — the full perform layer, panel shown or not.
            self._wire_keys()
        else:
            _register_quit(self.reg, self.ps, self.hud.toasts)

        os.makedirs("out", exist_ok=True)

        t0 = time.time(); fps = 0.0; count = 0
        black_streak = 0
        last_frame = None    # last good camera frame, held across read failures
        camera_lost = False
        last_t = time.monotonic()
        out = None
        try:
            while True:
                if self.pending_mode is not None:
                    mode_id, self.pending_mode = self.pending_mode, None
                    self._switch_mode(mode_id)
                    mode = self.mode

                ok, frame = self._source.read()
                # Still input (DESIGN.md §2.2: modes never special-case stills
                # — the shell substitutes the loaded still for the camera when
                # an accepts_still mode's input cycle selects it)
                use_still = (getattr(mode, "accepts_still", False)
                             and getattr(ui, "input_idx", 0) == 1)
                if use_still and self.still is None:
                    ui.input_idx = 0        # snap back; v1 has no file dialog
                    self.hud.toasts.hint("no still loaded - launch with --still PATH")
                    use_still = False
                if use_still:
                    frame, camera_lost = self.still, False
                elif not ok:
                    # Camera loss: hold the last good frame and say so on the HUD
                    # (DESIGN.md §6.4); recovery is automatic when reads resume.
                    # Modes never see None (DESIGN.md §2.2).
                    if last_frame is not None:
                        frame, camera_lost = last_frame, True
                    else:
                        # No frame has EVER arrived (§6.4): never a frozen,
                        # unquittable window — show an intentional black frame
                        # with the human fix and keep pumping keys through the
                        # normal path so q/quit works.
                        if self.show:
                            waiting = np.zeros((self.res[1], self.res[0], 3),
                                               np.uint8)
                            self._draw_waiting_note(waiting)
                            self.hud.draw(waiting, self.overlay,
                                          blackout=self.ps.blackout)
                            cv2.imshow(self.WIN, waiting)
                            self._route_key(cv2.waitKey(1) & 0xFF)
                            if self.ps.quit or ui.quit:
                                break
                            if cv2.getWindowProperty(
                                    self.WIN, cv2.WND_PROP_VISIBLE) < 0:
                                break
                        if self.max_frames is not None:
                            break
                        continue
                else:
                    last_frame, camera_lost = frame, False
                black_streak = (0 if use_still else
                                black_streak + 1 if float(frame.mean()) < 3.0 else 0)

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
                if ui.glitch:
                    if self.cb is None:
                        self.cb = CircuitBent(seed=self.seed)
                    cb = self.cb
                    cb.chroma_shift = ui.chroma
                    cb.scan_drift = ui.drift
                    cb.bit_crush = int(ui.crush)
                    cb.scanlines = ui.scanlines
                    # suppression rule (DESIGN.md §2.4): a mode that claims
                    # "dither" owns dithering — the rack runs minus its dither
                    claimed = frozenset(getattr(mode, "claims", ()))
                    cb.dither_mode = (None if "dither" in claimed
                                      or ui.dither_name == "off" else ui.dither_name)
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
                    # HUD/panel/menu draw AFTER the recorder write above —
                    # recordings never contain HUD, panel, or menu (the shipped
                    # invariant, kept; the boot card is the one deliberate
                    # recorded UI frame, DESIGN.md §3).
                    rw, rh = self.res
                    status = mode.status_line(self.cam_name)
                    dbg = f"{fps:4.1f}fps  {1000.0 / fps if fps > 0 else 0.0:5.1f}ms  {rw}x{rh}"
                    if self.menu.open:
                        # home menu (DESIGN.md §3): live camera through 1-bit
                        # blue noise + 65% scrim + mode cards; the running mode
                        # keeps stepping untouched behind it
                        ui._hot = []
                        self.menu.rects = draw_menu(bgr, frame, self.menu.cards,
                                                    self.menu.sel)
                        # HIDDEN-state HUD = toasts + blackout tick only
                        self.hud.draw(bgr, OverlayState.HIDDEN,
                                      blackout=self.ps.blackout)
                    else:
                        if self.panel and self.overlay is OverlayState.PANEL:
                            ui.draw(bgr, {"status": ""})
                        else:
                            ui._hot = []   # panel hidden: stale hit-rects must not eat clicks
                        self.hud.draw(bgr, self.overlay, status=status,
                                      debug_status=dbg,
                                      recording=(self.writer is not None),
                                      blackout=self.ps.blackout,
                                      camera_lost=(camera_lost or black_streak > 15))
                    if self.ps.help_open:
                        draw_help(bgr, self.help_rows)   # works in every overlay state
                    cv2.imshow(self.WIN, bgr)
                    key = cv2.waitKey(1) & 0xFF   # pump GUI + mouse
                    # menu → rename box → perform layer (see _route_key)
                    self._route_key(key)
                    if self.ps.quit or ui.quit:
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
