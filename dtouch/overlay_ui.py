"""In-frame control panel — a generic walker over a panel spec, one window.

The widget primitives (rows, sliders, section headers, cycles, rename box,
manage buttons, tooltip, scrollbar) live in dtouch.imgui; the controls
themselves are declared as data in a panel spec (dtouch.panelspec, DESIGN.md
§2.3). OverlayUI walks the spec and draws it with the toolkit — today's spec
is `build_particles_spec`, reproducing the shipped Particles panel exactly
(pinned by tests/test_goldens.py); a future `Mode.panel_spec()` returns pieces
of the same thing.

Immediate-mode GUI drawn with cv2 primitives (Tk won't paint when launched
headless on macOS; cv2 windows do). Everything is a drawn, boxed,
hover-highlighting control with click feedback, and aligned hit-testing via
setMouseCallback. Because it's just pixels, it can be rendered headless and
inspected.

Every pixel dimension is authored at a 1080p baseline and multiplied by a scale
factor derived from the output resolution (self._s, set each draw from the
frame height). The factor floors at 1.0 — so 720p/1080p render exactly as
authored — and grows to 2x at 4K, keeping the panel the same fraction of the
frame instead of shrinking to a tiny fixed pixel box (issue #3).
"""
from __future__ import annotations

import cv2
import numpy as np

from . import imgui
from .imgui import (PANEL, BTN, HOVER, INK, DIM, ACC, TRACK, HANDLE, RED, DARK,
                    in_rect as _in)
from .panelspec import (Slider, Toggle, Cycle, Action, Param, Readout,
                        PresetList, Section)

BASE_H = 1080   # resolution the layout literals are authored against

# Post-FX dither modes, in cycle order. "off" is a deliberate third choice rather than a
# disabled state — the glitch chain is still worth having without the dither.
DITHERS = ["bayer", "blue", "fs", "riemersma", "off"]

# SIGNAL rack dither-quality bias options (DESIGN.md §4.1): the ordered
# dithers' rounding direction — auto flips on dark frames, light/dark force it.
SIGNAL_BIASES = ["auto", "light", "dark"]
SIGNAL_BIAS_INVERT = {"auto": "auto", "light": False, "dark": True}

RES_OPTIONS = [("720p", 1280, 720), ("1080p", 1920, 1080),
               ("1440p", 2560, 1440), ("4K", 3840, 2160)]

_RANGES = {
    "fade": (0.50, 0.985), "exposure": (0.3, 4.0), "spark": (0.0, 1.0),
    "curl": (0.0, 1.0), "dot": (0.004, 0.020), "sens": (0.0, 3.0),
    "count": (0.1, 1.0), "damp": (0.82, 0.985), "pull": (8.0, 40.0), "reseed": (0.005, 0.15),
    "video_mix": (0.05, 1.0),
    # MOTION (boids gains) and SIGNAL (circuit-bent) — see dtouch.flock / dtouch.circuit_bent
    "cohere": (0.0, 1.0), "align": (0.0, 1.0), "separate": (0.0, 1.0),
    "chroma": (0.0, 60.0), "drift": (0.0, 40.0), "crush": (0.0, 8.0),
}

_SLIDERS = [
    ("Trails", "fade", "How long particle motion trails linger before fading."),
    ("Glow", "exposure", "Overall brightness and bloom of the particles."),
    ("Spark", "spark", "How much fast motion scatters particles into bursts of sparks."),
    ("Flow", "curl", "Swirling, turbulent drift of the particles."),
    ("Size", "dot", "Size of each individual particle dot."),
    ("Count", "count", "How many particles (density of the cloud)."),
    ("Glide", "damp", "How long particles keep gliding. High = they overshoot into sharp "
                      "contour lines (the sigil look); low = they stop quickly."),
    ("Pull", "pull", "How sharply particles are pulled to the shape. Low = tighter, brighter "
                     "edge bands."),
    ("Reseed", "reseed", "How fast particles respawn. Low = they persist and trace flowing "
                         "lines; high = a constantly refreshing spray."),
]


# UI attr -> serialized preset key, where they differ (the v1 names — so a
# migrated look and a fresh capture are byte-compatible; DESIGN.md §7).
_SAVE_KEYS = {"curl": "curl_amp", "dot": "base_size", "reseed": "reseed_frac",
              "pull": "pull_falloff"}


def _slider_spec(label, attr, tip, **kw):
    lo, hi = _RANGES[attr]
    kw.setdefault("save_key", _SAVE_KEYS.get(attr))
    return Slider(label, attr, lo, hi, tip=tip, **kw)


def build_particles_sections(palettes, mattes):
    """The Particles mode's own panel sections (DESIGN.md §4.1) — what
    ParticlesMode.panel_spec() returns. The shell appends the SIGNAL rack and
    the global rows below these (DESIGN.md §2.4)."""
    look_sliders = [_slider_spec(label, attr, tip) for label, attr, tip in _SLIDERS]
    return [
        Section("TEMPLATES", [PresetList()]),
        Section("SOURCE", [
            Cycle("matte", "matte_idx", list(mattes), save_key="matte",
                  status="matte {}"),
            Cycle("output", "res_idx", [n for n, _, _ in RES_OPTIONS],
                  key="res", save=False),
            Toggle("Video bg", "video_bg"),
            _slider_spec("Vid mix", "video_mix",
                         "How visible the raw camera footage is behind the particles.",
                         apply="keep"),
        ]),
        Section("LOOK", [
            Cycle("color", "palette_idx", list(palettes), gap=4,
                  save_key="palette", status="{}"),
        ] + look_sliders + [
            # captured-but-undrawn: saved since v1, no panel control yet
            Param("attract_speed"),
        ]),
        # MOTION — boids steering (dtouch.flock). The sliders stay visible while off so the
        # section reads as a thing you can turn on, not a thing that appears from nowhere.
        Section("MOTION", key_hint="F", widgets=[
            Toggle("Flock", "flock"),
            _slider_spec("Cohere", "cohere",
                         "Steer toward the local centre. Pulls the cloud into shoals."),
            _slider_spec("Align", "align",
                         "Match neighbours' direction. This is what makes it move as one."),
            _slider_spec("Separate", "separate",
                         "Push apart when crowded. Stops the shoal collapsing to a dot."),
        ], open=False),
    ]


def build_signal_section():
    """The shell's SIGNAL rack (DESIGN.md §2.4) — circuit-bent post-processing
    (dtouch.circuit_bent) applied to every mode's output after render + video
    composite, before recorder and panel (so the panel stays readable). Its
    state serializes under "signal" inside each mode's looks."""
    return Section("SIGNAL", key_hint="G", widgets=[
            Toggle("Glitch", "glitch"),
            Cycle("dither", "dither_idx", list(DITHERS), save_key="dither"),
            # dither-quality controls (DESIGN.md §2.4/§4.1: the audit's
            # quality controls grow the rack; a mode that claims them —
            # Dither Girl owns ALL dither quality — hides them)
            Slider("Bits", "sig_bits", 1.0, 4.0, fmt=".0f", save_key="bits",
                   tip="Dither bit depth. 1 = pure two-tone; higher keeps "
                       "more shades."),
            Toggle("Gamma", "sig_gamma", save_key="gamma",
                   tip="Dither in linear light so mid-tones keep their "
                       "perceived brightness. Off = the crushed retro look."),
            Cycle("bias", "sig_bias_idx", list(SIGNAL_BIASES), save_key="bias"),
            _slider_spec("Chroma", "chroma",
                         "Colour bleed: red and blue drift apart, slowly."),
            _slider_spec("Drift", "drift",
                         "Scan-line sync loss. Rows slip sideways; sometimes a whole band tears."),
            _slider_spec("Crush", "crush",
                         "Hard bit-depth reduction. 0 = off."),
            Toggle("Scanlines", "scanlines", on_text="on"),
        ], open=False, gap=6, store="signal")


def build_global_rows():
    """The shell's global rows (below every mode's sections) — the §4.1
    sketch: Sound react (A) - Sens - Record (R) - Mirror - Menu (M) - Quit.
    Key hints live in the labels; the Menu row is an Action posting the
    `menu.open` command through the walker's pending_commands mailbox."""
    return [
        Toggle("Sound react (A)", "audio"),
        _slider_spec("Sens", "sens", "How strongly sound drives the visuals.",
                     apply="keep", gap=4),
        Toggle("Record (R)", "record", save=False,
               label_fn=lambda v: "Stop recording (R)" if v else "Record (R)"),
        Toggle("Mirror", "mirror", on_text="on", save=False, gap=4),
        Action("Menu (M)", "menu.open"),
        Action("Quit", "quit"),
    ]


def build_particles_spec(presets, palettes, mattes):
    """Today's Particles panel as data — the exact shipped section order,
    names, labels, and vertical rhythm (goldens hold). Composed the same way
    the shell composes any mode's panel: mode sections + SIGNAL rack + global
    rows (DESIGN.md §2.4). `presets` is unused (PresetList reads the walker's
    live list) but kept for signature stability."""
    return (build_particles_sections(palettes, mattes)
            + [build_signal_section()] + build_global_rows())


class OverlayUI:
    def __init__(self, w, h, presets, palettes, mattes,
                 preset="abstract", matte="auto", palette="ice"):
        self.w, self.h = w, h
        self.presets, self.palettes, self.mattes = presets, palettes, mattes
        self.preset_idx = presets.index(preset) if preset in presets else 0
        self.matte_idx = mattes.index(matte) if matte in mattes else 0
        self.palette_idx = palettes.index(palette) if palette in palettes else 0
        self.fade, self.exposure, self.spark, self.curl, self.dot = 0.90, 1.4, 0.35, 0.5, 0.011
        self.damp, self.pull, self.reseed = 0.90, 22.0, 0.06   # motion physics (sigil knobs)
        self.count = 0.75   # fraction of the allocated particles to render
        self.mirror, self.audio, self.sens, self.record = True, False, 1.0, False
        self.video_bg, self.video_mix = False, 0.5   # raw footage behind the particles
        # MOTION — boids steering on the particle cloud (dtouch.flock). Off by default:
        # every gain at 0 short-circuits the solver, so the existing look is untouched.
        self.flock = False
        self.cohere, self.align, self.separate = 0.35, 0.45, 0.30
        # SIGNAL — circuit-bent post-processing on the rendered frame (dtouch.circuit_bent).
        # `dither_idx` indexes DITHERS; "off" is a real choice, not a disabled state.
        self.glitch = False
        self.chroma, self.drift, self.crush = 10.0, 8.0, 0.0
        self.dither_idx = 0
        # dither-quality controls (DESIGN.md §4.1): defaults match CircuitBent's
        # shipped behaviour (3-bit, gamma-correct, auto bias)
        self.sig_bits, self.sig_gamma, self.sig_bias_idx = 3.0, True, 0
        self.scanlines = True
        self.attract_speed = 4.5   # captured Param — no panel control yet
        self.res_options = list(RES_OPTIONS)
        self.res_idx = next((i for i, (_, rw, rh) in enumerate(self.res_options)
                             if (rw, rh) == (w, h)), None)
        if self.res_idx is None:
            # non-standard boot res (tests, custom rigs): a real option, so the
            # shell's res sync doesn't silently "correct" the output to 1080p
            self.res_options.insert(0, (f"{w}x{h}", w, h))
            self.res_idx = 0
        self.open = True
        self.quit = False
        self.scroll = 0          # panel scroll offset (px) — content taller than the window
        self._content_h = 0      # measured column height from the last draw
        self._scroll_drag = None
        self._thumb_drag = None  # grabbed the scrollbar thumb (not the content)
        self._tooltip = None
        self.pending_preset = None
        self.pending_save = False
        # per-mode perform state (DESIGN.md §7): explicit slot assignments and
        # setlist order for the ACTIVE mode, seeded by the shell. An empty
        # setlist means "all looks, load order".
        self.bank = {}               # {"1": name, ...} — digits 1-9 recall these
        self.setlist = []            # [ / ] walk this order
        self.pending_slot = None     # preset name whose slot badge was clicked
        self.pending_commands = []   # Action commands with no dedicated mailbox
        self.user_presets = set()    # names that can be renamed/deleted (saved looks)
        self.pending_delete = None   # name confirmed for deletion (live loop applies)
        self.pending_rename = None   # (old, new) committed via Enter (live loop applies)
        self.renaming = None         # name currently being renamed (typing mode)
        self.rename_buf = ""
        self._del_armed = None       # first x-click arms; second confirms
        self._blink = 0
        self.panel_w = 290           # base (1080p) panel width; scaled by self._s when drawn
        self._s = 1.0                # current UI scale (set each draw from the frame height)
        self._panel_px = 290         # actual drawn panel width in frame px (panel_w * self._s)
        self.mouse = (-1, -1)
        self._hot = []
        self._drag = None
        self._flash = 0          # frames of click feedback left
        self._flash_key = None   # (kind, payload-identity) — re-located each draw
        self._scrim = None       # cached PANEL-colored blend buffer (perf)
        # per-mode accent (DESIGN.md §5): exactly one accent on screen, owned by
        # the active mode. Default ACC green = the shipped Particles chrome, so
        # a bare OverlayUI (goldens, tests) renders identical pixels.
        self.accent = ACC
        # panel title (DESIGN.md §4.1/§4.2 sketch: 'dtouch - MODE TITLE',
        # ASCII hyphen). The shell sets it per mode; the bare default matches
        # the goldens' Particles panel.
        self.panel_title = "dtouch - PARTICLES"
        # still-image mailbox (DESIGN.md §2.1: the shell owns still sources) —
        # a path posted here is loaded by the shell's mailbox pump.
        self.pending_still_path = None
        self._gui = imgui.Gui()
        self.set_spec(build_particles_spec(presets, palettes, mattes))

    def set_spec(self, spec):
        """Bind a panel spec: build the sections state (collapsible headers keep
        their defaults from the spec — the two new families start closed so the
        panel opens looking like it always did, and the default column still
        fits a 1080p window; tests/test_overlay_scroll.py) and the activation
        maps the generic _activate routes through."""
        self.spec = spec
        self.sections = {s.title: s.open for s in spec if isinstance(s, Section)}
        # keyboard param-nudge selection (DESIGN.md §6.2): an index into the
        # spec-order list of nudgeable widgets (Sliders + Cycles). Rebinding
        # the spec (mode switch) resets it to the first control.
        self.nudge_idx = 0
        self._toggles = {}    # hit key -> Toggle
        self._cycles = {}     # hit key -> Cycle
        self._sliders = {}    # attr -> Slider
        for wdg in self.iter_widgets():
            if isinstance(wdg, Toggle):
                self._toggles[wdg.attr] = wdg
            elif isinstance(wdg, Cycle):
                if wdg.attr == "res_idx":
                    # the instance may carry a custom boot resolution option
                    wdg.options = [n for n, _, _ in self.res_options]
                self._cycles[wdg.hit_key] = wdg
            elif isinstance(wdg, Slider):
                self._sliders[wdg.attr] = wdg

    def iter_widgets(self):
        """Every widget in the spec, sections flattened (spec order)."""
        for item in self.spec:
            if isinstance(item, Section):
                yield from item.widgets
            else:
                yield item

    @property
    def typing(self): return self.renaming is not None
    @property
    def res_name(self): return self.res_options[self.res_idx][0]
    @property
    def res_wh(self): return self.res_options[self.res_idx][1:3]
    @property
    def matte_name(self): return self.mattes[self.matte_idx]
    @property
    def palette_name(self): return self.palettes[self.palette_idx]

    @property
    def dither_name(self): return DITHERS[self.dither_idx]
    @property
    def preset_name(self): return self.presets[self.preset_idx]

    def sync_from(self, pf, glow, matte):
        """Reflect engine state into the sliders (after a preset is applied)."""
        self.fade, self.exposure = glow.fade, glow.exposure
        self.spark, self.curl, self.dot = pf.spark, pf.curl_amp, pf.base_size
        self.damp, self.pull, self.reseed = pf.damp, pf.pull_falloff, pf.reseed_frac
        self.attract_speed = pf.attract_speed
        if matte in self.mattes: self.matte_idx = self.mattes.index(matte)
        if pf.palette in self.palettes: self.palette_idx = self.palettes.index(pf.palette)

    # ----- scaling -----
    def _S(self, n):
        """Scale a baseline pixel value by the current resolution factor (int for cv2)."""
        return self._gui.S(n)

    # ----- keyboard (rename typing) -----
    def on_key(self, key):
        """Feed a cv2.waitKey code. Returns True if consumed (a rename box is open),
        so the caller knows not to treat 'q' as quit while the user is typing.

        Contract (DESIGN.md §6.2): rename-typing consumes EVERY key — while
        renaming, no global keys fire. Esc is consumed too, and cancels the
        rename only (it must not also step the overlay state or anything else);
        `Esc` then `0` is the two-press panic escape hatch."""
        if self.renaming is None:
            return False
        if key in (13, 10):              # enter — commit
            new = self.rename_buf.strip()
            if new and new != self.renaming:
                self.pending_rename = (self.renaming, new)
            self.renaming = None
        elif key == 27:                  # esc — cancel
            self.renaming = None
        elif key in (8, 127):            # backspace / delete
            self.rename_buf = self.rename_buf[:-1]
        elif 32 <= key <= 126 and len(self.rename_buf) < 22:
            self.rename_buf += chr(key)
        return True

    # ----- layout -----
    def draw(self, frame, info):
        g = self._gui
        self._tooltip = None
        h, w = frame.shape[:2]
        # scale the whole panel by the output resolution, floored at the 1080p baseline so
        # 720p/1080p are unchanged and 4K renders at 2x (same fraction of the frame).
        self._s = max(1.0, h / BASE_H)
        self._hot = g.begin(self._s, self.mouse, self.accent)
        pw = g.S(self.panel_w)
        self._panel_px = pw
        if not self.open:
            r = (w - g.S(48), g.S(12), w - g.S(12), g.S(42))
            g.box(frame, r, HOVER if _in(r, self.mouse) else PANEL)
            for yy in (g.S(21), g.S(27), g.S(33)):
                cv2.line(frame, (w - g.S(40), yy), (w - g.S(20), yy), INK,
                         max(1, g.S(2)), cv2.LINE_AA)
            self._hot.append((r, "collapse", None))
            self._draw_status(frame, info)
            return frame

        px = w - pw
        # panel scrim restricted to the panel ROI (perf: the full-frame
        # copy+addWeighted was an identity op outside the panel — pixels
        # elsewhere are untouched, so the result is unchanged). The blend
        # runs on a contiguous copy of the ROI (cv2 on a column-sliced view
        # is slower than copying), against a cached PANEL-colored buffer.
        sx = max(px, 0)                # a frame narrower than the panel clamps
        roi = np.ascontiguousarray(frame[:, sx:])
        if self._scrim is None or self._scrim.shape != roi.shape:
            self._scrim = np.empty_like(roi)
            self._scrim[:] = PANEL
        cv2.addWeighted(self._scrim, 0.86, roi, 0.14, 0, dst=roi)
        frame[:, sx:] = roi
        # the column can be taller than the window (e.g. 720p) — scroll, clamped so it's
        # a no-op when everything fits. content height comes from the previous draw.
        self.scroll = imgui.clamp_scroll(self.scroll, self._content_h, h)
        x, cw, y = px + g.S(16), pw - g.S(32), g.S(30) - self.scroll
        g.text(frame, self.panel_title, x, y, self.accent, 0.62, 2)
        y += g.S(16)
        self._blink += 1

        # ----- the generic walk: sections, then the shell's global rows -----
        for item in self.spec:
            if isinstance(item, Section):
                y, sec_open = g.section(frame, item.title,
                                        self.sections.get(item.title, True), x, y, cw,
                                        key_hint=item.key_hint)
                if sec_open:
                    for wdg in item.widgets:
                        y = self._draw_widget(frame, wdg, x, y, cw, px)
                y += g.S(item.gap)
            else:
                y = self._draw_widget(frame, item, x, y, cw, px)
        self._content_h = y + self.scroll + g.S(8)   # column bottom incl. margin, unscrolled

        # collapse button drawn last so it stays fixed and clickable above scrolled content
        cr = (w - g.S(40), g.S(12), w - g.S(12), g.S(36))
        g.box(frame, cr, HOVER if _in(cr, self.mouse) else BTN)
        g.text(frame, ">", w - g.S(33), g.S(30), INK, 0.55, 2)
        self._hot.append((cr, "collapse", None))
        g.scrollbar(frame, px, h, self._content_h, self.scroll,
                    step=g.S(imgui.SCROLL_STEP))

        # click feedback: flash the last-clicked control — border-only (a fill
        # would hide the row's own content, e.g. the armed delete's 'sure?'),
        # re-located against THIS draw's hit rects so a res change, section
        # reflow, or scroll never leaves the flash on a stale rectangle. A
        # control that left the screen simply loses its flash.
        if self._flash > 0:
            rect = next((r for r, k, p in self._hot
                         if self._flash_id(k, p) == self._flash_key), None)
            if rect is not None:
                x0, y0, x1, y1 = rect
                cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 255, 255),
                              max(1, g.S(1)))   # crisp non-AA flash border
            self._flash -= 1
        self._tooltip = g.tooltip
        g.draw_tooltip(frame, self.h)
        self._draw_status(frame, info)
        return frame

    def _draw_widget(self, frame, wdg, x, y, cw, px):
        """Draw one spec widget at y; return the next y (the shipped rhythm:
        rows advance S(28+gap) in one rounding, sliders/cycles advance their own
        height then add S(gap) — exactly the shipped two-step literals)."""
        g = self._gui
        if isinstance(wdg, Slider):
            val = getattr(self, wdg.attr)
            y = g.slider(frame, wdg.label, wdg.attr, val, wdg.lo, wdg.hi, x, y, cw,
                         info=wdg.tip or None, fmt=wdg.fmt)
            return y + g.S(wdg.gap) if wdg.gap else y
        if isinstance(wdg, Toggle):
            val = bool(getattr(self, wdg.attr))
            label = (wdg.label_fn(val) if wdg.label_fn else
                     f"{wdg.label}: {wdg.on_text if val else wdg.off_text}")
            g.row(frame, label, wdg.attr, x, y, cw, active=val)
            return y + g.S(28 + wdg.gap)
        if isinstance(wdg, Cycle):
            idx = getattr(self, wdg.attr) % len(wdg.options)
            y = g.cycle(frame, wdg.label, wdg.options[idx], wdg.hit_key, x, y, cw)
            return y + g.S(wdg.gap) if wdg.gap else y
        if isinstance(wdg, Action):
            g.row(frame, wdg.label, wdg.command, x, y, cw)
            return y + g.S(28 + wdg.gap)
        if isinstance(wdg, PresetList):
            return self._draw_preset_list(frame, x, y, cw, px)
        if isinstance(wdg, Readout):
            return wdg.render(frame, g, x, y, cw)
        if isinstance(wdg, Param):
            return y                # captured, never drawn (no row, no pixels)
        raise TypeError(f"unknown panel-spec widget {wdg!r}")

    def _draw_preset_list(self, frame, x, y, cw, px):
        """TEMPLATES rows (slot badges, rename box, hover manage buttons) +
        Save current look. Badge hits go to the FRONT of `hot` (like the manage
        buttons) so they win over the full-row rect they sit on."""
        g = self._gui
        slots = {n: s for s, n in (self.bank or {}).items()}
        for i, name in enumerate(self.presets):
            if name == self.renaming:
                g.rename_box(frame, self.rename_buf, self._blink, x, y, cw, px)
                y += g.S(28)
                continue
            r = g.row(frame, name, "preset", x, y, cw,
                      active=(i == self.preset_idx), payload=i)
            br = g.slot_badge(frame, slots.get(name), x, y, cw)
            self._hot.insert(0, (br, "slot", name))
            if name in self.user_presets and (_in(r, self.mouse) or name == self._del_armed):
                g.manage_buttons(frame, name, x, y, cw, self._del_armed == name)
            y += g.S(28)
        g.row(frame, "+ Save current look", "save", x, y, cw)
        return y + g.S(30)

    def _draw_status(self, frame, info):
        g = self._gui
        g.text(frame, info.get("status", ""), g.S(12), g.S(24), self.accent, 0.5)
        if info.get("black"):
            h, w = frame.shape[:2]
            g.text(frame, "CAMERA IS BLACK - disable iPhone Continuity Camera",
                   g.S(12), h // 2, RED, 0.8, 2)
            g.text(frame, "iPhone: Settings > General > AirPlay & Handoff > Continuity Camera > Off",
                   g.S(12), h // 2 + g.S(28), (140, 140, 240), 0.5)
        if self.record:
            h, w = frame.shape[:2]
            off = (self._panel_px + g.S(22)) if self.open else g.S(70)
            cv2.circle(frame, (w - off, g.S(24)), g.S(7), RED, -1)

    # ----- mouse -----
    @staticmethod
    def _flash_id(kind, payload):
        """Stable identity for the click-flash relocation: a slider's payload
        carries track geometry that changes with res/reflow, so key on its
        attr; everything else's payload is already stable."""
        if kind == "slider":
            return (kind, payload[0])
        return (kind, payload)

    def on_mouse(self, event, x, y, flags, param=None):
        if event in (cv2.EVENT_MOUSEWHEEL, cv2.EVENT_MOUSEHWHEEL):
            # NOT a cursor position: on the Cocoa backend (x, y) IS the wheel
            # delta, so assigning it to self.mouse teleports every hover
            # highlight to the top-left corner mid-scroll. Decode and return.
            if event == cv2.EVENT_MOUSEWHEEL and self.open:
                self.scroll = imgui.wheel_scroll(
                    self.scroll, self._S(imgui.SCROLL_STEP),
                    imgui.wheel_delta(x, y, flags))
            return
        self.mouse = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            # Fixed chrome first: the collapse button floats ABOVE scrolled
            # content, so a row that scrolled underneath it (including a
            # hovered preset's delete button, which front-inserts its hit)
            # must never steal its click — that click-theft could arm and
            # confirm a preset delete in two presses (data loss).
            for rect, kind, payload in self._hot:
                if kind == "collapse" and _in(rect, (x, y)):
                    self._flash_key, self._flash = self._flash_id(kind, payload), 4
                    self._activate(kind, payload, x, y)
                    return
            for rect, kind, payload in self._hot:
                if kind == "collapse":
                    continue
                if _in(rect, (x, y)):
                    self._flash_key, self._flash = self._flash_id(kind, payload), 4
                    self._activate(kind, payload, x, y)
                    return
            if self.open and x >= self.w - self._panel_px:
                self._scroll_drag = (y, self.scroll)   # empty panel area: drag to scroll
        elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):
            if self._drag:
                attr, x0, x1, lo, hi = self._drag
                t = min(max((x - x0) / max(x1 - x0, 1), 0.0), 1.0)
                setattr(self, attr, lo + t * (hi - lo))
            elif self._thumb_drag:
                # the THUMB follows the finger (imgui's convention block)
                y0, s0, travel, span = self._thumb_drag
                self.scroll = imgui.thumb_scroll(s0, y0, y, travel, span)
            elif self._scroll_drag:
                # the CONTENT follows the finger (imgui's convention block)
                y0, s0 = self._scroll_drag
                self.scroll = imgui.drag_scroll(s0, y0, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self._drag = None
            self._scroll_drag = None
            self._thumb_drag = None

    def _activate(self, kind, payload, x, y=0):
        """Generic activation: toggles flip their attr, cycles rotate through
        their options, sliders set their attr from the track position, actions
        post to their mailbox. Routing comes from the spec (set_spec's maps)."""
        if kind != "del" and self._del_armed:
            self._del_armed = None       # any other click disarms a pending delete
        if kind == "scroll":
            # a scrollbar arrow: payload is the signed step (down = positive =
            # show me what is below — imgui's convention block). draw() clamps.
            self.scroll += payload
            return
        if kind == "scrolltrack":
            ty0, track_h, thumb_h, span, page = payload
            travel = track_h - thumb_h
            top = ty0 + int(travel * (min(max(self.scroll, 0), span) / span))
            if top <= y <= top + thumb_h:
                self._thumb_drag = (y, self.scroll, travel, span)
            else:
                self.scroll += page if y > top else -page
            return
        if kind == "del":
            self._del_armed, confirmed = imgui.arm_delete(self._del_armed, payload)
            if confirmed:
                self.pending_delete = confirmed
            return
        if kind == "ren":
            self.renaming = payload
            self.rename_buf = payload    # prefill with the current name
            return
        if kind == "slot":
            self.pending_slot = payload  # the shell assigns/clears + persists
            return
        if kind == "collapse":
            self.open = not self.open
        elif kind == "preset":
            self.preset_idx = payload
            self.pending_preset = self.presets[payload]
        elif kind == "cycle":
            key, d = payload
            wdg = self._cycles[key]
            setattr(self, wdg.attr, (getattr(self, wdg.attr) + d) % len(wdg.options))
        elif kind == "slider":
            attr, x0, x1, lo, hi = payload
            self._drag = payload
            t = min(max((x - x0) / max(x1 - x0, 1), 0.0), 1.0)
            setattr(self, attr, lo + t * (hi - lo))
        elif kind == "section":
            self.sections[payload] = not self.sections.get(payload, True)
        elif kind in self._toggles:
            attr = self._toggles[kind].attr
            setattr(self, attr, not getattr(self, attr))
        elif kind == "save":
            self.pending_save = True
        elif kind == "quit":
            self.quit = True
        else:
            self.pending_commands.append(kind)   # Action with no dedicated mailbox
