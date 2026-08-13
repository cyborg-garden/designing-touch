"""In-frame control panel — a collapsible sidebar drawn onto the render, one window.

Immediate-mode GUI drawn with cv2 primitives (Tk won't paint when launched headless on macOS;
cv2 windows do). Everything is a drawn, boxed, hover-highlighting control with click feedback,
and aligned hit-testing via setMouseCallback. Because it's just pixels, it can be rendered
headless and inspected. ASCII glyphs only — cv2's font can't render •/■/≡/— (they show as ???).

Every pixel dimension is authored at a 1080p baseline and multiplied by a scale factor derived
from the output resolution (self._s, set each draw from the frame height). The factor floors at
1.0 — so 720p/1080p render exactly as authored — and grows to 2x at 4K, keeping the panel the
same fraction of the frame instead of shrinking to a tiny fixed pixel box (issue #3).
"""
from __future__ import annotations

import cv2

PANEL = (34, 32, 30)
BTN = (54, 52, 50)
HOVER = (82, 86, 82)
INK = (215, 222, 218)
DIM = (140, 150, 145)
ACC = (120, 215, 140)
TRACK = (70, 74, 72)
HANDLE = (160, 230, 175)
RED = (70, 70, 235)
DARK = (20, 22, 20)

BASE_H = 1080   # resolution the layout literals are authored against

# Post-FX dither modes, in cycle order. "off" is a deliberate third choice rather than a
# disabled state — the glitch chain is still worth having without the dither.
DITHERS = ["bayer", "blue", "fs", "riemersma", "off"]


def _in(rect, p):
    x0, y0, x1, y1 = rect
    return x0 <= p[0] <= x1 and y0 <= p[1] <= y1


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
        self.scanlines = True
        self.res_options = [("720p", 1280, 720), ("1080p", 1920, 1080),
                            ("1440p", 2560, 1440), ("4K", 3840, 2160)]
        self.res_idx = next((i for i, (_, rw, rh) in enumerate(self.res_options)
                             if (rw, rh) == (w, h)), 1)
        self.open = True
        # Collapsible sections. The panel grew past the 1080p window when MOTION and SIGNAL
        # were added, breaking the "everything reachable without scrolling at 1080p" contract
        # (tests/test_overlay_scroll.py). Collapsing is the fix that keeps scaling: each new
        # family of effects costs one header row until you open it. The two new sections start
        # closed so the panel opens looking like it always did.
        self.sections = {"TEMPLATES": True, "SOURCE": True, "LOOK": True,
                         "MOTION": False, "SIGNAL": False}
        self.quit = False
        self.scroll = 0          # panel scroll offset (px) — content taller than the window
        self._content_h = 0      # measured column height from the last draw
        self._scroll_drag = None
        self._tooltip = None
        self.pending_preset = None
        self.pending_save = False
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
        self._flash = 0
        self._flash_rect = None

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
        if matte in self.mattes: self.matte_idx = self.mattes.index(matte)
        if pf.palette in self.palettes: self.palette_idx = self.palettes.index(pf.palette)

    # ----- scaling -----
    def _S(self, n):
        """Scale a baseline pixel value by the current resolution factor (int for cv2)."""
        return int(round(n * self._s))

    # ----- primitives -----
    def _text(self, img, s, x, y, color=INK, scale=0.46, thick=1):
        cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale * self._s, color,
                    max(1, int(round(thick * self._s))), cv2.LINE_AA)

    def _box(self, img, rect, fill, border=TRACK):
        x0, y0, x1, y1 = rect
        cv2.rectangle(img, (x0, y0), (x1, y1), fill, -1)
        cv2.rectangle(img, (x0, y0), (x1, y1), border, max(1, self._S(1)), cv2.LINE_AA)

    def _row(self, img, label, key, x, y, w, h=None, active=False, payload=None):
        if h is None:
            h = self._S(24)
        rect = (x, y, x + w, y + h)
        hovered = _in(rect, self.mouse)
        fill = ACC if active else (HOVER if hovered else BTN)
        self._box(img, rect, fill)
        self._text(img, label, x + self._S(10), y + h - self._S(8), DARK if active else INK, 0.46)
        self._hot.append(((x, y, x + w, y + h), key, payload))
        return rect

    def _slider(self, img, label, attr, x, y, w, info=None):
        lo, hi = _RANGES[attr]
        val = getattr(self, attr)
        rect = (x, y, x + w, y + self._S(26))
        hovered = _in(rect, self.mouse)
        self._box(img, rect, HOVER if hovered else BTN)
        # info badge
        ic = (x + self._S(14), y + self._S(13))
        irect = (x + self._S(4), y + self._S(3), x + self._S(24), y + self._S(23))
        cv2.circle(img, ic, self._S(8), (90, 110, 150), -1, cv2.LINE_AA)
        self._text(img, "i", ic[0] - self._S(2), ic[1] + self._S(5), (235, 240, 245), 0.42, 1)
        if info and _in(irect, self.mouse):
            self._tooltip = (info, x, y)
        self._text(img, label, x + self._S(30), y + self._S(17), DIM, 0.42)
        tx0, tx1 = x + self._S(96), x + w - self._S(44)
        cv2.line(img, (tx0, y + self._S(13)), (tx1, y + self._S(13)), TRACK, max(1, self._S(3)), cv2.LINE_AA)
        hx = int(tx0 + (val - lo) / (hi - lo) * (tx1 - tx0))
        cv2.circle(img, (hx, y + self._S(13)), self._S(6), HANDLE, -1, cv2.LINE_AA)
        self._text(img, f"{val:.2f}", x + w - self._S(38), y + self._S(17), INK, 0.42)
        self._hot.append(((tx0 - self._S(8), y, tx1 + self._S(8), y + self._S(26)),
                          "slider", (attr, tx0, tx1, lo, hi)))
        return y + self._S(30)

    def _section(self, img, title, x, y, w):
        """Clickable section header. Returns (next_y, is_open).

        ASCII markers only: cv2's Hershey font renders no glyph for the usual disclosure
        triangles, so they come out as '???' (see the module docstring).
        """
        is_open = self.sections.get(title, True)
        rect = (x - self._S(4), y - self._S(4), x + w, y + self._S(14))
        if _in(rect, self.mouse):
            self._box(img, rect, HOVER)
        self._text(img, ("- " if is_open else "+ ") + title, x, y + self._S(8),
                   INK if is_open else DIM, 0.4)
        self._hot.append((rect, "section", title))
        return y + self._S(16), is_open

    def _cycle(self, img, label, value, key, x, y, w):
        self._text(img, label, x, y + self._S(10), DIM, 0.42)
        ry = y + self._S(16)
        bw, bh = self._S(28), self._S(24)
        lb = (x, ry, x + bw, ry + bh)
        rb = (x + w - bw, ry, x + w, ry + bh)
        self._box(img, lb, HOVER if _in(lb, self.mouse) else BTN)
        self._box(img, rb, HOVER if _in(rb, self.mouse) else BTN)
        self._text(img, "<", x + self._S(9), ry + self._S(17), INK, 0.5, 2)
        self._text(img, ">", x + w - self._S(19), ry + self._S(17), INK, 0.5, 2)
        self._text(img, str(value), x + self._S(40), ry + self._S(17), ACC, 0.5)
        self._hot.append((lb, "cycle", (key, -1)))
        self._hot.append((rb, "cycle", (key, +1)))
        return ry + self._S(32)

    def _manage_buttons(self, frame, name, x, y, cw):
        """Hover-revealed rename (~) and delete (x) buttons on a saved look's row.
        Their hit rects go to the FRONT of _hot so they win over the full-row rect."""
        db = (x + cw - self._S(26), y + self._S(2), x + cw - self._S(4), y + self._S(22))
        rb = (x + cw - self._S(52), y + self._S(2), x + cw - self._S(30), y + self._S(22))
        armed = self._del_armed == name
        self._box(frame, rb, HOVER if _in(rb, self.mouse) else BTN)
        self._text(frame, "~", rb[0] + self._S(6), rb[3] - self._S(7), INK, 0.45)
        self._box(frame, db, RED if armed else (HOVER if _in(db, self.mouse) else BTN))
        self._text(frame, "x", db[0] + self._S(7), db[3] - self._S(7), INK, 0.45, 2 if armed else 1)
        if armed:
            self._text(frame, "sure? x again", x - self._S(118), y + self._S(16), RED, 0.42, 1)
        self._hot.insert(0, (db, "del", name))
        self._hot.insert(0, (rb, "ren", name))

    def _rename_box(self, frame, x, y, cw, px):
        """The row being renamed becomes a text input (Enter saves, Esc cancels)."""
        rect = (x, y, x + cw, y + self._S(24))
        self._box(frame, rect, DARK, border=ACC)
        cur = "_" if (self._blink // 12) % 2 == 0 else ""
        self._text(frame, self.rename_buf + cur, x + self._S(10), y + self._S(16), ACC, 0.46)
        self._text(frame, "type name: enter=save esc=cancel", px - self._S(240), y + self._S(16), ACC, 0.42)

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
        self._hot = []
        self._tooltip = None
        h, w = frame.shape[:2]
        # scale the whole panel by the output resolution, floored at the 1080p baseline so
        # 720p/1080p are unchanged and 4K renders at 2x (same fraction of the frame).
        self._s = max(1.0, h / BASE_H)
        pw = self._S(self.panel_w)
        self._panel_px = pw
        if not self.open:
            r = (w - self._S(48), self._S(12), w - self._S(12), self._S(42))
            self._box(frame, r, HOVER if _in(r, self.mouse) else PANEL)
            for yy in (self._S(21), self._S(27), self._S(33)):
                cv2.line(frame, (w - self._S(40), yy), (w - self._S(20), yy), INK,
                         max(1, self._S(2)), cv2.LINE_AA)
            self._hot.append((r, "collapse", None))
            self._draw_status(frame, info)
            return frame

        px = w - pw
        ov = frame.copy()
        cv2.rectangle(ov, (px, 0), (w, h), PANEL, -1)
        cv2.addWeighted(ov, 0.86, frame, 0.14, 0, frame)
        # the column can be taller than the window (e.g. 720p) — scroll, clamped so it's
        # a no-op when everything fits. content height comes from the previous draw.
        self.scroll = min(max(self.scroll, 0), max(0, self._content_h - h))
        x, cw, y = px + self._S(16), pw - self._S(32), self._S(30) - self.scroll
        self._text(frame, "dtouch", x, y, ACC, 0.62, 2)
        y += self._S(16)

        y, sec_open = self._section(frame, "TEMPLATES", x, y, cw)
        self._blink += 1
        for i, name in enumerate(self.presets if sec_open else []):
            if name == self.renaming:
                self._rename_box(frame, x, y, cw, px)
                y += self._S(28)
                continue
            r = self._row(frame, name, "preset", x, y, cw,
                          active=(i == self.preset_idx), payload=i)
            if name in self.user_presets and (_in(r, self.mouse) or name == self._del_armed):
                self._manage_buttons(frame, name, x, y, cw)
            y += self._S(28)
        if sec_open:
            self._row(frame, "+ Save current look", "save", x, y, cw); y += self._S(30)
        y += self._S(4)

        y, sec_open = self._section(frame, "SOURCE", x, y, cw)
        if sec_open:
            y = self._cycle(frame, "matte", self.matte_name, "matte", x, y, cw) + self._S(2)
            y = self._cycle(frame, "output", self.res_name, "res", x, y, cw) + self._S(2)
            self._row(frame, "Video bg: " + ("ON" if self.video_bg else "off"),
                      "video_bg", x, y, cw, active=self.video_bg); y += self._S(28)
            y = self._slider(frame, "Vid mix", "video_mix", x, y, cw,
                             "How visible the raw camera footage is behind the particles.")
        y += self._S(4)

        y, sec_open = self._section(frame, "LOOK", x, y, cw)
        if sec_open:
            y = self._cycle(frame, "color", self.palette_name, "color", x, y, cw) + self._S(4)
            for label, attr, tip in _SLIDERS:
                y = self._slider(frame, label, attr, x, y, cw, tip)
        y += self._S(4)

        # MOTION — boids steering (dtouch.flock). The sliders stay visible while off so the
        # section reads as a thing you can turn on, not a thing that appears from nowhere.
        y, sec_open = self._section(frame, "MOTION", x, y, cw)
        if sec_open:
            self._row(frame, "Flock: " + ("ON" if self.flock else "off"),
                      "flock", x, y, cw, active=self.flock); y += self._S(28)
            y = self._slider(frame, "Cohere", "cohere", x, y, cw,
                             "Steer toward the local centre. Pulls the cloud into shoals.")
            y = self._slider(frame, "Align", "align", x, y, cw,
                             "Match neighbours' direction. This is what makes it move as one.")
            y = self._slider(frame, "Separate", "separate", x, y, cw,
                             "Push apart when crowded. Stops the shoal collapsing to a dot.")
        y += self._S(4)

        # SIGNAL — circuit-bent post-processing (dtouch.circuit_bent), applied to the rendered
        # frame after the particles, before the panel is drawn (so the panel stays readable).
        y, sec_open = self._section(frame, "SIGNAL", x, y, cw)
        if sec_open:
            self._row(frame, "Glitch: " + ("ON" if self.glitch else "off"),
                      "glitch", x, y, cw, active=self.glitch); y += self._S(28)
            y = self._cycle(frame, "dither", self.dither_name, "dither", x, y, cw) + self._S(2)
            y = self._slider(frame, "Chroma", "chroma", x, y, cw,
                             "Colour bleed: red and blue drift apart, slowly.")
            y = self._slider(frame, "Drift", "drift", x, y, cw,
                             "Scan-line sync loss. Rows slip sideways; sometimes a whole band tears.")
            y = self._slider(frame, "Crush", "crush", x, y, cw,
                             "Hard bit-depth reduction. 0 = off.")
            self._row(frame, "Scanlines: " + ("on" if self.scanlines else "off"),
                      "scanlines", x, y, cw, active=self.scanlines); y += self._S(28)
        y += self._S(6)
        self._row(frame, "Sound react: " + ("ON" if self.audio else "off"),
                  "audio", x, y, cw, active=self.audio); y += self._S(28)
        y = self._slider(frame, "Sens", "sens", x, y, cw,
                         "How strongly sound drives the visuals.") + self._S(4)
        self._row(frame, ("Stop recording" if self.record else "Record"),
                  "record", x, y, cw, active=self.record); y += self._S(28)
        self._row(frame, "Mirror: " + ("on" if self.mirror else "off"),
                  "mirror", x, y, cw, active=self.mirror); y += self._S(32)
        self._row(frame, "Quit", "quit", x, y, cw)
        self._content_h = y + self.scroll + self._S(36)   # column bottom incl. margin, unscrolled

        # collapse button drawn last so it stays fixed and clickable above scrolled content
        cr = (w - self._S(40), self._S(12), w - self._S(12), self._S(36))
        self._box(frame, cr, HOVER if _in(cr, self.mouse) else BTN)
        self._text(frame, ">", w - self._S(33), self._S(30), INK, 0.55, 2)
        self._hot.append((cr, "collapse", None))
        if self._content_h > h:   # slim scrollbar on the panel's left edge
            bar_h = max(int(h * h / self._content_h), self._S(30))
            by = int((h - bar_h) * (self.scroll / max(self._content_h - h, 1)))
            cv2.rectangle(frame, (px + self._S(2), by), (px + self._S(5), by + bar_h), TRACK, -1)

        # click feedback: flash the last-clicked control
        if self._flash > 0 and self._flash_rect is not None:
            self._box(frame, self._flash_rect, BTN, border=(255, 255, 255))
            self._flash -= 1
        self._draw_tooltip(frame)
        self._draw_status(frame, info)
        return frame

    def _draw_tooltip(self, frame):
        if not self._tooltip:
            return
        text, sx, sy = self._tooltip
        words, lines, cur = text.split(), [], ""
        for wd in words:
            if len(cur) + len(wd) + 1 > 30:
                lines.append(cur); cur = wd
            else:
                cur = (cur + " " + wd).strip()
        if cur:
            lines.append(cur)
        bw, bh = self._S(250), self._S(16) * len(lines) + self._S(16)
        bx = max(sx - bw - self._S(14), self._S(10))
        by = max(min(sy, self.h - bh - self._S(10)), self._S(10))
        self._box(frame, (bx, by, bx + bw, by + bh), (44, 48, 56), border=(120, 140, 170))
        for i, ln in enumerate(lines):
            self._text(frame, ln, bx + self._S(10), by + self._S(20) + i * self._S(16), INK, 0.42)

    def _draw_status(self, frame, info):
        self._text(frame, info.get("status", ""), self._S(12), self._S(24), ACC, 0.5)
        if info.get("black"):
            h, w = frame.shape[:2]
            self._text(frame, "CAMERA IS BLACK - disable iPhone Continuity Camera",
                       self._S(12), h // 2, RED, 0.8, 2)
            self._text(frame, "iPhone: Settings > General > AirPlay & Handoff > Continuity Camera > Off",
                       self._S(12), h // 2 + self._S(28), (140, 140, 240), 0.5)
        if self.record:
            h, w = frame.shape[:2]
            off = (self._panel_px + self._S(22)) if self.open else self._S(70)
            cv2.circle(frame, (w - off, self._S(24)), self._S(7), RED, -1)

    # ----- mouse -----
    def on_mouse(self, event, x, y, flags, param=None):
        self.mouse = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            for rect, kind, payload in self._hot:
                if _in(rect, (x, y)):
                    self._flash_rect, self._flash = rect, 4
                    self._activate(kind, payload, x)
                    return
            if self.open and x >= self.w - self._panel_px:
                self._scroll_drag = (y, self.scroll)   # empty panel area: drag to scroll
        elif event == cv2.EVENT_MOUSEWHEEL and self.open:
            step = self._S(40)
            self.scroll += -step if flags > 0 else step   # flags>0 = wheel up (clamped in draw)
        elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):
            if self._drag:
                attr, x0, x1, lo, hi = self._drag
                t = min(max((x - x0) / max(x1 - x0, 1), 0.0), 1.0)
                setattr(self, attr, lo + t * (hi - lo))
            elif self._scroll_drag:
                y0, s0 = self._scroll_drag
                self.scroll = s0 + (y0 - y)
        elif event == cv2.EVENT_LBUTTONUP:
            self._drag = None
            self._scroll_drag = None

    def _activate(self, kind, payload, x):
        if kind != "del" and self._del_armed:
            self._del_armed = None       # any other click disarms a pending delete
        if kind == "del":
            if self._del_armed == payload:
                self.pending_delete = payload
                self._del_armed = None
            else:
                self._del_armed = payload
            return
        if kind == "ren":
            self.renaming = payload
            self.rename_buf = payload    # prefill with the current name
            return
        if kind == "collapse":
            self.open = not self.open
        elif kind == "preset":
            self.preset_idx = payload
            self.pending_preset = self.presets[payload]
        elif kind == "cycle":
            key, d = payload
            if key == "matte":
                self.matte_idx = (self.matte_idx + d) % len(self.mattes)
            elif key == "color":
                self.palette_idx = (self.palette_idx + d) % len(self.palettes)
            elif key == "res":
                self.res_idx = (self.res_idx + d) % len(self.res_options)
            elif key == "dither":
                self.dither_idx = (self.dither_idx + d) % len(DITHERS)
        elif kind == "slider":
            attr, x0, x1, lo, hi = payload
            self._drag = payload
            t = min(max((x - x0) / max(x1 - x0, 1), 0.0), 1.0)
            setattr(self, attr, lo + t * (hi - lo))
        elif kind == "audio":
            self.audio = not self.audio
        elif kind == "record":
            self.record = not self.record
        elif kind == "mirror":
            self.mirror = not self.mirror
        elif kind == "video_bg":
            self.video_bg = not self.video_bg
        elif kind == "section":
            self.sections[payload] = not self.sections.get(payload, True)
        elif kind == "flock":
            self.flock = not self.flock
        elif kind == "glitch":
            self.glitch = not self.glitch
        elif kind == "scanlines":
            self.scanlines = not self.scanlines
        elif kind == "save":
            self.pending_save = True
        elif kind == "quit":
            self.quit = True


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
