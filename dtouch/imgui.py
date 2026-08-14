"""Immediate-mode widget toolkit — the panel's drawing + hit-rect primitives.

Lifted out of OverlayUI (DESIGN.md §2.3 / §8 step 4) so any surface — the
Particles panel today, the home menu and per-mode panels later — draws with the
same widgets and gets the same pixels. Everything is a drawn, boxed,
hover-highlighting control with click feedback; hit-testing is a list of
(rect, kind, payload) tuples appended as widgets draw.

The toolkit is standalone: a `Gui` context carries the per-draw state (scale,
mouse, hit list, pending tooltip); widgets take an image + values + geometry
and append to `gui.hot`. Nothing here knows about OverlayUI, presets, or the
engine. ASCII glyphs only — cv2's Hershey font can't render •/■/≡/— (they show
as ???).

Every pixel dimension is authored at a 1080p baseline and multiplied by the
scale factor `Gui.s` (see `Gui.S`), floored at 1.0 by the caller so 720p/1080p
render exactly as authored and 4K renders at 2x (issue #3).
"""
from __future__ import annotations

import cv2

# BGR chrome — shared by the panel and every future toolkit surface.
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


def in_rect(rect, p):
    x0, y0, x1, y1 = rect
    return x0 <= p[0] <= x1 and y0 <= p[1] <= y1


# ----- scroll mechanics (pure) -----

def wheel_scroll(scroll, step, flags):
    """One wheel event: flags>0 = wheel up. Clamping happens at draw time."""
    return scroll + (-step if flags > 0 else step)


def clamp_scroll(scroll, content_h, view_h):
    """Clamp so scrolling is a no-op when everything fits."""
    return min(max(scroll, 0), max(0, content_h - view_h))


def drag_scroll(scroll0, y0, y):
    """Drag-on-empty-panel scrolling: content follows the finger."""
    return scroll0 + (y0 - y)


def arm_delete(armed, name):
    """Two-click destructive arm: first x-click arms, second confirms.
    Returns (new_armed, confirmed_name_or_None)."""
    if armed == name:
        return None, name
    return name, None


class Gui:
    """Per-draw immediate-mode context: resolution scale, mouse position, the
    hit list widgets append to, and the tooltip deferred to end-of-draw."""

    def __init__(self):
        self.s = 1.0            # current UI scale (1080p baseline; caller floors at 1.0)
        self.mouse = (-1, -1)
        self.hot = []           # (rect, kind, payload) — appended as widgets draw
        self.tooltip = None     # (text, x, y) — set on info-badge hover, drawn last
        self.accent = ACC       # per-mode accent (DESIGN.md §5): selected/active chrome

    def begin(self, s, mouse, accent=ACC):
        """Start a draw pass: set scale + mouse + accent, reset hit list and tooltip."""
        self.s = s
        self.mouse = mouse
        self.accent = accent
        self.hot = []
        self.tooltip = None
        return self.hot

    # ----- scaling -----
    def S(self, n):
        """Scale a baseline pixel value by the current resolution factor (int for cv2)."""
        return int(round(n * self.s))

    # ----- primitives -----
    def text(self, img, s, x, y, color=INK, scale=0.46, thick=1):
        cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale * self.s, color,
                    max(1, int(round(thick * self.s))), cv2.LINE_AA)

    def box(self, img, rect, fill, border=TRACK):
        x0, y0, x1, y1 = rect
        cv2.rectangle(img, (x0, y0), (x1, y1), fill, -1)
        cv2.rectangle(img, (x0, y0), (x1, y1), border, max(1, self.S(1)), cv2.LINE_AA)

    # ----- widgets -----
    def row(self, img, label, key, x, y, w, h=None, active=False, payload=None):
        """A full-width clickable row (button / toggle face)."""
        if h is None:
            h = self.S(24)
        rect = (x, y, x + w, y + h)
        hovered = in_rect(rect, self.mouse)
        fill = self.accent if active else (HOVER if hovered else BTN)
        self.box(img, rect, fill)
        self.text(img, label, x + self.S(10), y + h - self.S(8), DARK if active else INK, 0.46)
        self.hot.append(((x, y, x + w, y + h), key, payload))
        return rect

    def slider(self, img, label, attr, val, lo, hi, x, y, w, info=None, fmt=".2f"):
        """Labelled slider with an `i` tooltip badge. Hit payload carries the
        track geometry so the caller can turn a drag into a value. Returns next y."""
        rect = (x, y, x + w, y + self.S(26))
        hovered = in_rect(rect, self.mouse)
        self.box(img, rect, HOVER if hovered else BTN)
        # info badge
        ic = (x + self.S(14), y + self.S(13))
        irect = (x + self.S(4), y + self.S(3), x + self.S(24), y + self.S(23))
        cv2.circle(img, ic, self.S(8), (90, 110, 150), -1, cv2.LINE_AA)
        self.text(img, "i", ic[0] - self.S(2), ic[1] + self.S(5), (235, 240, 245), 0.42, 1)
        if info and in_rect(irect, self.mouse):
            self.tooltip = (info, x, y)
        self.text(img, label, x + self.S(30), y + self.S(17), DIM, 0.42)
        tx0, tx1 = x + self.S(96), x + w - self.S(44)
        cv2.line(img, (tx0, y + self.S(13)), (tx1, y + self.S(13)), TRACK,
                 max(1, self.S(3)), cv2.LINE_AA)
        hx = int(tx0 + (val - lo) / (hi - lo) * (tx1 - tx0))
        cv2.circle(img, (hx, y + self.S(13)), self.S(6), HANDLE, -1, cv2.LINE_AA)
        self.text(img, f"{val:{fmt}}", x + w - self.S(38), y + self.S(17), INK, 0.42)
        self.hot.append(((tx0 - self.S(8), y, tx1 + self.S(8), y + self.S(26)),
                         "slider", (attr, tx0, tx1, lo, hi)))
        return y + self.S(30)

    def section(self, img, title, is_open, x, y, w, key_hint=None):
        """Clickable section header. Returns (next_y, is_open).

        `key_hint` (DESIGN.md §4.1: MOTION (F), SIGNAL (G)) renders DIM,
        right-aligned in the header row. ASCII markers only: cv2's Hershey
        font renders no glyph for the usual disclosure triangles, so they
        come out as '???' (see module docstring)."""
        rect = (x - self.S(4), y - self.S(4), x + w, y + self.S(14))
        if in_rect(rect, self.mouse):
            self.box(img, rect, HOVER)
        self.text(img, ("- " if is_open else "+ ") + title, x, y + self.S(8),
                  INK if is_open else DIM, 0.4)
        if key_hint:
            hint = f"({key_hint})"
            (tw, _), _ = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX,
                                         0.4 * self.s, 1)
            self.text(img, hint, x + w - tw - self.S(6), y + self.S(8), DIM, 0.4)
        self.hot.append((rect, "section", title))
        return y + self.S(16), is_open

    def cycle(self, img, label, value, key, x, y, w):
        """< value > cycle control. Hits are ("cycle", (key, ±1)). Returns next y."""
        self.text(img, label, x, y + self.S(10), DIM, 0.42)
        ry = y + self.S(16)
        bw, bh = self.S(28), self.S(24)
        lb = (x, ry, x + bw, ry + bh)
        rb = (x + w - bw, ry, x + w, ry + bh)
        self.box(img, lb, HOVER if in_rect(lb, self.mouse) else BTN)
        self.box(img, rb, HOVER if in_rect(rb, self.mouse) else BTN)
        self.text(img, "<", x + self.S(9), ry + self.S(17), INK, 0.5, 2)
        self.text(img, ">", x + w - self.S(19), ry + self.S(17), INK, 0.5, 2)
        self.text(img, str(value), x + self.S(40), ry + self.S(17), self.accent, 0.5)
        self.hot.append((lb, "cycle", (key, -1)))
        self.hot.append((rb, "cycle", (key, +1)))
        return ry + self.S(32)

    def slot_badge(self, img, slot, x, y, cw):
        """Bank-slot badge on a preset row (DESIGN.md §4.1 / §6.3): an assigned
        row shows its digit in the accent; an unassigned row shows an empty
        well. Clicking assigns the next free slot (the shell owns the policy).
        Sits left of the hover manage buttons — no overlap. Returns its rect;
        the caller appends the hit with the row's name."""
        r = (x + cw - self.S(78), y + self.S(2), x + cw - self.S(56), y + self.S(22))
        if slot:
            self.box(img, r, DARK, border=self.accent)
            self.text(img, str(slot), r[0] + self.S(7), r[3] - self.S(6),
                      self.accent, 0.42)
        else:
            self.box(img, r, HOVER if in_rect(r, self.mouse) else BTN)
        return r

    def manage_buttons(self, img, name, x, y, cw, armed):
        """Hover-revealed rename (~) and delete (x) buttons on a saved look's row.
        Their hit rects go to the FRONT of `hot` so they win over the full-row rect.
        `armed` = this name's delete is one click from confirming (see arm_delete)."""
        db = (x + cw - self.S(26), y + self.S(2), x + cw - self.S(4), y + self.S(22))
        rb = (x + cw - self.S(52), y + self.S(2), x + cw - self.S(30), y + self.S(22))
        self.box(img, rb, HOVER if in_rect(rb, self.mouse) else BTN)
        self.text(img, "~", rb[0] + self.S(6), rb[3] - self.S(7), INK, 0.45)
        self.box(img, db, RED if armed else (HOVER if in_rect(db, self.mouse) else BTN))
        self.text(img, "x", db[0] + self.S(7), db[3] - self.S(7), INK, 0.45, 2 if armed else 1)
        if armed:
            self.text(img, "sure? x again", x - self.S(118), y + self.S(16), RED, 0.42, 1)
        self.hot.insert(0, (db, "del", name))
        self.hot.insert(0, (rb, "ren", name))

    def rename_box(self, img, buf, blink, x, y, cw, px):
        """The row being renamed becomes a text input (Enter saves, Esc cancels).
        `blink` is a frame counter driving the cursor; `px` is the panel's left edge."""
        rect = (x, y, x + cw, y + self.S(24))
        self.box(img, rect, DARK, border=self.accent)
        cur = "_" if (blink // 12) % 2 == 0 else ""
        self.text(img, buf + cur, x + self.S(10), y + self.S(16), self.accent, 0.46)
        self.text(img, "type name: enter=save esc=cancel", px - self.S(240), y + self.S(16),
                  self.accent, 0.42)

    # ----- deferred / chrome -----
    def draw_tooltip(self, img, clamp_h):
        """Draw the tooltip deferred by the last slider badge hover (if any)."""
        if not self.tooltip:
            return
        text, sx, sy = self.tooltip
        words, lines, cur = text.split(), [], ""
        for wd in words:
            if len(cur) + len(wd) + 1 > 30:
                lines.append(cur); cur = wd
            else:
                cur = (cur + " " + wd).strip()
        if cur:
            lines.append(cur)
        bw, bh = self.S(250), self.S(16) * len(lines) + self.S(16)
        bx = max(sx - bw - self.S(14), self.S(10))
        by = max(min(sy, clamp_h - bh - self.S(10)), self.S(10))
        self.box(img, (bx, by, bx + bw, by + bh), (44, 48, 56), border=(120, 140, 170))
        for i, ln in enumerate(lines):
            self.text(img, ln, bx + self.S(10), by + self.S(20) + i * self.S(16), INK, 0.42)

    def scrollbar(self, img, px, view_h, content_h, scroll):
        """Slim scrollbar on the panel's left edge — only when content overflows."""
        if content_h <= view_h:
            return
        bar_h = max(int(view_h * view_h / content_h), self.S(30))
        by = int((view_h - bar_h) * (scroll / max(content_h - view_h, 1)))
        cv2.rectangle(img, (px + self.S(2), by), (px + self.S(5), by + bar_h), TRACK, -1)
