"""Home menu + boot card — an in-app switcher screen, not a launcher (DESIGN.md §3).

The menu is a shell overlay state, not a Mode: one window, the live camera
behind it rendered through 1-bit blue-noise dither and dimmed under a 65%
scrim (the menu proves the camera works and demos the newest feature before a
single click). Cards come from the mode REGISTRY — one import + one line adds
a card — plus one dimmed reserved card for Flocking (DESIGN.md §2.5: the
landing site for issue #4's body-driven flock).

Navigation (DESIGN.md §3): the mode's letter or digit selects-and-enters,
','/'.' move selection, Enter commits, Esc returns to the running mode
untouched, mouse clicks a card. `Menu` is pure state (unit-testable without a
window); `draw_menu` renders and returns the click rects.

Mode switches draw a static boot card first (mode glyph + name in the mode's
accent, one frame, no animation — the cv2 loop is single-threaded, so nothing
can animate during GL teardown). The recorder captures the card, not a gray
flash.

ASCII text only (Hershey fonts render non-ASCII as '?').
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from .dither import blue_noise_dither
from .hud import DIM, INK, TITLE_SAFE, put_outlined, text_size, u
from .imgui import in_rect
from .modes import REGISTRY

MENU_SCRIM = 0.65               # DESIGN.md §5: menu scrim 65%


@dataclass
class Card:
    id: Optional[str]           # mode id, or None for a reserved card
    title: str
    blurb: str
    accent: tuple               # BGR
    key: Optional[str]          # select-and-enter letter, or None
    enabled: bool = True


def registry_cards():
    """One card per registered mode, plus the dimmed reserved Flocking card
    (DESIGN.md §2.5 — 'coming soon', not selectable)."""
    cards = [Card(m.id, m.title, getattr(m, "blurb", ""), m.accent, m.id[:1])
             for m in REGISTRY]
    cards.append(Card(None, "FLOCKING", "coming soon", DIM, None, enabled=False))
    return cards


class Menu:
    """Menu state machine — open flag, selection, key/click routing.

    `key()` returns ("switch", mode_id) when a card is committed,
    ("close", None) when the menu dismisses, (None, None) otherwise. Commit
    and dismiss both close the menu; the shell routes the switch."""

    def __init__(self, cards=None):
        self.cards = list(cards) if cards is not None else registry_cards()
        self.open = False
        self.sel = 0
        self.rects = []          # (rect, card) from the last draw — click targets

    # ----- lifecycle -----
    def show(self, active_id=None):
        self.open = True
        self.sel = next((i for i, c in enumerate(self.cards)
                         if c.enabled and c.id == active_id), self.sel)
        if not self._enabled(self.sel):
            self.sel = next(i for i, c in enumerate(self.cards) if c.enabled)

    def close(self):
        self.open = False

    def toggle(self, active_id=None):
        if self.open:
            self.close()
        else:
            self.show(active_id)

    # ----- navigation -----
    def _enabled(self, i):
        return 0 <= i < len(self.cards) and self.cards[i].enabled

    def move(self, d):
        """Step selection by d, skipping disabled cards, wrapping."""
        n = len(self.cards)
        i = self.sel
        for _ in range(n):
            i = (i + d) % n
            if self.cards[i].enabled:
                self.sel = i
                return

    def _commit(self, i):
        self.sel = i
        self.close()
        return "switch", self.cards[i].id

    def key(self, code):
        """Route one cv2.waitKey code. See class docstring for returns."""
        if code == 27 or code in (ord("m"), ord("M")):   # Esc / m: back untouched
            self.close()
            return "close", None
        if code in (13, 10):                             # Enter commits
            if self._enabled(self.sel):
                return self._commit(self.sel)
            return None, None
        if code == ord(","):
            self.move(-1)
            return None, None
        if code == ord("."):
            self.move(+1)
            return None, None
        if 32 <= code <= 126:
            ch = chr(code).lower()
            if ch.isdigit():                             # digit selects-and-enters
                i = int(ch) - 1
                if self._enabled(i):
                    return self._commit(i)
                return None, None
            for i, c in enumerate(self.cards):           # letter selects-and-enters
                if c.enabled and c.key and ch == c.key.lower():
                    return self._commit(i)
        return None, None

    def click(self, pt):
        """A mouse click at pt: commit the enabled card under it, or None."""
        for rect, card in self.rects:
            if card.enabled and in_rect(rect, pt):
                i = self.cards.index(card)
                return self._commit(i)[1]
        return None


def draw_menu(img, cam_bgr, cards, sel):
    """Render the menu over `img` (BGR, in place): 1-bit blue-noise-dithered
    live camera under a 65% scrim, 'dtouch' top-left, a centered row of mode
    cards. Returns the click rects [(rect, card), ...]."""
    h, w = img.shape[:2]
    uu = u(h)

    # background: the live camera through the newest feature (1-bit blue noise)
    if cam_bgr is not None:
        cam = cam_bgr if cam_bgr.shape[:2] == (h, w) else cv2.resize(cam_bgr, (w, h))
        gray = cv2.cvtColor(cam, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        lit = blue_noise_dither(gray, bits=1)
        shade = np.uint8(round(255 * (1.0 - MENU_SCRIM)))   # dimmed under the scrim
        img[:] = (lit * float(shade)).astype(np.uint8)[:, :, None]
    else:
        img[:] = 0

    ix, iy = int(w * TITLE_SAFE), int(h * TITLE_SAFE)
    tpx = int(1.4 * uu)
    put_outlined(img, "dtouch", (ix, iy + tpx), tpx, INK)

    # centered row of cards
    n = len(cards)
    cw, ch, gap = int(10.0 * uu), int(7.0 * uu), int(1.5 * uu)
    total = n * cw + (n - 1) * gap
    x0 = max((w - total) // 2, ix)
    y0 = (h - ch) // 2
    rects = []
    for i, c in enumerate(cards):
        x = x0 + i * (cw + gap)
        rect = (x, y0, x + cw, y0 + ch)
        color = c.accent if c.enabled else DIM
        selected = i == sel
        cv2.rectangle(img, (x, y0), (x + cw, y0 + ch), color,
                      max(2, int(0.2 * uu)) if selected else 1, cv2.LINE_AA)
        # title (centered), blurb, key hint — every state redundantly coded
        # (color + the selection border + the [key] hint)
        tp = int(1.0 * uu)
        tw = text_size(c.title, tp)[0]
        put_outlined(img, c.title, (x + (cw - tw) // 2, y0 + int(2.2 * uu)), tp,
                     color if (selected or not c.enabled) else INK)
        bp = int(0.62 * uu)
        for j, line in enumerate(c.blurb.split("\n")):
            bw = text_size(line, bp)[0]
            put_outlined(img, line, (x + (cw - bw) // 2,
                                     y0 + int(3.6 * uu) + j * int(1.0 * uu)),
                         bp, DIM)
        if c.key:
            hint = f"[{c.key.upper()}]"
            hw = text_size(hint, bp)[0]
            put_outlined(img, hint, (x + (cw - hw) // 2, y0 + ch - int(0.8 * uu)),
                         bp, color if selected else DIM)
        rects.append((rect, c))
    return rects


def render_boot_card(res, title, accent, glyph=None):
    """The static boot card shown for one frame during a mode switch
    (DESIGN.md §3): mode glyph + name in the mode's accent on black. Returns a
    BGR image at `res` — the caller shows it and (if recording) writes its RGB
    conversion, so the recorder captures the card, not a gray flash."""
    w, h = res
    img = np.zeros((h, w, 3), np.uint8)
    uu = u(h)
    glyph = (glyph or title[:1]).upper()
    gp = int(8.0 * uu)
    gw = text_size(glyph, gp)[0]
    put_outlined(img, glyph, ((w - gw) // 2, h // 2), gp, accent)
    tp = int(3.0 * uu)
    tw = text_size(title, tp)[0]
    put_outlined(img, title, ((w - tw) // 2, h // 2 + int(4.2 * uu)), tp, accent)
    return img
