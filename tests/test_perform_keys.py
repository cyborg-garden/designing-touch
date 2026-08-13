"""Perform keys + safety — DESIGN.md §6.2, migration step 3.

Drives the real registry wiring (`_wire_perform_keys`) and the real key path
(`_perform_key`) against a real OverlayUI and a fake-clock Hud/PerformState —
exactly the objects live_flow wires, minus the camera loop.
"""
import numpy as np
import pytest

from dtouch import hud as H
from dtouch.commands import CommandRegistry
from dtouch.hud import Hud, OverlayState, Toasts, draw_help
from dtouch.live import (MATTES, PerformState, _perform_key, _wire_perform_keys)
from dtouch.overlay_ui import OverlayUI
from dtouch.particles import PALETTES

PRESETS = ["abstract", "portrait", "textured", "embers", "aurora", "sigil"]


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


class Rig:
    def __init__(self):
        self.clock = Clock()
        self.ui = OverlayUI(1920, 1080, list(PRESETS), list(PALETTES), MATTES)
        self.hud = Hud(now=self.clock)
        self.ps = PerformState(now=self.clock)
        self.reg = CommandRegistry()
        self.applied = []          # preset names routed through the recall path
        self.overlay = OverlayState.HUD

        def recall(name):
            self.applied.append(name)
            self.ui.pending_preset = name   # same mailbox a panel click uses
        _wire_perform_keys(self.reg, self.ui, self.hud, self.ps, recall,
                           safe_look=lambda: "abstract")
        # the shell's default seeding for a mode with no stored bank:
        # built-ins on slots 1..9 in order, setlist empty (= all looks)
        self.ui.bank = {str(i + 1): n for i, n in enumerate(PRESETS)}
        self.ui.setlist = []

    def press(self, ch):
        key = ch if isinstance(ch, int) else ord(ch)
        self.overlay = _perform_key(key, self.overlay, self.ps, self.reg, self.hud)

    def center_toast(self):
        return self.hud.toasts._center

    def hints(self):
        return [t.text for t in self.hud.toasts._hints]


# ---------- Space: blackout ----------

def test_space_toggles_blackout_with_amber_flash():
    r = Rig()
    r.press(" ")
    assert r.ps.blackout is True
    assert r.center_toast().text == "BLACKOUT" and r.center_toast().color == H.AMBER
    r.press(" ")
    assert r.ps.blackout is False
    assert "blackout off" in r.hints()


# ---------- 0: panic ----------

def test_panic_applies_safe_look_disarms_blackout_and_glitch():
    """Amended DESIGN.md §6.2 '0': panic restores a known-good PICTURE — the
    mode's safe_look, blackout disarmed, SIGNAL rack (glitch) off."""
    r = Rig()
    r.ui.preset_idx = 3                     # embers is current
    r.ui.glitch = True                      # SIGNAL rack armed
    r.press(" ")                            # arm blackout
    r.press("0")
    assert r.ps.blackout is False
    assert r.ui.glitch is False             # rack off, not just params reset
    assert r.applied == ["abstract"]        # the safe look, not the current one
    assert r.ui.preset_idx == 0
    assert r.ui.pending_preset == "abstract"
    assert r.center_toast().text == "RESET" and r.center_toast().color == H.AMBER


# ---------- q: quit confirm ----------

def test_q_needs_two_presses_within_two_seconds():
    r = Rig()
    r.press("q")
    assert r.ps.quit is False
    assert "q again to quit" in r.hints()
    r.clock.t = 1.0
    r.press("q")
    assert r.ps.quit is True


def test_q_confirm_window_expires():
    r = Rig()
    r.press("q")
    r.clock.t = 3.0                         # too late — re-arms instead
    r.press("q")
    assert r.ps.quit is False
    r.clock.t = 4.0
    r.press("q")
    assert r.ps.quit is True


# ---------- ?: help overlay ----------

def test_question_mark_opens_help_and_any_key_closes_without_firing():
    r = Rig()
    r.press("?")
    assert r.ps.help_open is True
    r.press("f")                            # closes help; must NOT toggle flock
    assert r.ps.help_open is False
    assert r.ui.flock is False


@pytest.mark.parametrize("state", list(OverlayState))
def test_help_works_in_every_overlay_state(state):
    r = Rig()
    r.overlay = state
    r.press("?")
    assert r.ps.help_open is True
    r.press(9)                              # TAB closes too, without cycling
    assert r.ps.help_open is False
    assert r.overlay is state


def test_help_table_lists_the_perform_keys():
    r = Rig()
    keys = {k for k, _ in r.reg.table()}
    assert keys >= {" ", "0", "q", "?", "[", "]", "a", "r", "v", "f", "g", "i"}
    assert keys >= set("123456789")


def test_draw_help_darkens_the_frame():
    frame = np.full((1080, 1920, 3), 200, np.uint8)
    draw_help(frame, [("q", "Quit"), (" ", "Blackout")])
    assert frame.mean() < 120               # 65% scrim took the frame down


# ---------- digits + [ ] : bank recall + setlist (DESIGN.md §7, step 7) ----------

def test_digit_recalls_from_the_active_modes_bank_with_toast():
    r = Rig()
    r.press("3")
    assert r.ui.preset_idx == 2
    assert r.applied == ["textured"]
    assert r.center_toast().text == "3 - textured"


def test_digit_recall_uses_slot_assignments_not_list_positions():
    """Bank slots are explicit assignments (DESIGN.md §7) — a slot can point
    anywhere, and slot numbers survive around it."""
    r = Rig()
    r.ui.bank = {"2": "sigil"}
    r.press("2")
    assert r.applied == ["sigil"]
    assert r.center_toast().text == "2 - sigil"
    r.press("1")                            # unassigned now
    assert r.applied == ["sigil"]
    assert "no preset 1" in r.hints()


def test_empty_slot_is_a_gentle_hint():
    r = Rig()
    r.press("9")                            # 6 built-ins: slot 9 unassigned
    assert r.applied == []
    assert "no preset 9" in r.hints()


def test_slot_pointing_at_a_deleted_look_hints():
    r = Rig()
    r.ui.bank = {"4": "gone_look"}
    r.press("4")
    assert r.applied == []
    assert "no preset 4" in r.hints()


def test_brackets_walk_all_looks_when_setlist_empty():
    r = Rig()
    r.press("]")
    assert (r.ui.preset_idx, r.applied[-1]) == (1, "portrait")
    r.press("[")
    assert (r.ui.preset_idx, r.applied[-1]) == (0, "abstract")
    r.press("[")                            # wraps to the end
    assert (r.ui.preset_idx, r.applied[-1]) == (5, "sigil")


def test_brackets_walk_the_explicit_setlist():
    r = Rig()
    r.ui.setlist = ["embers", "abstract"]
    r.press("]")                            # abstract is current → next in setlist
    assert r.applied[-1] == "embers"
    assert r.center_toast().text == "4 - embers"    # slot badge in the toast
    r.press("]")
    assert r.applied[-1] == "abstract"
    r.press("[")
    assert r.applied[-1] == "embers"


def test_setlist_skips_names_that_no_longer_exist():
    r = Rig()
    r.ui.setlist = ["gone", "embers"]
    r.press("]")
    assert r.applied == ["embers"]


# ---------- layer / feature toggles ----------

@pytest.mark.parametrize("ch,attr", [
    ("a", "audio"), ("r", "record"), ("v", "video_bg"),
    ("f", "flock"), ("g", "glitch"),
])
def test_letter_toggles_flip_ui_attrs_with_toast(ch, attr):
    r = Rig()
    before = getattr(r.ui, attr)
    r.press(ch)
    assert getattr(r.ui, attr) is (not before)
    assert r.hud.toasts.active(), "silence-on-input is a bug"
    r.press(ch)
    assert getattr(r.ui, attr) is before


def test_shifted_letters_alias_unshifted():
    r = Rig()
    r.press("A")                            # case-folded routing, end to end
    assert r.ui.audio is True


def test_record_start_flashes_red():
    r = Rig()
    r.press("r")
    assert r.center_toast().text == "REC" and r.center_toast().color == H.RED


def test_panel_and_key_paths_share_state():
    """The panel toggles stay in sync: both paths set the same attr."""
    r = Rig()
    r.press("f")
    assert r.ui.flock is True
    r.ui._activate("flock", None, 0)        # panel click on the same toggle
    assert r.ui.flock is False


# ---------- i: debug ----------

def test_i_toggles_debug_status_variant():
    r = Rig()
    assert r.hud.debug is False
    r.press("i")
    assert r.hud.debug is True
    r.press("i")
    assert r.hud.debug is False


# ---------- unknown key ----------

def test_unknown_key_hints_question_mark():
    r = Rig()
    r.press("z")
    assert "? for keys" in r.hints()


# ---------- rename typing still consumes everything ----------

def test_no_global_keys_fire_while_renaming():
    r = Rig()
    r.ui.renaming = "mine"
    r.ui.rename_buf = "mine"
    # live_flow only routes keys the rename box did not consume
    for ch in ("q", "0", " ", "f", "1"):
        assert r.ui.on_key(ord(ch)) is True
    assert r.ps.quit is False and r.ps.blackout is False and r.applied == []
    assert r.ui.on_key(27) is True          # Esc cancels the rename only
    assert r.ui.renaming is None


# ---------- overlay states still reachable through the same path ----------

def test_tab_and_esc_route_before_the_registry():
    r = Rig()
    r.press(9)
    assert r.overlay is OverlayState.PANEL
    r.press(27)
    assert r.overlay is OverlayState.HUD
    r.press(27)
    assert r.overlay is OverlayState.HIDDEN
    r.press(27)                             # Esc in HIDDEN: nothing, never quits
    assert r.overlay is OverlayState.HIDDEN and r.ps.quit is False
