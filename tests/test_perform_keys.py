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
                           safe_look=lambda: "abstract",
                           get_overlay=lambda: self.overlay)
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


SHIPPED_RES = [(1280, 720), (1920, 1080), (3840, 2160)]


@pytest.mark.parametrize("w,h", SHIPPED_RES)
@pytest.mark.parametrize("n", [2, 33, 60])
def test_the_key_map_never_runs_off_the_frame(w, h, n):
    """The map had one fixed column and no fit at all: at 33 rows the last two
    — `TAB Cycle overlay` and `Esc Step toward hidden` — had baselines BELOW
    the frame at 720p (754) and 1080p (1114), and three rows fell off at 4K.
    The two keys that walk you back out of an overlay state were the two the
    map could not show."""
    rows = [("%d" % i, "Row number %d" % i) for i in range(n - 2)]
    rows += [("TAB", "Cycle overlay"), ("Esc", "Step toward hidden")]
    _, title_px, px, placed = H.help_layout(w, h, rows)

    assert len(placed) == n, "every row is placed, none dropped"
    ix, iy = int(w * H.TITLE_SAFE), int(h * H.TITLE_SAFE)
    for key, label, kx, lx, y in placed:
        assert iy <= y <= h - iy, f"{key} baseline {y} outside the frame"
        assert kx >= ix
        assert lx + H.text_size(label, px)[0] <= w - ix

    # ...and what actually gets drawn stays inside the frame too
    img = np.zeros((h, w, 3), np.uint8)
    draw_help(img, rows)
    ink = np.any(img > 6, axis=2)
    ys, xs = np.where(ink)
    assert len(ys) and ys.max() < h and xs.max() < w


def test_the_key_map_holds_full_size_type_at_the_shipped_row_count():
    """Fitting must not mean shrinking to unreadable: at the shipped 33 rows
    the answer is a second column, not smaller type."""
    rows = [("%d" % i, "Row number %d" % i) for i in range(33)]
    for w, h in SHIPPED_RES:
        _, _, px, placed = H.help_layout(w, h, rows)
        assert px == int(0.9 * H.u(h)), "type shrank before columns were used"
        assert len({p[2] for p in placed}) == 2


def test_the_key_map_is_readable_over_a_1_bit_picture():
    """The scrim is a MULTIPLY, so it darkens the picture without flattening
    it: a 1-bit output is 0 vs 255, and 65% of that is 0 vs 89 — still hard
    edges, still full contrast, at the same spatial scale as the glyph
    strokes. The table stayed legible but fought the ground the whole time,
    and this is the screen that teaches the keys, so it has to read over ANY
    output. Measured on pure 1-bit noise, in a strip of the block's padding
    where no glyph ever lands: the ground swung 0..89 (std 44.5), giving white
    ink only 2.9x contrast against the brightest pixel it sat on.
    """
    rows = [("%d" % i, "Row number %d" % i) for i in range(18)]
    w, h = 1920, 1080
    rng = np.random.default_rng(1)
    img = np.repeat(((rng.random((h, w)) > 0.5) * 255).astype(np.uint8)[:, :, None],
                    3, axis=2)
    _org, _tpx, px, placed = H.help_layout(w, h, rows)
    kx = min(p[2] for p in placed)
    ys, ye = placed[0][4], placed[-1][4]
    strip = (slice(ys, ye), slice(kx - int(0.8 * px), kx - 3))

    draw_help(img, rows)
    ground = img[strip]
    assert ground.std() < 12.0, "the key table still sits on a 1-bit checkerboard"
    assert ground.max() < 70, f"brightest ground pixel {ground.max()} fights white ink"
    # ...and the plate is a plate, not a blackout: the picture is still there
    # around it, so help never looks like the instrument stopped.
    assert img[:ys // 2].std() > 20.0


def test_draw_help_takes_the_mode_accent():
    """DESIGN.md §5 one-accent rule: help renders in the ACTIVE mode's accent,
    not a hard-coded green."""
    a = np.full((360, 640, 3), 120, np.uint8)
    b = a.copy()
    draw_help(a, [("q", "Quit")])
    draw_help(b, [("q", "Quit")], accent=(245, 140, 245))
    assert not np.array_equal(a, b)


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


# ---------- s: save current look (PANEL-state only — edit action) ----------

def test_s_saves_only_in_panel_state():
    """Amended DESIGN.md §6.2: 's' = preset.save, PANEL state only — saving is
    an edit action; elsewhere it hints instead of silently doing nothing."""
    r = Rig()
    r.press("s")                            # HUD: gated
    assert r.ui.pending_save is False
    assert r.hud.toasts.active(), "silence-on-input is a bug"
    r.overlay = OverlayState.PANEL
    r.press("s")
    assert r.ui.pending_save is True


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


@pytest.mark.parametrize("code,name", [
    (0, "up"), (1, "down"), (2, "left"), (3, "right"),
    (13, "enter"), (10, "lf"), (8, "backspace"), (127, "delete"),
])
@pytest.mark.parametrize("state", [OverlayState.HIDDEN, OverlayState.HUD,
                                   OverlayState.PANEL])
def test_the_arrows_and_enter_answer_in_every_overlay_state(code, name, state):
    """These did nothing, anywhere, silently — the hint window was 32..126 and
    macOS masks the arrows to 0..3, Enter is 13, Backspace 8, Delete 127. They
    stay not-load-bearing (DESIGN.md §6.2); they just stop pretending they are
    not there."""
    r = Rig()
    r.overlay = state
    r.press(code)
    assert "? for keys" in r.hints(), f"{name} is silent in {state}"


# ---------- param nudging + OSD (DESIGN.md §6.2) ----------

def _nudgeables(ui):
    from dtouch.panelspec import nudgeable
    return [w for w in ui.iter_widgets() if nudgeable(w)]


def test_the_nudge_keys_cannot_reach_the_output_resolution():
    """`.` `.` `=` used to resize the live window (DESIGN.md §6.2 nudging).

    `output` is a Cycle, so it was simply the 2nd of 23 stops in Particles and
    the 3rd of 15 in Dither Girl — two keys from the default selection, with no
    panel open. One `=` there recreated the window at 4K and took the frame
    rate with it, and `0` (panic) could not put it back: panic restores the
    mode's look, and the window size is not in the look. Nothing else the nudge
    keys can reach is unrecoverable like that, so nothing else opts out."""
    from dtouch.panelspec import Cycle, nudgeable, walk_spec
    from dtouch.modes.dithergirl import DitherGirlMode

    both = (("particles", list(Rig().ui.iter_widgets())),
            ("dithergirl", [w for _s, w in
                            walk_spec(DitherGirlMode().panel_spec())]))
    for label, widgets in both:
        cycles = [w for w in widgets if isinstance(w, Cycle)]
        assert "res_idx" in [w.attr for w in cycles], \
            f"{label}: the control must still exist on the edit surface"
        assert "res_idx" not in [w.attr for w in widgets if nudgeable(w)], \
            f"{label}: a bare key still resizes the window"
        # ...and it is the ONLY opt-out: everything else a bare key can reach,
        # a bare key can also take back.
        assert [w.attr for w in cycles if not nudgeable(w)] == ["res_idx"], \
            f"{label}: something else quietly left the nudge walk"


def test_nudge_selection_walks_the_spec_order_and_wraps():
    r = Rig()
    ws = _nudgeables(r.ui)
    assert r.ui.nudge_idx == 0
    r.press(".")
    assert r.ui.nudge_idx == 1
    assert r.hud.toasts is r.hud.toasts     # osd, not a toast:
    assert r.hud.osd._show is not None
    assert r.hud.osd._show[0] == ws[1].label
    r.press(",")
    r.press(",")
    assert r.ui.nudge_idx == len(ws) - 1    # wrapped backwards
    assert r.hud.osd._show[0] == ws[-1].label


def test_nudge_math_1_40th_of_range_x5_and_clamped():
    r = Rig()
    ws = _nudgeables(r.ui)
    i = next(i for i, w in enumerate(ws) if getattr(w, "attr", "") == "fade")
    r.ui.nudge_idx = i
    w = ws[i]
    step = (w.hi - w.lo) / 40.0
    r.ui.fade = 0.90
    r.press("=")
    assert r.ui.fade == pytest.approx(0.90 + step)
    r.press("_")                            # x5 down
    assert r.ui.fade == pytest.approx(0.90 + step - 5 * step)
    r.ui.fade = w.hi
    r.press("+")
    assert r.ui.fade == w.hi                # clamped at the top
    r.ui.fade = w.lo
    r.press("-")
    assert r.ui.fade == w.lo                # clamped at the bottom


def test_nudge_rotates_a_cycle_by_one_option_even_x5():
    r = Rig()
    ws = _nudgeables(r.ui)
    i = next(i for i, w in enumerate(ws) if getattr(w, "attr", "") == "matte_idx")
    r.ui.nudge_idx = i
    r.ui.matte_idx = 0
    r.press("=")
    assert r.ui.matte_idx == 1
    r.press("+")                            # cycles rotate by ONE, x5 or not
    assert r.ui.matte_idx == 2
    r.press("-")
    assert r.ui.matte_idx == 1
    r.ui.matte_idx = 0
    r.press("_")
    assert r.ui.matte_idx == len(MATTES) - 1   # wraps backwards
    # cycle OSD shows the option name, no bar
    name, text, fill, _ = r.hud.osd._show
    assert text == MATTES[r.ui.matte_idx] and fill is None


def test_nudge_osd_shows_name_value_and_bar_for_sliders():
    r = Rig()
    ws = _nudgeables(r.ui)
    i = next(i for i, w in enumerate(ws) if getattr(w, "attr", "") == "fade")
    r.ui.nudge_idx = i
    r.press("=")
    name, text, fill, _ = r.hud.osd._show
    assert name == "Trails"
    assert text == f"{r.ui.fade:.2f}"
    assert fill is not None and 0.0 <= fill <= 1.0


def test_nudging_works_in_hidden_and_the_osd_draws_there():
    r = Rig()
    r.press(9); r.press(27); r.press(27)     # to PANEL then back to HIDDEN
    assert r.overlay is OverlayState.HIDDEN
    before = r.ui.matte_idx
    r.press("=")
    assert r.ui.matte_idx == before + 1      # the key worked in HIDDEN
    img = np.zeros((360, 640, 3), np.uint8)
    r.hud.draw(img, OverlayState.HIDDEN)
    assert img.any(), "the OSD must draw in HIDDEN too"
    r.clock.t += 10.0                        # fades out -> provably clean again
    img2 = np.zeros((360, 640, 3), np.uint8)
    r.hud.draw(img2, OverlayState.HIDDEN)
    assert not img2.any()


def test_nudge_selection_resets_on_spec_rebind():
    """Mode switches rebind the spec (Host.set_mode -> ui.set_spec); the
    nudge selection must reset sanely with it."""
    from dtouch.overlay_ui import build_particles_spec
    r = Rig()
    r.ui.nudge_idx = 7
    r.ui.set_spec(build_particles_spec(PRESETS, list(PALETTES), MATTES))
    assert r.ui.nudge_idx == 0


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
