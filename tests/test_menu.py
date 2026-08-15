"""Home menu + boot card — DESIGN.md §3, migration step 8.

Menu is pure state (no window needed): selection movement with ','/'.',
Enter/Esc, direct letter + digit select-and-enter, disabled (reserved) cards
skipped everywhere, mouse-click commit through the drawn rects. draw_menu and
render_boot_card render onto plain numpy frames.
"""
import numpy as np
import pytest

from dtouch.hud import DIM, TITLE_SAFE, Toasts
from dtouch.menu import (Card, Menu, draw_menu, hint_baseline, menu_hint,
                         registry_cards, render_boot_card)
from dtouch.modes import REGISTRY


def _cards():
    return [
        Card("particles", "Particles", "webcam-matte\ninstrument", (120, 215, 140), "p"),
        Card("dithergirl", "Dither Girl", "live + still\ndithering", (245, 140, 245), "d"),
        Card(None, "FLOCKING", "coming soon", DIM, None, enabled=False),
    ]


def _menu():
    m = Menu(cards=_cards())
    m.show()
    return m


# ---------- cards from the registry ----------

def test_registry_cards_cover_registry_plus_reserved_flocking():
    cards = registry_cards()
    assert [c.id for c in cards[:-1]] == [m.id for m in REGISTRY]
    last = cards[-1]
    assert last.title == "FLOCKING" and last.blurb == "coming soon"
    assert last.enabled is False and last.id is None and last.key is None


def test_card_strings_are_ascii_only():
    """Hershey fonts render non-ASCII as '?' — 'coming soon' card uses '-' not
    an em-dash, and every card string must be ASCII."""
    for c in registry_cards():
        for s in (c.title, c.blurb, c.key or ""):
            assert s == s.encode("ascii", "replace").decode()


# ---------- selection movement ----------

def test_comma_period_move_selection_and_wrap():
    m = _menu()
    assert m.sel == 0
    m.key(ord("."))
    assert m.sel == 1
    m.key(ord("."))                      # flocking disabled -> wraps to 0
    assert m.sel == 0
    m.key(ord(","))                      # back over the disabled card
    assert m.sel == 1


def test_show_selects_the_active_mode_card():
    m = Menu(cards=_cards())
    m.show(active_id="dithergirl")
    assert m.sel == 1


# ---------- enter / esc ----------

def test_enter_commits_selected_card_and_closes():
    m = _menu()
    m.key(ord("."))
    assert m.key(13) == ("switch", "dithergirl")
    assert m.open is False


def test_esc_closes_without_committing():
    m = _menu()
    assert m.key(27) == ("close", None)
    assert m.open is False


def test_m_closes_too():
    m = _menu()
    assert m.key(ord("m")) == ("close", None)


# ---------- the boot menu (DESIGN.md §3, amended 2026-08-15) ----------

def test_show_marks_the_boot_menu_and_close_clears_it():
    m = Menu(cards=_cards())
    m.show(active_id="dithergirl", boot=True)
    assert m.boot is True and m.sel == 1
    m.close()
    assert m.boot is False
    m.show()                                  # a later open is not a boot menu
    assert m.boot is False


def test_esc_at_boot_commits_the_selection_instead_of_closing():
    """At boot there is no 'running mode' behind the menu to return to, so a
    dismiss that just closed would leave the operator on a mode they never
    chose — the dead end principle 5 forbids."""
    m = Menu(cards=_cards())
    m.show(active_id="particles", boot=True)
    assert m.key(27) == ("switch", "particles")
    assert m.open is False and m.boot is False


def test_m_at_boot_commits_too():
    m = Menu(cards=_cards())
    m.show(active_id="dithergirl", boot=True)
    assert m.key(ord("m")) == ("switch", "dithergirl")


def test_esc_at_boot_commits_what_the_operator_moved_to():
    m = Menu(cards=_cards())
    m.show(active_id="particles", boot=True)
    m.key(ord("."))
    assert m.key(27) == ("switch", "dithergirl")


def test_q_at_boot_still_quits_rather_than_committing():
    """q is 'leave', not 'pick' — the boot menu must not turn it into an
    entry (DESIGN.md §3: q closes the menu and arms the quit confirm)."""
    m = Menu(cards=_cards())
    m.show(boot=True)
    assert m.key(ord("q")) == ("quit", None)
    assert m.open is False


# ---------- direct keys ----------

def test_mode_letter_selects_and_enters():
    m = _menu()
    assert m.key(ord("d")) == ("switch", "dithergirl")
    assert m.open is False
    m.show()
    assert m.key(ord("P")) == ("switch", "particles")   # case-folded


def test_digit_selects_and_enters():
    m = _menu()
    assert m.key(ord("2")) == ("switch", "dithergirl")


def test_disabled_card_cannot_be_committed():
    m = _menu()
    assert m.key(ord("3")) == (None, None)               # flocking digit
    assert m.open is True
    assert m.key(ord("9")) == (None, None)               # out-of-range digit


def test_unknown_keys_report_unknown_for_the_hint_toast():
    """Amended DESIGN.md §3: silence-on-input is a bug inside the menu too —
    unknown keys surface as ("unknown", None) so the shell can hint."""
    m = _menu()
    assert m.key(ord("z")) == ("unknown", None)
    assert m.open is True and m.sel == 0


def test_q_closes_menu_and_reports_quit():
    """Amended DESIGN.md §3: q closes the menu AND arms the quit confirm."""
    m = _menu()
    assert m.key(ord("q")) == ("quit", None)
    assert m.open is False
    m.show()
    assert m.key(ord("Q")) == ("quit", None)             # case-folded


# ---------- mouse ----------

def test_click_commits_the_card_under_the_pointer():
    m = _menu()
    frame = np.zeros((720, 1280, 3), np.uint8)
    cam = np.full((36, 64, 3), 128, np.uint8)
    m.rects = draw_menu(frame, cam, m.cards, m.sel)
    (x0, y0, x1, y1), card = m.rects[1]
    assert m.click(((x0 + x1) // 2, (y0 + y1) // 2)) == "dithergirl"
    assert m.open is False


def test_click_on_disabled_card_or_background_does_nothing():
    m = _menu()
    frame = np.zeros((720, 1280, 3), np.uint8)
    m.rects = draw_menu(frame, np.full((36, 64, 3), 128, np.uint8), m.cards, m.sel)
    (x0, y0, x1, y1), card = m.rects[2]                  # flocking
    assert m.click(((x0 + x1) // 2, (y0 + y1) // 2)) is None
    assert m.click((5, 5)) is None
    assert m.open is True


# ---------- rendering ----------

def test_draw_menu_dithers_and_dims_the_camera_under_a_scrim():
    """DESIGN.md §3: live camera through 1-bit blue-noise dither, dimmed under
    a 65% scrim — a bright camera frame ends up dark, and the dithered ground
    holds only the two dimmed levels (plus the drawn chrome)."""
    frame = np.zeros((720, 1280, 3), np.uint8)
    cam = np.full((720, 1280, 3), 220, np.uint8)
    rects = draw_menu(frame, cam, _cards(), 0)
    assert len(rects) == 3
    assert frame.mean() < 90                             # 65% scrim took it down
    assert frame.max() <= 255 and frame.max() > 0


def test_bottom_hint_line_renders():
    """Amended DESIGN.md §3: the menu's bottom hint line
    ', . move - enter select - esc back' is part of the layout."""
    frame = np.zeros((720, 1280, 3), np.uint8)
    draw_menu(frame, None, _cards(), 0)
    y = hint_baseline(720)
    band = frame[y - 20:y + 6, 300:980]                  # bottom-center strip
    assert band.max() > 0, "the hint line must render bottom-center"


def test_boot_menu_hint_names_quit_instead_of_esc_back():
    """At boot Esc does not go 'back' — it enters the selection — so the line
    may not say it does. `q` is the other door, and it is the true one."""
    assert menu_hint(boot=False) == ", . move - enter select - esc back"
    assert menu_hint(boot=True) == ", . move - enter select - q quit"
    for s in (menu_hint(True), menu_hint(False)):
        assert s == s.encode("ascii", "replace").decode()   # Hershey is ASCII


def test_draw_menu_boot_flag_reaches_the_hint_line(monkeypatch):
    import dtouch.menu as M
    texts = []
    real = M.put_outlined

    def spy(img, text, *a, **k):
        texts.append(text)
        return real(img, text, *a, **k)
    monkeypatch.setattr(M, "put_outlined", spy)
    draw_menu(np.zeros((720, 1280, 3), np.uint8), None, _cards(), 0, boot=True)
    assert menu_hint(boot=True) in texts


@pytest.mark.parametrize("res", [(1280, 720), (1920, 1080), (3840, 2160)])
@pytest.mark.parametrize("n_hints", [1, 2, 3])
def test_menu_hint_never_shares_a_row_with_a_hud_hint_toast(res, n_hints):
    """Issue #18, at every resolution and every stack depth.

    The HUD stacks its hint toasts UP from the title-safe bottom inset, which
    is exactly where the menu's own hint line used to sit: at boot the doors
    hint and the menu hint were drawn centered on one baseline, so the first
    frame of the app showed two messages smeared into one unreadable line.

    Pixels, not coordinates: whatever rows the menu's hint paints, no toast in
    a full 3-deep stack may paint any of them."""
    w, h = res
    menu = np.zeros((h, w, 3), np.uint8)
    draw_menu(menu, None, _cards(), 0)
    y = hint_baseline(h)
    band = slice(max(0, y - int(0.08 * h)), h)           # bottom band only
    menu_rows = {r for r in range(band.start, h) if menu[r].any()}
    assert menu_rows, "the menu hint must actually paint something"

    toasts = np.zeros((h, w, 3), np.uint8)
    t = Toasts(now=lambda: 0.0)
    for text in ["m menu - TAB panel - ? keys", "enter resumes PARTICLES",
                 "q again to quit"][:n_hints]:
        t.hint(text, ttl=4.0)
    t.draw(toasts)
    toast_rows = {r for r in range(band.start, h) if toasts[r].any()}
    assert toast_rows, "the toast stack must actually paint something"

    assert not (menu_rows & toast_rows), (
        "menu hint and HUD toasts overprint on rows "
        f"{sorted(menu_rows & toast_rows)}")


def test_menu_hint_stays_inside_the_title_safe_box():
    """DESIGN.md §5: nothing crosses the inset — the lift may not push the
    line up out of the bottom of the frame either."""
    for h in (360, 720, 1080, 2160):
        y = hint_baseline(h)
        assert int(h * 0.5) < y <= h - int(h * TITLE_SAFE)


def test_draw_menu_survives_no_camera_frame():
    frame = np.full((720, 1280, 3), 99, np.uint8)
    draw_menu(frame, None, _cards(), 0)
    assert frame[0, 0].max() == 0                        # black ground, no crash


# ---------- boot card ----------

def test_card_titles_render_in_caps(monkeypatch):
    """DESIGN.md §3: the real mode cards render their titles in CAPS."""
    import dtouch.menu as M
    texts = []
    real = M.put_outlined

    def spy(img, text, *a, **k):
        texts.append(text)
        return real(img, text, *a, **k)
    monkeypatch.setattr(M, "put_outlined", spy)
    draw_menu(np.zeros((360, 640, 3), np.uint8), None, registry_cards(), 0)
    assert "PARTICLES" in texts and "DITHER GIRL" in texts
    assert "Particles" not in texts and "Dither Girl" not in texts


def test_reserved_card_border_is_dashed():
    """DESIGN.md §3 sketch: the FLOCKING card's frame is dashed (and dimmed);
    real cards keep a solid border."""
    img = np.zeros((360, 640, 3), np.uint8)
    rects = draw_menu(img, None, registry_cards(), 0)
    solid = next(r for r, c in rects if c.enabled and c.id == "dithergirl")
    dashed = next(r for r, c in rects if not c.enabled)

    def runs(rect):
        x0, y0, x1, _ = rect
        row = img[y0, x0:x1 + 1].any(axis=1)
        return row

    assert runs(solid).all(), "enabled card top border must be continuous"
    d = runs(dashed)
    assert d.any() and not d.all(), "reserved card top border must have gaps"


def test_boot_card_is_black_with_the_modes_accent():
    accent = (245, 140, 245)
    card = render_boot_card((640, 360), "Dither Girl", accent)
    assert card.shape == (360, 640, 3) and card.dtype == np.uint8
    assert card[0, 0].max() == 0                         # black ground
    flat = card.reshape(-1, 3)
    assert (flat == np.array(accent, np.uint8)).all(axis=1).any(), \
        "the glyph/name must render in the mode accent"
