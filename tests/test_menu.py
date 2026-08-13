"""Home menu + boot card — DESIGN.md §3, migration step 8.

Menu is pure state (no window needed): selection movement with ','/'.',
Enter/Esc, direct letter + digit select-and-enter, disabled (reserved) cards
skipped everywhere, mouse-click commit through the drawn rects. draw_menu and
render_boot_card render onto plain numpy frames.
"""
import numpy as np

from dtouch.hud import DIM
from dtouch.menu import Card, Menu, draw_menu, registry_cards, render_boot_card
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


def test_unknown_keys_are_ignored():
    m = _menu()
    assert m.key(ord("z")) == (None, None)
    assert m.open is True and m.sel == 0


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


def test_draw_menu_survives_no_camera_frame():
    frame = np.full((720, 1280, 3), 99, np.uint8)
    draw_menu(frame, None, _cards(), 0)
    assert frame[0, 0].max() == 0                        # black ground, no crash


# ---------- boot card ----------

def test_boot_card_is_black_with_the_modes_accent():
    accent = (245, 140, 245)
    card = render_boot_card((640, 360), "Dither Girl", accent)
    assert card.shape == (360, 640, 3) and card.dtype == np.uint8
    assert card[0, 0].max() == 0                         # black ground
    flat = card.reshape(-1, 3)
    assert (flat == np.array(accent, np.uint8)).all(axis=1).any(), \
        "the glyph/name must render in the mode accent"
