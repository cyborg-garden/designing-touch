"""Preset management — delete / rename user looks, and saves capture video + audio state."""
import cv2
import numpy as np

from dtouch import presets
from dtouch.overlay_ui import OverlayUI
from dtouch.particles import PALETTES
from dtouch.live import MATTES

CFG = dict(matte="auto", palette="fire", fade=0.91, exposure=1.7, spark=0.4,
           curl_amp=0.5, reseed_frac=0.06, base_size=0.012)


# ---------- presets.py ----------

def test_delete_user_preset(tmp_path):
    path = str(tmp_path / "p.json")
    presets.save("mine", CFG, path=path)
    assert presets.delete("mine", path=path) is True
    loaded = presets.load(path)
    assert "mine" not in loaded
    assert "abstract" in loaded                      # built-ins untouched


def test_delete_builtin_refused(tmp_path):
    path = str(tmp_path / "p.json")
    assert presets.delete("abstract", path=path) is False
    assert "abstract" in presets.load(path)


def test_rename_user_preset(tmp_path):
    path = str(tmp_path / "p.json")
    presets.save("mine_1234", CFG, path=path)
    assert presets.rename("mine_1234", "neon dancer", path=path) is True
    loaded = presets.load(path)
    assert "mine_1234" not in loaded
    assert loaded["neon dancer"]["palette"] == "fire"


def test_rename_refuses_collisions_and_builtins(tmp_path):
    path = str(tmp_path / "p.json")
    presets.save("a", CFG, path=path)
    presets.save("b", CFG, path=path)
    assert presets.rename("a", "b", path=path) is False         # user collision
    assert presets.rename("a", "abstract", path=path) is False  # builtin collision
    assert presets.rename("abstract", "z", path=path) is False  # builtins immune
    assert presets.rename("a", "  ", path=path) is False        # empty after strip


def test_save_captures_video_and_audio_state(tmp_path):
    path = str(tmp_path / "p.json")
    cfg = dict(CFG, video_bg=True, video_mix=0.7, audio=True, sens=1.8)
    presets.save("vid", cfg, path=path)
    loaded = presets.load(path)["vid"]
    assert loaded["video_bg"] is True and abs(loaded["video_mix"] - 0.7) < 1e-9
    assert loaded["audio"] is True and abs(loaded["sens"] - 1.8) < 1e-9


def test_builtins_do_not_carry_video_keys():
    # absence is the contract: loading a builtin must NOT reset the live video toggle
    for cfg in presets.BUILTIN.values():
        assert "video_bg" not in cfg and "audio" not in cfg


def test_user_names(tmp_path):
    path = str(tmp_path / "p.json")
    presets.save("mine", CFG, path=path)
    assert presets.user_names(path) == {"mine"}
    assert presets.user_names(str(tmp_path / "missing.json")) == set()


# ---------- panel UI ----------

def _ui(user=("mine",)):
    ui = OverlayUI(1920, 1080, ["abstract", "sigil"] + list(user), list(PALETTES), MATTES)
    ui.user_presets = set(user)
    ui.draw(np.zeros((1080, 1920, 3), np.uint8), {"status": ""})
    return ui


def _row_rect(ui, name):
    idx = ui.presets.index(name)
    return next(r for r, k, p in ui._hot if k == "preset" and p == idx)


def _hover(ui, name):
    r = _row_rect(ui, name)
    ui.mouse = ((r[0] + r[2]) // 2, (r[1] + r[3]) // 2)
    ui.draw(np.zeros((1080, 1920, 3), np.uint8), {"status": ""})


def _click(ui, kind, name):
    rect = next(r for r, k, p in ui._hot if k == kind and p == name)
    ui.on_mouse(cv2.EVENT_LBUTTONDOWN, (rect[0] + rect[2]) // 2,
                (rect[1] + rect[3]) // 2, 0)


def test_buttons_only_on_hovered_user_presets():
    ui = _ui()
    kinds = {k for _, k, _ in ui._hot}
    assert "del" not in kinds and "ren" not in kinds    # nothing hovered
    _hover(ui, "mine")
    assert any(k == "del" for _, k, _ in ui._hot)
    assert any(k == "ren" for _, k, _ in ui._hot)
    _hover(ui, "abstract")                              # builtin: never
    assert not any(k in ("del", "ren") for _, k, _ in ui._hot)


def test_delete_is_two_click():
    ui = _ui()
    _hover(ui, "mine")
    _click(ui, "del", "mine")
    assert ui.pending_delete is None                    # armed, not deleted
    _hover(ui, "mine")
    _click(ui, "del", "mine")
    assert ui.pending_delete == "mine"


def test_delete_disarms_on_other_click():
    ui = _ui()
    _hover(ui, "mine")
    _click(ui, "del", "mine")
    ui._activate("preset", 0, 0)                        # click something else
    _hover(ui, "mine")
    _click(ui, "del", "mine")
    assert ui.pending_delete is None                    # re-armed only


def test_rename_typing_flow():
    ui = _ui()
    _hover(ui, "mine")
    _click(ui, "ren", "mine")
    assert ui.typing
    for ch in "":  # clear prefill via backspaces
        pass
    for _ in range(len(ui.rename_buf)):
        ui.on_key(8)
    for ch in "neon":
        assert ui.on_key(ord(ch)) is True               # consumed while typing
    assert ui.on_key(ord("q")) is True                  # q must NOT leak to quit
    for _ in range(1):
        ui.on_key(8)                                    # backspace the q
    ui.on_key(13)                                       # enter commits
    assert ui.pending_rename == ("mine", "neon")
    assert not ui.typing


def test_rename_escape_cancels():
    ui = _ui()
    _hover(ui, "mine")
    _click(ui, "ren", "mine")
    ui.on_key(ord("x"))
    ui.on_key(27)
    assert not ui.typing and ui.pending_rename is None


def test_keys_ignored_when_not_typing():
    ui = _ui()
    assert ui.on_key(ord("q")) is False


# ---------- slot badges (DESIGN.md §4.1 / §6.3, step 7) ----------

def test_every_preset_row_has_a_slot_badge_hit():
    ui = _ui()
    names = [p for _, k, p in ui._hot if k == "slot"]
    assert names == list(reversed(ui.presets))      # front-inserted, all rows


def test_clicking_a_badge_posts_pending_slot():
    ui = _ui()
    _click(ui, "slot", "mine")
    assert ui.pending_slot == "mine"


def test_badge_wins_over_the_row_it_sits_on():
    """The badge rect overlaps the full-row preset hit; the badge must win."""
    ui = _ui()
    rect = next(r for r, k, p in ui._hot if k == "slot" and p == "abstract")
    before = ui.pending_preset
    ui.on_mouse(cv2.EVENT_LBUTTONDOWN, (rect[0] + rect[2]) // 2,
                (rect[1] + rect[3]) // 2, 0)
    assert ui.pending_slot == "abstract"
    assert ui.pending_preset == before              # the row click did NOT fire


def test_assigned_badge_shows_and_survives_redraw():
    ui = _ui()
    ui.bank = {"3": "mine"}
    ui.draw(np.zeros((1080, 1920, 3), np.uint8), {"status": ""})
    assert any(k == "slot" for _, k, _ in ui._hot)
