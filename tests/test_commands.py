"""Command registry — named commands, case-folded key routing, help table.

DESIGN.md §6.2: the table binds lowercase codes; shifted variants alias
unshifted. Table-driven where the contract is a mapping.
"""
import pytest

from dtouch.commands import Command, CommandRegistry


def _reg(*specs):
    reg = CommandRegistry()
    for name, key in specs:
        reg.add(name, name.split(".")[-1], key, lambda: None)
    return reg


# ---------- registration & lookup ----------

def test_register_and_lookup_by_name():
    reg = CommandRegistry()
    ran = []
    cmd = reg.register(Command("output.blackout", "Blackout", " ",
                               lambda: ran.append(1)))
    assert reg.get("output.blackout") is cmd
    assert reg.get("no.such") is None
    reg.dispatch(ord(" "))
    assert ran == [1]


def test_duplicate_name_refused():
    reg = _reg(("a.b", "a"))
    with pytest.raises(ValueError):
        reg.add("a.b", "again", "z", lambda: None)


def test_duplicate_key_refused_including_case_collision():
    reg = _reg(("a.b", "a"))
    with pytest.raises(ValueError):
        reg.add("c.d", "clash", "a", lambda: None)
    with pytest.raises(ValueError):
        reg.add("e.f", "shift-clash", "A", lambda: None)   # 'A' aliases 'a'


def test_multichar_key_refused():
    reg = CommandRegistry()
    with pytest.raises(ValueError):
        reg.add("a.b", "bad", "ab", lambda: None)


def test_unbound_command_is_allowed_and_not_in_table():
    reg = CommandRegistry()
    reg.add("internal.only", "Internal", None, lambda: None)
    assert reg.get("internal.only") is not None
    assert reg.table() == []


# ---------- case-folded key routing (table-driven) ----------

@pytest.mark.parametrize("bound,pressed,hits", [
    ("q", ord("q"), True),    # lowercase binds lowercase
    ("q", ord("Q"), True),    # shifted letter aliases unshifted
    ("Q", ord("q"), True),    # binding declared uppercase folds to lowercase
    ("Q", ord("Q"), True),
    ("[", ord("["), True),    # non-letters bind the exact character
    ("[", ord("{"), False),   # ...and shift variants of non-letters do NOT alias
    ("?", ord("?"), True),    # '?' is bound as itself (it IS the shifted key)
    ("?", ord("/"), False),
    ("0", ord("0"), True),
    (" ", ord(" "), True),
])
def test_key_routing(bound, pressed, hits):
    reg = CommandRegistry()
    ran = []
    reg.add("t.cmd", "T", bound, lambda: ran.append(1))
    assert reg.dispatch(pressed) is bool(hits)
    assert bool(ran) is bool(hits)


# ---------- unknown-key hook ----------

@pytest.mark.parametrize("code,fires", [
    (ord("z"), True),     # printable, unbound
    (ord("?"), True),
    (0, True),            # up arrow    (macOS 63232 & 0xFF)
    (1, True),            # down arrow  (63233 & 0xFF)
    (2, True),            # left arrow  (63234 & 0xFF)
    (3, True),            # right arrow (63235 & 0xFF)
    (13, True),           # Enter
    (10, True),           # LF
    (8, True),            # Backspace
    (127, True),          # Delete
    (9, False),           # TAB — not printable, caller's business
    (27, False),          # ESC
    (255, False),         # waitKey's no-key sentinel
])
def test_unknown_key_hook(code, fires):
    reg = _reg(("app.quit", "q"))
    unknown = []
    reg.on_unknown = unknown.append
    assert reg.dispatch(code) is False
    assert (unknown == [code]) is fires


def test_the_keys_people_press_first_get_an_answer():
    """The hook only fired for 32..126, so the arrows, Enter, Backspace and
    Delete did nothing ANYWHERE with no feedback at all — 40 of the 58 silent
    key x state cells per mode. Every printable unbound key hinted correctly,
    so the instrument was inconsistent rather than uniformly quiet, and the
    arrows and Enter are the first two things a child tries.

    Silence-on-input is a bug (DESIGN.md principle 4). These keys are still
    deliberately not load-bearing (§6.2 — their waitKey codes are
    platform-dependent); not load-bearing means they must SAY so, not vanish.
    """
    reg = _reg(("app.quit", "q"))
    answered = []
    reg.on_unknown = answered.append
    for code in (0, 1, 2, 3, 8, 10, 13, 127):
        reg.dispatch(code)
    assert answered == [0, 1, 2, 3, 8, 10, 13, 127]

    # ...and the two codes that must stay quiet still do: TAB and Esc are
    # consumed by the overlay stepper before dispatch is ever reached, and 255
    # is no keypress at all — hinting it would toast three times a second.
    answered.clear()
    for code in (9, 27, 255):
        reg.dispatch(code)
    assert answered == []


def test_bound_key_does_not_fire_unknown_hook():
    reg = _reg(("app.quit", "q"))
    unknown = []
    reg.on_unknown = unknown.append
    assert reg.dispatch(ord("q")) is True
    assert unknown == []


# ---------- help table ----------

def test_table_lists_bindings_in_registration_order():
    reg = CommandRegistry()
    reg.add("app.quit", "Quit", "q", lambda: None)
    reg.add("output.blackout", "Blackout", " ", lambda: None)
    reg.add("internal.only", "Internal", None, lambda: None)
    reg.add("preset.panic", "Panic reset", "0", lambda: None)
    assert reg.table() == [("q", "Quit"), (" ", "Blackout"), ("0", "Panic reset")]
