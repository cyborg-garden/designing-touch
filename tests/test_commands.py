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
