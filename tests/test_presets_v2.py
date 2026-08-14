"""Preset v2 store — per-mode looks/bank/setlist, v1 migration + backup
(DESIGN.md §7, §8 step 7; §9: migration is the one user-data-loss surface).

The v1 fixture is the shipped presets.json content, frozen in
tests/fixtures/presets_v1.json. Round-trip pins (v1 look → identical engine
state pre/post migration) live in test_preset_capture.py next to the
spec-derived capture/apply they exercise.
"""
import json
import os

import pytest

from dtouch import presets


@pytest.fixture(autouse=True)
def _clean_store_globals():
    """`_no_overwrite` and `_notes` are process globals — never let one test's
    poisoned path or queued note leak into the next."""
    yield
    presets._no_overwrite.clear()
    presets.take_notes()


def _make_unreadable(path):
    """A file that EXISTS but cannot be read (the permissions case, distinct
    from unparseable bytes). Returns a restore callable."""
    mode = os.stat(path).st_mode
    os.chmod(path, 0)
    return lambda: os.chmod(path, mode)


unreadable_supported = pytest.mark.skipif(
    os.geteuid() == 0, reason="root reads through a chmod 0 file")

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "presets_v1.json")
CFG = dict(matte="auto", palette="fire", fade=0.91, exposure=1.7, spark=0.4,
           curl_amp=0.5, reseed_frac=0.06, base_size=0.012)


def _v1_fixture():
    with open(FIXTURE) as f:
        return json.load(f)


def _write_v1(tmp_path, data=None):
    path = str(tmp_path / "presets.json")
    with open(path, "w") as f:
        json.dump(data if data is not None else _v1_fixture(), f, indent=2)
    return path


# ---------- v1 detection + in-memory migration ----------

def test_v1_file_detected_and_migrated_in_memory():
    raw = _v1_fixture()
    assert presets.is_v1(raw)
    v2 = presets.migrate_v1(raw)
    assert v2["version"] == 2
    assert set(v2["modes"]["particles"]["looks"]) == set(raw)


def test_v2_and_empty_files_are_not_v1(tmp_path):
    assert not presets.is_v1({})
    assert not presets.is_v1({"version": 2, "modes": {}})
    missing = str(tmp_path / "missing.json")
    assert presets.load_file(missing) == {"version": 2, "meta": {}, "modes": {}}


def test_migration_filters_through_v1_keys():
    """v1 load() filtered through KEYS; migration must be byte-equivalent —
    hand-edited junk keys don't survive into v2."""
    raw = {"mine": dict(CFG, junk=1, glitch=True)}
    looks = presets.migrate_v1(raw)["modes"]["particles"]["looks"]
    assert looks["mine"] == CFG                 # junk + unpersisted fx dropped


def test_load_over_v1_file_matches_v1_semantics(tmp_path):
    """The flat API over a v1 file returns exactly what v1 load() returned:
    built-ins merged under the user looks."""
    path = _write_v1(tmp_path)
    loaded = presets.load(path)
    for name in presets.BUILTIN:
        assert name in loaded
    for name, cfg in _v1_fixture().items():
        assert loaded[name] == {k: cfg[k] for k in presets.KEYS if k in cfg}


# ---------- first write: backup, then v2 on disk ----------

def test_first_save_backs_up_v1_then_writes_v2(tmp_path):
    path = _write_v1(tmp_path)
    original = _v1_fixture()
    presets.save("mine_new", CFG, path=path)

    bak = presets.backup_path(path)
    assert bak.endswith("presets.v1.bak.json")
    with open(bak) as f:
        assert json.load(f) == original         # byte-for-byte v1 content
    with open(path) as f:
        on_disk = json.load(f)
    assert on_disk["version"] == 2
    looks = on_disk["modes"]["particles"]["looks"]
    assert looks["mine_new"] == CFG
    assert set(original) <= set(looks)          # migrated looks carried over


def test_backup_written_only_once(tmp_path):
    path = _write_v1(tmp_path)
    presets.save("a", CFG, path=path)
    bak = presets.backup_path(path)
    before = os.path.getmtime(bak)
    with open(bak) as f:
        content = json.load(f)
    presets.save("b", CFG, path=path)           # second write: file already v2
    assert os.path.getmtime(bak) == before
    with open(bak) as f:
        assert json.load(f) == content


def test_v2_save_load_round_trips_nested_signal(tmp_path):
    path = str(tmp_path / "p.json")
    cfg = dict(CFG, signal={"glitch": True, "dither": "blue", "chroma": 22.0})
    presets.save("glitched", cfg, path=path)
    loaded = presets.load(path)["glitched"]
    assert loaded["signal"] == {"glitch": True, "dither": "blue", "chroma": 22.0}


def test_modes_are_isolated(tmp_path):
    path = str(tmp_path / "p.json")
    presets.save("mine", CFG, path=path, mode="particles")
    presets.save("girl", CFG, path=path, mode="dithergirl")
    assert presets.user_names(path, mode="particles") == {"mine"}
    assert presets.user_names(path, mode="dithergirl") == {"girl"}
    assert "girl" not in presets.load(path, mode="particles")


# ---------- corrupt file: backup + note, atomic writes (DESIGN.md §9) ----------

def test_corrupt_file_backed_up_with_note_and_survives_saves(tmp_path):
    path = str(tmp_path / "presets.json")
    garbage = "{not json"
    with open(path, "w") as f:
        f.write(garbage)
    loaded = presets.load(path)                 # built-ins still served
    assert "abstract" in loaded
    baks = list(tmp_path.glob("presets.corrupt.*.bak.json"))
    assert len(baks) == 1
    assert baks[0].read_text() == garbage       # original content preserved
    notes = presets.take_notes()
    assert notes and "backup" in notes[0]
    # a subsequent save proceeds with the empty store, never touches the backup
    presets.save("mine", CFG, path=path)
    assert baks[0].read_text() == garbage
    assert presets.load(path)["mine"] == CFG
    assert presets.take_notes() == []           # healthy file: no further notes


def test_corrupt_backup_not_duplicated_on_repeated_reads(tmp_path):
    path = str(tmp_path / "presets.json")
    with open(path, "w") as f:
        f.write("[broken")
    presets.load(path)
    presets.load(path)
    presets.user_names(path)
    assert len(list(tmp_path.glob("presets.corrupt.*.bak.json"))) == 1
    presets.take_notes()


def test_failed_backup_blocks_the_overwrite_and_says_so_loudly(tmp_path,
                                                               monkeypatch):
    """Backup copy fails (disk full) => the corrupt file is the user's ONLY
    copy. Saving over it would be silent unrecoverable data loss, so the write
    is refused with a note instead (DESIGN.md §9)."""
    path = str(tmp_path / "presets.json")
    garbage = '{"embers": {"fade": 0.93}'          # truncated but recoverable
    with open(path, "w") as f:
        f.write(garbage)

    def no_space(src, dst):
        raise OSError(28, "No space left on device")
    monkeypatch.setattr(presets.shutil, "copy2", no_space)
    presets.load(path)
    notes = presets.take_notes()
    assert any("NOT saving over it" in n for n in notes)
    assert not list(tmp_path.glob("presets.corrupt.*.bak.json"))

    presets.save("mine", CFG, path=path)           # no-op, never raises
    with open(path) as f:
        assert f.read() == garbage                 # the only copy survives
    assert any("not saved" in n for n in presets.take_notes())

    # …and once the disk frees up, the backup lands and the save goes through
    monkeypatch.undo()
    presets.save("mine", CFG, path=path)
    assert len(list(tmp_path.glob("presets.corrupt.*.bak.json"))) == 1
    assert presets.load(path)["mine"] == CFG
    presets.take_notes()


def test_partial_previous_backup_does_not_count_as_backed_up(tmp_path):
    """A copy2 that died midway leaves a short file. Its bytes differ, but the
    dedup must not need a full compare to notice — a size mismatch is enough,
    and the retry lands under a fresh name instead of clobbering it."""
    path = str(tmp_path / "presets.json")
    garbage = "{broken but long enough to truncate"
    with open(path, "w") as f:
        f.write(garbage)
    # a half-written backup, named for THIS second so the retry's default name
    # collides with it — the retry must not clobber the only artifact there is
    stamp = presets.time.strftime("%Y%m%d_%H%M%S")
    partial = tmp_path / ("presets.corrupt.%s.bak.json" % stamp)
    partial.write_text(garbage[:5])

    presets.load(path)
    baks = sorted(tmp_path.glob("presets.corrupt.*.bak.json"))
    assert len(baks) == 2                          # a real backup was attempted
    assert partial.read_text() == garbage[:5]      # the partial is not clobbered
    assert any(b.read_text() == garbage for b in baks)
    assert not presets._is_poisoned(path)
    presets.take_notes()


def test_write_is_atomic_old_content_survives_replace_failure(tmp_path,
                                                              monkeypatch):
    path = str(tmp_path / "p.json")
    presets.save("mine", CFG, path=path)
    with open(path) as f:
        old = f.read()

    def boom(src, dst):
        raise OSError("disk full")
    monkeypatch.setattr(presets.os, "replace", boom)
    with pytest.raises(OSError):
        presets.save("other", CFG, path=path)
    monkeypatch.undo()
    with open(path) as f:
        assert f.read() == old                  # old content intact, never truncated
    assert not list(tmp_path.glob("*.tmp"))     # temp file cleaned up


# ---------- write_file's boolean is the whole contract ----------

def test_write_file_returns_true_on_a_real_write(tmp_path):
    """It used to fall off the end returning None while its docstring promised
    True, so every caller reported success for a write that never happened."""
    path = str(tmp_path / "p.json")
    assert presets.write_file(presets._empty_v2(), path) is True
    assert os.path.exists(path)


def test_every_store_write_propagates_the_refusal(tmp_path, monkeypatch):
    """save/delete/rename/set_bank/set_setlist all hand back write_file's
    verdict — a poisoned file must not be reported as a successful edit."""
    path = str(tmp_path / "presets.json")
    presets.save("mine", CFG, path=path)
    presets.set_bank({"1": "mine"}, path=path)
    presets.take_notes()

    with open(path, "a") as f:
        f.write("<<<truncated")                   # now unparseable...

    def no_space(src, dst):
        raise OSError(28, "No space left on device")
    monkeypatch.setattr(presets.shutil, "copy2", no_space)   # ...and unbackupable

    assert presets.save("other", CFG, path=path) is False
    assert presets.set_bank({"2": "x"}, path=path) is False
    assert presets.set_setlist(["x"], path=path) is False
    # delete/rename read the (unusable) file first, so they refuse earlier —
    # what matters is that neither claims to have changed anything
    assert presets.delete("mine", path=path) is False
    assert presets.rename("mine", "neon", path=path) is False


# ---------- unreadable (not just unparseable) is the same data-loss event ----------

@unreadable_supported
def test_unreadable_file_is_poisoned_and_never_overwritten(tmp_path):
    """A presets.json that EXISTS but cannot be read used to slip past the
    corrupt-file recovery entirely (`_corrupt_backup` early-returned on the
    OSError before poisoning), so the next save silently destroyed every saved
    look with no backup and no warning."""
    path = str(tmp_path / "presets.json")
    presets.save("mine", CFG, path=path)
    presets.take_notes()
    original = open(path, "rb").read()
    restore = _make_unreadable(path)
    try:
        presets.load(path)                          # the read that discovers it
        notes = presets.take_notes()
        assert any("NOT saving over it" in n for n in notes), notes
        assert presets._is_poisoned(path)

        assert presets.save("other", CFG, path=path) is False
        assert any("not saved" in n for n in presets.take_notes())
    finally:
        restore()
    assert open(path, "rb").read() == original      # the only copy survives
    assert presets.load(path)["mine"] == CFG


# ---------- the poison clears when the file recovers (or goes away) ----------

@unreadable_supported
def test_repairing_the_file_clears_the_poison(tmp_path):
    """Both natural recoveries used to leave every later save a silent no-op
    for the rest of the process: `_no_overwrite` was only discarded inside
    `_corrupt_backup`, which only ran when json.load raised."""
    path = str(tmp_path / "presets.json")
    presets.save("mine", CFG, path=path)
    restore = _make_unreadable(path)
    presets.load(path)
    assert presets._is_poisoned(path)
    restore()                                       # the user fixes permissions

    assert presets.save("other", CFG, path=path) is True
    assert not presets._is_poisoned(path)
    assert set(presets.user_names(path)) == {"mine", "other"}


@unreadable_supported
def test_deleting_the_file_clears_the_poison(tmp_path):
    path = str(tmp_path / "presets.json")
    presets.save("mine", CFG, path=path)
    restore = _make_unreadable(path)
    presets.load(path)
    assert presets._is_poisoned(path)
    restore()
    os.remove(path)                                 # the user throws it away

    assert presets.save("fresh", CFG, path=path) is True
    assert not presets._is_poisoned(path)
    assert presets.user_names(path) == {"fresh"}


# ---------- hand-edited JSON: wrong shapes never reach a traceback ----------

@pytest.mark.parametrize("body", ["null", "[1, 2]", '"x"', "42", "true"])
def test_wrong_top_level_shape_routes_to_corrupt_recovery(tmp_path, body):
    """DESIGN.md §7 invites hand-editing these files. Valid JSON of the wrong
    shape used to reach callers untouched and raise AttributeError/TypeError
    several frames later — at boot, before any window existed."""
    path = str(tmp_path / "presets.json")
    with open(path, "w") as f:
        f.write(body)
    assert "abstract" in presets.load(path)         # built-ins still served
    assert presets.user_names(path) == set()
    assert presets.bank(path) is None
    assert presets.setlist(path) is None
    baks = list(tmp_path.glob("presets.corrupt.*.bak.json"))
    assert len(baks) == 1 and baks[0].read_text() == body
    assert any("backup" in n for n in presets.take_notes())


@pytest.mark.parametrize("body", ["null", "[1, 2]", '"x"', "42", "true"])
def test_wrong_top_level_shape_in_state_json_is_not_a_crash(tmp_path, body):
    path = str(tmp_path / "state.json")
    with open(path, "w") as f:
        f.write(body)
    assert presets.load_state(path) == {}
    assert list(tmp_path.glob("state.corrupt.*.bak.json"))
    presets.take_notes()


@pytest.mark.parametrize("body,checks", [
    ('{"version": 2, "modes": null}', "modes"),
    ('{"version": 2, "modes": {"particles": null}}', "entry"),
    ('{"version": 2, "modes": {"particles": {"looks": null}}}', "looks"),
    ('{"version": 2, "modes": {"particles": {"bank": null}}}', "bank"),
    ('{"version": 2, "modes": {"particles": {"setlist": null}}}', "setlist"),
    ('{"version": 2, "meta": null, "modes": {}}', "meta"),
])
def test_null_sub_objects_read_as_absent_instead_of_raising(tmp_path, body,
                                                            checks):
    """A null sub-object is shape-tolerated, not treated as corruption: the
    file is still recognisably a v2 store, it just has a hole in it."""
    path = str(tmp_path / "presets.json")
    with open(path, "w") as f:
        f.write(body)
    assert "abstract" in presets.load(path)
    assert presets.user_names(path) == set()
    # "never stored" (None) — a scrambled type is not an explicit emptying,
    # so the shell still seeds its defaults instead of booting an empty bank
    assert presets.bank(path) is None
    assert presets.setlist(path) is None
    assert not list(tmp_path.glob("presets.corrupt.*.bak.json"))
    # ...and the store is still writable through the hole
    assert presets.save("mine", CFG, path=path) is True
    assert presets.load(path)["mine"] == CFG


def test_non_dict_look_entries_are_dropped_not_crashed(tmp_path):
    path = str(tmp_path / "presets.json")
    with open(path, "w") as f:
        f.write('{"version": 2, "modes": {"particles": {"looks": '
                '{"bad": null, "good": {"fade": 0.5}}}}}')
    loaded = presets.load(path)
    assert loaded["good"] == {"fade": 0.5}
    assert "bad" not in loaded


# ---------- bank + setlist ----------

def test_bank_absent_vs_explicit_empty(tmp_path):
    path = str(tmp_path / "p.json")
    assert presets.bank(path) is None           # never stored → caller seeds
    presets.set_bank({}, path=path)
    assert presets.bank(path) == {}             # explicitly cleared → respected


def test_bank_slots_are_explicit_assignments(tmp_path):
    path = str(tmp_path / "p.json")
    presets.set_bank({"1": "abstract", "3": "embers"}, path=path)
    assert presets.bank(path) == {"1": "abstract", "3": "embers"}


def test_delete_clears_slot_but_keeps_other_numbers(tmp_path):
    """Slot stability (DESIGN.md §7): deletes never renumber other slots."""
    path = str(tmp_path / "p.json")
    presets.save("mine", CFG, path=path)
    presets.set_bank({"2": "mine", "5": "embers"}, path=path)
    presets.set_setlist(["mine", "embers"], path=path)
    assert presets.delete("mine", path=path)
    assert presets.bank(path) == {"5": "embers"}
    assert presets.setlist(path) == ["embers"]


def test_rename_follows_into_bank_and_setlist(tmp_path):
    path = str(tmp_path / "p.json")
    presets.save("mine", CFG, path=path)
    presets.set_bank({"4": "mine"}, path=path)
    presets.set_setlist(["abstract", "mine"], path=path)
    assert presets.rename("mine", "neon", path=path)
    assert presets.bank(path) == {"4": "neon"}
    assert presets.setlist(path) == ["abstract", "neon"]


def test_setlist_absent_returns_none(tmp_path):
    assert presets.setlist(str(tmp_path / "p.json")) is None


# ---------- autosave state ----------

def test_state_round_trip(tmp_path):
    path = str(tmp_path / "state.json")
    presets.save_state({"mode": "particles", "preset": "embers",
                        "bank": {"particles": {"1": "embers"}}}, path=path)
    assert presets.load_state(path)["preset"] == "embers"
    assert presets.load_state(str(tmp_path / "missing.json")) == {}


def test_state_save_failure_is_silent(tmp_path):
    """Autosave must never take the show down — a failing save raises
    nothing, writes nothing, and leaves the existing state intact."""
    # failure mode 1: unwritable path — no raise, no file appears
    bad = tmp_path / "no" / "dir" / "s.json"
    presets.save_state({"x": 1}, path=str(bad))
    assert not bad.exists() and not bad.parent.exists()
    # failure mode 2: unserializable payload over a GOOD existing file —
    # no raise, and the prior state survives byte-for-byte (atomic _write)
    path = str(tmp_path / "s.json")
    presets.save_state({"mode": "particles", "preset": "embers"}, path=path)
    before = open(path, "rb").read()
    presets.save_state({"bad": object()}, path=path)     # json.dump raises inside
    assert open(path, "rb").read() == before
    assert presets.load_state(path)["preset"] == "embers"
    # ...and no stray temp files were left behind
    assert {p.name for p in tmp_path.iterdir()} == {"s.json"}
