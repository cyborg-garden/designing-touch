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
