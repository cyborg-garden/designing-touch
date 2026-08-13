"""Presets — per-mode looks, banks, and setlists in one v2 file (DESIGN.md §7).

File schema (v2):

    {
      "version": 2,
      "meta": {},
      "modes": {
        "particles": {
          "looks":   {"embers": {…, "signal": {…}}},
          "bank":    {"1": "abstract", "3": "embers"},
          "setlist": ["abstract", "embers"]
        }
      }
    }

- Built-ins stay in code per mode (ParticlesMode.BUILTIN references this
  module's BUILTIN dict) — immune to rename/delete, bankable.
- The schema of a look is spec-derived (the panel spec's save/apply flags —
  DESIGN.md §2.1); this module stores what it is given and never filters.
- Bank slots are explicit assignments (not list position) — slot stability
  across saves/deletes. delete/rename keep bank + setlist consistent.
- The file stays in the launch dir (./presets.json), one file for all modes.

Migration: load detects a v1 file (top-level looks, no "version"), migrates in
memory (every v1 look → modes.particles.looks, filtered through the v1 KEYS
whitelist exactly as v1 load() did — v1 never persisted MOTION/SIGNAL keys, so
nothing changes meaning). The first WRITE copies the original file to
presets.v1.bak.json, then writes v2 (§9: preset migration is the one
user-data-loss surface — backup + round-trip pins mandatory).

last_mode: state.json (the autosave file, see load_state/save_state) is the
single authority for the active mode + preset across restarts; the v2 "meta"
object is reserved for future schema needs and deliberately NOT a second owner
(DESIGN §7 sketches meta.last_mode — resolved here in favor of state.json so
the diffable presets file doesn't churn on every mode switch).
"""
from __future__ import annotations

import glob
import json
import os
import shutil
import tempfile
import time
from typing import Dict, Optional

# The v1 whitelist of live-applicable keys — retained ONLY for migrating v1
# files (v1 load() filtered through it, so migration must too). The live
# schema authority is now the panel spec's save/apply flags (DESIGN.md §2.1).
KEYS = ("matte", "palette", "fade", "exposure", "spark", "curl_amp", "reseed_frac",
        "base_size", "damp", "pull_falloff", "attract_speed",
        "video_bg", "video_mix", "audio", "sens")

DEFAULT_MODE = "particles"

BUILTIN: Dict[str, dict] = {
    # the loved abstract cloud
    "abstract": dict(matte="auto", palette="ice", fade=0.90, exposure=1.4,
                     spark=0.35, curl_amp=0.5, reseed_frac=0.06, base_size=0.011),
    # recognizable: particles painted with the real footage, calm so the shape holds
    "portrait": dict(matte="person", palette="video", fade=0.74, exposure=2.1,
                     spark=0.0, curl_amp=0.05, reseed_frac=0.16, base_size=0.0045),
    # textured but subject-agnostic (no person model) — colors of whatever's salient/moving
    "textured": dict(matte="auto", palette="video", fade=0.76, exposure=2.1,
                     spark=0.04, curl_amp=0.10, reseed_frac=0.14, base_size=0.005),
    "embers": dict(matte="auto", palette="fire", fade=0.93, exposure=1.6,
                   spark=0.5, curl_amp=0.6, reseed_frac=0.06, base_size=0.012),
    "aurora": dict(matte="motion", palette="aurora", fade=0.92, exposure=1.5,
                   spark=0.4, curl_amp=0.5, reseed_frac=0.06, base_size=0.011),
    # "sigil": sharp inward pull + very low damping make particles overshoot into glowing
    # contour bands along the silhouette, with caustic streaks — cyber-sigil / techcore.
    "sigil": dict(matte="person", palette="mono", fade=0.96, exposure=1.5, spark=0.0,
                  curl_amp=0.08, reseed_frac=0.008, base_size=0.006, damp=0.975,
                  pull_falloff=12.0, attract_speed=5.0),
}


# ----- raw file I/O -----

# Warnings queued for the shell to surface as toasts (silence-on-data-loss is
# a bug — DESIGN.md §9: presets are the one user-data-loss surface).
_notes: list = []


def take_notes() -> list:
    """Drain the queued store warnings (the shell toasts them if it can)."""
    out = list(_notes)
    _notes.clear()
    return out


def _corrupt_backup(path: str):
    """An unreadable file is copied to {base}.corrupt.<ts>.bak.json before the
    store proceeds empty — the next write would otherwise overwrite the user's
    looks with no recovery path. Backed up once per distinct corruption (a
    re-read of the same bad bytes never stacks duplicates). Returns the
    warning note, or None when this corruption is already backed up."""
    try:
        with open(path, "rb") as f:
            bad = f.read()
    except OSError:
        return None
    base, ext = os.path.splitext(path)
    ext = ext or ".json"
    for bak in glob.glob(f"{glob.escape(base)}.corrupt.*.bak{ext}"):
        try:
            with open(bak, "rb") as f:
                if f.read() == bad:
                    return None
        except OSError:
            pass
    bak = f"{base}.corrupt.{time.strftime('%Y%m%d_%H%M%S')}.bak{ext}"
    try:
        shutil.copy2(path, bak)
    except OSError:
        return None
    return (f"{os.path.basename(path)} unreadable - "
            f"backup saved to {os.path.basename(bak)}")


def _read(path: str) -> dict:
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            note = _corrupt_backup(path)
            if note:
                print(note)
                _notes.append(note)
    return {}


def _write(data: dict, path: str):
    """Atomic write: temp file in the same directory, then os.replace — a
    crash mid-write leaves either the old or the new content, never a
    truncated file."""
    d = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(prefix=os.path.basename(path) + ".",
                               suffix=".tmp", dir=d)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# ----- v1 → v2 migration -----

def is_v1(raw: dict) -> bool:
    """A non-empty file with no "version" field is a v1 flat-looks file."""
    return bool(raw) and "version" not in raw


def migrate_v1(raw: dict) -> dict:
    """v1 flat looks → v2, in memory. Filters through the v1 KEYS whitelist —
    byte-equivalent to what v1 load() returned for the same file."""
    looks = {name: {k: cfg[k] for k in KEYS if k in cfg}
             for name, cfg in raw.items() if isinstance(cfg, dict)}
    return {"version": 2, "meta": {},
            "modes": {DEFAULT_MODE: {"looks": looks}}}


def _empty_v2() -> dict:
    return {"version": 2, "meta": {}, "modes": {}}


def backup_path(path: str) -> str:
    base, ext = os.path.splitext(path)          # presets.json → presets.v1.bak.json
    return f"{base}.v1.bak{ext or '.json'}"


def load_file(path: str = "presets.json") -> dict:
    """The whole store as v2 (migrating a v1 file in memory; disk untouched)."""
    raw = _read(path)
    if not raw:
        return _empty_v2()
    if is_v1(raw):
        return migrate_v1(raw)
    return raw


def write_file(data: dict, path: str = "presets.json"):
    """Persist v2. If the on-disk file is still v1, copy it to
    presets.v1.bak.json first (once) — the mandatory migration backup."""
    raw = _read(path)
    if is_v1(raw):
        bak = backup_path(path)
        if not os.path.exists(bak):
            shutil.copy2(path, bak)
    _write(data, path)


def _mode_entry(data: dict, mode: str) -> dict:
    return data.setdefault("modes", {}).setdefault(mode, {})


# ----- looks (flat API, per mode — default mode keeps v1 call sites working) -----

def load(path: str = "presets.json", mode: str = DEFAULT_MODE,
         builtin: Optional[dict] = None) -> Dict[str, dict]:
    """Built-in looks merged with the mode's saved looks (user wins on clash)."""
    builtin = BUILTIN if builtin is None else builtin
    presets = {k: dict(v) for k, v in builtin.items()}
    looks = load_file(path).get("modes", {}).get(mode, {}).get("looks", {})
    for name, cfg in looks.items():
        presets[name] = dict(cfg)
    return presets


def user_names(path: str = "presets.json", mode: str = DEFAULT_MODE) -> set:
    """Names of the user-saved looks (the ones that can be renamed/deleted)."""
    return set(load_file(path).get("modes", {}).get(mode, {}).get("looks", {}))


def save(name: str, cfg: dict, path: str = "presets.json",
         mode: str = DEFAULT_MODE) -> str:
    """Write/replace a user look, preserving everything else in the file."""
    data = load_file(path)
    _mode_entry(data, mode).setdefault("looks", {})[name] = dict(cfg)
    write_file(data, path)
    return path


def delete(name: str, path: str = "presets.json", mode: str = DEFAULT_MODE) -> bool:
    """Remove a user look (built-ins live in code, so they can't be deleted).
    Bank slots pointing at it are cleared (other slots keep their numbers —
    slot stability, DESIGN.md §7) and it leaves the setlist."""
    data = load_file(path)
    entry = data.get("modes", {}).get(mode, {})
    looks = entry.get("looks", {})
    if name not in looks:
        return False
    del looks[name]
    if "bank" in entry:
        entry["bank"] = {s: n for s, n in entry["bank"].items() if n != name}
    if "setlist" in entry:
        entry["setlist"] = [n for n in entry["setlist"] if n != name]
    write_file(data, path)
    return True


def rename(old: str, new: str, path: str = "presets.json",
           mode: str = DEFAULT_MODE, builtin: Optional[dict] = None) -> bool:
    """Rename a user look. Refuses built-ins, name clashes, and empty names.
    Bank slots and setlist entries follow the rename."""
    builtin = BUILTIN if builtin is None else builtin
    new = new.strip()
    data = load_file(path)
    entry = data.get("modes", {}).get(mode, {})
    looks = entry.get("looks", {})
    if old not in looks or not new or new == old:
        return False
    if new in looks or new in builtin:
        return False
    looks[new] = looks.pop(old)
    if "bank" in entry:
        entry["bank"] = {s: (new if n == old else n)
                         for s, n in entry["bank"].items()}
    if "setlist" in entry:
        entry["setlist"] = [new if n == old else n for n in entry["setlist"]]
    write_file(data, path)
    return True


# ----- bank + setlist (DESIGN.md §7: explicit slot assignments, not positions) -----

def bank(path: str = "presets.json", mode: str = DEFAULT_MODE) -> Optional[dict]:
    """The stored slot→name dict, or None when this mode never stored one
    (callers seed a default — an explicit empty dict is respected)."""
    entry = load_file(path).get("modes", {}).get(mode, {})
    return dict(entry["bank"]) if "bank" in entry else None


def set_bank(bank_dict: dict, path: str = "presets.json",
             mode: str = DEFAULT_MODE):
    data = load_file(path)
    _mode_entry(data, mode)["bank"] = {str(k): v for k, v in bank_dict.items()}
    write_file(data, path)


def setlist(path: str = "presets.json", mode: str = DEFAULT_MODE) -> Optional[list]:
    """The stored setlist order, or None when this mode never stored one."""
    entry = load_file(path).get("modes", {}).get(mode, {})
    return list(entry["setlist"]) if "setlist" in entry else None


def set_setlist(names: list, path: str = "presets.json",
                mode: str = DEFAULT_MODE):
    data = load_file(path)
    _mode_entry(data, mode)["setlist"] = list(names)
    write_file(data, path)


# ----- autosave state (DESIGN.md §6.4: crash restart resumes the same look) -----

def load_state(path: str = "state.json") -> dict:
    return _read(path)


def save_state(state: dict, path: str = "state.json"):
    try:
        _write(state, path)
    except Exception:
        pass    # autosave must never take the show down
