"""Spec-derived preset capture/apply (DESIGN.md §2.1/§7, step 7).

The panel spec's save/apply flags are the single schema authority: capture
walks save=True widgets (SIGNAL nests under "signal"), apply keeps
apply="keep" widgets untouched and merges apply="reset" widgets over the
mode's declared defaults. The v1 round-trip pins hold the migration contract:
every v1 fixture (the shipped presets.json content) applies to identical
engine state before and after migration.
"""
import json
import os

import pytest

from dtouch import presets
from dtouch.live import MATTES
from dtouch.modes.particles import ParticlesMode
from dtouch.overlay_ui import OverlayUI, DITHERS
from dtouch.panelspec import apply_look, capture_look
from dtouch.particles import PALETTES

PRESETS = ["abstract", "portrait", "textured", "embers", "aurora", "sigil"]
FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "presets_v1.json")

# every attr a look can influence — compared for state identity
STATE_ATTRS = ("matte_idx", "palette_idx", "fade", "exposure", "spark", "curl",
               "dot", "count", "damp", "pull", "reseed", "attract_speed",
               "video_bg", "video_mix", "audio", "sens",
               "flock", "cohere", "align", "separate",
               "glitch", "chroma", "drift", "crush", "dither_idx", "scanlines")


def _ui():
    return OverlayUI(1920, 1080, PRESETS, list(PALETTES), MATTES)


def _state(ui):
    return {a: getattr(ui, a) for a in STATE_ATTRS}


def _apply(ui, cfg):
    apply_look(ui, ui.spec, cfg, ParticlesMode.DEFAULTS)


# ---------- capture ----------

def test_capture_produces_v1_compatible_flat_keys_plus_nested_signal():
    ui = _ui()
    cfg = capture_look(ui, ui.spec)
    # the v1 keys a saved look always carried, under their v1 names
    for key in ("matte", "palette", "fade", "exposure", "spark", "curl_amp",
                "reseed_frac", "base_size", "damp", "pull_falloff",
                "attract_speed", "video_bg", "video_mix", "audio", "sens"):
        assert key in cfg, key
    # MOTION stays flat (mode-owned); SIGNAL nests (shell rack, DESIGN.md §2.4)
    for key in ("flock", "cohere", "align", "separate"):
        assert key in cfg, key
    assert set(cfg["signal"]) == {"glitch", "dither", "chroma", "drift",
                                  "crush", "scanlines"}
    # save=False widgets stay out
    for key in ("record", "mirror", "res", "res_idx"):
        assert key not in cfg


def test_capture_stores_cycle_values_not_indices():
    ui = _ui()
    ui.matte_idx = MATTES.index("person")
    ui.palette_idx = list(PALETTES).index("fire")
    ui.dither_idx = DITHERS.index("blue")
    cfg = capture_look(ui, ui.spec)
    assert cfg["matte"] == "person"
    assert cfg["palette"] == "fire"
    assert cfg["signal"]["dither"] == "blue"


def test_capture_apply_round_trips_to_identical_state():
    ui = _ui()
    ui.glitch = True; ui.chroma = 33.0; ui.flock = True; ui.count = 0.4
    ui.matte_idx = MATTES.index("luma")
    cfg = capture_look(ui, ui.spec)
    other = _ui()
    _apply(other, cfg)
    assert _state(other) == _state(ui)


# ---------- apply semantics ----------

def test_reset_widgets_merge_over_mode_defaults():
    """Switching looks fully resets physics — sigil's low damping must not
    leak into abstract (the shipped merge-over-defaults rule, now data)."""
    ui = _ui()
    _apply(ui, presets.BUILTIN["sigil"])
    assert ui.damp == pytest.approx(0.975)
    assert ui.attract_speed == pytest.approx(5.0)
    _apply(ui, presets.BUILTIN["abstract"])     # abstract records no damp
    assert ui.damp == pytest.approx(ParticlesMode.DEFAULTS["damp"])
    assert ui.pull == pytest.approx(ParticlesMode.DEFAULTS["pull_falloff"])
    assert ui.attract_speed == pytest.approx(4.5)


def test_keep_widgets_survive_template_hopping():
    """Hopping between built-ins never resets your live toggles or fx —
    absence is the contract (apply="keep" / reset-with-no-default)."""
    ui = _ui()
    ui.glitch = True; ui.chroma = 44.0; ui.video_bg = True; ui.audio = True
    ui.sens = 2.2; ui.flock = True; ui.cohere = 0.9; ui.count = 0.3
    ui.dither_idx = DITHERS.index("riemersma")
    _apply(ui, presets.BUILTIN["embers"])
    assert ui.glitch is True and ui.chroma == pytest.approx(44.0)
    assert ui.video_bg is True and ui.audio is True
    assert ui.sens == pytest.approx(2.2)
    assert ui.flock is True and ui.cohere == pytest.approx(0.9)
    assert ui.count == pytest.approx(0.3)
    assert ui.dither_name == "riemersma"


def test_recorded_keys_always_apply():
    ui = _ui()
    ui.glitch = False
    _apply(ui, {"palette": "mono", "signal": {"glitch": True, "dither": "fs"},
                "video_bg": True})
    assert ui.palette_name == "mono"
    assert ui.glitch is True and ui.dither_name == "fs"
    assert ui.video_bg is True


def test_unknown_cycle_values_are_ignored():
    ui = _ui()
    before = ui.matte_idx
    _apply(ui, {"matte": "hologram"})
    assert ui.matte_idx == before


# ---------- the v1 round-trip pins (DESIGN.md §7 Migration) ----------

def _v1_fixture():
    with open(FIXTURE) as f:
        return json.load(f)


@pytest.mark.parametrize("name", sorted(_v1_fixture()))
def test_v1_look_applies_identically_pre_and_post_migration(name, tmp_path):
    fixture = _v1_fixture()
    path = str(tmp_path / "presets.json")
    with open(path, "w") as f:
        json.dump(fixture, f)

    # pre-migration: the raw v1 cfg, applied directly
    pre = _ui()
    _apply(pre, fixture[name])

    # post-migration: force the v2 rewrite (first write backs up + migrates),
    # reload from the v2 file, apply the migrated look
    presets.save("trigger_migration", dict(matte="auto", palette="ice"), path=path)
    with open(path) as f:
        assert json.load(f)["version"] == 2
    migrated = presets.load(path)[name]
    post = _ui()
    _apply(post, migrated)

    assert _state(post) == _state(pre)
    assert capture_look(post, post.spec) == capture_look(pre, pre.spec)


def test_builtin_looks_apply_identically_through_migration_cycle():
    """Built-ins live in code and never migrate — pinned for completeness."""
    for name, cfg in presets.BUILTIN.items():
        a, b = _ui(), _ui()
        _apply(a, cfg)
        _apply(b, json.loads(json.dumps(cfg)))   # through a JSON round trip
        assert _state(a) == _state(b), name
