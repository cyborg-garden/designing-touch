"""Physarum mode + engine — protocol conformance, video coupling, key safety.

The mode tests mirror test_dithergirl.py (the full-frame-image exemplar). The
key-uniqueness test exists because mode-switch keys are only registered when a
window opens (show=True): a collision passes the whole headless suite and
kills the real app at boot, so it is asserted here at the derivation level.
"""
import numpy as np
import pytest

from dtouch.modes import REGISTRY, mode_by_id
from dtouch.modes.physarum import ACCENT, PALETTES_PH, PhysarumMode, _palette_lut
from dtouch.physarum import POINT_NAMES, POINTS, PhysarumField
from dtouch.shell import Host

from test_shell import SyntheticSource

RES = (192, 108)


def _paths(tmp_path):
    return dict(presets_path=str(tmp_path / "presets.json"),
                state_path=str(tmp_path / "state.json"))


def _host(tmp_path, **kw):
    kw.setdefault("res", RES)
    kw.setdefault("show", False)
    kw.setdefault("preset", None)
    return Host(PhysarumMode(), source=SyntheticSource(), **_paths(tmp_path), **kw)


def _booted(tmp_path, **kw):
    host = _host(tmp_path, max_frames=1, **kw)
    host.run()
    return host


def _white(h=36, w=64):
    return np.full((h, w, 3), 255, np.uint8)


# ---------- mode protocol conformance (DESIGN.md §2.2) ----------

def test_mode_protocol_attrs():
    m = PhysarumMode()
    assert m.id == "physarum" and m.title == "Physarum"
    assert m.accepts_still is False
    assert m.accent == ACCENT
    assert m.safe_look() in m.BUILTIN


def test_registered_in_registry():
    assert PhysarumMode in REGISTRY
    assert mode_by_id("physarum") is PhysarumMode


def test_mode_switch_keys_unique_and_off_the_global_row():
    """id[:1] / the `key` class attr must not collide — the registry raises
    only when a window opens, which no headless test exercises."""
    keys = [getattr(m, "key", m.id[:1]) for m in REGISTRY]
    assert len(set(keys)) == len(keys), f"mode-switch keys collide: {keys}"
    globals_taken = set("sgariqm")          # preset.save, glitch, audio,
    assert not set(keys) & globals_taken    # record, debug, quit, menu


def test_start_stop_idempotent_and_headless(tmp_path):
    host = _booted(tmp_path)
    m = host.mode
    m.stop()
    m.stop()                                # second stop must be a no-op
    m.start(host)
    out = m.step(_white(), None, 1 / 30)
    assert out.shape == (RES[1], RES[0], 3) and out.dtype == np.uint8
    m.stop()


def test_step_without_ui_uses_defaults(tmp_path):
    """The soak path: step before the shell builds the shared UI state."""
    host = _host(tmp_path)
    m = PhysarumMode()
    m.start(host)
    out = m.step(_white(), None, 1 / 30)
    assert out.shape == (RES[1], RES[0], 3) and out.dtype == np.uint8
    m.stop()


def test_status_line_is_spec_derived_and_ascii(tmp_path):
    host = _booted(tmp_path)
    s = host._status_line()
    assert s == s.encode("ascii", "replace").decode()
    assert s == ("PHYSARUM  matte auto  body fingers  field veins  arctic"
                 "  cam synthetic")


def test_panel_sections():
    spec = PhysarumMode().panel_spec()
    assert [sec.title for sec in spec] == ["TEMPLATES", "SOURCE", "MOLD", "LOOK"]


def test_swap_command_flips_the_points(tmp_path):
    host = _booted(tmp_path)
    ui = host.ui
    ui.ph_point_fg_idx, ui.ph_point_bg_idx = 2, 0
    host.mode.commands()["physarum.swap"].run()
    assert (ui.ph_point_fg_idx, ui.ph_point_bg_idx) == (0, 2)


def test_audio_pulses_the_picture(tmp_path):
    """Bass raises exposure for the frame — the engine must see it."""
    host = _booted(tmp_path)
    m = host.mode
    m.start(host)                           # run()'s finally stopped it
    m.step(_white(), {"bass": 1.0, "treble": 0.0}, 1 / 30)
    loud = m.pf.exposure
    m.step(_white(), {"bass": 0.0, "treble": 0.0}, 1 / 30)
    assert loud > m.pf.exposure


# ---------- engine (dtouch.physarum) ----------

def _field(**kw):
    kw.setdefault("n", 4000)
    kw.setdefault("gw", 96)
    kw.setdefault("gh", 54)
    kw.setdefault("seed", 7)
    return PhysarumField(**kw)


def _flat(f, v=0.0):
    return np.full((f.gh, f.gw), v, np.float32)


def test_update_deposits_trail_and_luminance_is_bounded():
    f = _field()
    for _ in range(5):
        f.update(_flat(f), _flat(f))
    assert f.trail.sum() > 0
    lum = f.luminance()
    assert lum.shape == (f.gh, f.gw) and lum.dtype == np.float32
    assert float(lum.min()) >= 0.0 and float(lum.max()) <= 1.0
    assert np.isfinite(f.trail).all()


def test_matte_is_the_pen_blends_toward_body_point():
    """matte=0 runs the field point, matte=1 the body point — stride follows.
    haze steps 0.7/frame, fingers 2.0, so mean per-frame displacement must
    roughly double between the two (wrap-aware distance)."""
    def mean_step(matte_value):
        f = _field(point_bg="haze", point_fg="fingers")
        m = _flat(f, matte_value)
        g = _flat(f)
        f.update(m, g)                       # warm-up (headings settle)
        px, py = f.px.copy(), f.py.copy()
        f.update(m, g)
        dx = np.minimum(np.abs(f.px - px), f.gw - np.abs(f.px - px))
        dy = np.minimum(np.abs(f.py - py), f.gh - np.abs(f.py - py))
        return float(np.hypot(dx, dy).mean())

    slow, fast = mean_step(0.0), mean_step(1.0)
    assert slow == pytest.approx(POINTS["haze"]["step"], rel=0.15)
    assert fast == pytest.approx(POINTS["fingers"]["step"], rel=0.15)


def test_luminance_is_food_draws_the_mold():
    """A bright band in the footage must accumulate more trail than the dark
    rest of the frame once food coupling is on."""
    f = _field(food=1.0, reseed_frac=0.05)
    gray = _flat(f)
    band = slice(f.gw // 3, 2 * f.gw // 3)
    gray[:, band] = 1.0
    for _ in range(40):
        f.update(_flat(f), gray)
    inside = float(f.trail[:, band].mean())
    outside = float(np.delete(f.trail, np.s_[band], axis=1).mean())
    assert inside > outside * 1.3


def test_agents_stay_on_grid_even_from_the_edge():
    """float32 wrap can land exactly ON the bound — indices must never
    escape. Park every agent at the last representable spot and step."""
    f = _field()
    f.px[:] = np.nextafter(np.float32(f.gw), np.float32(0))
    f.py[:] = np.nextafter(np.float32(f.gh), np.float32(0))
    f.update(_flat(f), _flat(f))             # must not raise
    assert (f.px >= 0).all() and (f.py >= 0).all()


def test_same_seed_same_trail():
    a, b = _field(), _field()
    for _ in range(3):
        a.update(_flat(a), _flat(a))
        b.update(_flat(b), _flat(b))
    assert np.array_equal(a.trail, b.trail)


def test_builtin_points_and_palettes_are_wired():
    for name in PhysarumMode.BUILTIN:
        look = PhysarumMode.BUILTIN[name]
        assert look["point_bg"] in POINT_NAMES
        assert look["point_fg"] in POINT_NAMES
        assert look["palette"] in PALETTES_PH
    for pal in PALETTES_PH:
        if pal != "video":
            lut = _palette_lut(pal)
            assert lut.shape == (256, 3) and lut.dtype == np.uint8
