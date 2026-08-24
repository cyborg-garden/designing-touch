"""Video-background compositing — raw footage screen-blended under the particles."""
import numpy as np

from dtouch.live import composite_video_bg
from dtouch.overlay_ui import OverlayUI
from dtouch.particles import PALETTES


def _particles(val=0, w=64, h=36):
    return np.full((h, w, 3), val, np.uint8)


def _frame_bgr(b=10, g=120, r=240, w=64, h=36):
    f = np.zeros((h, w, 3), np.uint8)
    f[:] = (b, g, r)
    return f


def test_mix_zero_leaves_particles_untouched():
    out = composite_video_bg(_particles(137), _frame_bgr(), 0.0)
    assert out.dtype == np.uint8
    assert np.array_equal(out, _particles(137))


def test_black_particles_full_mix_shows_video_in_rgb():
    # particles fully dark + mix 1.0 -> output IS the footage, converted BGR->RGB
    out = composite_video_bg(_particles(0), _frame_bgr(b=10, g=120, r=240), 1.0)
    assert abs(int(out[0, 0, 0]) - 240) <= 1   # R
    assert abs(int(out[0, 0, 1]) - 120) <= 1   # G
    assert abs(int(out[0, 0, 2]) - 10) <= 1    # B


def test_screen_blend_never_darkens_particles():
    rng = np.random.default_rng(7)
    p = rng.integers(0, 256, (36, 64, 3), np.uint8)
    out = composite_video_bg(p, _frame_bgr(), 0.7)
    assert (out.astype(int) >= p.astype(int) - 1).all()


def test_white_glow_stays_white():
    out = composite_video_bg(_particles(255), _frame_bgr(), 1.0)
    assert (out == 255).all()


def test_bg_resizes_to_render_resolution():
    # camera frame at a different resolution than the render target
    out = composite_video_bg(_particles(0, w=128, h=72), _frame_bgr(w=64, h=36), 1.0)
    assert out.shape == (72, 128, 3)


def test_overlay_ui_has_video_bg_controls():
    ui = OverlayUI(1280, 720, ["abstract"], list(PALETTES), ["auto"])
    assert ui.video_bg is False
    assert 0.0 < ui.video_mix <= 1.0
    ui._activate("video_bg", None, 0)
    assert ui.video_bg is True
    ui._activate("video_bg", None, 0)
    assert ui.video_bg is False


def test_overlay_ui_draws_video_bg_hit_targets():
    ui = OverlayUI(1280, 720, ["abstract"], list(PALETTES), ["auto"])
    canvas = np.zeros((720, 1280, 3), np.uint8)
    ui.draw(canvas, {"status": ""})
    kinds = [k for _, k, _ in ui._hot]
    assert "video_bg" in kinds
    # Vid mix only acts while Video bg is ON, so it only draws then
    # (panelspec.visible; magic-over-control, 2026-08-24)
    slider_attrs = [p[0] for _, k, p in ui._hot if k == "slider"]
    assert "video_mix" not in slider_attrs
    ui.video_bg = True
    ui.draw(canvas, {"status": ""})
    slider_attrs = [p[0] for _, k, p in ui._hot if k == "slider"]
    assert "video_mix" in slider_attrs
