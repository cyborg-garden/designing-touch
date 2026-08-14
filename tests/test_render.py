"""Integration smoke test: synthetic source -> field -> GPU render -> non-black frame."""
import numpy as np
import pytest

from dtouch import (SyntheticSource, make_grid, displace_z, random_scale,
                    random_euler, pack_instances, Renderer, ShadowRenderer)


def _gl_available():
    try:
        import moderngl
        ctx = moderngl.create_standalone_context()
        ctx.release()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _gl_available(),
                                reason="no GL context available (CI)")


def test_pipeline_renders_nonblack_frame():
    gx, gy = 64, 36
    n = gx * gy
    luma = SyntheticSource(gx, gy, frames=1).read()
    assert luma is not None and luma.shape == (gy, gx)

    buf = pack_instances(
        displace_z(make_grid(gx, gy), luma, depth_scale=1.2),
        random_scale(n, seed=0), random_euler(n, seed=1),
    )
    r = Renderer(320, 180, n, base_size=0.7 / max(gx, gy), depth_scale=1.2)
    try:
        frame = r.render(buf)
    finally:
        r.release()

    assert frame.shape == (180, 320, 3) and frame.dtype == np.uint8
    lit = int((frame.max(axis=2) > 20).sum())
    assert lit > 200, f"expected many lit pixels, got {lit}"


def test_shadow_renderer_produces_ground_and_shadow():
    gx, gy = 48, 28
    n = gx * gy
    luma = SyntheticSource(gx, gy, frames=1).read()
    buf = pack_instances(
        displace_z(make_grid(gx, gy), luma, depth_scale=1.2),
        random_scale(n, seed=0), random_euler(n, seed=1),
    )
    r = ShadowRenderer(320, 180, n, base_size=1.5 / max(gx, gy), depth_scale=1.2,
                       shadow_size=1024)
    try:
        frame = r.render(buf)
    finally:
        r.release()
    assert frame.shape == (180, 320, 3)
    gray = frame.max(axis=2)
    # bright cubes exist, a lit ground exists, and some ground is shadowed (a range of values)
    assert int((gray > 120).sum()) > 100      # bright cubes
    assert int((gray > 20).sum()) > 5000       # ground fills much of the frame
    assert gray.std() > 15                      # contrast from shadowing/relief

