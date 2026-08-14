"""Smoke test for the glow renderer (needs a GL context)."""
import numpy as np
import pytest

from dtouch.glow import GlowRenderer
from dtouch.particles import ParticleFlow


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


def test_glow_renders_nonblack_and_trails_accumulate():
    gw, gh, n = 96, 54, 4000
    pf = ParticleFlow(n=n, gw=gw, gh=gh, seed=0)
    matte = np.zeros((gh, gw), np.float32)
    matte[gh // 3:2 * gh // 3, gw // 3:2 * gw // 3] = 1.0
    gray = np.zeros((gh, gw), np.float32)

    gr = GlowRenderer(256, 144, n, fade=0.9, exposure=1.4)
    try:
        frame = None
        for _ in range(20):
            pf.update(matte, gray)
            frame = gr.render(pf.render_data())
    finally:
        gr.release()

    assert frame.shape == (144, 256, 3) and frame.dtype == np.uint8
    lit = int((frame.max(axis=2) > 15).sum())
    assert lit > 150, f"expected glowing particles, got {lit} lit pixels"
