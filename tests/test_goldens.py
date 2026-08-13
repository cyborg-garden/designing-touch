"""Golden pins for the shipped panel — DESIGN.md §8 step 0.

Renders today's OverlayUI panel (default state, empty status) over two fixture
frames (bright: 230-gray with a white gradient; dark: flat 20-gray) at 720p,
1080p, and 4K, and compares against stored PNGs in tests/goldens/.

Comparison is mean-abs-diff with tolerance 0.15 (per-image, float, all pixels):
cv2 text rasterization varies slightly across builds (anti-aliased glyph edges
only), which measures well under 0.1 mean-abs, while the smallest realistic
layout change measured 0.29 (one row label's text) and a 2 px panel shift 0.70.
Same-build renders are byte-identical (measured 0.0 twice in a row).

Regenerate deliberately with:  GOLDEN_REGEN=1 pytest tests/test_goldens.py
"""
import os

import cv2
import numpy as np
import pytest

from dtouch.overlay_ui import OverlayUI
from dtouch.particles import PALETTES
from dtouch.live import MATTES

PRESETS = ["abstract", "portrait", "textured", "embers", "aurora", "sigil"]
GOLDEN_DIR = os.path.join(os.path.dirname(__file__), "goldens")
TOLERANCE = 0.15
RESOLUTIONS = [(1280, 720), (1920, 1080), (3840, 2160)]


def _bright(w, h):
    """230-gray with a horizontal white gradient — worst-case light background."""
    base = np.full((h, w, 3), 230, np.uint8)
    grad = np.linspace(0, 25, w).astype(np.int16)[None, :, None]
    return np.clip(base.astype(np.int16) + grad, 0, 255).astype(np.uint8)


def _dark(w, h):
    return np.full((h, w, 3), 20, np.uint8)


FIXTURES = {"bright": _bright, "dark": _dark}


def _render_panel(w, h, frame):
    ui = OverlayUI(w, h, PRESETS, list(PALETTES), MATTES)
    img = frame.copy()
    ui.draw(img, {"status": ""})
    return img


def _mean_abs_diff(a, b):
    return float(np.abs(a.astype(np.float32) - b.astype(np.float32)).mean())


@pytest.mark.parametrize("w,h", RESOLUTIONS)
@pytest.mark.parametrize("fixture", sorted(FIXTURES))
def test_panel_matches_golden(w, h, fixture):
    img = _render_panel(w, h, FIXTURES[fixture](w, h))
    path = os.path.join(GOLDEN_DIR, f"panel_{h}p_{fixture}.png")
    if os.environ.get("GOLDEN_REGEN") == "1":
        os.makedirs(GOLDEN_DIR, exist_ok=True)
        cv2.imwrite(path, img)
    golden = cv2.imread(path, cv2.IMREAD_COLOR)
    assert golden is not None, (
        f"missing golden {path} — generate with GOLDEN_REGEN=1 pytest tests/test_goldens.py"
    )
    assert golden.shape == img.shape
    diff = _mean_abs_diff(img, golden)
    assert diff <= TOLERANCE, (
        f"panel drifted from golden {os.path.basename(path)}: mean-abs-diff {diff:.3f} "
        f"> {TOLERANCE} (regenerate deliberately with GOLDEN_REGEN=1 if the change is intended)"
    )


def test_default_panel_fits_1080p_without_scrolling():
    """Existing contract, pinned here with the goldens: the default panel's whole
    control column fits a 1080p window with no scrolling."""
    ui = OverlayUI(1920, 1080, PRESETS, list(PALETTES), MATTES)
    ui.draw(np.zeros((1080, 1920, 3), np.uint8), {"status": ""})
    assert ui._content_h <= 1080
