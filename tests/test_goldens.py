"""Golden pins for the shipped panel — DESIGN.md §8 step 0.

Renders today's OverlayUI panel (default state, empty status) over two fixture
frames (bright: 230-gray with a white gradient; dark: flat 20-gray) at 720p,
1080p, and 4K, and compares against stored PNGs in tests/goldens/.

Comparison is two-tier:

- PRIMARY: mean-abs-diff over the PANEL ROI only (the right sidebar — the
  only region the panel draws into), tolerance 0.9. Concentrating the diff
  where the pixels are raises both sensitivity and margin: cross-build AA
  glyph-edge noise measures well under 0.1 whole-frame, and since it all
  lives in the panel (15-23% of the frame) that bounds ROI noise at ~0.66;
  the smallest realistic layout change measured 0.29 whole-frame = >=1.28
  over the ROI. 0.9 sits between those bounds with margin on both sides
  (the old whole-frame 0.15 had the noise ceiling and signal floor only
  1.9x apart; the ROI check makes it ~2x with wider absolute margin).
- SECONDARY: whole-frame mean-abs-diff, tolerance 0.15 (loose) — catches a
  regression that paints OUTSIDE the panel, which the ROI check would miss.

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
TOLERANCE = 0.15        # whole-frame (secondary, loose)
ROI_TOLERANCE = 0.9     # panel ROI (primary — see module docstring)
PANEL_W = 290           # OverlayUI.panel_w, scaled like the panel itself
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


def _panel_roi(img, h):
    """The right-sidebar region the panel draws into (same scaling rule as
    OverlayUI: 290 px at the 1080p baseline, floored at 1.0)."""
    pw = int(round(PANEL_W * max(1.0, h / 1080)))
    return img[:, img.shape[1] - pw:]


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
    # primary: panel ROI (sensitive — see module docstring for the rationale)
    roi_diff = _mean_abs_diff(_panel_roi(img, h), _panel_roi(golden, h))
    assert roi_diff <= ROI_TOLERANCE, (
        f"panel drifted from golden {os.path.basename(path)}: ROI mean-abs-diff "
        f"{roi_diff:.3f} > {ROI_TOLERANCE} (regenerate deliberately with "
        f"GOLDEN_REGEN=1 if the change is intended)"
    )
    # secondary: whole frame (loose — catches paint OUTSIDE the panel)
    diff = _mean_abs_diff(img, golden)
    assert diff <= TOLERANCE, (
        f"pixels outside the panel drifted in {os.path.basename(path)}: "
        f"whole-frame mean-abs-diff {diff:.3f} > {TOLERANCE}"
    )


def test_default_panel_fits_1080p_without_scrolling():
    """Existing contract, pinned here with the goldens: the default panel's whole
    control column fits a 1080p window with no scrolling."""
    ui = OverlayUI(1920, 1080, PRESETS, list(PALETTES), MATTES)
    ui.draw(np.zeros((1080, 1920, 3), np.uint8), {"status": ""})
    assert ui._content_h <= 1080
