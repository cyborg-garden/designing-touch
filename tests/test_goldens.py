"""Golden pins for the shipped panel — DESIGN.md §8 step 0.

Renders today's OverlayUI panel (empty status) over two fixture frames
(bright: 230-gray with a white gradient; dark: flat 20-gray) at 720p, 1080p,
and 4K, and compares against stored PNGs in tests/goldens/. Three panel states
are pinned, and between them every widget in the spec is drawn into some
golden:

- **closed** (`panel_{h}p_{fixture}.png`) — the boot panel, MOTION and SIGNAL
  collapsed, at all three resolutions. This is the top of the control column.
- **closed, scrolled to the bottom** (`panel_720p_{fixture}_bottom.png`) —
  720p only, and the reason is below.
- **open, scrolled to the bottom** (`panel_{h}p_{fixture}_open_bottom.png`) —
  MOTION and SIGNAL expanded, at 1080p and 4K. Nothing else pins the MOTION
  rows, SIGNAL's Bits/Gamma/Bias, or the circuit-bent rows: measured against
  the closed goldens, renaming `Bits`, `Cohere`, `Chroma` or `Scanlines` moves
  0.0000 — a collapsed section simply does not draw. Open, the column is
  1434 px at 1080p, so it must be scrolled for its lower half to be on screen
  at all (unscrolled, everything below `bias` is clipped and still measures
  0.0000). 720p is omitted there: the open column at 720p would show the same
  rows this pair already pins, for two more binaries.

**Why 720p needs a second, scrolled pin.** The panel is authored at the 1080p
baseline, and the closed column measures 1012 px — which fits 1080p and 4K
with room over, and leaves **292 px below the fold at 720p**. Everything in
that last 292 px is clipped, so the unscrolled 720p pair is close to blind
over the bottom third of its own subject: measured, dropping the `(F)`/`(G)`
section key hints moves **0.0000** there (it is caught only by 1080p/4K), and
deleting the entire `Menu (M)` row moves 0.0026 bright — 1.3x tolerance, i.e.
one rounding away from passing. A resolution whose pin cannot see a deleted
row is not pinning that resolution; it is pinning 1080p twice. Scrolled to the
bottom, the same two mutations measure 0.0482/0.0673 and 15.26/19.66 — the
second is that large because removing a row shortens the column, and a
bottom-anchored view shifts every remaining row up. That sensitivity is the
point of the pin and it cuts both ways: an intentional row add or remove will
move these two goldens far more than the closed ones, and they are meant to be
regenerated with the rest.

Comparison is two-tier:

- PRIMARY: mean-abs-diff over the PANEL ROI only (the right sidebar — the only
  region the panel draws into), tolerance 0.002.
- SECONDARY: whole-frame mean-abs-diff, tolerance 0.01 — catches a regression
  that paints OUTSIDE the panel, which the ROI check would miss.

**Measured, on this repo (Apple Silicon, py3.9):**

- *Noise floor is 0.0, not "small".* Same-build re-render: 0.000000 at every
  resolution and fixture. Cross-build within cv2 4.x: opencv-python 4.10.0.84
  reproduces goldens captured on 4.13.0 **byte-identically** — 0.000000 on all
  six closed cases. Hershey rasterisation is fixed-point integer work; it did
  not drift across three minor versions.
- *Signal floor* (worst-case smallest real change, ROI diff at 1080p/4K):
  renaming one slider `Trails`->`Trail` 0.0067, `Glow`->`Bloom` 0.0227,
  dropping the `(F)`/`(G)` section key hints 0.0321, deleting the whole
  `Menu (M)` row 0.3615, and on the open goldens `Bits`->`Bit` 0.0066,
  `Chroma`->`Chrom` 0.0071, `Cohere`->`Cohesion` 0.0219.
- *Tolerance choice:* 0.002 is a token allowance (~14 fully-flipped glyph
  pixels, or ~1900 pixels off by one, over a 1080p ROI) above a measured-zero
  noise floor, and sits 3.3x below the smallest change we want to catch
  (0.0067). The previous 0.9 was ~135x too loose: deleting the entire
  `Menu (M)` row measured 0.744 ROI / 0.112 whole-frame and **passed**, as did
  every label rename and the key hints. Its stated justification ("the
  smallest realistic layout change measured 0.29 whole-frame") did not hold —
  the numbers above replace it.

Regenerate deliberately with:  GOLDEN_REGEN=1 pytest tests/test_goldens.py
(only meaningful on cv2 4.x — see GOLDEN_CV2_MAJOR below.)
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
TOLERANCE = 0.01        # whole-frame (secondary — outside-the-panel paint)
ROI_TOLERANCE = 0.002   # panel ROI (primary — see module docstring)
PANEL_W = 290           # OverlayUI.panel_w, scaled like the panel itself
RESOLUTIONS = [(1280, 720), (1920, 1080), (3840, 2160)]
OPEN_RESOLUTIONS = [(1920, 1080), (3840, 2160)]
# Resolutions whose CLOSED column overflows the frame, so the unscrolled pin
# cannot see its own lower rows (see the module docstring). Only 720p does:
# the closed column is 1012 px against 1080 and 2160.
BOTTOM_RESOLUTIONS = [(1280, 720)]

# The cv2 major the goldens were captured on. cv2 5 rasterises Hershey glyphs
# differently (measured: up to 5.2 ROI diff on these same fixtures) — that is a
# rendering-stack change, not a panel regression, so the pins skip rather than
# fail. Regenerating them there would just break everyone on cv2 4.
GOLDEN_CV2_MAJOR = 4
_CV2_MAJOR = int(cv2.__version__.split(".")[0])
needs_golden_cv2 = pytest.mark.skipif(
    _CV2_MAJOR != GOLDEN_CV2_MAJOR,
    reason=(f"goldens were captured on cv2 {GOLDEN_CV2_MAJOR}.x; this is cv2 "
            f"{cv2.__version__} — text rasterisation differs, so these pins "
            f"say nothing about the panel. Install opencv-python<5 to run "
            f"them; do NOT regenerate."),
)


def _bright(w, h):
    """230-gray with a horizontal white gradient — worst-case light background."""
    base = np.full((h, w, 3), 230, np.uint8)
    grad = np.linspace(0, 25, w).astype(np.int16)[None, :, None]
    return np.clip(base.astype(np.int16) + grad, 0, 255).astype(np.uint8)


def _dark(w, h):
    return np.full((h, w, 3), 20, np.uint8)


FIXTURES = {"bright": _bright, "dark": _dark}

# Sections the "open" goldens expand (they boot collapsed — DESIGN.md §4.1).
OPEN_SECTIONS = ("MOTION", "SIGNAL")


def _render_panel(w, h, frame, sections_open=False, scrolled=False):
    ui = OverlayUI(w, h, PRESETS, list(PALETTES), MATTES)
    img = frame.copy()
    if sections_open:
        for title in OPEN_SECTIONS:
            assert title in ui.sections, f"{title} section vanished from the spec"
            ui.sections[title] = True
    if sections_open or scrolled:
        # The column is taller than the frame (always when open; at 720p even
        # closed). Scroll is clamped inside draw() against the LAST measured
        # column height, so draw once on a scratch frame to measure, then
        # scroll past the end (clamped to the bottom) and draw for real.
        # Deterministic: same scroll, byte-identical renders, verified twice
        # in a row.
        ui.draw(np.zeros_like(img), {"status": ""})
        ui.scroll = 10 ** 6
        ui.draw(img, {"status": ""})
        assert ui.scroll > 0, (
            f"{h}p panel did not scroll (column {ui._content_h}px fits the "
            f"frame) — this golden would duplicate the unscrolled one"
        )
        return img
    ui.draw(img, {"status": ""})
    return img


def _mean_abs_diff(a, b):
    return float(np.abs(a.astype(np.float32) - b.astype(np.float32)).mean())


def _panel_roi(img, h):
    """The right-sidebar region the panel draws into (same scaling rule as
    OverlayUI: 290 px at the 1080p baseline, floored at 1.0)."""
    pw = int(round(PANEL_W * max(1.0, h / 1080)))
    return img[:, img.shape[1] - pw:]


def _compare_to_golden(w, h, fixture, suffix="", sections_open=False,
                       scrolled=False):
    img = _render_panel(w, h, FIXTURES[fixture](w, h), sections_open, scrolled)
    path = os.path.join(GOLDEN_DIR, f"panel_{h}p_{fixture}{suffix}.png")
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
        f"{roi_diff:.4f} > {ROI_TOLERANCE} (regenerate deliberately with "
        f"GOLDEN_REGEN=1 if the change is intended)"
    )
    # secondary: whole frame (catches paint OUTSIDE the panel)
    diff = _mean_abs_diff(img, golden)
    assert diff <= TOLERANCE, (
        f"pixels outside the panel drifted in {os.path.basename(path)}: "
        f"whole-frame mean-abs-diff {diff:.4f} > {TOLERANCE}"
    )


@needs_golden_cv2
@pytest.mark.parametrize("w,h", RESOLUTIONS)
@pytest.mark.parametrize("fixture", sorted(FIXTURES))
def test_panel_matches_golden(w, h, fixture):
    """The boot panel: MOTION and SIGNAL collapsed, as it opens."""
    _compare_to_golden(w, h, fixture)


@needs_golden_cv2
@pytest.mark.parametrize("w,h", BOTTOM_RESOLUTIONS)
@pytest.mark.parametrize("fixture", sorted(FIXTURES))
def test_panel_closed_bottom_matches_golden(w, h, fixture):
    """The boot panel at 720p, scrolled to the bottom of its column — the only
    pin on the 292 px that hang below a 720p fold. Without it, dropping the
    `(F)`/`(G)` key hints moves 0.0000 at this resolution and deleting the
    whole `Menu (M)` row moves 0.0026, so 720p was carried by the 1080p/4K
    pins rather than pinning itself (see the module docstring)."""
    _compare_to_golden(w, h, fixture, suffix="_bottom", scrolled=True)


@needs_golden_cv2
@pytest.mark.parametrize("w,h", OPEN_RESOLUTIONS)
@pytest.mark.parametrize("fixture", sorted(FIXTURES))
def test_panel_sections_open_matches_golden(w, h, fixture):
    """MOTION and SIGNAL expanded, scrolled to the bottom of the column — the
    only pin on the boids rows, on SIGNAL's Bits / Gamma / Bias, and on the
    circuit-bent rows (DESIGN.md §2.4/§4.1)."""
    _compare_to_golden(w, h, fixture, suffix="_open_bottom", sections_open=True)


def test_default_panel_fits_1080p_without_scrolling():
    """Existing contract, pinned here with the goldens: the default panel's whole
    control column fits a 1080p window with no scrolling."""
    ui = OverlayUI(1920, 1080, PRESETS, list(PALETTES), MATTES)
    ui.draw(np.zeros((1080, 1920, 3), np.uint8), {"status": ""})
    assert ui._content_h <= 1080
