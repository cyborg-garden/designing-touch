"""Matte operators — extract an "interest field" from a frame, subject-agnostic.

A matte is a float32 (H, W) array in [0, 1]: how much each pixel belongs to the thing the
particles should form. It is NOT person-specific — it keys on what stands out or moves, so it
works for a dancer, a crowd, a boat, a passing hand, anything.

Strategies:
- SaliencyMatte  — spectral-residual saliency (Hou & Zhang), pure NumPy FFT. Works on a single
  still frame, any subject. The robust default.
- MotionMatte    — MOG2 background subtraction. Whatever moves becomes the subject (great for a
  dancer or a moving boat on a static camera). Needs a few frames to warm up.
- EdgeMatte      — Canny edges, dilated. Outlines of whatever is there.
- LumaMatte       — plain brightness.
- PersonMatte    — MediaPipe selfie segmentation (multi-person). One *option*, not the core.
- AutoMatte      — motion ∪ saliency: moving things dominate, static-but-salient things still
  register. The recommended general default.

All return the same (H, W) shape as the input frame.
"""
from __future__ import annotations

import os
import urllib.request

import cv2
import numpy as np

from .hud import AMBER

_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/image_segmenter/"
              "selfie_segmenter/float16/latest/selfie_segmenter.tflite")
_MODEL_PATH = os.path.join(os.path.dirname(__file__), "assets", "selfie_segmenter.tflite")


def _norm(a: np.ndarray) -> np.ndarray:
    lo, hi = float(a.min()), float(a.max())
    return ((a - lo) / (hi - lo + 1e-9)).astype(np.float32)


def _gray(frame_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0


class SaliencyMatte:
    """Spectral-residual saliency — subject-agnostic, single-frame, cheap."""

    def __init__(self, small: int = 64):
        self.small = small

    def compute(self, frame_bgr: np.ndarray) -> np.ndarray:
        h, w = frame_bgr.shape[:2]
        g = cv2.resize(_gray(frame_bgr), (self.small, self.small))
        f = np.fft.fft2(g)
        log_amp = np.log(np.abs(f) + 1e-9)
        phase = np.angle(f)
        residual = log_amp - cv2.blur(log_amp, (3, 3))
        sal = np.abs(np.fft.ifft2(np.exp(residual + 1j * phase))) ** 2
        sal = cv2.GaussianBlur(sal, (0, 0), 2.0)
        return _norm(cv2.resize(sal, (w, h)))


class MotionMatte:
    """MOG2 background subtraction — moving things become the subject."""

    def __init__(self, history: int = 200, var_threshold: float = 25.0):
        self._bg = cv2.createBackgroundSubtractorMOG2(
            history=history, varThreshold=var_threshold, detectShadows=False)
        self.frames = 0

    def compute(self, frame_bgr: np.ndarray) -> np.ndarray:
        fg = self._bg.apply(frame_bgr)
        self.frames += 1
        m = cv2.GaussianBlur(fg.astype(np.float32) / 255.0, (0, 0), 3.0)
        return np.clip(m, 0.0, 1.0)

    @property
    def warm(self) -> bool:
        return self.frames > 12


class EdgeMatte:
    def compute(self, frame_bgr: np.ndarray) -> np.ndarray:
        g = (_gray(frame_bgr) * 255).astype(np.uint8)
        e = cv2.Canny(g, 60, 160)
        e = cv2.dilate(e, np.ones((3, 3), np.uint8), iterations=1)
        return cv2.GaussianBlur(e.astype(np.float32) / 255.0, (0, 0), 1.5)


class LumaMatte:
    def compute(self, frame_bgr: np.ndarray) -> np.ndarray:
        return _gray(frame_bgr)


class PersonMatte:
    """MediaPipe selfie segmentation (multi-person). Optional, person-specific."""

    def __init__(self, model_path: str = _MODEL_PATH):
        ensure_person_model(model_path)
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        self._mp = mp
        self._seg = vision.ImageSegmenter.create_from_options(
            vision.ImageSegmenterOptions(
                base_options=python.BaseOptions(model_asset_path=model_path),
                output_confidence_masks=True))

    def compute(self, frame_bgr: np.ndarray) -> np.ndarray:
        rgb = np.ascontiguousarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        res = self._seg.segment(img)
        return res.confidence_masks[0].numpy_view().astype(np.float32)


class AutoMatte:
    """Motion ∪ saliency: movement dominates, static-but-salient things still register."""

    def __init__(self):
        self.motion = MotionMatte()
        self.saliency = SaliencyMatte()

    def compute(self, frame_bgr: np.ndarray) -> np.ndarray:
        sal = self.saliency.compute(frame_bgr)
        mot = self.motion.compute(frame_bgr)
        if not self.motion.warm:
            return sal
        return np.clip(np.maximum(mot, 0.6 * sal), 0.0, 1.0)


def ensure_person_model(path: str = _MODEL_PATH):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        urllib.request.urlretrieve(_MODEL_URL, path)
    return path


KINDS = {
    "auto": AutoMatte, "saliency": SaliencyMatte, "motion": MotionMatte,
    "edges": EdgeMatte, "luma": LumaMatte, "person": PersonMatte,
}

# Plain-language fixes for the optional mattes (DESIGN.md §6.4: a human
# summary, never a traceback). Only `person` has a dependency beyond cv2.
UNAVAILABLE_HINT = {
    "person": "person matte needs the [person] extra",
}


class MatteUnavailable(RuntimeError):
    """An optional matte cannot be built here — its dependency is missing, or
    its model could not be fetched. Distinct from a bad `kind` (KeyError):
    this one is an install/network fact about the machine, so it is a thing to
    say out loud and step back from, not a bug (DESIGN.md §6.4)."""


def make_matte(kind: str = "auto"):
    """Build a matte operator. Raises MatteUnavailable when the optional
    dependency is missing or its model cannot be fetched — a `person` matte on
    a start.command install with no mediapipe used to raise ModuleNotFoundError
    straight out of a mode's step() and end the show with a traceback."""
    factory = KINDS[kind]                       # unknown kind stays a KeyError
    try:
        return factory()
    except (ImportError, OSError) as e:         # missing extra / model fetch
        raise MatteUnavailable(
            UNAVAILABLE_HINT.get(kind, f"{kind} matte is not available here")
        ) from e


def select_matte(ui, attr, options, current, toasts=None):
    """Switch the matte cycle at ``ui.<attr>`` to its selected option.

    A matte that cannot be built here is not an error the operator can act on
    mid-performance, so it is not allowed to reach the frame loop: the cycle
    REVERTS to `current`, the plain-language fix is toasted in amber, and the
    caller keeps the operator it already had. Returns (operator, kind) with
    operator None when the build was refused — the shipped `portrait` and
    `sigil` templates both select `person`, so this is one click away on any
    install without the [person] extra."""
    kind = options[int(getattr(ui, attr, 0)) % len(options)]
    try:
        return make_matte(kind), kind
    except MatteUnavailable as e:
        if current in options:
            setattr(ui, attr, options.index(current))
        if toasts is not None:
            toasts.flash(str(e), AMBER)
        print("matte unavailable:", e)
        return None, current
