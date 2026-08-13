"""Shell-level regression pins for the verified UI-overhaul bug list.

Everything here drives the real Host loop headless with DitherGirlMode (pure
numpy/cv2 — no GL, so these run everywhere test_dithergirl.py does). GUI-path
tests monkeypatch cv2's window calls so `show=True` code paths run headless.
"""
import numpy as np
import pytest

import cv2

from dtouch import presets
from dtouch.hud import OverlayState
from dtouch.modes.dithergirl import DitherGirlMode
from dtouch.shell import Host

RES = (192, 108)


class SyntheticSource:
    """Deterministic frame source (seeded per read) with an optional per-read
    hook and a fail-from mode (fail_after=0 = never yields a frame)."""

    def __init__(self, w=64, h=36, on_read=None, fail_after=None):
        self.w, self.h = w, h
        self.reads = 0
        self.on_read = on_read
        self.fail_after = fail_after
        self.released = False
        self.name = "synthetic"

    def read(self):
        self.reads += 1
        if self.on_read:
            self.on_read(self.reads)
        if self.fail_after is not None and self.reads > self.fail_after:
            return False, None
        rng = np.random.default_rng(self.reads)
        return True, rng.integers(0, 256, (self.h, self.w, 3), np.uint8)

    def release(self):
        self.released = True


class FakeWriter:
    def __init__(self):
        self.frames = []
        self.closed = False

    def append_data(self, f):
        assert not self.closed
        self.frames.append(np.asarray(f).copy())

    def close(self):
        self.closed = True


def _paths(tmp_path):
    return dict(presets_path=str(tmp_path / "presets.json"),
                state_path=str(tmp_path / "state.json"))


def _host(tmp_path, mode=None, src=None, **kw):
    kw.setdefault("res", RES)
    kw.setdefault("show", False)
    kw.setdefault("preset", None)
    return Host(mode or DitherGirlMode(), source=src or SyntheticSource(),
                **_paths(tmp_path), **kw)


def _booted(tmp_path, **kw):
    host = _host(tmp_path, max_frames=1, **kw)
    host.run()
    return host


def _hints(host):
    return [t.text for t in host.hud.toasts._hints]


def _patch_gui(monkeypatch, keys=(), shown=None):
    """Run show=True paths headless: no-op the cv2 window calls; waitKey pops
    from `keys` (then 255); imshow appends to `shown` when given."""
    seq = list(keys)
    monkeypatch.setattr(cv2, "namedWindow", lambda *a, **k: None)
    monkeypatch.setattr(cv2, "setMouseCallback", lambda *a, **k: None)
    monkeypatch.setattr(cv2, "imshow",
                        (lambda win, img: shown.append(img.copy()))
                        if shown is not None else (lambda *a: None))
    monkeypatch.setattr(cv2, "waitKey",
                        lambda ms=0: seq.pop(0) if seq else 255)
    monkeypatch.setattr(cv2, "getWindowProperty", lambda *a: 1.0)
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)


# ---------- corrupt presets file: the note reaches the toasts ----------

def test_corrupt_presets_note_reaches_the_toasts(tmp_path):
    p = _paths(tmp_path)
    with open(p["presets_path"], "w") as f:
        f.write("{broken")
    host = _booted(tmp_path)
    assert any("backup" in t for t in _hints(host))
    assert list(tmp_path.glob("presets.corrupt.*.bak.json"))
