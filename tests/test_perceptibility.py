"""Perceptibility floors — the panel may not show a control that bends nothing.

The magic-over-control decision (2026-08-24): a visible control that does
nothing perceptible in the current state is a bug — hide it (panelspec's
`show_when` visibility gate) or make it matter. These tests hold BOTH sides
of that contract against the real rendered picture, so the surface cannot
silently go dead (or a gate silently go stale) again:

- **visible ⇒ perceptible**: every control the panel draws in a state must
  move the rendered frames above a floor somewhere in its range, in that
  state. A new widget added without a perceptible effect fails here.
- **gated-off ⇒ imperceptible**: every `show_when`-gated control must
  genuinely do nothing where it hides (that is the gate's justification),
  and must clear the floor in its gate-ON state (the gate must not hide a
  live control).

Method: the real Host loop runs fully headless (synthetic deterministic
source, injected writer, faked clock so dt is exactly 1/30 — no camera, no
window, no audio), one control changed per render, mean-abs-diff over the
tail frames. Everything is deterministic, so the diff is exactly the
control's effect.

The 2026-08-25 audit sweep (36 frames at 192x108, all candidate pairs) that
these floors were distilled from measured, e.g.: Chroma/Drift/Crush/
Scanlines/dither/Bits/Gamma/bias = 0.000 in every mode with Glitch off
(13.7-56.0 with it on); Cohere/Align/Separate = 0.000 with Flock off
(15.5-32.5 on); Vid mix = 0.000 with Video bg off (31.4-51.1 on); Dither
mode's bias = 0.000 under the error-diffusion algorithms (72.9 under Bayer)
and its Hue = 0.000 at Tint 0 (9.1 with Tint up).
"""
import itertools
import os
import time as _time
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

import dtouch.shell as shell_mod
from dtouch.modes.dithergirl import DitherGirlMode
from dtouch.modes.particles import ParticlesMode
from dtouch.modes.physarum import PhysarumMode
from dtouch.panelspec import Cycle, Slider, Toggle, visible, walk_spec
from dtouch.shell import Host

RES = (160, 90)
FRAMES = 24
TAIL = 8
# mean-abs-diff (uint8 scale) over the tail frames. Real effects measured
# 4.6-244; a dead control measures exactly 0.0 (the runs are deterministic).
FLOOR = 0.75


# ----- deterministic headless rig ------------------------------------------

_clock = {"t": 0.0}


class _FakeTime:
    """Stands in for dtouch.shell's `time` during a render: monotonic
    advances 1/30 s per source frame (the source ticks it), so dt — and with
    it every simulation trajectory — is identical run to run."""

    @staticmethod
    def monotonic():
        return _clock["t"]

    @staticmethod
    def perf_counter():
        return _clock["t"]

    @staticmethod
    def time():
        return _clock["t"]

    @staticmethod
    def sleep(_s):
        pass

    def __getattr__(self, name):
        return getattr(_time, name)


class _DetSource:
    """Deterministic animated scene: a drifting bright blob over
    frame-indexed noise. Same frames every run, motion for the mattes."""
    name = "det"

    def __init__(self, on_read=None, w=96, h=54):
        self.w, self.h = w, h
        self.i = 0
        self.on_read = on_read

    def read(self):
        self.i += 1
        _clock["t"] += 1.0 / 30.0
        if self.on_read:
            self.on_read(self.i)
        rng = np.random.default_rng(self.i)
        yy, xx = np.mgrid[0:self.h, 0:self.w].astype(np.float32)
        cx = self.w * (0.45 + 0.20 * np.sin(self.i / 9.0))
        cy = self.h * (0.5 + 0.10 * np.cos(self.i / 13.0))
        g = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2)
                   / (2 * (self.h * 0.28) ** 2))
        img = np.clip(g * 215.0 + rng.integers(0, 40, (self.h, self.w)),
                      0, 255).astype(np.uint8)
        return True, cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    def release(self):
        pass


class _Writer:
    def __init__(self):
        self.frames = []

    def append_data(self, f):
        self.frames.append(f.copy())

    def close(self):
        pass


_MODES = {
    "particles": lambda: ParticlesMode(n=12000, grid=(96, 54), seed=1),
    "physarum": lambda: PhysarumMode(engine="cpu", grid=RES, n=16000,
                                     seed=7, matte="luma"),
    "dithergirl": lambda: DitherGirlMode(),
}

_render_cache = {}


def _render(mode_id, overrides, tmp_path):
    """Tail frames (float32) of a headless run with `overrides` written onto
    the shared UI state before the first frame. Cached per settings."""
    key = (mode_id, tuple(sorted((k, repr(v)) for k, v in overrides.items())))
    if key in _render_cache:
        return _render_cache[key]
    _clock["t"] = 0.0
    real_time, holder, wr = shell_mod.time, {}, _Writer()

    def fake_start(self):
        self.writer, self.rec_path = wr, "fake"

    def on_read(i):
        if i == 1:
            ui = holder["host"].ui
            for k, v in overrides.items():
                setattr(ui, k, v)
            ui.record = True          # routes frames into the fake writer

    shell_mod.time = _FakeTime()
    real_start = Host._start_recording
    Host._start_recording = fake_start
    try:
        host = Host(_MODES[mode_id](), source=_DetSource(on_read=on_read),
                    res=RES, show=False, max_frames=FRAMES, preset=None,
                    presets_path=str(tmp_path / f"{len(_render_cache)}p.json"),
                    state_path=str(tmp_path / f"{len(_render_cache)}s.json"))
        holder["host"] = host
        host.run()
    finally:
        shell_mod.time = real_time
        Host._start_recording = real_start
    assert len(wr.frames) >= TAIL, (mode_id, overrides, len(wr.frames))
    out = [f.astype(np.float32) for f in wr.frames[-TAIL:]]
    _render_cache[key] = out
    return out


def _diff(mode_id, state, a, b, tmp_path):
    fa = _render(mode_id, {**state, **a}, tmp_path)
    fb = _render(mode_id, {**state, **b}, tmp_path)
    return float(np.mean([np.abs(x - y).mean() for x, y in zip(fa, fb)]))


def _max_diff(mode_id, state, attr, values, tmp_path):
    return max(_diff(mode_id, state, {attr: a}, {attr: b}, tmp_path)
               for a, b in itertools.combinations(values, 2))


# ----- what gets swept ------------------------------------------------------

# Controls the sweep must not touch, with the reason. Their perceptibility is
# either meaningless (system surface) or unverifiable headless (hardware).
SKIP = {
    "res_idx": "system: output resolution (already nudge=False)",
    "record": "system: starts the recorder (the harness rides it)",
    "quit": "system",
    "audio": "hardware: toggling it would open the microphone",
    "sens": "audio-gated; the mic never opens here — visibility gate on "
            "`audio` is asserted in test_gate_states_match_visibility",
    "input_idx": "system: still-image picker",
    "ph_quality_idx": "system: sim quality tier (save=False, nudge=False)",
}

# attr -> the state overrides under which a show_when-gated control acts.
# The off state is always the mode's boot default ({}), where every one of
# these measured exactly 0.000 in the audit sweep.
GATE_ON = {
    "dither_idx": {"glitch": True},
    "sig_px": {"glitch": True},
    "sig_bits": {"glitch": True},
    "sig_gamma": {"glitch": True},
    "sig_bias_idx": {"glitch": True},
    "chroma": {"glitch": True},
    "drift": {"glitch": True},
    "crush": {"glitch": True},
    "scanlines": {"glitch": True},
    "cohere": {"flock": True},
    "align": {"flock": True},
    "separate": {"flock": True},
    "video_mix": {"video_bg": True},
    "ph_video_mix": {"ph_video_bg": True},
    "dg_matte_black": {"dg_matte_idx": 2},     # motion matte on
    "dg_hue": {"dg_tint": 0.6},
    "dg_bias_idx": {"dg_algo_idx": 0},         # Bayer (ordered)
}

# (mode, attr) -> extra overrides for the gate-ON perceptibility check.
# Dither mode's default mono palette is pure black/white, on which bit-crush
# is mathematically an identity at every depth; amber's off-white ink moves.
ON_EXTRA = {
    ("dithergirl", "crush"): {"dg_palette_idx": 1},
}

# Cycle candidates: default first three options; matte cycles skip the
# heavyweight person/saliency entries (model downloads, nondeterminism).
CYCLE_CANDS = {
    "matte_idx": [0, 1, 5],          # auto / motion / luma
    "ph_matte_idx": [0, 1, 5],
    "dg_matte_idx": [0, 2, 6],       # off / motion / luma
    "dg_algo_idx": [0, 2, 4],        # Bayer / Floyd-Steinberg / ASCII
}


def _candidates(w, on_check):
    """Values to probe a control at. Extremes alone are not enough: Hue's
    ends are the same colour (0 == 360) and Crush's hi (8 bits) is a near
    no-op, so sliders get lo / one-step / mid / hi."""
    if isinstance(w, Slider):
        vals = [w.lo, (w.lo + w.hi) / 2.0, w.hi]
        if w.step:
            vals.append(min(w.lo + w.step, w.hi))
        return sorted(set(vals)) if on_check else [w.lo, w.hi]
    if isinstance(w, Toggle):
        return [False, True]
    if isinstance(w, Cycle):
        cands = CYCLE_CANDS.get(w.attr, [0, 1, 2])
        return [c for c in cands if c < len(list(w.options))]
    return []


def _composed_spec(mode_id):
    mode = _MODES[mode_id]()
    host = Host(mode, source=_DetSource(), res=RES, show=False, max_frames=0,
                preset=None, presets_path="unused-p.json",
                state_path="unused-s.json")
    return host.compose_spec(mode)


def _controls(mode_id):
    """(widget, gated) for every sweepable control on the mode's panel."""
    out = []
    for _section, w in walk_spec(_composed_spec(mode_id)):
        attr = getattr(w, "attr", None)
        if attr is None or attr in SKIP:
            continue
        if not isinstance(w, (Slider, Toggle, Cycle)):
            continue
        out.append((w, getattr(w, "show_when", None) is not None))
    return out


# ----- the contract ---------------------------------------------------------

@pytest.mark.parametrize("mode_id", sorted(_MODES))
def test_every_visible_control_is_perceptible(mode_id, tmp_path):
    """visible ⇒ perceptible: in its gate-ON state (boot state for ungated
    controls), every drawn control must move the picture above the floor."""
    failures = []
    for w, gated in _controls(mode_id):
        state = dict(GATE_ON[w.attr]) if gated else {}
        state.update(ON_EXTRA.get((mode_id, w.attr), {}))
        d = _max_diff(mode_id, state, w.attr,
                      _candidates(w, on_check=True), tmp_path)
        if d < FLOOR:
            failures.append(f"{w.attr} ({w.label}) in {state or 'base'}: "
                            f"{d:.3f} < {FLOOR}")
    assert not failures, (
        f"[{mode_id}] controls the panel draws but the picture ignores "
        f"(hide them with show_when, or fix the wiring):\n  "
        + "\n  ".join(failures))


@pytest.mark.parametrize("mode_id", sorted(_MODES))
def test_gated_controls_are_dead_exactly_where_hidden(mode_id, tmp_path):
    """gated-off ⇒ imperceptible: where a show_when gate hides a control
    (the boot state, for every current gate), the control must truly move
    nothing — otherwise the gate is hiding a live control."""
    failures = []
    for w, gated in _controls(mode_id):
        if not gated:
            continue
        assert w.attr in GATE_ON, (
            f"[{mode_id}] {w.attr} declares show_when but has no gate-ON "
            f"state registered here — add one so both sides stay tested")
        d = _max_diff(mode_id, {}, w.attr,
                      _candidates(w, on_check=False), tmp_path)
        if d >= FLOOR:
            failures.append(f"{w.attr} ({w.label}): {d:.3f} in the hidden "
                            f"state — the gate is hiding a live control")
    assert not failures, f"[{mode_id}]:\n  " + "\n  ".join(failures)


@pytest.mark.parametrize("mode_id", sorted(_MODES))
def test_gate_states_match_visibility(mode_id):
    """The show_when predicates agree with the states the sweep uses: hidden
    in the boot state, shown in the registered gate-ON state. Pure spec — no
    rendering — so it also covers the mic-gated Sens row."""
    for w, gated in _controls(mode_id):
        if not gated:
            continue
        assert not visible(SimpleNamespace(), w), \
            f"[{mode_id}] {w.attr} should hide in the boot state"
        on = SimpleNamespace(**GATE_ON[w.attr])
        assert visible(on, w), \
            f"[{mode_id}] {w.attr} should draw in {GATE_ON[w.attr]}"
    # the one hardware-gated row the render sweep must not exercise
    if mode_id == "particles":
        sens = next(w for _s, w in walk_spec(_composed_spec(mode_id))
                    if getattr(w, "attr", None) == "sens")
        assert not visible(SimpleNamespace(), sens)
        assert visible(SimpleNamespace(audio=True), sens)
