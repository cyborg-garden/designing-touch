"""Physarum — the slime-mold instrument (Mode protocol, DESIGN.md §2.2).

The trail map IS the picture: a Jones-model mold (dtouch.physarum) grows vein
networks, and the live video steers it two ways — the matte spatially blends
every simulation parameter between a "field" behavior point and a "body"
behavior point (whoever is in frame runs different physics than the room), and
the footage's luminance is food the sensors are drawn toward.

Two engines run the same model behind the same contract: the GPU field
(dtouch.physarum_gl — millions of agents on the full grid, moderngl
ping-pong) is tried first and the numpy field (dtouch.physarum) is the
fallback when no GL context can be had, toasted in amber (DESIGN.md §6.4: the
show still has to start). Headless-testable either way. The mode owns its
engine + matte, its panel sections (TEMPLATES / SOURCE / MOLD / LOOK), one
mode-local command (X swaps the two behavior points, the single most
theatrical move the instrument has), and its defaults + built-in looks.
"""
from __future__ import annotations

import cv2
import numpy as np

from ..commands import Command
from ..hud import AMBER
from ..matte import MatteUnavailable, make_matte, select_matte
from ..panelspec import Cycle, PresetList, Section, Slider, Toggle
from ..physarum import POINT_NAMES, PhysarumField
from ..physarum_gl import PhysarumFieldGL
from .particles import MATTE_H, MATTE_W, MATTES, composite_video_bg

ACCENT = (40, 190, 250)          # BGR — amber, the color of the reference mold

PALETTES_PH = ["arctic", "fire", "aurora", "violet", "toxic", "rose", "mono", "video"]

# gradient stops (RGB), interpolated to a 256-entry LUT per palette
_PALETTE_STOPS = {
    "arctic": [(0, 0, 0), (18, 28, 88), (40, 118, 200), (120, 220, 255), (255, 255, 255)],
    "fire":   [(0, 0, 0), (80, 10, 0), (200, 60, 10), (255, 160, 20), (255, 250, 180)],
    "aurora": [(0, 0, 0), (8, 60, 48), (20, 180, 120), (140, 120, 220), (240, 240, 255)],
    "violet": [(0, 0, 0), (58, 10, 88), (160, 40, 180), (255, 120, 220), (255, 255, 255)],
    "toxic":  [(0, 0, 0), (6, 40, 8), (30, 140, 30), (150, 240, 80), (240, 255, 220)],
    "rose":   [(0, 0, 0), (70, 12, 40), (190, 40, 90), (255, 120, 150), (255, 235, 240)],
    "mono":   [(0, 0, 0), (255, 255, 255)],
}
_LUT_CACHE = {}


def _fmt_agents(n):
    """400000 -> '400k', 2000000 -> '2.0M' — the status line's agent count."""
    if n >= 1_000_000:
        return f"{n / 1e6:.1f}M"
    return f"{n // 1000}k"


def _palette_lut(name):
    """256x3 uint8 RGB ramp for a named palette, cached forever (tiny)."""
    lut = _LUT_CACHE.get(name)
    if lut is None:
        stops = np.array(_PALETTE_STOPS[name], np.float32)
        xs = np.linspace(0.0, 1.0, len(stops))
        t = np.linspace(0.0, 1.0, 256)
        lut = np.stack([np.interp(t, xs, stops[:, c]) for c in range(3)],
                       axis=1).astype(np.uint8)
        _LUT_CACHE[name] = lut
    return lut


def _colorize(lum, name):
    """lum float32 [0,1] (h, w) -> uint8 RGB (h, w, 3) through the palette.

    cv2.applyColorMap with the ramp as a user colormap, not `lut[idx]`:
    numpy's fancy index costs 4.4 ms on the GPU engine's 1280x736 grid,
    more than the whole simulation; applyColorMap is 0.09 ms and
    pixel-identical (it indexes the 256 triplets as given, no channel swap).
    """
    idx = (lum * 255.0).astype(np.uint8)
    return cv2.applyColorMap(idx, _palette_lut(name).reshape(256, 1, 3))


class PhysarumMode:
    """Live video as the pen inside a slime-mold simulation."""

    id = "physarum"
    title = "Physarum"
    key = "o"                    # 'p' belongs to Particles; o as in Organism/Ooze
    accent = ACCENT
    accepts_still = False
    blurb = "slime-mold veins\neat the light"

    # option lists the shell's panel walker needs
    palettes = list(PALETTES_PH)
    mattes = list(MATTES)
    points = list(POINT_NAMES)

    # Ordered by playability — the first 9 seed bank slots 1-9.
    BUILTIN = {
        "veinwork":  dict(point_bg="veins", point_fg="fingers", palette="arctic",
                          matte="auto", food=0.35, gain=1.0, decay=0.94, exposure=3.5),
        "amoeba":    dict(point_bg="cells", point_fg="storm", palette="fire",
                          matte="motion", food=0.50, gain=1.1, decay=0.92, exposure=3.0),
        "ghost":     dict(point_bg="haze", point_fg="web", palette="mono",
                          matte="person", food=0.60, gain=0.9, decay=0.96, exposure=4.5),
        "lightning": dict(point_bg="web", point_fg="fingers", palette="violet",
                          matte="edges", food=0.45, gain=1.4, decay=0.90, exposure=3.0),
        "breath":    dict(point_bg="haze", point_fg="cells", palette="aurora",
                          matte="luma", food=0.30, gain=0.8, decay=0.95, exposure=4.0),
    }

    # apply="reset" merges a look over these; matte / video_bg / video_mix are
    # deliberately absent (keep semantics — rig switches survive look hops).
    DEFAULTS = dict(point_bg="veins", point_fg="fingers", palette="arctic",
                    food=0.35, gain=1.0, decay=0.94, exposure=3.5, grain=0.5)

    _UI_DEFAULTS = dict(ph_matte_idx=0, ph_food=0.35, ph_video_bg=False,
                        ph_video_mix=0.5, ph_point_bg_idx=0, ph_point_fg_idx=2,
                        ph_gain=1.0, ph_decay=0.94, ph_palette_idx=0,
                        ph_exposure=3.5, ph_grain=0.5)

    # Per-engine sizing. The CPU field is budgeted at ~21 ms/frame on the
    # working grid; the GPU field runs 2M agents on a 1280x736 grid in ~8 ms
    # (experiments/08-physarum-gl). An explicit grid / n overrides both.
    ENGINES = ("auto", "gl", "cpu")
    CPU_GRID, CPU_N = (576, 324), 400_000
    GL_GRID, GL_N = (1280, 736), 2_000_000

    def __init__(self, matte="auto", grid=None, n=None, seed=1, engine="auto"):
        if engine not in self.ENGINES:
            raise ValueError(f"engine must be one of {self.ENGINES}, got {engine!r}")
        self.matte_kind = matte
        self._grid = tuple(grid) if grid is not None else None
        self._n = n
        self.grid = self._grid or self.CPU_GRID
        self.n = self._n or self.CPU_N
        self.seed = seed
        self.engine_pref = engine
        self.engine = None           # "gl" / "cpu" once started
        self._px_scale = 1.0         # grid-px per CPU-ground-truth-px (set per engine)
        self.host = None
        self.pf = None
        self.mat = None
        self._burst_pending = False
        self._wave_pending = False

    # ----- lifecycle -----
    def start(self, host):
        self.host = host
        try:
            self.mat = make_matte(self.matte_kind)
        except MatteUnavailable as e:
            # missing optional dep: say so and boot on the default matte —
            # the show still has to start (DESIGN.md §6.4)
            host.hud.toasts.flash(str(e))
            self.matte_kind = "auto"
            self.mat = make_matte(self.matte_kind)
        self.pf = self._build_field(host)

    def _build_field(self, host):
        """The GPU field when it can be had, else the CPU field — sized per
        engine unless the caller fixed grid / n. GL failing to come up is
        not a reason to lose the show (DESIGN.md §6.4): it is toasted in
        amber, printed for the headless log, and the mold runs on numpy."""
        if self.engine_pref != "cpu":
            self.grid = self._grid or self.GL_GRID
            self.n = self._n or self.GL_N
            gw, gh = self.grid
            try:
                pf = PhysarumFieldGL(n=self.n, gw=gw, gh=gh, seed=self.seed)
                self.engine = "gl"
                # The look/point parameters (sense, step — and the blur that
                # sets vein thickness) are calibrated in CPU-grid pixels. On
                # the GL grid a pixel covers ~1/2.2 as much of the frame, so
                # driving the raw values halved the mold's relative scale:
                # every look collapsed into the same fine wire-mesh and the
                # veins fell below what the projector/dither stage resolves.
                # Scale lengths by the grid ratio so the GL field renders the
                # CPU field's composition at higher fidelity.
                self._px_scale = gw / self.CPU_GRID[0]
                pf.diffuse = max(1, round(pf.diffuse * self._px_scale))
                return pf
            except Exception as e:                   # noqa: BLE001 — §6.4
                msg = f"GPU physarum unavailable, running on CPU: {e}"
                host.hud.toasts.flash(msg[:80], AMBER)
                print(msg)
        self.grid = self._grid or self.CPU_GRID
        self.n = self._n or self.CPU_N
        gw, gh = self.grid
        self.engine = "cpu"
        self._px_scale = 1.0
        return PhysarumField(n=self.n, gw=gw, gh=gh, seed=self.seed)

    def stop(self):
        """Release the GPU field if that is what booted; idempotent."""
        if self.pf is not None:
            self.pf.release()
        self.pf = None
        self.mat = None

    def on_resize(self, w, h):
        pass                     # everything derives from host.res per frame

    def configure_ui(self, ui):
        """Seed this mode's ph_* attrs on the shared UI state without
        clobbering anything a look already applied (DitherGirl's pattern)."""
        for k, v in self._UI_DEFAULTS.items():
            if not hasattr(ui, k):
                setattr(ui, k, v)

    # ----- panel / commands -----
    def panel_spec(self):
        pts = list(POINT_NAMES)
        return [
            Section("TEMPLATES", [PresetList()]),
            Section("SOURCE", [
                Cycle("matte", "ph_matte_idx", list(MATTES), save_key="matte",
                      status="matte {}"),
                Slider("Food", "ph_food", 0.0, 1.5, save_key="food",
                       tip="How strongly the footage's light pulls the mold. "
                           "High: the network chases whatever is bright."),
                Toggle("Video bg", "ph_video_bg", save_key="video_bg"),
                Slider("Vid mix", "ph_video_mix", 0.0, 1.0, save_key="video_mix",
                       apply="keep",
                       tip="How visible the raw camera footage is under the veins."),
            ]),
            Section("MOLD", [
                Cycle("body", "ph_point_fg_idx", pts, save_key="point_fg",
                      status="body {}"),
                Cycle("field", "ph_point_bg_idx", pts, save_key="point_bg",
                      status="field {}"),
                Slider("Tempo", "ph_gain", 0.4, 2.5, save_key="gain",
                       tip="Global speed — scales every agent's stride and reach."),
                Slider("Decay", "ph_decay", 0.80, 0.99, save_key="decay",
                       tip="How long trails persist. High: durable veins. "
                           "Low: nervous, fast-forgetting lace."),
            ]),
            Section("LOOK", [
                Cycle("color", "ph_palette_idx", list(PALETTES_PH), gap=4,
                      save_key="palette", status="{}"),
                Slider("Exposure", "ph_exposure", 0.5, 8.0, save_key="exposure",
                       tip="Brightness curve on the trail. High burns the "
                           "veins white; low keeps only the trunk lines."),
                Slider("Grain", "ph_grain", 0.0, 1.5, save_key="grain",
                       tip="This frame's raw agent dust over the smooth "
                           "trail. Zero is airbrushed; high is sandstorm."),
            ]),
        ]

    def commands(self):
        """X swaps body/field points; B pours agents onto the subject;
        W ripples the whole organism outward. Burst/wave land at the matte's
        bright centroid, resolved on the next step (commands run between
        frames, and the shell owns the mouse)."""
        ui, toasts = self.host.ui, self.host.hud.toasts
        pts = self.points

        def _swap():
            ui.ph_point_bg_idx, ui.ph_point_fg_idx = (ui.ph_point_fg_idx,
                                                      ui.ph_point_bg_idx)
            toasts.flash("SWAP  body %s / field %s"
                         % (pts[ui.ph_point_fg_idx % len(pts)],
                            pts[ui.ph_point_bg_idx % len(pts)]))

        def _burst():
            self._burst_pending = True
            toasts.flash("BURST")

        def _wave():
            self._wave_pending = True
            toasts.flash("WAVE")
        return {"physarum.swap": Command("physarum.swap",
                                         "Swap body/field points", "x", _swap),
                "physarum.burst": Command("physarum.burst",
                                          "Spawn burst on the subject", "b", _burst),
                "physarum.wave": Command("physarum.wave",
                                         "Radial wave", "w", _wave)}

    def safe_look(self):
        return "veinwork"

    def status_tail(self, cam_name):
        return f"{self.engine or 'cpu'} {_fmt_agents(self.n)}  cam {cam_name[:16]}"

    def _ui(self, attr, default):
        """Read a live value defensively — step() can run before the shell
        builds the shared UI state (the soak/test path)."""
        ui = self.host.ui if self.host is not None else None
        return getattr(ui, attr, default) if ui is not None else default

    # ----- per-frame -----
    def step(self, frame_bgr, audio_levels, dt):
        """Matte + luma from the frame, one sim frame, colorize the trail.

        The frame is already mirrored by the shell and never None; all float
        work happens at the sim grid, with ONE upscale to host.res at the end.
        """
        ui = self.host.ui if self.host is not None else None
        pf = self.pf
        gw, gh = self.grid
        rw, rh = self.host.res

        if ui is not None and getattr(ui, "ph_matte_idx", None) is not None:
            want = self.mattes[ui.ph_matte_idx % len(self.mattes)]
            if want != self.matte_kind:
                mat, self.matte_kind = select_matte(
                    ui, "ph_matte_idx", self.mattes, self.matte_kind,
                    self.host.hud.toasts)
                if mat is not None:
                    self.mat = mat

        pts = self.points
        pf.point_fg = pts[int(self._ui("ph_point_fg_idx", 2)) % len(pts)]
        pf.point_bg = pts[int(self._ui("ph_point_bg_idx", 0)) % len(pts)]
        pf.decay = float(self._ui("ph_decay", 0.94))
        pf.food = float(self._ui("ph_food", 0.35))
        pf.grain = float(self._ui("ph_grain", 0.5))
        gain = float(self._ui("ph_gain", 1.0))
        exposure = float(self._ui("ph_exposure", 3.5))

        # audio rides on top of the sliders for this frame only: bass pulses
        # the exposure, treble quickens the mold
        if audio_levels is not None:
            sens = float(self._ui("sens", 1.0))
            exposure *= 1.0 + 1.6 * sens * audio_levels["bass"]
            gain *= 1.0 + 0.8 * sens * audio_levels["treble"]
        # gain multiplies sense + step (both in grid px) in either engine, so
        # it doubles as the length-unit conversion onto the GL grid
        pf.gain = gain * self._px_scale
        pf.exposure = exposure

        small = cv2.resize(frame_bgr, (MATTE_W, MATTE_H))
        m = cv2.resize(self.mat.compute(small), (gw, gh))
        gray = cv2.resize(
            cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0,
            (gw, gh))

        if self._burst_pending or self._wave_pending:
            # land the impulse on the lit subject: weighted centroid of
            # matte*luma, falling back to frame center on an empty matte
            w = m * np.clip(gray, 0.05, 1.0)
            tot = float(w.sum())
            if tot > 1e-3:
                cy, cx = np.mgrid[0:gh, 0:gw]
                px_c = float((cx * w).sum() / tot)
                py_c = float((cy * w).sum() / tot)
            else:
                px_c, py_c = gw / 2.0, gh / 2.0
            if self._burst_pending:
                # the burst's footprint is sized in CPU-grid pixels too
                pf.spawn_burst(px_c, py_c, radius=6.0 * self._px_scale)
            if self._wave_pending:
                pf.wave(px_c, py_c)
            self._burst_pending = self._wave_pending = False

        pf.update(m, gray)
        lum = pf.luminance()

        pal = self.palettes[int(self._ui("ph_palette_idx", 0)) % len(self.palettes)]
        if pal == "video":
            # veins lit by the footage's own color — the mold as a lampshade
            color = cv2.cvtColor(cv2.resize(small, (gw, gh)),
                                 cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            out_small = (lum[..., None] * (0.25 + 0.75 * color) * 255.0).astype(np.uint8)
        else:
            out_small = _colorize(lum, pal)
        out = cv2.resize(out_small, (rw, rh), interpolation=cv2.INTER_LINEAR)

        if bool(self._ui("ph_video_bg", False)):
            out = composite_video_bg(out, frame_bgr,
                                     float(self._ui("ph_video_mix", 0.5)))
        return np.ascontiguousarray(out)
