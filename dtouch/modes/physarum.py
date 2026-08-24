"""Physarum — the slime-mold instrument (Mode protocol, DESIGN.md §2.2).

The trail map IS the picture: a Jones-model mold (dtouch.physarum) grows vein
networks, and the live video steers it two ways — the matte spatially blends
every simulation parameter between a "field" behavior point and a "body"
behavior point (whoever is in frame runs different physics than the room), and
the footage's luminance is food the sensors are drawn toward.

Three feel knobs sit on top of the model (MOLD section): **Weave** drives the
engine's anti-thoroughfare levers (sensor saturation, heading jitter,
sub-population sense split, reseed churn, diffuse trim) so the picture leans
reticulated-network instead of a few fat canals; **Evolve** breathes the
point geometry on slow incommensurate sine walks and marches a phantom food
blob around the frame so a still scene keeps reorganizing; **React** turns
motion into carving — motion history feeds the sensors, holds trail decay
where you swept, and fires local gather impulses at the gesture.

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

from ..circuit_bent import CircuitBent
from ..commands import Command
from ..hud import AMBER
from ..matte import MatteUnavailable, make_matte, select_matte
from ..overlay_ui import RES_OPTIONS, sync_signal
from ..panelspec import Cycle, PresetList, Section, Slider, Toggle
from ..physarum import POINT_NAMES, PhysarumField
from ..physarum_gl import PhysarumFieldGL
from ..rack_gl import PhysarumOutGL
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
    # weave/evolve/react per look keep each identity while tilting the whole
    # instrument toward network-not-thoroughfares (owner feedback 2026-08):
    # veinwork and lightning web hardest; breath stays the calmest.
    BUILTIN = {
        "veinwork":  dict(point_bg="veins", point_fg="fingers", palette="arctic",
                          matte="auto", food=0.35, gain=1.0, decay=0.94, exposure=3.5,
                          weave=0.7, evolve=0.5, react=0.7),
        "amoeba":    dict(point_bg="cells", point_fg="storm", palette="fire",
                          matte="motion", food=0.50, gain=1.1, decay=0.92, exposure=3.0,
                          weave=0.55, evolve=0.6, react=0.8),
        "ghost":     dict(point_bg="haze", point_fg="web", palette="mono",
                          matte="person", food=0.60, gain=0.9, decay=0.96, exposure=4.5,
                          weave=0.5, evolve=0.75, react=0.6),
        "lightning": dict(point_bg="web", point_fg="fingers", palette="violet",
                          matte="edges", food=0.45, gain=1.4, decay=0.90, exposure=3.0,
                          weave=0.65, evolve=0.6, react=0.8),
        "breath":    dict(point_bg="haze", point_fg="cells", palette="aurora",
                          matte="luma", food=0.30, gain=0.8, decay=0.95, exposure=4.0,
                          weave=0.45, evolve=0.7, react=0.5),
    }

    # apply="reset" merges a look over these; matte / video_bg / video_mix are
    # deliberately absent (keep semantics — rig switches survive look hops).
    DEFAULTS = dict(point_bg="veins", point_fg="fingers", palette="arctic",
                    food=0.35, gain=1.0, decay=0.94, exposure=3.5, grain=0.5,
                    weave=0.6, evolve=0.5, react=0.7)

    _UI_DEFAULTS = dict(ph_matte_idx=0, ph_food=0.35, ph_video_bg=False,
                        ph_video_mix=0.5, ph_point_bg_idx=0, ph_point_fg_idx=2,
                        ph_gain=1.0, ph_decay=0.94, ph_palette_idx=0,
                        ph_exposure=3.5, ph_grain=0.5,
                        ph_weave=0.6, ph_evolve=0.5, ph_react=0.7,
                        ph_quality_idx=0)

    # Per-engine sizing. The CPU field is budgeted at ~21 ms/frame on the
    # working grid; the GPU field runs 2M agents on a 1280x736 grid in ~8 ms
    # (experiments/08-physarum-gl). An explicit grid / n overrides both.
    ENGINES = ("auto", "gl", "cpu")
    CPU_GRID, CPU_N = (576, 324), 400_000
    GL_GRID, GL_N = (1280, 736), 2_000_000

    # Render-quality tiers (GL engine only; the CPU fallback has no headroom).
    # perform = the shipped sizing; higher tiers raise the sim grid + agent
    # pool so the veins stay crisp on a 1440p/4K projector. Switching tiers
    # rebuilds the field live — the trail regrows in a couple of seconds.
    QUALITY = {
        "perform": ((1280, 736), 2_000_000),
        "balance": ((1920, 1104), 3_000_000),
        "quality": ((2560, 1472), 4_000_000),
    }
    QUALITY_NAMES = list(QUALITY)

    def __init__(self, matte="auto", grid=None, n=None, seed=1, engine="auto"):
        if engine not in self.ENGINES:
            raise ValueError(f"engine must be one of {self.ENGINES}, got {engine!r}")
        self.matte_kind = matte
        self._grid = tuple(grid) if grid is not None else None
        self._n = n
        self._quality = "perform"
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
        self._t = 0.0                # evolve clock (sums clamped dt)
        self._prev_gray = None       # last grid-res luma (react's motion diff)
        self._motion = None          # lingering motion-energy map (react)
        # SIGNAL-on-GPU state (dtouch.rack_gl): the output composer + rack
        # on the field's context, and the per-frame flag telling the shell
        # its CPU rack already ran here (DESIGN.md §2.4)
        self._glout = None
        self._glout_key = None
        self._rack_gl_ok = True      # one GL failure disables the path (§6.4)
        self.signal_done = False

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
        # any prior GL output composer belonged to the old field/context
        self._glout = None
        self._glout_key = None
        if self.engine_pref != "cpu":
            q_grid, q_n = self.QUALITY.get(self._quality, self.QUALITY["perform"])
            self.grid = self._grid or q_grid
            self.n = self._n or q_n
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
                self._base_diffuse = pf.diffuse
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
        pf = PhysarumField(n=self.n, gw=gw, gh=gh, seed=self.seed)
        self._base_diffuse = pf.diffuse
        return pf

    def stop(self):
        """Release the GPU field if that is what booted; idempotent."""
        if self.pf is not None:
            self.pf.release()      # the composer's objects die with the ctx
        self.pf = None
        self.mat = None
        self._glout = None
        self._glout_key = None

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
                      status="matte {}",
                      tip="How the camera finds you: your whole body, only "
                          "what moves, edges, or the brightest parts."),
                Slider("Food", "ph_food", 0.0, 1.5, save_key="food",
                       tip="How strongly the footage's light pulls the mold. "
                           "High: the network chases whatever is bright."),
                Toggle("Video bg", "ph_video_bg", save_key="video_bg"),
                # only acts while Video bg is on (both engines gate the
                # composite on ph_video_bg) — hidden otherwise
                # (panelspec.visible; magic-over-control, 2026-08-24)
                Slider("Vid mix", "ph_video_mix", 0.0, 1.0, save_key="video_mix",
                       apply="keep",
                       show_when=lambda s: bool(getattr(s, "ph_video_bg", False)),
                       tip="How visible the raw camera footage is under the veins."),
                Cycle("output", "res_idx", [n for n, _, _ in RES_OPTIONS],
                      key="res", save=False, nudge=False,
                      tip="The window / projector resolution."),
                Cycle("quality", "ph_quality_idx", list(self.QUALITY_NAMES),
                      save=False, nudge=False,
                      tip="How finely the mold itself is simulated. Higher "
                          "keeps veins crisp on a big projector, and costs "
                          "speed. Switching regrows the field in seconds."),
            ]),
            Section("MOLD", [
                Cycle("body", "ph_point_fg_idx", pts, save_key="point_fg",
                      status="body {}",
                      tip="How the mold behaves ON you — where the camera "
                          "sees you, it grows in this style."),
                Cycle("field", "ph_point_bg_idx", pts, save_key="point_bg",
                      status="field {}",
                      tip="How the mold behaves in the rest of the room, "
                          "away from you."),
                Slider("Tempo", "ph_gain", 0.4, 2.5, save_key="gain",
                       tip="Global speed — scales every agent's stride and reach."),
                Slider("Decay", "ph_decay", 0.80, 0.99, save_key="decay",
                       tip="How long trails persist. High: durable veins. "
                           "Low: nervous, fast-forgetting lace."),
                Slider("Weave", "ph_weave", 0.0, 1.0, save_key="weave",
                       tip="Fine webbing. Low: a few bold canals. High: a "
                           "dense net of thin threads and crossings."),
                Slider("Evolve", "ph_evolve", 0.0, 1.0, save_key="evolve",
                       tip="The mold rearranges itself over time, even when "
                           "nothing moves. Zero holds one structure."),
                Slider("React", "ph_react", 0.0, 1.0, save_key="react",
                       tip="How hard your movement carves it. High: motion "
                           "pours mold into the path you sweep."),
            ]),
            Section("LOOK", [
                Cycle("color", "ph_palette_idx", list(PALETTES_PH), gap=4,
                      save_key="palette", status="{}",
                      tip="The color the veins glow in. 'video' lights them "
                          "with the camera's own colors."),
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

    @staticmethod
    def signal_dither_rows(out_h):
        """SIGNAL-rack dither working rows for this mode: 1/6 of the output
        height (a ~6-px cell at any resolution) instead of the rack's fixed
        72 rows, which turn the mold's smooth veins into boulder-sized grain
        at 1080p+. The floor keeps the cell look at small windows. Expressed
        in rows so a GPU dither pass can mirror it as a quantized-UV cell."""
        return max(96, out_h // 6)

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

        # quality tier switch: rebuild the GL field at the new sizing (the
        # trail regrows in a couple of seconds; a fixed grid/n override and
        # the CPU fallback both ignore the cycle — no headroom there)
        want_q = self.QUALITY_NAMES[int(self._ui("ph_quality_idx", 0))
                                    % len(self.QUALITY_NAMES)]
        if (want_q != self._quality and self.engine == "gl"
                and self._grid is None and self._n is None):
            self._quality = want_q
            self.pf.release()
            self.pf = self._build_field(self.host)
            if self.engine == "gl":
                g_w, g_h = self.grid
                self.host.hud.toasts.flash(
                    f"quality {want_q}  {g_w}x{g_h} {_fmt_agents(self.n)}")
        elif want_q != self._quality:
            self._quality = want_q      # remember; applies if GL boots later

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
        decay = float(self._ui("ph_decay", 0.94))
        food = float(self._ui("ph_food", 0.35))
        pf.grain = float(self._ui("ph_grain", 0.5))
        gain = float(self._ui("ph_gain", 1.0))
        exposure = float(self._ui("ph_exposure", 3.5))
        weave = min(max(float(self._ui("ph_weave", 0.6)), 0.0), 1.0)
        evolve = min(max(float(self._ui("ph_evolve", 0.5)), 0.0), 1.0)
        react = min(max(float(self._ui("ph_react", 0.7)), 0.0), 1.0)

        # audio rides on top of the sliders for this frame only: bass pulses
        # the exposure, treble quickens the mold
        if audio_levels is not None:
            sens = float(self._ui("sens", 1.0))
            exposure *= 1.0 + 1.6 * sens * audio_levels["bass"]
            gain *= 1.0 + 0.8 * sens * audio_levels["treble"]

        # evolve: the mold drifts through its own parameter space on slow,
        # incommensurate sine walks (13-31 s periods — never repeats), so the
        # field visibly reorganizes over ~10-30 s even on a still scene. The
        # structural multipliers (sense / turn / spread geometry) are what
        # actually re-knit the topology; gain/food/decay breathe on top.
        # Our own design: continuous modulation of THIS mode's own
        # parameters, no external parameter tables.
        self._t += dt
        weave_base = weave
        if evolve > 0:
            tau = self._t * (2.0 * np.pi)
            # the structural movers ride sqrt(evolve): a woven mesh anchors
            # itself hard (measured: mid evolve barely decorrelated a dense
            # web at 10 s lag), so mid-slider needs near-full reorganizing
            # strength while the top stays the same
            es = float(np.sqrt(evolve))
            gain *= 1.0 + 0.22 * evolve * np.sin(tau / 19.0)
            food *= 1.0 + 0.45 * evolve * np.sin(tau / 23.0 + 4.2)
            decay = min(max(decay + 0.02 * evolve * np.sin(tau / 29.0 + 2.1),
                            0.80), 0.995)
            weave = min(max(weave + 0.30 * evolve * np.sin(tau / 31.0 + 1.0),
                            0.0), 1.0)
            pf.mod_sense = 1.0 + 0.50 * es * np.sin(tau / 17.0 + 0.7)
            pf.mod_turn = 1.0 + 0.35 * es * np.sin(tau / 27.0 + 3.4)
            pf.mod_spread = 1.0 + 0.30 * es * np.sin(tau / 13.0 + 5.5)
        else:
            pf.mod_sense = pf.mod_turn = pf.mod_spread = 1.0

        # weave: one knob onto the engine's anti-thoroughfare levers, tuned
        # offline (junction density several-x between 0 and 1 on a static
        # scene while veins stay coherent). 0 is the legacy bold-canal
        # behavior. The diffuse trim rides the SLIDER value, not the evolve-
        # modulated one — an integer blur radius popping mid-oscillation
        # would beat visibly.
        pf.sat = 0.25 * weave
        pf.jitter = 0.38 * weave
        pf.hetero = weave
        pf.reseed_frac = 0.004 + 0.022 * weave * weave
        pf.diffuse = max(1, round(self._base_diffuse * (1.0 - 0.45 * weave_base)))

        pf.decay = decay
        pf.food = food
        # gain multiplies sense + step (both in grid px) in either engine, so
        # it doubles as the length-unit conversion onto the GL grid
        pf.gain = gain * self._px_scale
        pf.exposure = exposure

        small = cv2.resize(frame_bgr, (MATTE_W, MATTE_H))
        m = cv2.resize(self.mat.compute(small), (gw, gh))
        gray = cv2.resize(
            cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0,
            (gw, gh))

        # react: motion carves. Frame-differenced luma builds a lingering
        # motion-energy map; it is poured into the food channel (sensors
        # chase it), respawn traffic is steered onto it (trails linger where
        # you swept), and a strong gesture fires one theatrical burst at its
        # centroid — episodic, so the pour reads as a cast spell instead of
        # continuously draining the rest of the organism.
        if self._prev_gray is None or self._prev_gray.shape != gray.shape:
            self._prev_gray = gray
            self._motion = np.zeros_like(gray)
            self._cy, self._cx = np.mgrid[0:gh, 0:gw]
            # half-res coordinate grids for the evolve phantom (see below)
            self._hcy, self._hcx = np.mgrid[0:(gh + 1) // 2, 0:(gw + 1) // 2]
            self._hcy = (self._hcy * 2).astype(np.float32)
            self._hcx = (self._hcx * 2).astype(np.float32)
        # deadband under the diff: real cameras hold ~0.01-0.03 of per-pixel
        # sensor noise, which would otherwise read as permanent full-frame
        # "motion" and fire gather pulses at the frame centroid forever
        motion = np.maximum(np.abs(gray - self._prev_gray) - np.float32(0.04),
                            np.float32(0.0))
        self._prev_gray = gray
        np.maximum(self._motion * np.float32(0.975),
                   np.minimum(motion * np.float32(3.0), np.float32(1.0)),
                   out=self._motion)
        # evolve's phantom: a slow-wandering invisible food blob the mold
        # chases. Parameter breathing alone cannot re-knit the network — the
        # laid trail is an attractor and the layout locks onto the scene's
        # light — but a migrating attractor drags veins across the frame and
        # the field re-organizes behind it (~30-40 s circuits). Computed at
        # half res (one exp on a quarter of the cells) and upsampled.
        if evolve > 0:
            tau_p = self._t * (2.0 * np.pi)
            fx = gw * (0.5 + 0.38 * np.sin(tau_p / 37.0 + 0.9))
            fy = gh * (0.5 + 0.38 * np.sin(tau_p / 41.0 + 2.6))
            sig = 0.16 * min(gw, gh)
            blob = np.exp(((self._hcx - fx) ** 2 + (self._hcy - fy) ** 2)
                          * np.float32(-1.0 / (2.0 * sig * sig)))
            # sqrt(evolve), same reasoning as the structural movers above
            gray = gray + (1.1 * np.sqrt(evolve)) * cv2.resize(blob, (gw, gh))

        self._burst_cool = max(getattr(self, "_burst_cool", 0.0) - dt, 0.0)
        keep = None
        if react > 0:
            gray = gray + (5.0 * react) * self._motion
            # swept paths linger: motion history becomes a per-pixel decay
            # boost, so the veins you carve stay painted for a few seconds
            keep = react * self._motion
            energy = float(motion.mean())
            if energy > 8e-4 and self._burst_cool <= 0.0:
                tot = float(motion.sum())
                mx = float((self._cx * motion).sum() / tot)
                my = float((self._cy * motion).sum() / tot)
                pf.gather(mx, my,
                          frac=min(0.9, react * (0.3 + 60.0 * energy)),
                          radius=40.0 * self._px_scale)
                self._burst_cool = 0.5

        if self._burst_pending or self._wave_pending:
            # land the impulse on the lit subject: weighted centroid of
            # matte*luma, falling back to frame center on an empty matte
            w = m * np.clip(gray, 0.05, 1.0)
            tot = float(w.sum())
            if tot > 1e-3:
                px_c = float((self._cx * w).sum() / tot)
                py_c = float((self._cy * w).sum() / tot)
            else:
                px_c, py_c = gw / 2.0, gh / 2.0
            if self._burst_pending:
                # the burst's footprint is sized in CPU-grid pixels too
                pf.spawn_burst(px_c, py_c, radius=6.0 * self._px_scale)
            if self._wave_pending:
                pf.wave(px_c, py_c)
            self._burst_pending = self._wave_pending = False

        pf.update(m, gray, keep)

        pal = self.palettes[int(self._ui("ph_palette_idx", 0)) % len(self.palettes)]
        # SIGNAL on the GPU (DESIGN.md §2.4; PR #26's measured port list):
        # with the GL engine and the rack ON, tonemap + colorize + upscale +
        # video composite + the whole rack run as fragment passes on the
        # field's own context — the trail never round-trips through numpy,
        # and the ONE readback is the final composed uint8 frame.
        # signal_done tells the shell its CPU rack already happened here.
        self.signal_done = False
        if (self.engine == "gl" and self._rack_gl_ok and self.host is not None
                and bool(self._ui("glitch", False))):
            out = self._step_gpu_rack(small, frame_bgr, pal)
            if out is not None:
                self.signal_done = True
                return out

        lum = pf.luminance()
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

    # ----- SIGNAL on the GPU -----
    def _ensure_glout(self, rw, rh):
        """The GL output composer + rack, rebuilt when the field, grid, or
        output resolution changes (all of them size its textures). Must be
        called with the field's context bound."""
        key = (id(self.pf), self.grid, (rw, rh))
        if self._glout is not None and self._glout_key != key:
            if self._glout.ctx is self.pf.ctx:
                self._glout.release()   # same live ctx: free the old textures
            self._glout = None
        if self._glout is None:
            gw, gh = self.grid
            self._glout = PhysarumOutGL(self.pf.ctx, gw, gh, rw, rh)
            self._glout_key = key
        return self._glout

    def _step_gpu_rack(self, small, frame_bgr, pal):
        """One frame of the ported output path: tonemap into the field's
        luminance texture, colorize + upscale + video composite, then the
        SIGNAL rack, all as fragment passes — one uint8 RGB readback at
        output res. Uses the shell's own CircuitBent (host.cb) for the
        rack's stochastic plan, so toggling engines stays continuous.

        Returns None on any GL failure and disables the path — the CPU
        rack takes over, toasted in amber (DESIGN.md §6.4: the show still
        has to start)."""
        host = self.host
        try:
            rw, rh = host.res
            cb = getattr(host, "cb", None)
            if cb is None:
                cb = CircuitBent(seed=getattr(host, "seed", 0))
                host.cb = cb
            sync_signal(cb, host.ui, self, rh)
            video_small = (cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                           if pal == "video" else None)
            cam, mix = None, 0.0
            if bool(self._ui("ph_video_bg", False)):
                mix = float(self._ui("ph_video_mix", 0.5))
                if mix > 0.0:
                    cam = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pf = self.pf
            with pf.ctx:
                glout = self._ensure_glout(rw, rh)
                pf.luminance_into_tex()
                lut = None if pal == "video" else _palette_lut(pal)
                src = glout.compose(pf.tex_lum, lut, video_small, cam, mix)
                glout.rack.run(cb, cb.plan(rh, rw), src)
                return glout.rack.read()
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:               # noqa: BLE001 — §6.4
            self._rack_gl_ok = False
            self._glout = None
            msg = f"GPU rack unavailable, rack running on CPU: {e}"
            if host is not None:
                host.hud.toasts.flash(msg[:80], AMBER)
            print(msg)
            return None
