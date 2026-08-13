"""Particles — the continuity mode (DESIGN.md §2.2, §4.1), transplanted from live_flow.

The mode owns its engine objects (ParticleFlow + GlowRenderer + matte), its
per-frame image, its panel sections (TEMPLATES / SOURCE / LOOK / MOTION), its
mode-local commands, and its defaults + built-in looks. The per-frame UI→engine
sync does not disappear — it relocates *inside* step() (DESIGN.md §2.2: the
honest accounting; a params namespace does not remove the need to copy values
onto pf/glow each frame).

The shell owns everything else: window, capture, key/mouse routing, audio,
recorder, preset CRUD, overlay state machine, the SIGNAL rack (CircuitBent is
post-FX on ANY mode's output — §2.4), blackout, quit.
"""
from __future__ import annotations

import cv2
import numpy as np

from ..glow import GlowRenderer
from ..imgui import ACC
from ..matte import make_matte
from ..overlay_ui import build_particles_sections
from ..particles import ParticleFlow, PALETTES
from .. import presets as _presets

MATTES = ["auto", "motion", "saliency", "person", "edges", "luma"]

# matte compute resolution — fixed independent of render/grid res (the shipped
# 416x234 constant, now named instead of buried in live_flow's body)
MATTE_W, MATTE_H = 416, 234


def composite_video_bg(particles_rgb, frame_bgr, mix):
    """Screen-blend the raw camera footage under the additive particle render.

    Screen (1 - (1-p)(1-bg*mix)) can only brighten, so the glow stays luminous on top
    and nothing blows out; mix=0 returns the particles untouched. All-uint8 cv2 ops to
    stay cheap at full render resolution.
    """
    if mix <= 0.0:
        return particles_rgb
    h, w = particles_rgb.shape[:2]
    bg = cv2.cvtColor(cv2.resize(frame_bgr, (w, h)), cv2.COLOR_BGR2RGB)
    bg = cv2.convertScaleAbs(bg, alpha=float(mix))
    inv = cv2.multiply(cv2.bitwise_not(particles_rgb), cv2.bitwise_not(bg), scale=1.0 / 255.0)
    return cv2.bitwise_not(inv)


class ParticlesMode:
    """The webcam-matte particle instrument as a shell plugin (Mode protocol)."""

    id = "particles"
    title = "Particles"
    accent = ACC                 # keeps the shipped ACC green (DESIGN.md §5 continuity)
    accepts_still = False

    # Built-in looks stay in code, per mode (DESIGN.md §7) — immune to
    # rename/delete, bankable. Today's dict, owned here.
    BUILTIN = _presets.BUILTIN

    # option lists the shell's panel walker needs (cycle options + name lookups)
    palettes = list(PALETTES)
    mattes = list(MATTES)

    # apply="reset" merges a look over these defaults so switching looks fully
    # resets physics params (otherwise e.g. sigil's low damping would leak into
    # the next look) — live_flow's `d` dict, now the mode's declared defaults.
    # Keys ABSENT here (matte, count, fx...) are left untouched when a look
    # doesn't record them — today's asymmetric keep semantics.
    DEFAULTS = dict(palette="ice", spark=0.35, curl_amp=0.5, reseed_frac=0.06,
                    base_size=0.011, damp=0.90, pull_falloff=22.0,
                    attract_speed=4.5, fade=0.90, exposure=1.4)

    def __init__(self, matte="auto", grid=(416, 234), n=200000, seed=1,
                 video_bg=False, video_mix=0.5, flock=False, glitch=False):
        self.matte_kind = matte
        self.grid = tuple(grid)
        self.n = n
        self.seed = seed
        # CLI boot opt-ins, applied to the shared UI state in configure_ui
        self.boot_video_bg = video_bg
        self.boot_video_mix = video_mix
        self.boot_flock = flock
        self.boot_glitch = glitch
        self.host = None
        self.pf = None
        self.glow = None
        self.mat = None

    # ----- lifecycle -----
    def start(self, host):
        """Build engines at the host's resolution. May raise — the shell
        catches, toasts, and reinstates the previous mode (DESIGN.md §2.2)."""
        self.host = host
        gw, gh = self.grid
        rw, rh = host.res
        self.mat = make_matte(self.matte_kind)
        self.pf = ParticleFlow(n=self.n, gw=gw, gh=gh, seed=self.seed)
        self.glow = GlowRenderer(rw, rh, self.n, fade=0.90, exposure=1.4)

    def stop(self):
        """Release GL/matte; idempotent (DESIGN.md §2.2 — the soak-test contract)."""
        if self.glow is not None:
            self.glow.release()
            self.glow = None
        self.pf = None
        self.mat = None

    def on_resize(self, w, h):
        if self.glow is not None:
            self.glow.resize(w, h)

    # ----- panel / commands -----
    def panel_spec(self):
        """The mode's own sections; the shell appends SIGNAL + global rows."""
        return build_particles_sections(list(PALETTES), MATTES)

    def commands(self):
        """Mode-local perform keys (DESIGN.md §6.2): F flock, V video bg."""
        from ..shell import ui_toggle_command
        ui, toasts = self.host.ui, self.host.hud.toasts
        return {
            "layer.flock": ui_toggle_command(ui, toasts, "layer.flock", "Flock",
                                             "f", "flock", "FLOCK"),
            "video_bg.toggle": ui_toggle_command(ui, toasts, "video_bg.toggle",
                                                 "Video background", "v",
                                                 "video_bg", "VIDEO BG"),
        }

    def safe_look(self):
        """The panic target — the mode's known-good look (DESIGN.md §6.2 '0')."""
        return "abstract"

    def configure_ui(self, ui):
        """Apply the CLI boot opt-ins to the shared UI state: turn the effect on
        AND open its section, so --glitch doesn't leave someone hunting for the
        controls behind a collapsed header (shipped behavior, kept)."""
        ui.flock = ui.flock or self.boot_flock
        ui.glitch = ui.glitch or self.boot_glitch
        if self.boot_flock:
            ui.sections["MOTION"] = True
        if self.boot_glitch:
            ui.sections["SIGNAL"] = True
        ui.video_bg = bool(ui.video_bg or self.boot_video_bg)

    def status_line(self, cam_name):
        return f"matte={self.matte_kind}  color={self.pf.palette}  cam={cam_name[:16]}"

    # ----- per-frame -----
    def step(self, frame_bgr, audio_levels, dt):
        """UI→engine sync, sim, render, video composite → RGB at host.res.

        The frame is already mirrored by the shell (so the footage lines up
        with the particles in the video composite); it is never None — on
        camera loss the shell holds the last good frame (DESIGN.md §2.2).
        """
        ui = self.host.ui
        pf, glow = self.pf, self.glow
        gw, gh = self.grid

        if ui is not None:
            if ui.matte_name != self.matte_kind:
                self.matte_kind = ui.matte_name
                self.mat = make_matte(self.matte_kind)
            pf.palette = ui.palette_name
            pf.curl_amp = ui.curl
            pf.base_size = ui.dot
            pf.damp = ui.damp
            pf.pull_falloff = ui.pull
            pf.reseed_frac = ui.reseed
            pf.attract_speed = ui.attract_speed
            # flocking: gains go to 0 when the toggle is off, which short-circuits
            # the solver entirely — the off state costs nothing.
            pf.flock_cohesion = ui.cohere if ui.flock else 0.0
            pf.flock_alignment = ui.align if ui.flock else 0.0
            pf.flock_separation = ui.separate if ui.flock else 0.0
            glow.fade = ui.fade

        small = cv2.resize(frame_bgr, (MATTE_W, MATTE_H))
        m = cv2.resize(self.mat.compute(small), (gw, gh))
        gray = cv2.resize(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0,
                          (gw, gh))
        color = cv2.cvtColor(cv2.resize(small, (gw, gh)),
                             cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        # base look from the panel; audio modulates glow/spark on top for this frame.
        # Spark is modulated MULTIPLICATIVELY off the slider, so when Spark is 0 (e.g.
        # sigil) sound adds no spark/blur — it only pulses brightness with the bass.
        glow.exposure = ui.exposure if ui is not None else glow.exposure
        pf.spark = ui.spark if ui is not None else pf.spark
        if audio_levels is not None:
            sens = ui.sens if ui is not None else 1.0
            glow.exposure = glow.exposure * (1.0 + 1.6 * sens * audio_levels["bass"])
            pf.spark = pf.spark * (1.0 + 2.0 * sens * audio_levels["treble"])

        # weight density by luminance for portraits (video palette OR person matte) so
        # the face's tones resolve as particle density — on ANY color, not just video.
        if pf.palette == "video" or self.matte_kind == "person":
            density = m * np.clip((gray - 0.06) * 1.5, 0.05, 1.0)
        else:
            density = None
        pf.update(m, gray, color, density=density)
        buf = pf.render_data()
        if ui is not None and ui.count < 0.999:
            buf = buf[: int(ui.count * self.n) * 7]   # live dot-count control
        out = glow.render(buf)

        # video-bg composite is a Particles concern (its toggle lives in SOURCE)
        if ui is not None:
            video_bg, video_mix = ui.video_bg, ui.video_mix
        else:
            video_bg, video_mix = self.boot_video_bg, self.boot_video_mix
        if video_bg:
            out = composite_video_bg(out, frame_bgr, video_mix)
        return out

