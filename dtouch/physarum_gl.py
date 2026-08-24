"""Physarum on the GPU — the same Jones-model mold as dtouch.physarum, run as
fragment-shader ping-pong GPGPU on moderngl.

macOS caps OpenGL at 4.1, so there are no compute shaders here and none are
needed: this is the standard WebGL physarum architecture, and it runs on GL
3.3 core.

- **Agents live in a float texture.** RGBA32F, one texel per agent
  (x, y, heading, spare). The *update* pass draws one full-screen triangle
  over the agent texture and, per texel, does exactly what PhysarumField
  does per row: read the matte under the agent (the pen), blend the two
  behavior points, sense the trail+food at three points, turn, step, wrap,
  and — for a trickle of agents — respawn onto the lit subject. It writes the
  next agent texture; the two agent textures ping-pong.
- **Deposit is a point draw.** An attribute-less GL_POINTS draw of N vertices
  whose vertex shader texelFetches its own agent by gl_VertexID and lands a
  1px point on that agent's cell, additively blended into a float "laid"
  texture. That is bincount on the GPU.
- **Diffuse + decay** is a separable box blur (H then V) that adds the laid
  deposits on the way in and multiplies by decay on the way out, ping-ponging
  the trail.
- **The picture is tonemapped on the GPU** and read back as uint8 — the mode
  quantizes luminance to a 256-entry LUT index anyway, so an 8-bit readback
  loses nothing and is 4x smaller than the float trail. The two statistics
  the tonemap needs (the trail's 95th percentile, the deposits' mean) come
  from a 1/4-res subsample pass read back as a small float texture.

The GLSL lives in `dtouch/shaders/physarum/` as standalone files, without a
`#version` line, in the GLSL 3.30 ∩ GLSL ES 3.00 subset: they are the single
source of truth for this engine AND for the WebGL2 port on the public site,
which vendors them verbatim (see that directory's README). `load_shader`
prepends the desktop version line.

Same interface as PhysarumField — `update(matte, gray)`, `luminance()`,
`spawn_burst`, `wave`, the matte-blended POINTS — so PhysarumMode can select
either. Known, deliberate differences from the CPU field: the blur wraps at
the edges (the agents already do; cv2.boxFilter reflects); deposit weight is
read at the agent's post-move cell rather than its pre-move one; the
percentile is estimated on a stride-4 subsample; reseeding is per-agent
Bernoulli(reseed_frac) with rejection sampling against matte*luma instead of
a CDF draw. None of these is visible on the instrument.

Clean-room note: the model is Jeff Jones's (sense/rotate/move/deposit/
diffuse/decay) and the semantics are this repo's dtouch/physarum.py. No code
or parameter tables from any CC BY-NC-SA physarum project were used.
"""
from __future__ import annotations

import math
import os

import numpy as np

from .physarum import POINTS

# Agent texture width. One texel per agent; the height is ceil(n / width).
# 2048 x 16384 (the GL_MAX_TEXTURE_SIZE floor on anything that runs this)
# is 33M agents, well past what the deposit draw can afford anyway.
AGENT_TEX_W = 2048

# Stride of the statistics subsample (percentile + mean readback).
STATS_STRIDE = 4

SHADER_DIR = os.path.join(os.path.dirname(__file__), "shaders", "physarum")
SHADER_FILES = ("fullscreen.vert", "update.frag", "deposit.vert", "deposit.frag",
                "blur.frag", "stats.frag", "tonemap.frag", "impulse.frag")

# What moderngl gets in front of every file. The browser prepends its own
# `#version 300 es` + precision lines; the files carry neither.
GLSL_VERSION_LINE = "#version 330 core\n"


def load_shader(name, version_line=GLSL_VERSION_LINE):
    """Source of `dtouch/shaders/physarum/<name>` with the version line
    prepended. A `#line 1` follows it so compile errors keep file-true
    line numbers."""
    with open(os.path.join(SHADER_DIR, name), "r", encoding="utf-8") as fh:
        body = fh.read()
    return version_line + "#line 1\n" + body


class PhysarumGLUnavailable(RuntimeError):
    """Raised by PhysarumFieldGL.__init__ when no GL context / float render
    target can be had. The mode catches it and falls back to the CPU field."""


class PhysarumFieldGL:
    """PhysarumField's contract on a moderngl standalone context.

    Raises PhysarumGLUnavailable when the context or the float render
    targets cannot be created; releases whatever it built first.

    One live field at a time (the same rule as GlowRenderer): moderngl
    issues GL calls on whichever standalone context was created last, so a
    second live field silently aliases the first one's textures and
    programs. Release a field before building another."""

    def __init__(self, n=1_000_000, gw=1280, gh=736, seed=0,
                 point_bg="veins", point_fg="fingers",
                 decay=0.94, diffuse=1, food=0.35, exposure=3.5,
                 grain=0.5, reseed_frac=0.004, gain=1.0):
        if n <= 0:
            raise ValueError(f"n must be > 0, got {n}")
        if gw <= 0 or gh <= 0:
            raise ValueError(f"grid must be positive, got {gw}x{gh}")
        if not (0.0 <= decay <= 1.0):
            raise ValueError(f"decay must be in [0, 1], got {decay}")
        if not (0.0 <= reseed_frac <= 1.0):
            raise ValueError(f"reseed_frac must be in [0, 1], got {reseed_frac}")
        self.n, self.gw, self.gh = n, gw, gh
        self.point_bg = point_bg
        self.point_fg = point_fg
        self.decay = decay
        self.diffuse = diffuse
        self.food = food
        self.exposure = exposure
        self.grain = grain
        self.reseed_frac = reseed_frac
        self.gain = gain
        self.seed = seed
        self.frame = 0
        self.ctx = None
        self._rng = np.random.default_rng(seed)
        self._salt = np.random.default_rng(seed + 1)
        self.aw = min(n, AGENT_TEX_W)
        self.ah = int(math.ceil(n / self.aw))
        try:
            self._build()
        except Exception as e:                          # noqa: BLE001
            self.release()
            raise PhysarumGLUnavailable(f"GPU physarum unavailable: {e}") from e

    # ----- GL setup -----
    def _build(self):
        import moderngl
        self._gl = moderngl
        ctx = self.ctx = moderngl.create_standalone_context()
        gw, gh, aw, ah = self.gw, self.gh, self.aw, self.ah

        fs_vs = load_shader("fullscreen.vert")

        def prog(frag, vert=fs_vs):
            return ctx.program(vertex_shader=vert, fragment_shader=load_shader(frag))
        self.p_update = prog("update.frag")
        self.p_deposit = prog("deposit.frag", load_shader("deposit.vert"))
        self.p_blur = prog("blur.frag")
        self.p_stats = prog("stats.frag")
        self.p_tonemap = prog("tonemap.frag")
        self.p_impulse = prog("impulse.frag")

        tri = np.array([-1, -1, 3, -1, -1, 3], np.float32)
        self.tri_vbo = ctx.buffer(tri.tobytes())
        self.vao_update = ctx.vertex_array(self.p_update, [(self.tri_vbo, "2f", "in_vert")])
        self.vao_blur = ctx.vertex_array(self.p_blur, [(self.tri_vbo, "2f", "in_vert")])
        self.vao_stats = ctx.vertex_array(self.p_stats, [(self.tri_vbo, "2f", "in_vert")])
        self.vao_tonemap = ctx.vertex_array(self.p_tonemap, [(self.tri_vbo, "2f", "in_vert")])
        self.vao_impulse = ctx.vertex_array(self.p_impulse, [(self.tri_vbo, "2f", "in_vert")])
        # attribute-less point draw: the vertex shader is fed by gl_VertexID
        self.vao_deposit = ctx.vertex_array(self.p_deposit, [])

        def ftex(size, comps, data=None):
            t = ctx.texture(size, comps, data=data, dtype="f4")
            t.filter = (moderngl.NEAREST, moderngl.NEAREST)
            t.repeat_x = t.repeat_y = False
            return t

        # agents: same initial distribution as the CPU field (same seed)
        rng = self._rng
        init = np.zeros((ah * aw, 4), np.float32)
        init[:self.n, 0] = rng.uniform(0, gw, self.n)
        init[:self.n, 1] = rng.uniform(0, gh, self.n)
        init[:self.n, 2] = rng.uniform(0, 2 * np.pi, self.n)
        self.tex_agents_a = ftex((aw, ah), 4, init.tobytes())
        self.tex_agents_b = ftex((aw, ah), 4)
        self.fbo_agents_a = ctx.framebuffer(color_attachments=[self.tex_agents_a])
        self.fbo_agents_b = ctx.framebuffer(color_attachments=[self.tex_agents_b])

        # the shaders read/write .r only, so the trail-sized targets are R32F
        # here; the browser backs the same GLSL with RGBA16F/32F
        self.tex_trail_a = ftex((gw, gh), 1)
        self.tex_trail_b = ftex((gw, gh), 1)
        self.tex_tmp = ftex((gw, gh), 1)
        self.tex_laid = ftex((gw, gh), 1)
        self.fbo_trail_a = ctx.framebuffer(color_attachments=[self.tex_trail_a])
        self.fbo_trail_b = ctx.framebuffer(color_attachments=[self.tex_trail_b])
        self.fbo_tmp = ctx.framebuffer(color_attachments=[self.tex_tmp])
        self.fbo_laid = ctx.framebuffer(color_attachments=[self.tex_laid])
        for f in (self.fbo_trail_a, self.fbo_trail_b, self.fbo_tmp, self.fbo_laid):
            f.use()
            ctx.clear(0.0, 0.0, 0.0, 1.0)

        self.tex_matte = ftex((gw, gh), 1)
        self.tex_gray = ftex((gw, gh), 1)

        self.sw = int(math.ceil(gw / STATS_STRIDE))
        self.sh = int(math.ceil(gh / STATS_STRIDE))
        self.tex_stats = ftex((self.sw, self.sh), 2)
        self.fbo_stats = ctx.framebuffer(color_attachments=[self.tex_stats])
        self.tex_lum = ctx.texture((gw, gh), 1)             # uint8 readback target
        self.fbo_lum = ctx.framebuffer(color_attachments=[self.tex_lum])

        # constant uniforms + sampler units
        for p in (self.p_update, self.p_deposit, self.p_impulse):
            p["u_aw"].value = aw
        for p in (self.p_update, self.p_deposit, self.p_blur, self.p_stats,
                  self.p_impulse):
            p["u_grid"].value = (gw, gh)
        self.p_update["u_agents"].value = 0
        self.p_update["u_trail"].value = 1
        self.p_update["u_matte"].value = 2
        self.p_update["u_gray"].value = 3
        self.p_deposit["u_agents"].value = 0
        self.p_deposit["u_matte"].value = 2
        self.p_blur["u_src"].value = 0
        self.p_blur["u_add"].value = 1
        self.p_stats["u_trail"].value = 0
        self.p_stats["u_laid"].value = 1
        self.p_stats["u_stride"].value = STATS_STRIDE
        self.p_tonemap["u_trail"].value = 0
        self.p_tonemap["u_laid"].value = 1
        self.p_impulse["u_agents"].value = 0

        # a render pass to prove float targets + blending actually work here
        # (a context can exist and still refuse them); a failure surfaces as
        # PhysarumGLUnavailable from __init__ rather than a black show later
        self._deposit(1.0, 1.0)
        self.fbo_laid.use()
        ctx.clear(0.0, 0.0, 0.0, 1.0)

    def _salt_value(self):
        return int(self._salt.integers(0, 2**32, dtype=np.uint32))

    # ----- parameter blending -----
    def _points(self):
        return POINTS[self.point_bg], POINTS[self.point_fg]

    def swap_points(self):
        self.point_bg, self.point_fg = self.point_fg, self.point_bg

    # ----- passes -----
    def _deposit(self, dep_bg, dep_fg):
        ctx, gl = self.ctx, self._gl
        self.fbo_laid.use()
        ctx.clear(0.0, 0.0, 0.0, 1.0)
        ctx.enable(gl.BLEND)
        ctx.blend_func = (gl.ONE, gl.ONE)
        self.tex_agents_a.use(0)
        self.tex_matte.use(2)
        self.p_deposit["u_deposit"].value = (dep_bg, dep_fg)
        self.vao_deposit.render(gl.POINTS, vertices=self.n)
        ctx.disable(gl.BLEND)

    def _blur_decay(self):
        ctx, gl = self.ctx, self._gl
        r = int(self.diffuse) if self.diffuse > 0 else 0
        k = 2 * r + 1
        p = self.p_blur
        p["u_radius"].value = r
        # H: trail + laid -> tmp
        self.fbo_tmp.use()
        self.tex_trail_a.use(0)
        self.tex_laid.use(1)
        p["u_use_add"].value = 1
        p["u_dir"].value = (1, 0)
        p["u_scale"].value = 1.0 / k
        self.vao_blur.render(gl.TRIANGLES, vertices=3)
        # V: tmp -> trail_b, times decay
        self.fbo_trail_b.use()
        self.tex_tmp.use(0)
        p["u_use_add"].value = 0
        p["u_dir"].value = (0, 1)
        p["u_scale"].value = self.decay / k
        self.vao_blur.render(gl.TRIANGLES, vertices=3)
        self.tex_trail_a, self.tex_trail_b = self.tex_trail_b, self.tex_trail_a
        self.fbo_trail_a, self.fbo_trail_b = self.fbo_trail_b, self.fbo_trail_a

    def _swap_agents(self):
        self.tex_agents_a, self.tex_agents_b = self.tex_agents_b, self.tex_agents_a
        self.fbo_agents_a, self.fbo_agents_b = self.fbo_agents_b, self.fbo_agents_a

    # ----- one simulation frame -----
    def update(self, matte, gray):
        """matte, gray: float32 (gh, gw) in [0,1]. Advances agents one frame
        and rebuilds the trail map — three GPU passes, no readback."""
        gl = self._gl
        gh, gw = self.gh, self.gw
        if matte.shape != (gh, gw) or gray.shape != (gh, gw):
            raise ValueError(f"matte/gray must be {(gh, gw)}, got "
                             f"{matte.shape} / {gray.shape}")
        matte = np.ascontiguousarray(matte, dtype=np.float32)
        gray = np.ascontiguousarray(gray, dtype=np.float32)
        self.tex_matte.write(matte)
        self.tex_gray.write(gray)

        a, b = self._points()
        p = self.p_update
        p["u_sense"].value = (a["sense"], b["sense"])
        p["u_spread"].value = (a["spread"], b["spread"])
        p["u_turn"].value = (a["turn"], b["turn"])
        p["u_step"].value = (a["step"], b["step"])
        p["u_gain"].value = float(self.gain)
        p["u_food"].value = max(float(self.food), 0.0)
        p["u_reseed"].value = float(self.reseed_frac)
        # rejection-sampling bound for the respawn weight matte*clip(gray):
        # an upper bound keeps the draw exact; <= 0 means "nothing lit" and
        # the CPU field's uniform fallback
        mmax = float(matte.max())
        wmax = mmax * min(1.0, max(float(gray.max()), 0.05)) if mmax > 0 else 0.0
        p["u_wmax"].value = wmax
        p["u_salt"].value = self._salt_value()

        self.fbo_agents_b.use()
        self.tex_agents_a.use(0)
        self.tex_trail_a.use(1)
        self.tex_matte.use(2)
        self.tex_gray.use(3)
        self.vao_update.render(gl.TRIANGLES, vertices=3)
        self._swap_agents()

        self._deposit(a["deposit"], b["deposit"])
        self._blur_decay()
        self.frame += 1

    # ----- interactions -----
    def _impulse(self, mode, x, y, frac=0.0, radius=0.0):
        gl = self._gl
        p = self.p_impulse
        p["u_mode"].value = mode
        p["u_center"].value = (float(x), float(y))
        p["u_frac"].value = float(frac)
        p["u_radius"].value = float(radius)
        p["u_salt"].value = self._salt_value()
        self.fbo_agents_b.use()
        self.tex_agents_a.use(0)
        self.vao_impulse.render(gl.TRIANGLES, vertices=3)
        self._swap_agents()

    def spawn_burst(self, x, y, frac=0.08, radius=6.0):
        """Teleport a fraction of the pool into a tight gaussian at (x, y)
        with fresh random headings — 'pour more mold HERE'."""
        if frac <= 0:
            return
        self._impulse(1, x, y, frac, radius)

    def wave(self, x, y):
        """Point every agent's heading away from (x, y) — one radial impulse."""
        self._impulse(2, x, y)

    # ----- picture -----
    def _stats(self):
        """(95th percentile of the trail, mean of this frame's deposits),
        estimated on a stride-STATS_STRIDE subsample read back as floats."""
        gl = self._gl
        self.fbo_stats.use()
        self.tex_trail_a.use(0)
        self.tex_laid.use(1)
        self.vao_stats.render(gl.TRIANGLES, vertices=3)
        raw = self.fbo_stats.read(components=2, dtype="f4")
        s = np.frombuffer(raw, np.float32).reshape(self.sh, self.sw, 2)
        return float(np.percentile(s[..., 0], 95.0)), float(s[..., 1].mean())

    def luminance(self):
        """Tonemapped trail in [0,1] float32 (gh, gw) — same curve as the CPU
        field (trail normalized by its 95th percentile, `grain` mixing this
        frame's raw deposits over it, 1 - exp(-exposure * x)), evaluated on
        the GPU and read back as 8-bit."""
        gl = self._gl
        norm, lmean = self._stats()
        if norm <= 0:
            return np.zeros((self.gh, self.gw), np.float32)
        gnorm = lmean * 4.0
        p = self.p_tonemap
        p["u_inv_norm"].value = 1.0 / norm
        p["u_grain"].value = float(self.grain) if gnorm > 0 else 0.0
        p["u_inv_gnorm"].value = (1.0 / gnorm) if gnorm > 0 else 0.0
        p["u_exposure"].value = float(self.exposure)
        self.fbo_lum.use()
        self.tex_trail_a.use(0)
        self.tex_laid.use(1)
        self.vao_tonemap.render(gl.TRIANGLES, vertices=3)
        raw = self.fbo_lum.read(components=1)
        lum = np.frombuffer(raw, np.uint8).reshape(self.gh, self.gw)
        return lum.astype(np.float32) * np.float32(1.0 / 255.0)

    # ----- readbacks (slow; tests and diagnostics) -----
    def _read_f4(self, fbo, components):
        """fbo's contents as float32, binding it with use() first. Apple's GL
        needs the bind: once the tonemap has rendered into the uint8 target,
        the second and every later glReadPixels of a float target returns
        the uint8 target's contents (all ones / [0,1] values) with no GL
        error, until some framebuffer is bound for drawing again. The
        per-frame path never hit it — luminance() reads each target right
        after rendering into it — but trail / agents() after luminance()
        did."""
        fbo.use()
        raw = fbo.read(components=components, dtype="f4")
        return np.frombuffer(raw, np.float32).copy()

    @property
    def trail(self):
        """The full float trail, read back from the GPU (gh, gw) float32."""
        return self._read_f4(self.fbo_trail_a, 1).reshape(self.gh, self.gw)

    def agents(self):
        """(px, py, heading) float32 arrays of length n, read back."""
        a = self._read_f4(self.fbo_agents_a, 4).reshape(self.ah * self.aw, 4)[:self.n]
        return a[:, 0].copy(), a[:, 1].copy(), a[:, 2].copy()

    @property
    def px(self):
        return self.agents()[0]

    @property
    def py(self):
        return self.agents()[1]

    @property
    def heading(self):
        return self.agents()[2]

    # ----- lifecycle -----
    def release(self):
        """Release the context; idempotent (the mode's stop() contract)."""
        ctx, self.ctx = self.ctx, None
        if ctx is not None:
            ctx.release()
