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

import numpy as np

from .physarum import POINTS

# Agent texture width. One texel per agent; the height is ceil(n / width).
# 2048 x 16384 (the GL_MAX_TEXTURE_SIZE floor on anything that runs this)
# is 33M agents, well past what the deposit draw can afford anyway.
AGENT_TEX_W = 2048

# Stride of the statistics subsample (percentile + mean readback).
STATS_STRIDE = 4


class PhysarumGLUnavailable(RuntimeError):
    """Raised by PhysarumFieldGL.__init__ when no GL context / float render
    target can be had. The mode catches it and falls back to the CPU field."""


_FS_VS = """
#version 330
in vec2 in_vert;
void main(){ gl_Position = vec4(in_vert, 0.0, 1.0); }
"""

# Integer hash + per-agent random stream. A per-frame salt goes into the seed
# so each frame's coin flips are fresh; the stream is advanced by re-hashing.
_HASH_GLSL = """
uint hash(uint x){
    x ^= x >> 16u; x *= 0x7feb352du;
    x ^= x >> 15u; x *= 0x846ca68bu;
    x ^= x >> 16u;
    return x;
}
float rnd(inout uint s){ s = hash(s); return float(s) * (1.0 / 4294967296.0); }
"""

_UPDATE_FS = """
#version 330
uniform sampler2D u_agents;
uniform sampler2D u_trail;
uniform sampler2D u_matte;
uniform sampler2D u_gray;
uniform ivec2 u_grid;
uniform int u_aw;
uniform vec2 u_sense, u_spread, u_turn, u_step;   // (field point, body point)
uniform float u_gain, u_food, u_reseed, u_wmax;
uniform uint u_salt;
out vec4 f_agent;
""" + _HASH_GLSL + """
ivec2 cell(vec2 p){ return ivec2(mod(floor(p), vec2(u_grid))); }
float food(vec2 p){
    ivec2 c = cell(p);
    return texelFetch(u_trail, c, 0).r + u_food * texelFetch(u_gray, c, 0).r;
}
void main(){
    ivec2 ac = ivec2(gl_FragCoord.xy);
    uint idx = uint(ac.y * u_aw + ac.x);
    vec4 a = texelFetch(u_agents, ac, 0);
    vec2 p = a.xy;
    float h = a.z;
    uint s = hash(idx * 0x9e3779b9u + u_salt);

    // the pen: blend field -> body by the matte under the agent
    float t = texelFetch(u_matte, cell(p), 0).r;
    float sense = mix(u_sense.x, u_sense.y, t) * u_gain;
    float spread = mix(u_spread.x, u_spread.y, t);
    float turn = mix(u_turn.x, u_turn.y, t);
    float stp = mix(u_step.x, u_step.y, t) * u_gain;

    // Jones steering: hold when ahead wins; coin flip when ahead loses to
    // both sides; otherwise turn toward the stronger side.
    float fc = food(p + vec2(cos(h), sin(h)) * sense);
    float fl = food(p + vec2(cos(h - spread), sin(h - spread)) * sense);
    float fr = food(p + vec2(cos(h + spread), sin(h + spread)) * sense);
    float dir;
    if (fc > fl && fc > fr) dir = 0.0;
    else if (fc < fl && fc < fr) dir = (rnd(s) < 0.5) ? -1.0 : 1.0;
    else dir = (fl > fr) ? -1.0 : 1.0;
    h += dir * turn;

    p += vec2(cos(h), sin(h)) * stp;
    p = mod(p, vec2(u_grid));

    // recycle a trickle of agents onto the lit subject (matte * luma)
    if (u_reseed > 0.0 && rnd(s) < u_reseed) {
        vec2 g = vec2(u_grid);
        if (u_wmax <= 0.0) {
            p = vec2(rnd(s), rnd(s)) * g;
            h = rnd(s) * 6.2831853;
        } else {
            for (int i = 0; i < 16; i++) {
                vec2 c = vec2(rnd(s), rnd(s)) * g;
                ivec2 ci = cell(c);
                float w = texelFetch(u_matte, ci, 0).r
                        * clamp(texelFetch(u_gray, ci, 0).r, 0.05, 1.0);
                if (rnd(s) * u_wmax < w) {
                    p = c;
                    h = rnd(s) * 6.2831853;
                    break;
                }
            }
        }
    }
    f_agent = vec4(p, h, a.w);
}
"""

_DEPOSIT_VS = """
#version 330
uniform sampler2D u_agents;
uniform sampler2D u_matte;
uniform ivec2 u_grid;
uniform int u_aw;
uniform vec2 u_deposit;
out float v_dep;
void main(){
    ivec2 ac = ivec2(gl_VertexID % u_aw, gl_VertexID / u_aw);
    vec4 a = texelFetch(u_agents, ac, 0);
    ivec2 c = ivec2(mod(floor(a.xy), vec2(u_grid)));
    v_dep = mix(u_deposit.x, u_deposit.y, texelFetch(u_matte, c, 0).r);
    vec2 ndc = (vec2(c) + 0.5) / vec2(u_grid) * 2.0 - 1.0;
    gl_Position = vec4(ndc, 0.0, 1.0);
}
"""
_DEPOSIT_FS = """
#version 330
in float v_dep;
out vec4 f_color;
void main(){ f_color = vec4(v_dep, 0.0, 0.0, 1.0); }
"""

# One axis of the box blur. `u_add` (the laid deposits) is folded in on the
# first axis; `u_scale` carries 1/(2r+1) on both and decay on the second.
_BLUR_FS = """
#version 330
uniform sampler2D u_src;
uniform sampler2D u_add;
uniform int u_use_add;
uniform ivec2 u_dir;
uniform int u_radius;
uniform ivec2 u_grid;
uniform float u_scale;
out vec4 f_color;
void main(){
    ivec2 c = ivec2(gl_FragCoord.xy);
    float acc = 0.0;
    for (int i = -u_radius; i <= u_radius; i++) {
        ivec2 q = c + u_dir * i;
        q = ((q % u_grid) + u_grid) % u_grid;
        acc += texelFetch(u_src, q, 0).r;
        if (u_use_add == 1) acc += texelFetch(u_add, q, 0).r;
    }
    f_color = vec4(acc * u_scale, 0.0, 0.0, 1.0);
}
"""

# stride-STATS_STRIDE subsample of (trail, laid) for the tonemap statistics
_STATS_FS = """
#version 330
uniform sampler2D u_trail;
uniform sampler2D u_laid;
uniform int u_stride;
uniform ivec2 u_grid;
out vec4 f_color;
void main(){
    ivec2 c = ivec2(gl_FragCoord.xy) * u_stride;
    c = min(c, u_grid - 1);
    f_color = vec4(texelFetch(u_trail, c, 0).r, texelFetch(u_laid, c, 0).r, 0.0, 1.0);
}
"""

_TONEMAP_FS = """
#version 330
uniform sampler2D u_trail;
uniform sampler2D u_laid;
uniform float u_inv_norm, u_grain, u_inv_gnorm, u_exposure;
out vec4 f_color;
void main(){
    ivec2 c = ivec2(gl_FragCoord.xy);
    float x = texelFetch(u_trail, c, 0).r * u_inv_norm
            + u_grain * texelFetch(u_laid, c, 0).r * u_inv_gnorm;
    f_color = vec4(vec3(1.0 - exp(-u_exposure * x)), 1.0);
}
"""

# burst (mode 1): a fraction of agents teleport to a gaussian around the
# center with fresh headings; wave (mode 2): every heading points away from it
_IMPULSE_FS = """
#version 330
uniform sampler2D u_agents;
uniform ivec2 u_grid;
uniform int u_aw;
uniform int u_mode;
uniform vec2 u_center;
uniform float u_frac, u_radius;
uniform uint u_salt;
out vec4 f_agent;
""" + _HASH_GLSL + """
void main(){
    ivec2 ac = ivec2(gl_FragCoord.xy);
    uint idx = uint(ac.y * u_aw + ac.x);
    vec4 a = texelFetch(u_agents, ac, 0);
    if (u_mode == 1) {
        uint s = hash(idx * 0x9e3779b9u + u_salt);
        if (rnd(s) < u_frac) {
            float u1 = max(rnd(s), 1e-7), u2 = rnd(s) * 6.2831853;
            float r = sqrt(-2.0 * log(u1)) * u_radius;      // Box-Muller
            a.xy = mod(u_center + vec2(cos(u2), sin(u2)) * r, vec2(u_grid));
            a.z = rnd(s) * 6.2831853;
        }
    } else if (u_mode == 2) {
        a.z = atan(a.y - u_center.y, a.x - u_center.x);
    }
    f_agent = a;
}
"""


class PhysarumFieldGL:
    """PhysarumField's contract on a moderngl standalone context.

    Raises PhysarumGLUnavailable when the context or the float render
    targets cannot be created; releases whatever it built first."""

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

        def prog(fs, vs=_FS_VS):
            return ctx.program(vertex_shader=vs, fragment_shader=fs)
        self.p_update = prog(_UPDATE_FS)
        self.p_deposit = ctx.program(vertex_shader=_DEPOSIT_VS,
                                     fragment_shader=_DEPOSIT_FS)
        self.p_blur = prog(_BLUR_FS)
        self.p_stats = prog(_STATS_FS)
        self.p_tonemap = prog(_TONEMAP_FS)
        self.p_impulse = prog(_IMPULSE_FS)

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
    @property
    def trail(self):
        """The full float trail, read back from the GPU (gh, gw) float32."""
        raw = self.fbo_trail_a.read(components=1, dtype="f4")
        return np.frombuffer(raw, np.float32).reshape(self.gh, self.gw).copy()

    def agents(self):
        """(px, py, heading) float32 arrays of length n, read back."""
        raw = self.fbo_agents_a.read(components=4, dtype="f4")
        a = np.frombuffer(raw, np.float32).reshape(self.ah * self.aw, 4)[:self.n]
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
