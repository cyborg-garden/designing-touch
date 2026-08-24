"""Physarum simulation — a slime-mold trail network that eats the live video.

A fixed pool of N agents follows the classic Jones transport-network model
(Jeff Jones, "Characteristics of pattern formation and evolution in
approximations of physarum transport networks", Artificial Life 16(2), 2010):
each agent senses the trail map at three points ahead of it (left / center /
right), turns toward the strongest reading, steps forward, and deposits trail;
the trail map is then diffused and decayed. Out of nothing but that loop,
vein networks grow, merge, and re-route.

Two couplings make it a *video* instrument rather than a screensaver:

- **The matte is the pen.** Every simulation parameter exists as a PAIR — a
  background value and a subject value — and each agent runs on the blend of
  the two by the matte value under it. Body and background literally run
  different physics, and the silhouette becomes a negotiated boundary between
  two organisms. (This spatial-interpolation idea is inspired by Etienne
  Jacob's interactive-physarum "pen"; the implementation here is independent
  and shares no code or parameter tables with it — that project is
  CC BY-NC-SA and this one is AGPL.)
- **Luminance is food.** The camera's luma is added to what the sensors read,
  so the network is drawn toward bright regions, and recycled agents respawn
  preferentially onto the lit subject. Wave a light around and the mold
  chases it.

Pure NumPy + a little OpenCV (box blur), all on a small working grid, same as
ParticleFlow. The trail map itself is the picture.
"""
from __future__ import annotations

import cv2
import numpy as np

# Behavior points — named parameter sets an agent can run on. Both ends of the
# matte blend (background / subject) pick from this table. Hand-tuned on the
# live instrument; the names describe the texture each one settles into.
# Re-spread 2026-08 (owner feedback: presets read too alike): each point now
# owns a distinct morphology niche — veins = the anchor mesh, cells = slow
# rounded compartments, fingers = fast bright darting runners, haze = pure
# fog, web = long straight strands, storm = wide-angle chaos. Verified by
# pairwise morphology distance on a static scene (min pair +38%).
#   sense:   sensor distance, in grid pixels
#   spread:  sensor half-angle, radians (how wide the left/right feelers sit)
#   turn:    rotation applied when a side sensor wins, radians
#   step:    move distance per frame, grid pixels
#   deposit: trail laid down per agent per frame
POINTS = {
    "veins":   dict(sense=9.0,  spread=0.39, turn=0.79, step=1.4, deposit=1.0),
    "cells":   dict(sense=14.0, spread=1.10, turn=0.30, step=0.8, deposit=1.0),
    "fingers": dict(sense=2.5,  spread=0.20, turn=0.70, step=3.4, deposit=2.0),
    "haze":    dict(sense=1.2,  spread=1.50, turn=0.08, step=0.35, deposit=0.25),
    "web":     dict(sense=34.0, spread=0.50, turn=1.10, step=1.8, deposit=0.6),
    "storm":   dict(sense=10.0, spread=1.55, turn=1.70, step=3.2, deposit=1.4),
}
POINT_NAMES = list(POINTS)

_PARAM_KEYS = ("sense", "spread", "turn", "step", "deposit")

# Sense-range multipliers for the `hetero` sub-populations: a third of the
# pool feels close, a third mid, a third far. Multi-scale sensing grows
# multi-scale structure — filigree between the trunk lines.
HETERO_MULTS = np.array([0.45, 1.0, 1.9], np.float32)


class PhysarumField:
    def __init__(self, n=250_000, gw=480, gh=270, seed=0,
                 point_bg="veins", point_fg="fingers",
                 decay=0.94, diffuse=1, food=0.35, exposure=3.5,
                 grain=0.5, reseed_frac=0.004, gain=1.0,
                 sat=0.0, jitter=0.0, hetero=0.0):
        if n <= 0:
            raise ValueError(f"n must be > 0, got {n}")
        if gw <= 0 or gh <= 0:
            raise ValueError(f"grid must be positive, got {gw}x{gh}")
        if not (0.0 <= decay <= 1.0):
            raise ValueError(f"decay must be in [0, 1], got {decay}")
        if not (0.0 <= reseed_frac <= 1.0):
            raise ValueError(f"reseed_frac must be in [0, 1], got {reseed_frac}")
        self.n, self.gw, self.gh = n, gw, gh
        rng = np.random.default_rng(seed)
        self.px = rng.uniform(0, gw, n).astype(np.float32)
        self.py = rng.uniform(0, gh, n).astype(np.float32)
        self.heading = rng.uniform(0, 2 * np.pi, n).astype(np.float32)
        self.trail = np.zeros((gh, gw), np.float32)
        # params
        self.point_bg = point_bg
        self.point_fg = point_fg
        self.decay = decay
        self.diffuse = diffuse          # box-blur radius in pixels; 0 = off
        self.food = food                # how strongly video luma attracts sensors
        self.exposure = exposure        # tonemap gain on the trail
        self.grain = grain              # raw per-frame deposits in the picture
        self.reseed_frac = reseed_frac
        self.gain = gain                # global multiplier on sense+step (tempo)
        # Anti-thoroughfare levers (the "weave" family — see PhysarumMode):
        self.sat = sat                  # sensor saturation: sensed trail is
                                        # softly capped at sat x the trail's
                                        # bright end, so a fat vein stops
                                        # out-competing thin ones. 0 = off.
        self.jitter = jitter            # random per-step heading wobble (rad)
        self.hetero = hetero            # 0..1 blend toward a 3-sub-population
                                        # sense-range split (short/mid/long)
        # slow structural modulation (the mode's `evolve` drives these):
        # multipliers on the blended point parameters — oscillating the
        # sense/turn/spread geometry re-organizes the network topology,
        # where gain alone only re-scales its tempo
        self.mod_sense = 1.0
        self.mod_turn = 1.0
        self.mod_spread = 1.0
        self._rng = rng
        self._laid = np.zeros((gh, gw), np.float32)
        self._norm = 0.0                # last luminance() percentile (sat cap ref)
        # per-agent sense-range multiplier groups for `hetero` (1/3 each)
        self._sense_group = HETERO_MULTS[np.arange(n) % 3].astype(np.float32)

    # ----- parameter blending -----
    def _blend_params(self, t):
        """Per-agent parameter arrays: background point blended toward the
        subject point by t = matte value under each agent (the pen)."""
        a, b = POINTS[self.point_bg], POINTS[self.point_fg]
        out = {}
        for k in _PARAM_KEYS:
            lo, hi = np.float32(a[k]), np.float32(b[k])
            out[k] = lo + (hi - lo) * t
        if self.mod_sense != 1.0:
            out["sense"] = out["sense"] * np.float32(self.mod_sense)
        if self.mod_turn != 1.0:
            out["turn"] = out["turn"] * np.float32(self.mod_turn)
        if self.mod_spread != 1.0:
            out["spread"] = out["spread"] * np.float32(self.mod_spread)
        return out

    def swap_points(self):
        self.point_bg, self.point_fg = self.point_fg, self.point_bg

    # ----- one simulation frame -----
    def update(self, matte, gray, keep=None):
        """matte, gray: float32 (gh, gw) in [0,1]. Advances agents one frame
        and rebuilds the trail map.

        `keep` (optional, same shape, [0,1]): per-pixel decay boost — where
        keep is 1 the trail decays at 0.995 instead of `decay`, so swept
        paths linger (the mode's react machinery paints it from motion)."""
        gw, gh = self.gw, self.gh
        n = self.n
        px, py, h = self.px, self.py, self.heading

        # float32 modulo can land exactly ON the upper bound (a hair below gw
        # rounds up in float32), so every grid index wraps in INT space
        t = matte[py.astype(np.int32) % gh, px.astype(np.int32) % gw]
        p = self._blend_params(t)
        sense_d = p["sense"] * self.gain
        step_d = p["step"] * self.gain
        if self.hetero > 0:
            # blend each agent's sense range toward its sub-population's
            sense_d = sense_d * (1.0 + (self._sense_group - 1.0) * self.hetero)

        # what the sensors read: laid trail plus the footage's light as food.
        # `sat` softly caps the sensed trail at sat x its own bright end, so
        # a saturated fat vein reads the same as a merely strong thin one —
        # the single strongest anti-thoroughfare lever (fat veins stop
        # winning every recruitment contest).
        sensed = self.trail
        cap = float(self.sat) * self._norm
        if cap > 0:
            sensed = cap * (1.0 - np.exp(sensed * np.float32(-1.0 / cap)))
        food = sensed if self.food <= 0 else sensed + self.food * gray

        def _sample(angle_off):
            sx = px + np.cos(h + angle_off) * sense_d
            sy = py + np.sin(h + angle_off) * sense_d
            return food[sy.astype(np.int32) % gh, sx.astype(np.int32) % gw]

        f_c = _sample(0.0)
        f_l = _sample(-p["spread"])
        f_r = _sample(p["spread"])

        # Jones steering: hold when ahead wins; flip a coin when ahead loses to
        # both sides; otherwise turn toward the stronger side.
        rand_sign = np.where(self._rng.random(n) < 0.5, -1.0, 1.0).astype(np.float32)
        turn = np.where(
            (f_c > f_l) & (f_c > f_r), 0.0,
            np.where((f_c < f_l) & (f_c < f_r), rand_sign,
                     np.where(f_l > f_r, -1.0, 1.0))).astype(np.float32)
        h += turn * p["turn"]
        if self.jitter > 0:
            # per-step heading wobble: highways stop being perfectly straight
            # attractors and the mold keeps probing sideways
            h += ((self._rng.random(n, dtype=np.float32) * 2.0 - 1.0)
                  * np.float32(self.jitter))

        px += np.cos(h) * step_d
        py += np.sin(h) * step_d
        px %= gw
        py %= gh

        # deposit — one bincount over flattened grid indices, the cheap way to
        # scatter-add N agents without np.add.at's per-element dispatch
        idx = (py.astype(np.int32) % gh) * gw + (px.astype(np.int32) % gw)
        laid = np.bincount(idx, weights=p["deposit"], minlength=gw * gh)
        self._laid = laid.reshape(gh, gw).astype(np.float32)
        self.trail += self._laid

        # diffuse + decay
        if self.diffuse > 0:
            k = 2 * int(self.diffuse) + 1
            self.trail = cv2.boxFilter(self.trail, -1, (k, k))
        if keep is None:
            self.trail *= self.decay
        else:
            self.trail *= (np.float32(self.decay)
                           + np.float32(0.995 - self.decay)
                           * np.clip(keep, 0.0, 1.0).astype(np.float32))

        # recycle a trickle of agents onto the lit subject, so the network
        # keeps finding whoever is in frame instead of ossifying
        budget = int(self.reseed_frac * n)
        if budget > 0:
            pick = self._rng.integers(0, n, budget)
            rx, ry = self._sample_weighted(matte * np.clip(gray, 0.05, 1.0), budget)
            self.px[pick] = rx
            self.py[pick] = ry
            self.heading[pick] = self._rng.uniform(0, 2 * np.pi, budget).astype(np.float32)

    def _sample_weighted(self, weight, k):
        w = weight.ravel().astype(np.float64)
        s = w.sum()
        if s <= 0:
            return (self._rng.uniform(0, self.gw, k).astype(np.float32),
                    self._rng.uniform(0, self.gh, k).astype(np.float32))
        cdf = np.cumsum(w) / s
        idx = np.searchsorted(cdf, self._rng.random(k))
        gy_i, gx_i = np.divmod(idx, self.gw)
        return ((gx_i + self._rng.random(k)).astype(np.float32),
                (gy_i + self._rng.random(k)).astype(np.float32))

    # ----- interactions -----
    def spawn_burst(self, x, y, frac=0.08, radius=6.0):
        """Teleport a fraction of the pool into a tight gaussian at (x, y)
        with fresh random headings — Bleuje-style particle spawning, the
        theatrical 'pour more mold HERE' move."""
        k = int(frac * self.n)
        if k <= 0:
            return
        pick = self._rng.integers(0, self.n, k)
        self.px[pick] = (x + self._rng.normal(0, radius, k)).astype(np.float32) % self.gw
        self.py[pick] = (y + self._rng.normal(0, radius, k)).astype(np.float32) % self.gh
        self.heading[pick] = self._rng.uniform(0, 2 * np.pi, k).astype(np.float32)

    def wave(self, x, y):
        """Point every agent's heading away from (x, y) — one radial impulse
        that ripples the whole organism outward, then the mold reknits."""
        self.heading = np.arctan2(self.py - y, self.px - x).astype(np.float32)

    def gather(self, x, y, frac=0.5, radius=60.0):
        """Rush agents already within `radius` of (x, y) into a tight knot
        there — a LOCAL impulse that leaves the rest of the organism alone
        (the react 'spell' move; spawn_burst teleports from the whole pool)."""
        if frac <= 0 or radius <= 0:
            return
        dx = self.px - np.float32(x)
        dy = self.py - np.float32(y)
        dx -= self.gw * np.floor(dx / self.gw + 0.5)     # shortest torus offset
        dy -= self.gh * np.floor(dy / self.gh + 0.5)
        near = dx * dx + dy * dy < np.float32(radius * radius)
        pick = np.flatnonzero(near & (self._rng.random(self.n) < frac))
        k = pick.size
        if k == 0:
            return
        r = 0.15 * radius
        self.px[pick] = (x + self._rng.normal(0, r, k)).astype(np.float32) % self.gw
        self.py[pick] = (y + self._rng.normal(0, r, k)).astype(np.float32) % self.gh
        self.heading[pick] = self._rng.uniform(0, 2 * np.pi, k).astype(np.float32)

    # ----- lifecycle -----
    def release(self):
        """Nothing to free — here so both engines share one stop() contract."""

    # ----- picture -----
    def luminance(self):
        """Tonemapped trail in [0,1] float32 (gh, gw) — the mode colorizes it.

        The raw trail's scale depends on agent density and decay (equilibrium
        is roughly agents-per-cell x deposit x decay/(1-decay)), so it is
        normalized by its own bright end before the exposure curve — the
        exposure slider then means the same thing at any agent count.

        `grain` mixes the CURRENT frame's raw deposits (pre-blur agent
        positions) over the smooth diffused trail — that per-frame dust is
        what makes the organism look granular and alive instead of airbrushed."""
        norm = float(np.percentile(self.trail, 95.0))
        self._norm = norm               # feedback for the `sat` sensing cap
        if norm <= 0:
            return np.zeros((self.gh, self.gw), np.float32)
        x = self.trail * (1.0 / norm)
        if self.grain > 0:
            gnorm = float(self._laid.mean()) * 4.0
            if gnorm > 0:
                x = x + self.grain * (self._laid * (1.0 / gnorm))
        return (1.0 - np.exp(-self.exposure * x)).astype(np.float32)
