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
#   sense:   sensor distance, in grid pixels
#   spread:  sensor half-angle, radians (how wide the left/right feelers sit)
#   turn:    rotation applied when a side sensor wins, radians
#   step:    move distance per frame, grid pixels
#   deposit: trail laid down per agent per frame
POINTS = {
    "veins":   dict(sense=9.0,  spread=0.39, turn=0.79, step=1.4, deposit=1.0),
    "cells":   dict(sense=16.0, spread=0.79, turn=0.52, step=1.1, deposit=0.9),
    "fingers": dict(sense=5.0,  spread=0.26, turn=1.05, step=2.0, deposit=1.1),
    "haze":    dict(sense=3.0,  spread=1.05, turn=0.26, step=0.7, deposit=0.6),
    "web":     dict(sense=24.0, spread=0.52, turn=0.91, step=1.6, deposit=0.8),
    "storm":   dict(sense=12.0, spread=1.40, turn=1.40, step=2.4, deposit=1.2),
}
POINT_NAMES = list(POINTS)

_PARAM_KEYS = ("sense", "spread", "turn", "step", "deposit")


class PhysarumField:
    def __init__(self, n=250_000, gw=480, gh=270, seed=0,
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
        self._rng = rng
        self._laid = np.zeros((gh, gw), np.float32)

    # ----- parameter blending -----
    def _blend_params(self, t):
        """Per-agent parameter arrays: background point blended toward the
        subject point by t = matte value under each agent (the pen)."""
        a, b = POINTS[self.point_bg], POINTS[self.point_fg]
        out = {}
        for k in _PARAM_KEYS:
            lo, hi = np.float32(a[k]), np.float32(b[k])
            out[k] = lo + (hi - lo) * t
        return out

    def swap_points(self):
        self.point_bg, self.point_fg = self.point_fg, self.point_bg

    # ----- one simulation frame -----
    def update(self, matte, gray):
        """matte, gray: float32 (gh, gw) in [0,1]. Advances agents one frame
        and rebuilds the trail map."""
        gw, gh = self.gw, self.gh
        n = self.n
        px, py, h = self.px, self.py, self.heading

        # float32 modulo can land exactly ON the upper bound (a hair below gw
        # rounds up in float32), so every grid index wraps in INT space
        t = matte[py.astype(np.int32) % gh, px.astype(np.int32) % gw]
        p = self._blend_params(t)
        sense_d = p["sense"] * self.gain
        step_d = p["step"] * self.gain

        # what the sensors read: laid trail plus the footage's light as food
        food = self.trail if self.food <= 0 else self.trail + self.food * gray

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
        self.trail *= self.decay

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
        if norm <= 0:
            return np.zeros((self.gh, self.gw), np.float32)
        x = self.trail * (1.0 / norm)
        if self.grain > 0:
            gnorm = float(self._laid.mean()) * 4.0
            if gnorm > 0:
                x = x + self.grain * (self._laid * (1.0 / gnorm))
        return (1.0 - np.exp(-self.exposure * x)).astype(np.float32)
