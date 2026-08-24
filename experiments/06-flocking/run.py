#!/usr/bin/env python3
"""Flocking system — emergent order from local rules, rendered as oriented instances.

Reynolds boids (separation, alignment, cohesion) in 3D, with a soft spherical
boundary and a slow global swirl. No central controller — every agent sees only
its neighbours within a radius, and coherent motion falls out of the three local
forces. The same shape as a swarm of agents tending a shared commons: order that
is produced, not imposed.

Pipeline (TouchDesigner graph, as code):
    boid state (POP: pos+vel) -> neighbour forces (separation/alignment/cohesion)
    -> integrate -> heading euler + speed->scale -> pack -> instanced cubes (Copy SOP)
    -> directional light + depth (Light/Render) -> MP4 + PNG

Examples:
    python run.py                              # 700 boids, 240 frames -> out/flock.mp4
    python run.py --boids 1200 --frames 600
    python run.py --neighbor 0.5 --separation 1.8 --swirl 0.25
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import imageio.v2 as imageio

from dtouch import pack_instances, Renderer


def _normalize(v, eps=1e-9):
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.maximum(n, eps)


def _limit(v, max_len):
    n = np.linalg.norm(v, axis=1, keepdims=True)
    scale = np.minimum(1.0, max_len / np.maximum(n, 1e-9))
    return v * scale


def heading_to_euler(vel):
    """Euler (x_pitch, y, z_yaw) so a cube points along its velocity. Yaw from the
    XY heading, pitch from the Z climb — enough for the instances to read as a
    directed shoal rather than a static cloud."""
    vx, vy, vz = vel[:, 0], vel[:, 1], vel[:, 2]
    yaw = np.arctan2(vy, vx)
    pitch = np.arctan2(vz, np.sqrt(vx * vx + vy * vy) + 1e-9)
    euler = np.zeros((vel.shape[0], 3), dtype=np.float32)
    euler[:, 0] = pitch
    euler[:, 2] = yaw
    return euler


def step(pos, vel, p):
    """One flocking update. O(n^2) neighbour pass — fine for a few thousand boids,
    and honest about where the cost is rather than hiding it behind a grid."""
    n = pos.shape[0]
    # pairwise offsets + distances
    diff = pos[:, None, :] - pos[None, :, :]          # (n, n, 3): i minus j
    dist2 = np.sum(diff * diff, axis=2)               # (n, n)
    np.fill_diagonal(dist2, np.inf)
    within = dist2 < (p.neighbor * p.neighbor)        # neighbour mask

    counts = np.maximum(within.sum(axis=1, keepdims=True), 1)

    # cohesion: steer toward neighbour centroid
    centroid = (within[:, :, None] * pos[None, :, :]).sum(axis=1) / counts
    cohesion = _normalize(centroid - pos)

    # alignment: match neighbour mean heading
    mean_vel = (within[:, :, None] * vel[None, :, :]).sum(axis=1) / counts
    alignment = _normalize(mean_vel)

    # separation: push from too-close neighbours, weighted by 1/dist
    close = dist2 < (p.sep_radius * p.sep_radius)
    np.fill_diagonal(close, False)
    w = close / (dist2 + 1e-6)
    sep = (w[:, :, None] * diff).sum(axis=1)          # diff already points i<-j
    separation = _normalize(sep)

    # slow global swirl about Z — a breath of shared weather over local rules
    swirl = np.stack([-pos[:, 1], pos[:, 0], np.zeros(n)], axis=1)
    swirl = _normalize(swirl) * p.swirl

    # soft spherical containment
    r = np.linalg.norm(pos, axis=1, keepdims=True)
    inward = -_normalize(pos) * np.maximum(0.0, r - p.bound) * 2.5

    # interactive attractor/repulsor (live mode): the cursor as a hand in the field
    attract = 0.0
    att = getattr(p, "_attractor", None)
    if att is not None:
        to = np.zeros((n, 3), dtype=np.float32)
        to[:, 0] = att[0] - pos[:, 0]
        to[:, 1] = att[1] - pos[:, 1]
        attract = _normalize(to) * (getattr(p, "_attract_sign", 1.0) * p.attract_strength)

    acc = (p.cohesion * cohesion + p.alignment * alignment +
           p.separation * separation + swirl + inward + attract)
    vel = _limit(vel + acc * p.dt, p.max_speed)
    vel = np.where(np.linalg.norm(vel, axis=1, keepdims=True) < p.min_speed,
                   _normalize(vel) * p.min_speed, vel)
    pos = pos + vel * p.dt
    if p.recenter:
        pos = pos - pos.mean(axis=0, keepdims=True)  # camera tracks the shoal's centroid
    return pos, vel


SHAPES = ["cube", "star", "bird"]


def _xstretch(shape):
    """Cubes get a motion-streak along the heading; stars/birds keep their proportions."""
    return 1.8 if shape == "cube" else 1.0


MOODS = {
    "1 murmuration": dict(cohesion=0.7, alignment=1.5, separation=1.6, swirl=0.10, neighbor=0.5),
    "2 scatter":     dict(cohesion=0.3, alignment=0.8, separation=2.6, swirl=0.05, neighbor=0.4),
    "3 vortex":      dict(cohesion=0.55, alignment=1.2, separation=1.7, swirl=0.55, neighbor=0.45),
}


def live(p):
    """Real-time interactive flocking instrument. A new base mode for lighteater:
    self-driving boids instead of a video-driven matte. Keys tune the three Reynolds
    forces + the swirl live; the mouse is a hand in the field (drag to attract, right-drag
    to scatter). Emergent order you can push around."""
    import cv2
    import signal as _signal

    rw, rh = (int(x) for x in p.res.lower().split("x"))
    n = p.boids
    rng = np.random.default_rng(p.seed)
    pos = rng.normal(0.0, 0.5, size=(n, 3)).astype(np.float32)
    vel = _normalize(rng.normal(0.0, 1.0, size=(n, 3)).astype(np.float32)) * p.min_speed
    base_size = 1.4 / np.sqrt(n)
    renderer = Renderer(rw, rh, n, base_size=base_size, depth_scale=1.0, extent=1.7, geometry=p.shape)
    p.recenter = True
    p._attractor = None
    p._attract_sign = 1.0
    extent = 1.7

    win = "lighteater · flocking"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, rw, rh)

    state = {"mx": rw // 2, "my": rh // 2, "down": 0, "hud": True, "freeze": False, "quit": False}

    # Make the window actually closable: macOS cv2 windows often render no close button
    # and 'q' needs window focus, so a backgrounded launch can trap the user. Catch
    # SIGTERM (dock Quit) and SIGINT (Ctrl-C) and break the loop cleanly.
    def _on_sig(*_a):
        state["quit"] = True
    for _sig in (_signal.SIGTERM, _signal.SIGINT):
        try:
            _signal.signal(_sig, _on_sig)
        except Exception:
            pass

    def on_mouse(event, x, y, flags, _):
        state["mx"], state["my"] = x, y
        if event == cv2.EVENT_LBUTTONDOWN: state["down"] = 1     # attract
        elif event == cv2.EVENT_RBUTTONDOWN: state["down"] = -1  # repel
        elif event in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP): state["down"] = 0
    cv2.setMouseCallback(win, on_mouse)

    def screen_to_world(mx, my):
        wx = (mx / rw - 0.5) * 2.0 * extent
        wy = -(my / rh - 0.5) * 2.0 * extent
        return (wx, wy)

    def apply_mood(name):
        for k, v in MOODS[name].items():
            setattr(p, k.replace("-", "_"), v)

    while True:
        if state["quit"]:
            break
        if state["down"] != 0:
            p._attractor = screen_to_world(state["mx"], state["my"])
            p._attract_sign = float(state["down"])
        else:
            p._attractor = None

        if not state["freeze"]:
            pos, vel = step(pos, vel, p)

        speed = np.linalg.norm(vel, axis=1, keepdims=True)
        s = np.repeat(0.7 + 1.6 * (speed / p.max_speed), 3, axis=1).astype(np.float32)
        s[:, 0] *= _xstretch(p.shape)
        euler = heading_to_euler(vel)
        frame = renderer.render(pack_instances(pos.astype(np.float32), s, euler))
        frame = np.ascontiguousarray(frame[:, :, ::-1])  # RGB -> BGR for cv2

        if state["hud"]:
            lines = [
                "flocking  ·  drag=attract  right-drag=scatter  space=freeze  h=hud  q=quit",
                f"[c/C] cohesion {p.cohesion:.2f}   [a/A] alignment {p.alignment:.2f}   [s/S] separation {p.separation:.2f}",
                f"[w/W] swirl {p.swirl:.2f}   [g] shape: {p.shape}   [1/2/3] moods   [r] reset   boids {n}",
            ]
            for i, t in enumerate(lines):
                cv2.putText(frame, t, (14, 26 + i * 24), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow(win, frame)
        # quit if the window was closed via the title bar / window manager (macOS often
        # renders no close button — this makes closing the window actually stop the loop)
        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q") or k == 27: break
        elif k == ord("g"):
            p.shape = SHAPES[(SHAPES.index(p.shape) + 1) % len(SHAPES)]
            renderer.set_geometry(p.shape)
        elif k == ord("c"): p.cohesion = max(0.0, p.cohesion - 0.05)
        elif k == ord("C"): p.cohesion += 0.05
        elif k == ord("a"): p.alignment = max(0.0, p.alignment - 0.05)
        elif k == ord("A"): p.alignment += 0.05
        elif k == ord("s"): p.separation = max(0.0, p.separation - 0.05)
        elif k == ord("S"): p.separation += 0.05
        elif k == ord("w"): p.swirl = max(0.0, p.swirl - 0.02)
        elif k == ord("W"): p.swirl += 0.02
        elif k == ord("1"): apply_mood("1 murmuration")
        elif k == ord("2"): apply_mood("2 scatter")
        elif k == ord("3"): apply_mood("3 vortex")
        elif k == ord("h"): state["hud"] = not state["hud"]
        elif k == ord(" "): state["freeze"] = not state["freeze"]
        elif k == ord("r"):
            pos = rng.normal(0.0, 0.5, size=(n, 3)).astype(np.float32)
            vel = _normalize(rng.normal(0.0, 1.0, size=(n, 3)).astype(np.float32)) * p.min_speed

    renderer.release()
    cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description="Emergent flocking -> instanced particle render")
    ap.add_argument("--boids", type=int, default=700)
    ap.add_argument("--res", default="960x540", help="output resolution WxH")
    ap.add_argument("--frames", type=int, default=240)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--neighbor", type=float, default=0.45, help="neighbour radius")
    ap.add_argument("--sep-radius", type=float, default=0.2, help="separation radius")
    ap.add_argument("--cohesion", type=float, default=0.55)
    ap.add_argument("--alignment", type=float, default=1.25)
    ap.add_argument("--separation", type=float, default=2.0)
    ap.add_argument("--swirl", type=float, default=0.13)
    ap.add_argument("--bound", type=float, default=1.5, help="containment sphere radius")
    ap.add_argument("--max-speed", type=float, default=0.9)
    ap.add_argument("--min-speed", type=float, default=0.3)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--recenter", action=argparse.BooleanOptionalAction, default=True,
                    help="keep the shoal centroid framed (camera tracks it)")
    ap.add_argument("--live", action="store_true",
                    help="real-time interactive instrument (window + key/mouse controls)")
    ap.add_argument("--attract-strength", type=float, default=2.2,
                    help="pull/push strength of the mouse hand in --live")
    ap.add_argument("--shape", default="cube", choices=SHAPES,
                    help="instance geometry: cube | star (Matariki ✦) | bird")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "out", "flock.mp4"))
    p = ap.parse_args()

    if p.live:
        live(p)
        return

    rw, rh = (int(x) for x in p.res.lower().split("x"))
    n = p.boids
    os.makedirs(os.path.dirname(p.out) or ".", exist_ok=True)

    rng = np.random.default_rng(p.seed)
    pos = rng.normal(0.0, 0.5, size=(n, 3)).astype(np.float32)
    vel = _normalize(rng.normal(0.0, 1.0, size=(n, 3)).astype(np.float32)) * p.min_speed

    base_size = 1.4 / np.sqrt(n)            # cube size shrinks as the shoal grows
    renderer = Renderer(rw, rh, n, base_size=base_size, depth_scale=1.0, extent=1.7, geometry=p.shape)

    writer = imageio.get_writer(p.out, fps=p.fps, macro_block_size=8)
    png_path = os.path.splitext(p.out)[0] + "_frame0.png"
    mid_png = os.path.splitext(p.out)[0] + "_mid.png"

    for i in range(p.frames):
        pos, vel = step(pos, vel, p)
        speed = np.linalg.norm(vel, axis=1, keepdims=True)
        # length-stretch along heading via per-axis scale: faster boids read longer
        s = np.repeat(0.7 + 1.6 * (speed / p.max_speed), 3, axis=1).astype(np.float32)
        s[:, 0] *= _xstretch(p.shape)                       # elongate the leading axis
        euler = heading_to_euler(vel)
        frame = renderer.render(pack_instances(pos.astype(np.float32), s, euler))
        writer.append_data(frame)
        if i == 0:
            imageio.imwrite(png_path, frame)
        if i == p.frames // 2:
            imageio.imwrite(mid_png, frame)

    writer.close()
    renderer.release()
    print(f"rendered {p.frames} frames of {n} boids -> {p.out}")
    print(f"frames -> {png_path}, {mid_png}")


if __name__ == "__main__":
    main()
