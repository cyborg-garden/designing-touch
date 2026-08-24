"""Physarum CPU vs GPU — same model, same machine, same method.

Drives dtouch.physarum.PhysarumField (numpy) and dtouch.physarum_gl.
PhysarumFieldGL (moderngl ping-pong) through identical synthetic input — a
gaussian "subject" that drifts around the grid as both matte and luma — and
times, per frame:

  sim       update() alone. For the GPU this includes a ctx.finish() so the
            number is the GPU's real cost, not the time to enqueue it.
  picture   luminance(): tonemap + the readback to numpy (stats + uint8 on
            the GPU; percentile + exp on the CPU).
  frame     update() + luminance() back to back, no extra sync — what the
            mode pays per step, before colorize/upscale.

Best-of-N and median are both printed: the minimum is the code's cost, the
median is what a show sees. Raw glReadPixels costs for the three readback
formats are measured separately so "readback-bound" is a claim with a number.

Headless: standalone GL context only, no window, no camera, no audio. Writes
out/physarum_gl.png (the last GPU frame, colorized) for eyeballing.

    python bench.py                       # the table
    python bench.py --frames 120          # longer run
    python bench.py --gpu-only --agents 4000000
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from dtouch.physarum import PhysarumField                       # noqa: E402
from dtouch.physarum_gl import PhysarumFieldGL, PhysarumGLUnavailable  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def synth(gw, gh, i):
    """Drifting gaussian blob as matte and (slightly dimmer, wider) luma."""
    yy, xx = np.mgrid[0:gh, 0:gw].astype(np.float32)
    cx = gw * (0.5 + 0.3 * np.sin(i * 0.05))
    cy = gh * (0.5 + 0.3 * np.cos(i * 0.037))
    r2 = (xx - cx) ** 2 + (yy - cy) ** 2
    sig = min(gw, gh) * 0.15
    matte = np.exp(-r2 / (2 * sig * sig)).astype(np.float32)
    gray = (0.15 + 0.7 * np.exp(-r2 / (2 * (1.6 * sig) ** 2))).astype(np.float32)
    return matte, gray


def _stats(xs):
    xs = np.array(xs) * 1000.0
    return float(xs.min()), float(np.median(xs))


def bench(field, gw, gh, frames, warm, sync=None):
    inputs = [synth(gw, gh, i) for i in range(frames + warm)]
    for i in range(warm):
        field.update(*inputs[i])
        field.luminance()
    sim, pic, full = [], [], []
    lum = None
    for i in range(warm, warm + frames):
        m, g = inputs[i]
        t0 = time.perf_counter()
        field.update(m, g)
        if sync is not None:
            sync()
        t1 = time.perf_counter()
        lum = field.luminance()
        t2 = time.perf_counter()
        sim.append(t1 - t0)
        pic.append(t2 - t1)
    # the un-synced pair, as the mode pays it
    for i in range(warm, warm + frames):
        m, g = inputs[i]
        t0 = time.perf_counter()
        field.update(m, g)
        lum = field.luminance()
        full.append(time.perf_counter() - t0)
    return dict(sim=_stats(sim), pic=_stats(pic), frame=_stats(full)), lum


def readback_costs(gw, gh, runs=30):
    """Raw glReadPixels of a gw x gh target: R32F, R8, RGB8."""
    import moderngl
    ctx = moderngl.create_standalone_context()
    out = {}
    try:
        for label, comps, dtype in (("R32F", 1, "f4"), ("R8", 1, "f1"), ("RGB8", 3, "f1")):
            tex = ctx.texture((gw, gh), comps, dtype=dtype)
            fbo = ctx.framebuffer(color_attachments=[tex])
            fbo.use()
            ctx.clear(0.3, 0.2, 0.1, 1.0)
            ctx.finish()
            best = np.inf
            for _ in range(runs):
                t0 = time.perf_counter()
                raw = fbo.read(components=comps, dtype=dtype)
                np.frombuffer(raw, np.float32 if dtype == "f4" else np.uint8)
                best = min(best, time.perf_counter() - t0)
            out[label] = (best * 1000.0, len(raw) / 1e6)
            fbo.release()
            tex.release()
    finally:
        ctx.release()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--warm", type=int, default=10)
    ap.add_argument("--agents", type=int, default=None,
                    help="extra GPU row at this agent count on the full grid")
    ap.add_argument("--gpu-only", action="store_true")
    args = ap.parse_args()

    rows = []
    small = (576, 324)
    big = (1280, 736)

    if not args.gpu_only:
        f = PhysarumField(n=400_000, gw=small[0], gh=small[1], seed=1)
        r, _ = bench(f, *small, args.frames, args.warm)
        rows.append(("CPU", 400_000, small, r))

    try:
        gl_rows = [(400_000, small), (1_000_000, big), (2_000_000, big), (4_000_000, big)]
        if args.agents:
            gl_rows.append((args.agents, big))
        last = None
        for n, grid in gl_rows:
            f = PhysarumFieldGL(n=n, gw=grid[0], gh=grid[1], seed=1)
            try:
                r, lum = bench(f, *grid, args.frames, args.warm, sync=f.ctx.finish)
            finally:
                f.release()
            rows.append(("GPU", n, grid, r))
            if grid == big:
                last = lum
        rb = readback_costs(*big)
    except PhysarumGLUnavailable as e:
        print("GPU unavailable:", e)
        rb, last = {}, None

    print()
    print("engine   agents     grid       sim ms (best/med)   picture ms   frame ms      fps(med)")
    for eng, n, (gw, gh), r in rows:
        fps = 1000.0 / r["frame"][1]
        print(f"{eng:<6} {n:>9,}   {gw}x{gh:<5}   {r['sim'][0]:6.2f} / {r['sim'][1]:6.2f}   "
              f"{r['pic'][0]:5.2f} / {r['pic'][1]:5.2f}   {r['frame'][0]:5.2f} / "
              f"{r['frame'][1]:5.2f}   {fps:6.1f}")
    if rb:
        print()
        print(f"raw glReadPixels @ {big[0]}x{big[1]} (best of 30):")
        for k, (ms, mb) in rb.items():
            print(f"  {k:<5} {mb:5.2f} MB  {ms:5.2f} ms")

    if last is not None:
        from dtouch.modes.physarum import _palette_lut
        img = _palette_lut("arctic")[(last * 255.0).astype(np.uint8)]
        os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
        path = os.path.join(HERE, "out", "physarum_gl.png")
        cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        print("\nwrote", path)


if __name__ == "__main__":
    main()
