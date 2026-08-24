"""SIGNAL rack CPU vs GPU at 4K — the PR #26 port list, measured.

Three tables, all headless (standalone GL contexts only — no camera, no
window, no audio):

1. Rack stages at 4K (3840x2160). CPU = the numpy stage cost inside
   CircuitBent.apply_plan; GPU = the same stage as SignalRackGL fragment
   passes, enabled alone, with a ctx.finish() so the number is the GPU's
   real cost. Best-of-N is the code's cost (the repo's bench convention);
   the median shows what a loaded machine sees.
2. The physarum GL output path, per phase and tier: sim update, tonemap +
   stats, colorize + upscale, the full default rack, and THE readback (the
   one uint8 RGB transfer of the composed 4K frame).
3. Full PhysarumMode.step at 4K per quality tier, rack ON: the ported GPU
   path vs the classic CPU path (mode step + shell-style cb.process).

    python bench.py               # all tables
    python bench.py --frames 90   # longer run
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from dtouch.circuit_bent import CircuitBent                     # noqa: E402
from dtouch.modes.physarum import PhysarumMode, _palette_lut    # noqa: E402
from dtouch.overlay_ui import sync_signal                       # noqa: E402
from dtouch.physarum_gl import PhysarumFieldGL, PhysarumGLUnavailable  # noqa: E402
from dtouch.rack_gl import PhysarumOutGL, SignalRackGL          # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
W4K, H4K = 3840, 2160
TIERS = PhysarumMode.QUALITY


def _ms(ts):
    xs = np.array(ts) * 1000.0
    return float(xs.min()), float(np.median(xs))


def synth_frame(i, w=1280, h=720):
    base = np.zeros((h, w, 3), np.uint8)
    base[:] = np.linspace(20, 50, w, dtype=np.uint8)[None, :, None]
    cv2.circle(base, (int(w * (0.3 + 0.4 * (i % 40) / 40)), h // 2), 100,
               (235, 235, 235), -1)
    return cv2.GaussianBlur(base, (15, 15), 0)


def synth_field_input(gw, gh, i):
    yy, xx = np.mgrid[0:gh, 0:gw].astype(np.float32)
    cx = gw * (0.5 + 0.3 * np.sin(i * 0.05))
    cy = gh * (0.5 + 0.3 * np.cos(i * 0.037))
    r2 = (xx - cx) ** 2 + (yy - cy) ** 2
    sig = min(gw, gh) * 0.15
    matte = np.exp(-r2 / (2 * sig * sig)).astype(np.float32)
    gray = (0.15 + 0.7 * np.exp(-r2 / (2 * (1.6 * sig) ** 2))).astype(np.float32)
    return matte, gray


# One rack config per stage row: everything else off. The first row is the
# CPU path's fixed framing cost (uint8 -> float32 in, clip + float32 ->
# uint8 out, ~30 ms at 4K) that every CPU stage number below also carries;
# the GPU path's equivalent (upload is the mode's texture, readback is row
# `readback` in table 2) is the driftchroma+compose passes alone.
STAGE_CONFIGS = [
    ("framing (f32<->u8 alone)", dict()),
    ("scan-drift", dict(scan_drift=8)),
    ("chroma", dict(chroma_shift=10)),
    ("glitch (prob=1)", dict(glitch_prob=1.0, glitch_hold=4)),
    ("bit crush", dict(bit_crush=3)),
    ("bayer dither @ rows", dict(dither_mode="bayer")),
    ("scanlines + u8 encode", dict(scanlines=True)),
    ("default chain", dict(scan_drift=8, chroma_shift=10, glitch_prob=0.10,
                           glitch_hold=4, scanlines=True,
                           dither_mode="bayer")),
]
OFF = dict(chroma_shift=0, scan_drift=0, glitch_prob=0.0, bit_crush=0,
           scanlines=False, dither_mode=None)


def bench_rack_stages(frames_n):
    print(f"\n== 1. rack stages @ {W4K}x{H4K}, ms (best / median of "
          f"{frames_n}) ==")
    print(f"{'stage':24s} {'CPU':>15s} {'GPU':>15s}")
    frame = cv2.resize(synth_frame(0), (W4K, H4K))
    rows = PhysarumMode.signal_dither_rows(H4K)
    import moderngl
    ctx = moderngl.create_standalone_context()
    src = ctx.texture((W4K, H4K), 4, dtype="f1")
    src.write(np.ascontiguousarray(
        np.dstack([frame, np.full((H4K, W4K), 255, np.uint8)])))
    for name, over in STAGE_CONFIGS:
        kw = {**OFF, **over}
        if kw.get("dither_mode"):
            kw["dither_size"] = rows
        cpu_cb = CircuitBent(seed=7, **kw)
        gpu_cb = CircuitBent(seed=7, **kw)
        rack = SignalRackGL(ctx, W4K, H4K)
        t_cpu, t_gpu = [], []
        for _ in range(frames_n):
            t0 = time.perf_counter()
            cpu_cb.process(frame)
            t_cpu.append(time.perf_counter() - t0)
        for _ in range(frames_n):
            plan = gpu_cb.plan(H4K, W4K)
            ctx.finish()
            t0 = time.perf_counter()
            rack.run(gpu_cb, plan, src)
            ctx.finish()
            t_gpu.append(time.perf_counter() - t0)
        rack.release()
        cb, cm = _ms(t_cpu)
        gb, gm = _ms(t_gpu)
        print(f"{name:24s} {cb:6.1f} / {cm:6.1f} {gb:6.1f} / {gm:6.1f}")
    src.release()
    ctx.release()


def bench_gl_phases(frames_n):
    print(f"\n== 2. GL output path phases @ {W4K}x{H4K}, ms best "
          f"(median) of {frames_n} ==")
    print(f"{'tier':8s} {'sim':>12s} {'tone+stats':>12s} {'color+up':>12s} "
          f"{'rack':>12s} {'readback':>12s} {'sum(best)':>10s}")
    for tier, ((gw, gh), n) in TIERS.items():
        try:
            pf = PhysarumFieldGL(n=n, gw=gw, gh=gh, seed=1)
        except PhysarumGLUnavailable as e:
            print(f"no GL: {e}")
            return
        cb = CircuitBent(seed=7)
        cb.dither_size = PhysarumMode.signal_dither_rows(H4K)
        with pf.ctx:
            glout = PhysarumOutGL(pf.ctx, gw, gh, W4K, H4K)
        lut = _palette_lut("arctic")
        cols = [[] for _ in range(5)]
        for i in range(frames_n):
            matte, gray = synth_field_input(gw, gh, i)
            with pf.ctx:
                t0 = time.perf_counter()
                pf.update(matte, gray)
                pf.ctx.finish()
                t1 = time.perf_counter()
                pf.luminance_into_tex()
                pf.ctx.finish()
                t2 = time.perf_counter()
                srct = glout.compose(pf.tex_lum, lut, None, None, 0.0)
                pf.ctx.finish()
                t3 = time.perf_counter()
                glout.rack.run(cb, cb.plan(H4K, W4K), srct)
                pf.ctx.finish()
                t4 = time.perf_counter()
                glout.rack.read()
                t5 = time.perf_counter()
            for c, v in zip(cols, np.diff([t0, t1, t2, t3, t4, t5])):
                c.append(v)
        parts = [_ms(c) for c in cols]
        total = sum(b for b, _ in parts)
        print(f"{tier:8s} " + " ".join(f"{b:5.1f} ({m:5.1f})"
                                       for b, m in parts)
              + f" {total:9.1f}")
        with pf.ctx:
            glout.release()
        pf.release()


class _StubToasts:
    def flash(self, *a, **k):
        pass


class _StubHost:
    def __init__(self, res):
        self.res = res
        self.ui = type("Ui", (), {})()
        self.hud = type("Hud", (), {})()
        self.hud.toasts = _StubToasts()


def _signal_ui(ui, glitch):
    ui.glitch = glitch
    ui.chroma, ui.drift, ui.crush = 10.0, 8.0, 0.0
    ui.sig_bits, ui.sig_gamma, ui.sig_bias_idx = 3.0, True, 0
    ui.scanlines = True
    ui.dither_name = "bayer"
    ui.ph_matte_idx = 5          # luma
    ui.ph_video_bg = False


def bench_full_pipeline(frames_n):
    print(f"\n== 3. full PhysarumMode.step @ {W4K}x{H4K}, rack ON "
          f"(best / median of {frames_n}) ==")
    print(f"{'tier':8s} {'path':9s} {'ms':>16s} {'fps best':>9s} "
          f"{'fps med':>8s}")
    cams = [synth_frame(i) for i in range(40)]
    for tier_idx, tier in enumerate(TIERS):
        for gpu in (True, False):
            m = PhysarumMode(matte="luma", seed=1, engine="gl")
            host = _StubHost((W4K, H4K))
            m.start(host)
            if m.engine != "gl":
                print("no GL — skipping full-pipeline table")
                m.stop()
                return
            _signal_ui(host.ui, glitch=gpu)
            host.ui.ph_quality_idx = tier_idx
            cb = CircuitBent(seed=7)
            ts = []
            out = None
            for i in range(frames_n + 4):
                f = cams[i % len(cams)]
                t0 = time.perf_counter()
                out = m.step(f, None, 1 / 30)
                if not gpu:
                    sync_signal(cb, host.ui, m, H4K)
                    out = cb.process(out)
                if i >= 4:                    # skip tier-rebuild warmup
                    ts.append(time.perf_counter() - t0)
            if gpu and out is not None:
                os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
                cv2.imwrite(os.path.join(HERE, "out", f"rack_gl_{tier}.png"),
                            cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
            m.stop()
            b, md = _ms(ts)
            lab = "GPU rack" if gpu else "CPU rack"
            print(f"{tier:8s} {lab:9s} {b:6.1f} / {md:6.1f} "
                  f"{1000.0 / b:8.1f} {1000.0 / md:8.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=60)
    args = ap.parse_args()
    la = os.getloadavg()
    print(f"load average: {la[0]:.0f} {la[1]:.0f} {la[2]:.0f} "
          f"on {os.cpu_count()} cores")
    bench_rack_stages(args.frames)
    bench_gl_phases(args.frames)
    bench_full_pipeline(args.frames)


if __name__ == "__main__":
    main()
