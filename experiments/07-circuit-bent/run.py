#!/usr/bin/env python3
"""Experiment 07 — circuit-bent camera effect (live or synthetic).

Applies stochastic circuit-bent glitching (chroma bleed, scan-line drift,
glitch blocks, bit-crush, CRT scan lines) and Bayer/Floyd-Steinberg dithering
to a live webcam feed or a synthetic source for headless testing.

    python run.py                          # live, built-in camera, all effects on
    python run.py --device 1              # specific camera index
    python run.py --source synthetic      # headless synthetic source (no camera)
    python run.py --dither fs             # Floyd-Steinberg dithering
    python run.py --bits 2 --crush 2      # 2-bit dither + 2-bit hard quantisation
    python run.py --no-scanlines          # disable CRT scan-line overlay
    python run.py --frames 60 --no-show   # render 60 frames, save last to out/

Controls (live):
    q / close window  quit
    d                 cycle dither mode (bayer → fs → none)
    b                 toggle bit-crush
    s                 toggle scan lines
    g                 toggle glitch blocks
    +/-               increase/decrease chroma shift
    [/]               increase/decrease scan drift
"""
from __future__ import annotations

import argparse
import os
import time

import cv2
import numpy as np

from dtouch.circuit_bent import CircuitBent
from dtouch.sources import SyntheticSource


_DITHER_MODES = ["bayer", "fs", None]
_WIN = "lighteater - circuit bent"


def _read_camera(cap):
    ok, frame = cap.read()
    return frame if ok else None


def _open_capture(device):
    if isinstance(device, int):
        return cv2.VideoCapture(device, cv2.CAP_AVFOUNDATION)
    return cv2.VideoCapture(device)


def run(
    source="builtin",
    dither="bayer",
    dither_bits=3,
    dither_size=72,
    bit_crush=0,
    chroma_shift=10,
    scan_drift=8,
    glitch_prob=0.10,
    scanlines=True,
    scanline_strength=0.35,
    seed=0,
    mirror=True,
    show=True,
    max_frames=None,
    out_dir="out",
):
    """Main loop: source → circuit-bent → display / save.

    Returns (frame_count, last_output_frame_or_None).
    """
    use_synthetic = (source == "synthetic")

    # --- source setup ---
    if use_synthetic:
        gx, gy = 416, 234
        src = SyntheticSource(gx, gy, frames=max_frames or 9999)
        cap = None
    else:
        device = int(source) if source.isdigit() else source
        if device == "builtin":
            from dtouch.camera import open_camera
            cap, _ = open_camera(device)
        else:
            cap = _open_capture(device if isinstance(device, int) else device)
        src = None

    cb = CircuitBent(
        seed=seed,
        chroma_shift=chroma_shift,
        scan_drift=scan_drift,
        glitch_prob=glitch_prob,
        bit_crush=bit_crush,
        scanlines=scanlines,
        scanline_strength=scanline_strength,
        dither_mode=dither,
        dither_bits=dither_bits,
        dither_size=dither_size,
    )

    dither_idx = _DITHER_MODES.index(dither) if dither in _DITHER_MODES else 0

    if show:
        cv2.namedWindow(_WIN, cv2.WINDOW_AUTOSIZE)

    os.makedirs(out_dir, exist_ok=True)

    count = 0
    t0 = time.time()
    fps = 0.0
    last_out = None

    try:
        while True:
            # --- read frame ---
            if use_synthetic:
                luma = src.read()
                if luma is None:
                    break
                # convert float32 luma → BGR uint8
                rgb = np.stack([luma] * 3, axis=2)
                # add a faint colour tint so chroma-shift is visible
                rgb[:, :, 0] *= 0.6
                rgb[:, :, 2] *= 1.4
                frame = np.clip(rgb * 255, 0, 255).astype(np.uint8)
            else:
                ok, frame = cap.read()
                if not ok:
                    if max_frames is None:
                        continue
                    break
                if mirror:
                    frame = cv2.flip(frame, 1)

            # --- apply effect ---
            out_bgr = cb.process(frame)
            last_out = out_bgr
            count += 1

            if count % 10 == 0:
                now = time.time()
                fps = 10.0 / max(now - t0, 1e-6)
                t0 = now

            if show:
                label = (
                    f"{fps:4.1f}fps  dither={cb.dither_mode}  "
                    f"bits={cb.dither_bits}  crush={cb.bit_crush}  "
                    f"chroma={cb.chroma_shift}  drift={cb.scan_drift}"
                )
                disp = out_bgr.copy()
                cv2.putText(disp, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                            0.45, (180, 255, 180), 1, cv2.LINE_AA)
                cv2.imshow(_WIN, disp)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                if cv2.getWindowProperty(_WIN, cv2.WND_PROP_VISIBLE) < 0:
                    break
                # live key bindings
                if key == ord('d'):
                    dither_idx = (dither_idx + 1) % len(_DITHER_MODES)
                    cb.dither_mode = _DITHER_MODES[dither_idx]
                elif key == ord('b'):
                    cb.bit_crush = 0 if cb.bit_crush else 2
                elif key == ord('s'):
                    cb.scanlines = not cb.scanlines
                elif key == ord('g'):
                    cb.glitch_prob = 0.0 if cb.glitch_prob else 0.10
                elif key == ord('+') or key == ord('='):
                    cb.chroma_shift = min(cb.chroma_shift + 2, 40)
                elif key == ord('-'):
                    cb.chroma_shift = max(cb.chroma_shift - 2, 0)
                elif key == ord(']'):
                    cb.scan_drift = min(cb.scan_drift + 2, 40)
                elif key == ord('['):
                    cb.scan_drift = max(cb.scan_drift - 2, 0)

            if max_frames is not None and count >= max_frames:
                break
    finally:
        if cap is not None:
            cap.release()
        if show:
            cv2.destroyAllWindows()
            cv2.waitKey(1)

    if last_out is not None:
        path = os.path.join(out_dir, "circuit_bent_last.png")
        cv2.imwrite(path, last_out)

    return count, last_out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="builtin",
                    help="'builtin' (laptop cam), int index, 'synthetic', or a file path")
    ap.add_argument("--dither", default="bayer", choices=["bayer", "fs", "none"])
    ap.add_argument("--bits", type=int, default=3, dest="dither_bits",
                    help="dither bit depth (1–8, default 3)")
    ap.add_argument("--dither-size", type=int, default=72,
                    help="dither at this image height then upscale (0 = full res)")
    ap.add_argument("--crush", type=int, default=0, dest="bit_crush",
                    help="hard quantisation bit depth (0 = off)")
    ap.add_argument("--chroma", type=int, default=10, dest="chroma_shift",
                    help="max chroma bleed offset in pixels")
    ap.add_argument("--drift", type=int, default=8, dest="scan_drift",
                    help="max scan-line drift in pixels")
    ap.add_argument("--glitch", type=float, default=0.10, dest="glitch_prob",
                    help="per-frame glitch block probability (0–1)")
    ap.add_argument("--no-scanlines", action="store_false", dest="scanlines")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-mirror", action="store_true")
    ap.add_argument("--no-show", action="store_true",
                    help="headless mode (no cv2 window)")
    ap.add_argument("--frames", type=int, default=None,
                    help="stop after N frames (default: run until quit)")
    args = ap.parse_args()

    dither = None if args.dither == "none" else args.dither
    dither_size = args.dither_size if args.dither_size > 0 else None

    count, _ = run(
        source=args.source,
        dither=dither,
        dither_bits=args.dither_bits,
        dither_size=dither_size,
        bit_crush=args.bit_crush,
        chroma_shift=args.chroma_shift,
        scan_drift=args.scan_drift,
        glitch_prob=args.glitch_prob,
        scanlines=args.scanlines,
        seed=args.seed,
        mirror=not args.no_mirror,
        show=not args.no_show,
        max_frames=args.frames,
    )
    print(f"rendered {count} frames")


if __name__ == "__main__":
    main()
