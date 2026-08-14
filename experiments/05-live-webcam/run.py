#!/usr/bin/env python3
"""Experiment 05 — real-time live preview (interactive).

Default mode 'flow': camera -> subject-agnostic matte (whatever moves/stands out) -> a smoothly
flowing cloud of glowing particles. Works for a dancer, a crowd, a boat — not person-only.
Mode 'grid' is the older luminance-displaced grid.

    python run.py                       # flow, built-in laptop camera, auto matte
    python run.py --matte motion        # key on motion only (great for a dancer)
    python run.py --matte person        # multi-person segmentation
    python run.py --device 1            # a specific camera index
    python run.py --mode grid           # the old displacement-grid effect

    python run.py --flock               # start with the particle cloud flocking
    python run.py --glitch              # start with the circuit-bent signal chain on

    python run.py --mode dithergirl     # boot into Dither Girl (live dithering)
    python run.py --still photo.jpg     # load a still and imply dithergirl

Controls (flow): q quit · n cycle matte · m mirror · [ ] trail length · -/= glow · space freeze

The control panel groups everything into collapsible sections: TEMPLATES, SOURCE, LOOK,
MOTION (flocking) and SIGNAL (glitch + dithering). Click a header to open it.
"""
from __future__ import annotations

import argparse

from dtouch.camera import list_cameras
from dtouch.live import live
from dtouch.modes.dithergirl import DitherGirlMode
from dtouch.modes.particles import ParticlesMode
from dtouch.shell import Host


def parse_wh(s):
    a, b = s.lower().split("x")
    return int(a), int(b)


# Engine flags that only Particles/flow can honour. A value different from the
# default is an explicit request, so it means "boot into flow" (DESIGN.md §3:
# CLI flags mean "boot into") — otherwise the flag would be silently dropped by
# the state.json resume path, which constructs the mode with DEFAULT args.
ENGINE_DEFAULTS = dict(matte="auto", grid="416x234", particles=200000,
                       flock=False, glitch=False)


def engine_flags_given(args):
    """The particles-implying flags the user actually set, as CLI spellings."""
    return ["--" + name for name, default in ENGINE_DEFAULTS.items()
            if getattr(args, name) != default]


def boot_mode_name(mode_arg, still_arg, particles_flags=False):
    """Boot precedence: --mode > --still > engine flags > state.json.

    --still PATH implies dithergirl unless --mode is given (DESIGN.md §3: CLI
    flags mean 'boot into'). Any particles-implying engine flag set to a
    non-default value (--matte/--grid/--particles/--flock/--glitch) boots flow
    WITH those args. With no mode-implying flag at all, returns None — the
    shell resumes the last-used mode from state.json (§3: launch goes straight
    into the last-used mode; first run: Particles)."""
    if mode_arg:
        return mode_arg
    if still_arg:
        return "dithergirl"
    return "flow" if particles_flags else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default=None,
                    choices=["flow", "grid", "dithergirl"])
    ap.add_argument("--still", default=None, metavar="PATH",
                    help="load a still image (implies --mode dithergirl)")
    ap.add_argument("--ui", default="panel", choices=["panel", "keys"],
                    help="panel = render + slider window (default); keys = render only")
    ap.add_argument("--audio", action="store_true", help="start with mic reactivity on")
    ap.add_argument("--device", default="builtin",
                    help="'builtin' (laptop cam), an index, or a name substring")
    ap.add_argument("--matte", default="auto",
                    choices=["auto", "motion", "saliency", "person", "edges", "luma"])
    ap.add_argument("--preset", default=None,
                    help="named look: abstract | portrait | textured | embers | aurora | <your saved>")
    ap.add_argument("--res", default="1920x1080")
    ap.add_argument("--grid", default="416x234")
    ap.add_argument("--particles", type=int, default=200000,
                    help="max particles allocated; the Count slider scales how many render")
    ap.add_argument("--list-cameras", action="store_true")
    ap.add_argument("--flock", action="store_true",
                    help="start with boids steering on the particle cloud (MOTION panel)")
    ap.add_argument("--glitch", action="store_true",
                    help="start with the circuit-bent post-FX chain on (SIGNAL panel)")
    ap.add_argument("--no-mirror", action="store_true")
    args = ap.parse_args()

    if args.list_cameras:
        cams = list_cameras()
        if not cams:
            print("  (AVFoundation enumeration unavailable)")
        for i, name, dtype in cams:
            tag = dtype.replace("AVCaptureDeviceType", "")
            print(f"  [{i}] {name}  ({tag})")
        return

    device = int(args.device) if args.device.isdigit() else args.device
    given = engine_flags_given(args)
    boot = boot_mode_name(args.mode, args.still, particles_flags=bool(given))
    if given and boot != "flow":
        # a higher-precedence flag won; say so instead of dropping the flag
        print("note: %s ignored - booting %s" % (", ".join(given), boot))
    if boot == "grid":
        live(device=device, res=parse_wh(args.res), mirror=not args.no_mirror)
        return
    # thin launcher: shell + mode (DESIGN.md §8 step 6)
    preset = args.preset
    if boot is None:
        mode = None      # shell resumes the last-used mode from state.json (§3)
    elif boot == "dithergirl":
        mode = DitherGirlMode(still=bool(args.still))
    else:
        mode = ParticlesMode(matte=args.matte, grid=parse_wh(args.grid),
                             n=args.particles, flock=args.flock, glitch=args.glitch)
    host = Host(mode, device=device, res=parse_wh(args.res),
                preset=preset, audio=args.audio, still=args.still,
                panel=(args.ui == "panel"), mirror=not args.no_mirror)
    host.run()


if __name__ == "__main__":
    main()
