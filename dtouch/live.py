"""Live entry points — live_flow is now a thin shim over the shell + ParticlesMode.

The instrument itself lives in dtouch.shell (the Host: window, capture, key/mouse
routing, recorder, SIGNAL rack, overlay/HUD, preset CRUD, mode lifecycle) and
dtouch.modes.particles (the engines + per-frame image) — DESIGN.md §2 / §8 step 6.
live_flow keeps its exact signature and (count, last_frame) return so existing
callers and tests keep working.

The names tests and callers import from here (MATTES, PerformState,
_perform_key, _wire_perform_keys, composite_video_bg) are re-exported from
their new homes.
"""
from __future__ import annotations

import cv2

from .camera import open_camera
from .modes.particles import MATTES, ParticlesMode, composite_video_bg  # noqa: F401
from .shell import (Host, PerformState, _overlay_key,  # noqa: F401
                    _perform_key, _register_quit, _wire_perform_keys)


def live_flow(device="builtin", matte="auto", res=(1920, 1080), grid=(416, 234),
              n=200000, mirror=True, seed=1, preset="abstract", audio=False,
              panel=True, show=True, max_frames=None, video_bg=False, video_mix=0.5,
              flock=False, glitch=False, source=None,
              presets_path="presets.json", state_path="state.json"):
    """The particle instrument — shell + ParticlesMode (DESIGN.md §8 step 6)."""
    mode = ParticlesMode(matte=matte, grid=grid, n=n, seed=seed,
                         video_bg=video_bg, video_mix=video_mix,
                         flock=flock, glitch=glitch)
    host = Host(mode, source=source, device=device, res=res, mirror=mirror,
                seed=seed, preset=preset, audio=audio, panel=panel, show=show,
                max_frames=max_frames, presets_path=presets_path,
                state_path=state_path)
    return host.run()


def live(device="builtin", res=(1024, 576), grid=(130, 73), depth=1.3, mirror=True,
         show=True, max_frames=None):
    """Legacy luminance-displaced grid effect (kept as a simple fallback)."""
    import numpy as np
    from .field import make_grid, displace_z, random_scale, random_euler, pack_instances
    from .render import Renderer
    gx, gy = grid; rw, rh = res; n = gx * gy
    cap, _ = open_camera(device)
    g = make_grid(gx, gy); s = random_scale(n, 0); e = random_euler(n, 1)
    r = Renderer(rw, rh, n, base_size=0.8 / max(gx, gy), depth_scale=depth)
    if show:
        cv2.namedWindow("lighteater - grid", cv2.WINDOW_NORMAL)
    count = 0; out = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                if max_frames is None: continue
                break
            if mirror: frame = cv2.flip(frame, 1)
            luma = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (gx, gy)).astype(np.float32) / 255.0
            out = r.render(pack_instances(displace_z(g, luma, depth), s, e))
            if show:
                cv2.imshow("lighteater - grid", cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
                if cv2.getWindowProperty("lighteater - grid", cv2.WND_PROP_VISIBLE) < 1:
                    break
                cv2.waitKey(1)
            count += 1
            if max_frames is not None and count >= max_frames:
                break
    finally:
        cap.release(); r.release()
        if show:
            cv2.destroyAllWindows(); cv2.waitKey(1)
    return count, out
