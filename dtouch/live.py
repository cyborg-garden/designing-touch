"""Real-time live preview — camera -> flowing particle cloud, in ONE OpenCV window.

The cv2 window paints reliably on macOS (unlike Tk launched headless). The control panel is an
in-frame collapsible sidebar (dtouch.overlay_ui) drawn onto the render with mouse hit-testing,
so it's one window, mouse-driven, and self-verifiable. Quitting is via the window's close (red
X) button or double-tapped 'q'. ESC steps the overlay toward HIDDEN and never quits — safe by
design, not ignored (DESIGN.md §6.1: Esc always walks one step toward a clean output).

Detects all-black camera frames (the symptom of iPhone Continuity stealing the built-in camera)
and says so on screen instead of showing a silent blank.
"""
from __future__ import annotations

import os
import time

import cv2
import numpy as np
import imageio.v2 as imageio

from .camera import open_camera
from .matte import make_matte
from .particles import ParticleFlow, PALETTES
from .glow import GlowRenderer
from .audio import LiveMic
from .overlay_ui import OverlayUI, DITHERS
from .circuit_bent import CircuitBent
from .commands import CommandRegistry
from .hud import Hud, OverlayState, cycle_overlay, esc_overlay
from . import presets as _presets

MATTES = ["auto", "motion", "saliency", "person", "edges", "luma"]


def _overlay_key(key, state, toasts):
    """TAB/Esc overlay-state stepping (DESIGN.md §6.1). Returns (state, handled).

    TAB cycles HIDDEN -> HUD -> PANEL -> HIDDEN; Esc steps one toward HIDDEN and
    in HIDDEN does nothing — Esc never quits. Entering HIDDEN emits one final
    toast, then the output is provably clean once it fades.
    """
    if key == 9:       # TAB
        new = cycle_overlay(state)
    elif key == 27:    # Esc
        new = esc_overlay(state)
    else:
        return state, False
    if new is OverlayState.HIDDEN and state is not OverlayState.HIDDEN:
        toasts.hint("overlay hidden - TAB to show")
    return new, True


def composite_video_bg(particles_rgb, frame_bgr, mix):
    """Screen-blend the raw camera footage under the additive particle render.

    Screen (1 - (1-p)(1-bg*mix)) can only brighten, so the glow stays luminous on top
    and nothing blows out; mix=0 returns the particles untouched. All-uint8 cv2 ops to
    stay cheap at full render resolution.
    """
    if mix <= 0.0:
        return particles_rgb
    h, w = particles_rgb.shape[:2]
    bg = cv2.cvtColor(cv2.resize(frame_bgr, (w, h)), cv2.COLOR_BGR2RGB)
    bg = cv2.convertScaleAbs(bg, alpha=float(mix))
    inv = cv2.multiply(cv2.bitwise_not(particles_rgb), cv2.bitwise_not(bg), scale=1.0 / 255.0)
    return cv2.bitwise_not(inv)


def _apply_fx(ui, raw):
    """Restore MOTION + SIGNAL state from a saved look, if it recorded any.

    Every key is optional: the built-in presets predate these effects and saved looks from
    before this change have none of them. A missing key must leave the live toggle alone
    rather than resetting it, which is the same rule video_bg/audio already follow — otherwise
    hopping between built-in templates would silently switch your glitch off.
    """
    from .overlay_ui import DITHERS
    if "flock" in raw: ui.flock = bool(raw["flock"])
    if "cohere" in raw: ui.cohere = float(raw["cohere"])
    if "align" in raw: ui.align = float(raw["align"])
    if "separate" in raw: ui.separate = float(raw["separate"])
    if "glitch" in raw: ui.glitch = bool(raw["glitch"])
    if "chroma" in raw: ui.chroma = float(raw["chroma"])
    if "drift" in raw: ui.drift = float(raw["drift"])
    if "crush" in raw: ui.crush = float(raw["crush"])
    if "scanlines" in raw: ui.scanlines = bool(raw["scanlines"])
    if raw.get("dither") in DITHERS: ui.dither_idx = DITHERS.index(raw["dither"])


def _open_capture(device):
    if isinstance(device, int):
        return cv2.VideoCapture(device, cv2.CAP_AVFOUNDATION)
    return cv2.VideoCapture(device)


def live_flow(device="builtin", matte="auto", res=(1920, 1080), grid=(416, 234),
              n=200000, mirror=True, seed=1, preset="abstract", audio=False,
              panel=True, show=True, max_frames=None, video_bg=False, video_mix=0.5,
             flock=False, glitch=False):
    rw, rh = res
    gw, gh = grid
    mw, mh = 416, 234

    cap, cam_name = open_camera(device)
    matte_kind = matte
    mat = make_matte(matte_kind)
    pf = ParticleFlow(n=n, gw=gw, gh=gh, seed=seed)
    glow = GlowRenderer(rw, rh, n, fade=0.90, exposure=1.4)
    all_presets = _presets.load()
    preset_names = list(all_presets.keys())

    def apply_preset(name):
        nonlocal matte_kind, mat
        if name not in all_presets:
            return {}
        # merge over defaults so switching presets fully resets physics params
        # (otherwise e.g. sigil's low damping would leak into the next look)
        d = dict(palette="ice", spark=0.35, curl_amp=0.5, reseed_frac=0.06, base_size=0.011,
                 damp=0.90, pull_falloff=22.0, attract_speed=4.5, fade=0.90, exposure=1.4)
        cfg = {**d, **all_presets[name]}
        if cfg.get("matte") and cfg["matte"] != matte_kind:
            matte_kind = cfg["matte"]; mat = make_matte(cfg["matte"])
        pf.palette = cfg["palette"]; pf.spark = cfg["spark"]; pf.curl_amp = cfg["curl_amp"]
        pf.reseed_frac = cfg["reseed_frac"]; pf.base_size = cfg["base_size"]; pf.damp = cfg["damp"]
        pf.pull_falloff = cfg["pull_falloff"]; pf.attract_speed = cfg["attract_speed"]
        glow.fade = cfg["fade"]; glow.exposure = cfg["exposure"]
        # video/audio state is applied only when the preset recorded it (saved looks do;
        # built-ins don't), so template-hopping never resets the live toggles.
        return all_presets[name]

    raw0 = apply_preset(preset) if preset in all_presets else {}
    video_bg = bool(raw0.get("video_bg", video_bg))
    video_mix = float(raw0.get("video_mix", video_mix))
    audio = bool(raw0.get("audio", audio))

    ui = None
    win = "dtouch - flow"
    if show:
        # AUTOSIZE: the window is fixed at the render resolution so the OS can't maximize/scale
        # it — that scaling was tanking fps (display upscaling) and breaking click mapping.
        # To go bigger, switch the 'output' resolution; the window resizes to match natively.
        cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
        if panel:
            ui = OverlayUI(rw, rh, preset_names, PALETTES, MATTES,
                           preset=preset, matte=matte_kind, palette=pf.palette)
            ui.sync_from(pf, glow, matte_kind)
            ui.mirror = mirror
            ui.audio = audio
            # CLI opt-ins: turn the effect on AND open its section, so --glitch doesn't leave
            # someone hunting for the controls behind a collapsed header.
            ui.flock = ui.flock or flock
            ui.glitch = ui.glitch or glitch
            if flock: ui.sections["MOTION"] = True
            if glitch: ui.sections["SIGNAL"] = True
            ui.video_bg = video_bg
            ui.video_mix = video_mix
            if "sens" in raw0:
                ui.sens = float(raw0["sens"])
            _apply_fx(ui, raw0)      # the look loaded at startup, same as a live switch
            ui.user_presets = _presets.user_names()
            cv2.setMouseCallback(win, ui.on_mouse)

    # Key routing goes through the command registry (DESIGN.md principle 7).
    # Today's behavior only: 'q' quits (case-folded, so 'Q' no longer silently
    # demands releasing Shift). The perform-layer commands land in step 3.
    reg = CommandRegistry()
    _want_quit = [False]
    reg.add("app.quit", "Quit", "q", lambda: _want_quit.__setitem__(0, True))

    # Perform-surface renderer + overlay state machine. Boot state: HUD — the
    # instrument boots into perform, already playing (DESIGN.md §6.1).
    hud = Hud()
    overlay = OverlayState.HUD

    mic = None
    if audio:
        mic = LiveMic(); mic.start()
    writer = None; rec_path = None
    os.makedirs("out", exist_ok=True)

    cb = None            # circuit-bent post-FX, built on first use
    t0 = time.time(); fps = 0.0; count = 0
    black_streak = 0
    out = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                if max_frames is None:
                    continue
                break
            black_streak = black_streak + 1 if float(frame.mean()) < 3.0 else 0

            if ui is not None:
                if ui.pending_preset:
                    raw = apply_preset(ui.pending_preset)
                    ui.sync_from(pf, glow, matte_kind)
                    if "video_bg" in raw: ui.video_bg = bool(raw["video_bg"])
                    if "video_mix" in raw: ui.video_mix = float(raw["video_mix"])
                    if "audio" in raw: ui.audio = bool(raw["audio"])
                    if "sens" in raw: ui.sens = float(raw["sens"])
                    _apply_fx(ui, raw)
                    ui.pending_preset = None
                if ui.pending_save:
                    name = "mine_%s" % time.strftime("%H%M%S")
                    _presets.save(name, dict(matte=matte_kind, palette=pf.palette,
                                  fade=glow.fade, exposure=ui.exposure, spark=ui.spark,
                                  curl_amp=ui.curl, reseed_frac=ui.reseed, base_size=ui.dot,
                                  damp=ui.damp, pull_falloff=ui.pull,
                                  attract_speed=pf.attract_speed,
                                  video_bg=ui.video_bg, video_mix=ui.video_mix,
                                  audio=ui.audio, sens=ui.sens,
                                  # MOTION + SIGNAL travel with the look; without these a
                                  # saved glitch preset would come back clean.
                                  flock=ui.flock, cohere=ui.cohere, align=ui.align,
                                  separate=ui.separate,
                                  glitch=ui.glitch, chroma=ui.chroma, drift=ui.drift,
                                  crush=ui.crush, dither=ui.dither_name,
                                  scanlines=ui.scanlines))
                    all_presets = _presets.load()
                    preset_names = list(all_presets.keys())
                    ui.presets = preset_names
                    ui.user_presets = _presets.user_names()
                    if name in preset_names:
                        ui.preset_idx = preset_names.index(name)
                    print("saved preset", name)
                    ui.pending_save = False
                if ui.pending_delete:
                    sel = ui.preset_name if ui.preset_idx < len(ui.presets) else None
                    if _presets.delete(ui.pending_delete):
                        all_presets = _presets.load()
                        preset_names = list(all_presets.keys())
                        ui.presets = preset_names
                        ui.user_presets = _presets.user_names()
                        ui.preset_idx = (preset_names.index(sel)
                                         if sel in preset_names else 0)
                        print("deleted preset", ui.pending_delete)
                    ui.pending_delete = None
                if ui.pending_rename:
                    old, new = ui.pending_rename
                    sel = ui.preset_name if ui.preset_idx < len(ui.presets) else None
                    if _presets.rename(old, new):
                        all_presets = _presets.load()
                        preset_names = list(all_presets.keys())
                        ui.presets = preset_names
                        ui.user_presets = _presets.user_names()
                        target = new if sel == old else sel
                        ui.preset_idx = (preset_names.index(target)
                                         if target in preset_names else 0)
                        print("renamed preset", old, "->", new)
                    else:
                        print("rename refused (name taken or invalid):", old, "->", new)
                    ui.pending_rename = None
                if ui.matte_name != matte_kind:
                    matte_kind = ui.matte_name; mat = make_matte(matte_kind)
                pf.palette = ui.palette_name
                pf.curl_amp = ui.curl
                pf.base_size = ui.dot
                pf.damp = ui.damp
                pf.pull_falloff = ui.pull
                pf.reseed_frac = ui.reseed
                # flocking: gains go to 0 when the toggle is off, which short-circuits the
                # solver entirely — the off state costs nothing, not even a pass over the array.
                pf.flock_cohesion = ui.cohere if ui.flock else 0.0
                pf.flock_alignment = ui.align if ui.flock else 0.0
                pf.flock_separation = ui.separate if ui.flock else 0.0
                glow.fade = ui.fade
                mirror = ui.mirror
                nw, nh = ui.res_wh
                if (nw, nh) != (rw, rh):
                    rw, rh = nw, nh
                    glow.resize(rw, rh)
                    ui.w, ui.h = rw, rh   # AUTOSIZE window refits on next imshow
                if ui.audio and mic is None:
                    mic = LiveMic(); mic.start()
                elif not ui.audio and mic is not None:
                    mic.stop(); mic = None
                if ui.record and writer is None:
                    rec_path = os.path.join("out", "rec_%s.mp4" % time.strftime("%Y%m%d_%H%M%S"))
                    writer = imageio.get_writer(rec_path, fps=24, macro_block_size=8)
                elif not ui.record and writer is not None:
                    writer.close(); print("saved", rec_path); writer = None

            if mirror:
                frame = cv2.flip(frame, 1)
            small = cv2.resize(frame, (mw, mh))
            m = cv2.resize(mat.compute(small), (gw, gh))
            gray = cv2.resize(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0,
                              (gw, gh))
            color = cv2.cvtColor(cv2.resize(small, (gw, gh)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

            # base look from the panel; audio modulates glow/spark on top for this frame.
            # Spark is modulated MULTIPLICATIVELY off the slider, so when Spark is 0 (e.g. sigil)
            # sound adds no spark/blur — it only pulses brightness with the bass.
            glow.exposure = ui.exposure if ui is not None else glow.exposure
            pf.spark = ui.spark if ui is not None else pf.spark
            if mic is not None and mic.available:
                lv = mic.levels(); sens = ui.sens if ui is not None else 1.0
                glow.exposure = glow.exposure * (1.0 + 1.6 * sens * lv["bass"])
                pf.spark = pf.spark * (1.0 + 2.0 * sens * lv["treble"])

            # video palette = textured/recognizable: weight particle density by the footage's
            # luminance inside the subject so the face's tones resolve as a pointillist portrait.
            # weight density by luminance for portraits (video palette OR person matte) so the
            # face's tones resolve as particle density — on ANY color, not just video.
            if pf.palette == "video" or matte_kind == "person":
                density = m * np.clip((gray - 0.06) * 1.5, 0.05, 1.0)
            else:
                density = None
            pf.update(m, gray, color, density=density)
            buf = pf.render_data()
            if ui is not None and ui.count < 0.999:
                buf = buf[: int(ui.count * n) * 7]   # live dot-count control
            out = glow.render(buf)
            if ui is not None:
                video_bg, video_mix = ui.video_bg, ui.video_mix
            if video_bg:
                # frame is already mirrored here, so the footage lines up with the particles
                out = composite_video_bg(out, frame, video_mix)
            # SIGNAL post-FX. Applied here on purpose: after the particles and the video
            # composite (so it bends the whole picture), before the recorder (so captures match
            # what you see) and before ui.draw (so the control panel never gets glitched into
            # unreadability). CircuitBent is documented for BGR; `out` is RGB, which only swaps
            # which channel drifts left vs right — the offsets are independent symmetric draws,
            # so the look is identical. Constructed lazily so a session that never enables it
            # pays nothing.
            if ui is not None and ui.glitch:
                if cb is None:
                    cb = CircuitBent(seed=seed)
                cb.chroma_shift = ui.chroma
                cb.scan_drift = ui.drift
                cb.bit_crush = int(ui.crush)
                cb.scanlines = ui.scanlines
                cb.dither_mode = None if ui.dither_name == "off" else ui.dither_name
                out = cb.process(out)
            if writer is not None:
                writer.append_data(out)
            bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

            count += 1
            if count % 10 == 0:
                now = time.time(); fps = 10.0 / (now - t0); t0 = now

            if show:
                # HUD/panel draw AFTER the recorder write above — recordings never
                # contain HUD or panel (the existing ordering invariant, kept).
                status = f"matte={matte_kind}  color={pf.palette}  cam={cam_name[:16]}"
                dbg = f"{fps:4.1f}fps  {1000.0 / fps if fps > 0 else 0.0:5.1f}ms  {rw}x{rh}"
                if ui is not None and overlay is OverlayState.PANEL:
                    ui.draw(bgr, {"status": "", "black": black_streak > 15})
                elif ui is not None:
                    ui._hot = []   # panel hidden: stale hit-rects must not eat clicks
                hud.draw(bgr, overlay, status=status, debug_status=dbg,
                         recording=(writer is not None))
                cv2.imshow(win, bgr)
                key = cv2.waitKey(1) & 0xFF   # pump GUI + mouse
                # rename-typing consumes every key; Esc only cancels the rename —
                # while renaming, no global keys fire (DESIGN.md §6.2)
                consumed = ui.on_key(key) if (ui is not None and key != 255) else False
                if not consumed and key != 255:
                    overlay, handled = _overlay_key(key, overlay, hud.toasts)
                    if not handled:
                        reg.dispatch(key)
                if _want_quit[0] or (ui is not None and ui.quit):
                    break
                # quit only when the window is actually destroyed (red X) -> property is -1.
                # A minimized window reports 0, so this does NOT quit on minimize.
                if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 0:
                    break
            if max_frames is not None and count >= max_frames:
                break
    finally:
        if writer is not None:
            writer.close(); print("saved", rec_path)
        if mic is not None:
            mic.stop()
        cap.release()
        glow.release()
        if show:
            cv2.destroyAllWindows(); cv2.waitKey(1)
    return count, out


def live(device="builtin", res=(1024, 576), grid=(130, 73), depth=1.3, mirror=True,
         show=True, max_frames=None):
    """Legacy luminance-displaced grid effect (kept as a simple fallback)."""
    from .field import make_grid, displace_z, random_scale, random_euler, pack_instances
    from .render import Renderer
    gx, gy = grid; rw, rh = res; n = gx * gy
    cap, _ = open_camera(device)
    g = make_grid(gx, gy); s = random_scale(n, 0); e = random_euler(n, 1)
    r = Renderer(rw, rh, n, base_size=0.8 / max(gx, gy), depth_scale=depth)
    if show:
        cv2.namedWindow("dtouch - grid", cv2.WINDOW_NORMAL)
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
                cv2.imshow("dtouch - grid", cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
                if cv2.getWindowProperty("dtouch - grid", cv2.WND_PROP_VISIBLE) < 1:
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
