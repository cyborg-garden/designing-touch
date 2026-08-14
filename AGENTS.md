# AGENTS.md — working on designing-touch

Guide for AI agents (and humans) developing this repo. Read this before changing code.

## What this is

A real-time, modular engine for turning **video and/or sound into flowing particle visuals** —
recreating TouchDesigner-style effects as text-first, CLI/GUI-drivable Python instead of a GUI
node graph. The headline experience is **experiment 05** (`experiments/05-live-webcam/`), now a
thin launcher over a **shell that hosts modes** (`dtouch/shell.py` + `dtouch/modes/`): a live
webcam instrument with an in-frame control panel, saved looks ("templates"), a perform key
layer, and two modes — Particles and Dither Girl.

**[DESIGN.md](DESIGN.md) is the design authority for anything UI/UX.** Every PR that touches
the instrument cites the principle it serves; read §2 (the mode-shell contract) before adding
a control, a key, or a mode.

## Architecture

The engine is the `dtouch/` package (installable via `pip install -e .`). Each module is one
small, composable operator — the "node graph as code":

| Module             | Role                                                                |
|--------------------|---------------------------------------------------------------------|
| `sources.py`       | video/webcam/image/synthetic → luminance grids (TOP)                |
| `matte.py`         | subject-agnostic interest field: motion / saliency / edges / luma / person |
| `field.py`         | grid, luminance Z-displacement, seeded randoms, packing (TOP→POP)   |
| `particles.py`     | `ParticleFlow` — particles fill a matte, advect by optical flow + curl |
| `render.py`        | `Renderer` — headless instanced-cube GPU renderer (lights + depth)  |
| `shadow.py`        | `ShadowRenderer` — adds a depth-from-light shadow map               |
| `glow.py`          | `GlowRenderer` — additive soft-particle renderer with trail feedback |
| `audio.py`         | `analyze_block`, `SyntheticAudio`/`WavAudio`, `LiveMic` (CHOP)       |
| `fluid.py`         | `Fluid2D` — 2D stable-fluids solver                                 |
| `camera.py`        | select the built-in laptop camera by device type (macOS-safe)       |
| `flock.py`         | Reynolds boids steering on the particle cloud (MOTION section)      |
| `dither.py`        | Bayer / blue-noise / Floyd-Steinberg / Riemersma, gamma-correct     |
| `circuit_bent.py`  | `CircuitBent` — the SIGNAL post-FX rack (glitch chain + dither)     |
| `pipeline.py`      | `Op`/`Graph` — wire operators by threading a context dict           |

The instrument on top of those operators (DESIGN.md §2 — the shell owns the show,
modes rack into it):

| Module             | Role                                                                |
|--------------------|---------------------------------------------------------------------|
| `shell.py`         | `Host` — window, capture, key/mouse routing, recorder, SIGNAL rack, overlay state machine, preset CRUD, mode lifecycle |
| `modes/__init__.py`| the `Mode` protocol + `REGISTRY` (one line adds a mode)             |
| `modes/particles.py` | Particles mode: engines, panel sections, `MATTES`, per-frame image |
| `modes/dithergirl.py` | Dither Girl mode: dithering as the primary image (§4.2)          |
| `commands.py`      | `Command`/`CommandRegistry` — every action is named and bindable    |
| `hud.py`           | overlay states, toasts, OSD, status line, `?` key map, u-unit text  |
| `menu.py`          | home menu + boot cards (a shell overlay state, not a Mode)          |
| `panelspec.py`     | controls declared as data (`Slider`/`Toggle`/`Cycle`/`Section`…)    |
| `imgui.py`         | the cv2 widget toolkit the panel and menu draw with                 |
| `overlay_ui.py`    | `OverlayUI` — a generic walker over a panel spec (the sidebar)      |
| `presets.py`       | per-mode looks, banks, setlists (v2 JSON) + `state.json` autosave   |
| `live.py`          | thin shim: `live_flow` (shell + ParticlesMode) and `live` (legacy grid) |

Rendering is **headless moderngl** (`create_standalone_context`) → renders to an offscreen FBO,
read back as a NumPy array, then blitted to an OpenCV window. This is why it's all self-verifiable.

## The core convention: headless self-verification

See `docs/autonomy-pattern.md`. Every effect must render to a file you can inspect, and every
input has a synthetic/file fallback so it runs with no camera/mic/display. When changing visuals,
**render a frame and look at it** (read the PNG) — don't claim it works from code alone. TDD the
deterministic NumPy cores (`field`, `audio`, `fluid`, `particles`, `matte`); smoke-test rendering.

## Run & test

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip   # stock macOS pip 21.2 predates PEP 660 (no -e)
pip install -e .            # engine
pip install -e ".[person]"  # + mediapipe for the person matte
pytest tests/ -q            # the full suite, a few seconds, no camera needed
python experiments/05-live-webcam/run.py     # the live instrument (or double-click start.command)
python experiments/05-live-webcam/run.py --mode dithergirl --still photo.jpg
```

**opencv-python is pinned `<5`** in `pyproject.toml`. The panel is drawn entirely with cv2
primitives, and cv2 5 rasterises Hershey text differently — enough to move every panel pixel.
`tests/test_goldens.py` skips itself (rather than failing, or inviting a `GOLDEN_REGEN` that
would break everyone else) on any other cv2 major.

## How to extend

- **New color palette:** add its name to `PALETTES` in `particles.py` and a branch in
  `ParticleFlow._colorize`. It auto-appears in the panel's color cycler.
- **New matte (subject source):** add a class with `.compute(frame_bgr) -> float (H,W) [0,1]` in
  `matte.py`, register it in `make_matte` and the `MATTES` list in `modes/particles.py`
  (re-exported from `live.py` for existing callers).
- **New template (look):** add an entry to the mode's `BUILTIN` — `presets.BUILTIN` for
  Particles, `DitherGirlMode.BUILTIN` for Dither Girl. Users save their own to `presets.json`
  (gitignored) via the panel or the `s` key.
- **New control:** declare it once in the mode's `panel_spec()` as a `Slider`/`Toggle`/`Cycle`
  (`dtouch/panelspec.py`) — the walker draws it, the `save`/`apply` flags serialize it, and the
  `status` flag puts it on the HUD line. Then read the UI attr in the mode's `step()` and give
  it a default on `OverlayUI`. Do **not** hand-write preset capture/apply code: the spec is the
  schema authority (DESIGN.md §2.1), and hand-written per-mode preset code is how `KEYS` drifted
  the first time. Changing anything the panel draws means regenerating `tests/goldens/`
  deliberately — read that test's docstring first.
- **New key:** register a `Command` (name, label, key, run) — mode-local ones in the mode's
  `commands()`, global ones in `shell.py`. The `?` overlay is generated from the registry, so a
  registered key documents itself.
- **New mode:** a class satisfying the `Mode` protocol in `modes/`, plus one line in `REGISTRY`.
  That is the whole wiring: menu card, `p`/`d`-style key, preset namespace, panel.
- **New experiment:** `experiments/NN-name/` with a `run.py` that composes `dtouch` operators and
  writes `out/*.mp4` + a `_frame0.png`; commit a sample frame under `docs/`.

## Gotchas (hard-won)

- **macOS camera:** an iPhone Continuity Camera forces the built-in camera to return **all-black**
  frames. `camera.py` selects the built-in by device type; the live app detects black frames and
  says so. The real fix is disabling Continuity Camera on the iPhone.
- **cv2 window is `WINDOW_AUTOSIZE`** (fixed at render res). Maximizing a smaller window upscales
  every frame (kills fps) and breaks mouse-coordinate mapping (controls go unresponsive). "Bigger"
  = switch the `output` resolution, not OS-maximize.
- **Overlay panel is ASCII-only.** cv2's Hershey font renders `•/■/≡/—` as `???`; use ASCII or
  draw shapes.
- **Tkinter doesn't paint** when launched headless on macOS — that's why the panel is drawn on the
  cv2 frame, not a Tk window. (`gui.py` was a drifted Tk duplicate of the instrument and is
  **deleted**; the overlay panel is the one UI — DESIGN.md §4.3.)
- **Quit** is the panel's Quit button, the window's close box (detected via
  `WND_PROP_VISIBLE < 0`, which does NOT fire on minimize), or **`q` twice within 2 s** — a
  single `q` only toasts `q again to quit`, so a mis-hit mid-set can't end the show. `Esc` never
  quits: it steps one state toward HIDDEN (DESIGN.md §6.1).
- **Nothing but the output is drawn in HIDDEN**, and the recorder writes post-FX, pre-overlay —
  recordings never contain the panel, HUD or menu (the mode-switch boot card is the one
  deliberate recorded UI frame). Both are contracts, not habits (DESIGN.md principle 2); a
  change that leaks UI into either is a release blocker.
- **`presets.json`, `state.json`, their `.bak.json` backups, and `dtouch/assets/*.tflite`** are
  gitignored (user data / downloaded model).
