# designing-touch

Turn **video and sound into flowing particle visuals**, in real time — a modular,
code-first take on the kind of generative effects you'd build in TouchDesigner, but driven from a terminal or a simple control panel instead of a GUI node graph.

Point your webcam at yourself and you dissolve into a luminous, flowing cloud of particles. Customisable and genuinely fun.

Dance, play music, and it moves with you.

<img width="1600" height="894" alt="image" src="https://github.com/user-attachments/assets/0f4a036a-ab8a-445a-9d43-00d871481a93" />


## Quick start

```bash
git clone https://github.com/NimbleCoAI/designing-touch
cd designing-touch
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[person]"        # engine + person segmentation
python experiments/05-live-webcam/run.py
```

On macOS you can also just **double-click `start.command`** (it sets up the environment on first
run, then opens the window). Grant your terminal **Camera** (and **Microphone**, for sound) access
in System Settings → Privacy & Security.

## The live instrument

One window: the live render plus a collapsible control panel (click the `>` to fold it away).
The panel is grouped into **collapsible sections** — click a header to open it. `MOTION` and
`SIGNAL` start closed, so the panel opens at the size it always did.

- **Templates** — one-click looks: `abstract` (glowing cloud), `portrait`/`textured`
  (recognizable, painted with your real colors), `embers`, `aurora`, and **`sigil`** (sharp
  fractal contour lines — move, then hold still and watch them form). Save your own with
  "Save current look" — it captures everything, including Video bg / Vid mix and Sound
  react / Sens. Hover a saved look for `~` (rename — type, Enter saves) and `x` (delete —
  click twice). Built-ins can't be renamed or deleted, and they never touch your live
  video/sound toggles when you switch to them.
- **Source** — `matte` (what becomes particles: motion / saliency / person / edges / luma),
  `output` resolution (720p → 4K), and **Video bg** — show the raw camera footage behind the
  particles (screen-blended so the glow stays on top), with a `Vid mix` slider for how present
  it is. Recordings capture it too.
- **Look** — color palette + sliders for Trails, Glow, Spark, Flow, Size, Count, and the sigil
  knobs Glide / Pull / Reseed. Every slider has an `i` tooltip.
- **Motion** — **Flock**: Reynolds boids steering (cohesion / alignment / separation) applied
  to the particle cloud, so it organises into shoals and murmurations while still being shaped
  by the camera. Costs ~4 ms at 200k particles — the neighbourhood is approximated on a spatial
  grid, because true all-pairs boids at this particle count is ~4x10^10 pair terms.
- **Signal** — **Glitch**: the circuit-bent chain (chroma bleed, scan drift and sync tears,
  glitch blocks, bit-crush, CRT scan lines) plus Bayer / blue-noise / Floyd-Steinberg /
  Riemersma **dithering** (sRGB gamma-correct — quantised in linear light), applied
  to the finished frame. Costs ~6 ms at 720p, ~14 ms at 1080p. The control panel is drawn
  *after* it, so the panel never glitches into unreadability, and recordings capture what you
  see. Start it from the CLI with `--glitch` (or `--flock`).
- **Audio** — toggle sound reactivity (bass pulses brightness; treble adds spark where Spark > 0)
  and a sensitivity slider.
- **Record** — capture an MP4 of your session.

Quit via the **Quit** button, `q`, or the window's close box.

## The engine (`dtouch/`)

A small package of composable operators — sources, mattes, a particle-flow simulation, GPU glow
renderer, a 2D stable-fluids solver, audio analysis, and a tiny node-graph spine. Rendering is
headless (moderngl) so everything is scriptable and inspectable. See **[AGENTS.md](AGENTS.md)** for
the architecture and how to extend it, and **[docs/autonomy-pattern.md](docs/autonomy-pattern.md)**
for the self-verifying design philosophy.

## Experiments

Standalone studies, each in `experiments/NN-name/` with its own README and `run.py`:

| #  | Experiment        | What it explores                                        |
|----|-------------------|---------------------------------------------------------|
| 01 | displacement      | luminance displacement, instanced boxes, light/depth    |
| 02 | shadows           | depth-from-light shadow mapping                          |
| 03 | audio-reactive    | sound → displacement                                     |
| 04 | fluid             | stable-fluids advection of a particle field             |
| 05 | live-webcam       | the real-time interactive instrument (above)            |
| 06 | flocking          | Reynolds boids — order grown from local rules            |
| 07 | circuit-bent      | stochastic glitch + dithering: emulating bent hardware   |

Experiments 06 and 07 are also **built into the live instrument** as the `MOTION` and `SIGNAL`
panel sections — the standalone versions remain as focused studies of each idea (06 renders
boids in 3D through the instanced renderer; 07 bends the raw camera feed rather than the
particle render).

## Tests

```bash
pytest tests/ -q
```

Verified on Apple Silicon (Metal-backed GL 4.1). The render tests need a GPU/GL context.

## License

[GNU AGPL-3.0](LICENSE). If you run a modified version as a network service, you must offer users
its source. Contributions welcome — see [AGENTS.md](AGENTS.md).
