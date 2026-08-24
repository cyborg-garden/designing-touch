# 06 — Flocking

![flocking](../../docs/06-flocking.png)

**TouchDesigner feature targeted:** POP-based particle simulation with per-point
neighbour forces (the kind you'd wire as a feedback loop of `Particle SOP` /
custom POP VEXpressions), driving an instanced `Copy SOP` render. Here the whole
graph is code: a Reynolds boids solver in NumPy → oriented instanced cubes through
the shared `dtouch` renderer.

## What it is

Classic Reynolds flocking in 3D — **separation, alignment, cohesion** — over a soft
spherical boundary, with a slow global swirl. No leader, no script. Every boid reads
only its neighbours inside a radius, and the coherent shoal (with breakaway
sub-flocks and scouts) is produced by those three local rules alone. Each instance is
elongated and oriented along its velocity, so motion reads as a directed shoal rather
than a drifting cloud.

It is the simplest honest picture of a polycentric system: order that is *grown* from
local interaction, not imposed from a center. Many centers, not one.

## Live instrument (`--live`)

A new base mode for lighteater: instead of a video-driven matte, the picture drives
*itself* — self-organising boids you can push around. Open a real-time window and play:

```bash
python experiments/06-flocking/run.py --live                # ~700 boids, interactive
python experiments/06-flocking/run.py --live --boids 400    # fewer = smoother on a laptop
```

| Control | Effect |
|---------|--------|
| **drag** (left) | a hand in the field — the flock is **attracted** to the cursor |
| **right-drag** | the flock **scatters** from the cursor |
| `c` / `C` | cohesion − / + |
| `a` / `A` | alignment − / + |
| `s` / `S` | separation − / + |
| `w` / `W` | swirl − / + |
| `1` `2` `3` | mood presets: murmuration · scatter · vortex |
| `g` | cycle shape: cube → **star ✦** → bird |
| `space` | freeze · `r` reset · `h` toggle HUD · `q` quit (or just close the window) |

Launch straight into stars (Matariki):

```bash
python experiments/06-flocking/run.py --live --shape star
```

Shapes: `cube` (the original, with a motion-streak), `star` (a 3D 5-pointed star that
twinkles as it tumbles — a flock of stars), `bird` (a crude swept-wing delta that points
where it flies). Set with `--shape` or cycle live with `g`.

The forces fight each other; the interesting looks live on the edges between them. Drag the
flock into a wall, let go, watch it reorganise.

## Render to file

```bash
python experiments/06-flocking/run.py                       # 700 boids, 240 frames -> mp4
python experiments/06-flocking/run.py --boids 1200 --frames 600
python experiments/06-flocking/run.py --cohesion 0.4 --separation 2.4   # looser, more lanes
python experiments/06-flocking/run.py --no-recenter         # let the shoal wander out of frame
```

Writes `out/flock.mp4` plus `out/flock_frame0.png` and `out/flock_mid.png`.

## Knobs

| Flag | Effect |
|------|--------|
| `--boids` | flock size (O(n²) neighbour pass — comfortable into the low thousands) |
| `--neighbor` / `--sep-radius` | how far a boid sees / how close is "too close" |
| `--cohesion` / `--alignment` / `--separation` | the three Reynolds weights — the whole character lives here |
| `--swirl` | a breath of shared global weather over the local rules |
| `--bound` | containment-sphere radius |
| `--recenter` / `--no-recenter` | track the shoal's centroid (framed) vs. let it wander |

Tuning note: raise `--separation` relative to `--cohesion` and the single mass breaks
into lanes and competing sub-flocks; reverse it and they fuse into one ball. The
interesting regime is the edge between the two.
