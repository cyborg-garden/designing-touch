# Why lighteater looks like electric tendrils and the reference looks alive

Diagnosis, 2026-08-26. Evidence: the C++ reference observed running (clean-room,
behaviour only), lighteater rendered headless at 2.0M agents, the browser port
observed live, and a full read of the sim path in both.

## The short version

The instrument is not badly tuned. It is **physically prevented from forming
structure**, by five independent mechanisms, each of which alone would flatten
it. Every "look" therefore converges on the same object: a uniform mesh of
identical-width saturated filaments over an even speckle. Changing look, point
or palette changes the colour and the mesh scale. It cannot change the
character, because the character is set by the bugs, not by the parameters.

## The five suppressors (all measured, all in the shipped defaults)

### 1. The camera is a 1% signal
`update.frag:49` — `return t + u_food * texelFetch(u_gray, c, 0).r;`

`t` is raw trail: mean ~33, p95 ~120 absolute units. `u_gray` is 0..1 and
`u_food` defaults to 0.35. So the video's luminance perturbs the sensed field by
**0.35 against 33** — about 1%. `u_satcap` was made scale-aware
(`physarum_gl.py:352` multiplies by `_last_norm`); `u_food` never was.

Consequence: the mold chases light only where no trail exists yet. The moment
structure forms anywhere, the camera stops steering it. This is the root cause
of "it looks similar no matter what I do in front of it", and it also kills
evolve's phantom food blob (0.16 units, 0.5% of signal) and most of react's
motion-carving (~2% typical).

### 2. There is no empty space
`GL_N / (1280x736) = 2_000_000 / 942_080 = 2.12 agents per cell per frame`
(`modes/physarum.py:197-198`). Equilibrium mean trail is
`2.12 * decay/(1-decay) = 33 units everywhere`, including "background".
Jones reticulation lives at 0.05-0.3 agents/cell. Measured vein/floor contrast
lands near 2-4x where a network normally reaches 20-50x.

Veins need dark to be veins. There is none; the floor is a gas of
freshly-reseeded, jitter-randomised agents.

### 3. `weave` blinds the agents it was meant to help
`weave` was added to fight thoroughfare-dominance. Three of its five levers
destroy network formation, and every shipped look sits at 0.45-0.70, deep in
the destructive band (`modes/physarum.py:606-612`).

- **satcap.** At weave 0.6, `satcap ~= 26.5`. Sensed trail: t=33 -> 18.9,
  t=60 -> 23.7, t=120 -> 26.2, t=200 -> 26.5. Everything above ~30 units
  compresses into a 3-unit band; gradient slope inside structure falls to ~5%
  of nominal. **Agents inside the network are steering on noise.**
- **jitter.** +-0.368 rad uniform, sigma 0.212 rad/frame. Heading decorrelates
  in 22 frames = 0.37 s. Inside a converged vein the centre sensor wins, so
  `dir = 0` and jitter is the *only* thing turning the agent. Persistence length
  caps at ~5% of frame width, for every point.
- **reseed.** `0.004 + 0.022*w^2` -> the entire population recycles about once
  per second, to a random position with a **random heading**. Persistence needs
  22-50 frames; agents live 63. The picture's statistics are pinned to the
  reseed distribution, not to any self-organised attractor. This is churn
  without selection — the exact inverse of the population dynamics that produce
  reticulation.

### 4. Two of six points cannot steer at all
`sense`/`step` are grid pixels, scaled by `_px_scale = 1280/576 = 2.222`.
Accumulated trail smoothing at diffuse r=1 over a 16.7-frame memory is
**sigma = 3.33 px**. Left/right sensor separation is `2*sense*sin(spread)`:

| point   | sense px | L/R sep px | step px | verdict |
|---------|---------:|-----------:|--------:|---------|
| veins   | 20.00 | 15.21 | 3.11 | ok |
| cells   | 31.11 | 55.45 | 1.78 | ok |
| fingers |  5.56 |  **2.21** | **7.56** | blind (sep < sigma) AND step > sense |
| haze    |  2.67 |  5.32 | 0.78 | marginal (sense ~= sigma) |
| web     | 75.56 | 72.45 | 4.00 | ok |
| storm   | 22.22 | 44.43 | 7.11 | ok |

`fingers` reads the same smoothed value at both side sensors, so `fl == fr`
resolves to a constant one-way turn: a biased random walk, not Jones steering.
**`fingers` is the default body point.** In the browser its separation is
0.99 px — sub-texel, with `floor()`+`texelFetch`, so the bias is exact.

Sensing is `texelFetch` at NEAREST, which quantises away sub-pixel gradients and
independently degrades every short-sense point.

### 5. The display erases what survives
- `deposit` is a **mathematical no-op**. `tonemap.frag:21-22` normalises trail by
  p95(trail) and grain by 4*mean(laid); both are linear in deposit, so it divides
  out exactly. The 8x spread across the point table produces zero visual
  difference, and the whole `deposit` column of `REGIMES` is inert.
- p95 auto-gain renormalises every look to the same top end, removing all global
  density and contrast difference between them.
- `lum = 1 - exp(-3.5x)` puts the median pixel at 0.65 and clips 5% to white by
  construction. Morphology lives in the mid-tones; the curve spends its range
  above them.
- `grain=0.5` adds sigma=0.086 x-units of identical full-frame speckle to every
  look. That is a large part of the literal "electric" reading.

## What the reference has that we do not

Observed behaviour only. Its own caption credits mxsage's "36 points".

The decisive difference is **not** tuning and not vein width. It is that several
*qualitatively different morphologies coexist in one frame*: radial colony discs
with concentric banding and dark depletion halos, immediately beside fine
reticulated mesh, with hard fronts between them, over a very wide vein-width
range.

lighteater has exactly **two** parameter points (field/body) blended **smoothly**
by the matte — i.e. one averaged regime across the whole canvas. With no camera
the matte is near-uniform, so it is literally one regime. PR #26/#28 added
regimes on the *time* axis (cruise/surge/shatter/knit/flood/drift, 15-45 s dwell)
which raised change-over-time above the reference's, but left morphology uniform,
because a global multiplier moves the whole frame together.

The missing axis is **spatial**. Many parameter sets in a mosaic with sharp
boundaries; agents crossing a boundary abruptly change behaviour. Fronts, halos
and mixed morphology are what that produces.

Secondary: our trail is a single R32F scalar (`physarum_gl.py:199-203`; every
shader says "only .r is read"), so colour is a LUT over one quantity and can
never separate structures. Multi-species with signed cross-terms — negative
off-diagonals — is the only mechanism that yields **cell walls**, and it is
structurally impossible in a one-channel attract-only model. The agent texture
already carries an unused fourth float (`update.frag:55,112` reads and writes
`a.w` untouched) — a free per-agent species/lineage slot.

## Order of work

Phase 1 — restore the physics (shared shaders; local and web both):
1. Scale `u_food` by `_last_norm` as `u_satcap` already is. One line.
2. Cut density ~4x (2.12 -> ~0.5 agents/cell).
3. Reframe satcap so it caps only the fat canals, not the whole field.
4. Decouple weave from jitter and reseed; cap both hard.
5. Bilinear sensing.
6. Re-spread POINTS so no point is steering-blind; add a test asserting
   `sense*scale >= sigma_blur`, `2*sense*scale*sin(spread) >= 1.5*sigma_blur`
   and `step < sense`, FOR BOTH ENGINES. (The first draft of this line said
   3*sigma. Nothing in the shipped table clears that — `veins` misses it on
   the GL grid — so the criterion is 1.5, and the number here was wrong rather
   than the points. The engine-pair matters more than the constant: checking
   only the GL scale let `fingers` ship at 2.36 px of separation on the CPU
   fallback, worse than the 2.21 px this document calls blind.)
7. Tonemap: EMA the norm; cut grain. (A black floor and an output gamma were
   planned here and NOT built — cutting density fixed the tonality they were
   meant to rescue, so the curve was left alone. Grain went 0.5 -> 0.2 on both
   engines, not to zero: at the new density it reads as spore dust rather than
   full-frame static.)

Phase 2 — aliveness:
8. Multi-species: RGB trail, species assigned once at spawn in `a.w`, signed 3x3
   interaction matrix. `weave` becomes off-diagonal strength.
9. Per-agent persistent variation + left/right sensor asymmetry (chirality).
10. Selection-based churn: death probability falls with local trail.
11. Spatial regime mosaic (the reference gap).
12. Anisotropic diffusion along the trail.

Phase 3 — the product asks: local auto mode + home-menu entry; Dither defaults
ordering; multi-colour Dither.

## Bonus: how the CCCCCC preset gets blue, purple and black

Not the palette. `presets.json` CCCCCC is `palette: magenta`, `hue 238`,
`tint 0.61`, `bits 2` -> four colours on a straight sRGB line:
`#080112 / #3F295C / #7752A5 / #AE7AEF`. Then `signal.crush = 1` runs
`round(out*1)/1` (`circuit_bent.py:234-236`) — a hard per-channel binarisation at
127.5:

| level | before | after | reads as |
|------:|--------|-------|----------|
| 0.000 | (8,1,18)     | (0,0,0)     | black |
| 0.333 | (63,41,92)   | (0,0,0)     | black |
| 0.667 | (119,82,165) | (0,0,255)   | **pure blue** |
| 1.000 | (174,122,239)| (255,0,255) | **pure magenta** |

Green never crosses threshold. `chroma 55.4` then rolls R and B independently,
manufacturing red/blue fringes at every edge. Dither's palette system never made
three colours — the SIGNAL rack's 1-bit crush did. That is the mechanism to make
native and choosable.

---

# What changed

## Phase 1 — the physics could not form structure

| fix | where | before -> after |
|---|---|---|
| food is in trail units | `physarum_gl._food_norm`, `update.frag`, `physarum.py` | light was ~1% of the sensed field; now `food=0.35` means "light is worth 35% of the trail's bright end" |
| agent density | `modes/physarum.py` `GL_N` / `CPU_N` / `QUALITY` | 2.12 -> ~0.5 agents per cell; there is dark for a vein to be a vein against |
| sensor saturation | `modes/physarum.py`, `update.frag` | cap sat at ~0.22x the bright end (flattened everything into a 3-unit band) -> above it, so only fat canals compress. weave 0 disables it outright, as the legacy contract says |
| heading jitter | `modes/physarum.py` | 0.50*w -> 0.14*w; heading decorrelated in 0.37 s, now ~1.3 s |
| reseed churn | `modes/physarum.py` | `0.004 + 0.022w^2` -> `0.004 + 0.006w^2`; the whole population recycled every second |
| bilinear sensing | `update.frag` `trail_at` | `texelFetch` quantized the field to whole cells, so sub-texel sensors returned identical values |
| `fingers`, `haze` re-spread | `physarum.py` POINTS, `looks.json` | `fingers` had 2.21 px of sensor separation on a 3.33 px blur and a stride longer than its reach — a constant-turn random walk, and the default body point |
| exposure reference | `physarum_gl` `NORM_EMA` | per-frame p95 renormalized against its own noise (a slow pump); now EMA'd, which is most of what made the browser build read as crisper |
| grain | defaults + `looks.json` | 0.5 -> 0.2; it added the same full-frame speckle to every look |

Pinned by `tests/test_physarum_points.py`: every point must sense past its own
blur, resolve a lateral gradient, not out-stride its own reach, and sit a real
distance from every other point in log-parameter space.

## Phase 2 — aliveness

**Three species, one signed matrix.** The trail is RGB now: each population
lays into its own channel and reads all three through a row of a 3x3 matrix.
Diagonal attracts; each species is repelled by the next and mildly drawn to
the previous. The negative terms are the mechanism for exclusion membranes —
the thin dark walls between territories — which a single-channel attract-only
field cannot produce at any parameter setting. The asymmetry is what makes
those walls travel rather than freeze. Species lives in the agent texture's
previously unused `.w`, assigned once at spawn: a lineage, not a lookup.

**The spatial mosaic.** The grid is cut into ~7-across drifting Voronoi zones,
and each zone runs ONE of six regimes outright — mesh, trunk, bloom, drift,
knot, calm (`physarum.SPATIAL_REGIMES`). Hard seams, not a blend, because a
blend averages six characters back into one. This is the axis the instrument
was missing: PR #26/#28 gave it regimes in TIME, which made the whole frame
change together and still read as one texture everywhere.

**Lateral inhibition.** The diffusion pass is centre-surround now: a strong
vein suppresses its own neighbourhood. That sharpens the vein and digs the
dark halo around it.

It is SPLIT ACROSS BOTH AXES, half strength on each, in both engines. The
three ways to get this wrong, in the order I got them wrong: full strength on
both axes doubles the operator and collapses the picture into a pixel-scale
Turing dot pattern; one axis only is a directional operator and the picture
laminates along it; and — the one that survived longest — GL doing one thing
while the CPU port did the other, with the comments in both files asserting
the version that was no longer true. An earlier draft of this document said
"on the vertical axis only", which by then described neither engine.

## What the knobs do now

- **weave** — one organism or three. 0 leaves the field single-species with the
  cap off: the legacy bold-canal transport network. Up, the populations carve
  each other and the picture becomes a woven, walled mane. Visible in seconds.
- **evolve** — how unlike itself it gets, in time (the existing regime hops)
  and now in space (mosaic strength). 0 is genuinely uniform.
- **react** — unchanged; its keep-map was already the best-scaled lever in the
  instrument, and the food-scale fix makes its motion channel matter too.

## Performance

Measured on this M4 Max, GL field only, 1280x736, sim + tonemap, mean of 120
frames after a 30-frame warmup:

| | agents | ms/frame |
|---|---:|---:|
| main @ c97b61f | 2,000,000 | 7.26 |
| this branch | 500,000 | 4.55 |

1.6x faster, with the mosaic's Voronoi lookup (9 lattice probes per agent per
frame) and the wider inhibition kernel both added. Cutting density paid for
everything and left change. The headroom is deliberately unspent: the sim was
never the expensive half of the pipeline.

## Known limits

- The CPU field is the no-GL fallback. Species, mosaic and inhibition are
  ported to it so both engines express the same model, but it is not
  bit-exact with the GPU — the mosaic is a precomputed zone map rather than a
  per-agent Voronoi.
- Selection-based churn (death probability falling with local trail, which is
  what prunes redundant loops and thickens load-bearing ones) is designed but
  not implemented. It is the next lever for vein-width hierarchy.
