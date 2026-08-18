# designing-touch — DESIGN

**Thesis:** designing-touch is a stage rig, not a settings panel. Every performance
action is one bare key with big momentary feedback, the output frame is sacred, and
modes are plugins racked into a shell that owns the show.

This document is the design authority for the UI/UX overhaul (issue #7, the
2026-08-04 overhaul handoff). Every PR that touches UX cites the principle it
serves. Method: three competing directions ("Stagecraft" — performance-first,
"Cartridges" — identity-first, "Contract" — architecture-first) were drafted and
judged against the end-user personas, feasibility, and product coherence.
Stagecraft won; the best of the other two is grafted in and credited inline.

Primary persona: **the mid-performance operator** — one hand, dark room, the
laptop screen may be the projector, cannot mouse-hunt, cannot ever see a
traceback. Secondary personas: the dancer inside the frame, the OBS streamer,
the code-first tinkerer, the workshop teacher.

---

## 1. Principles (the eight)

1. **One Key, One Outcome.** Every performance-critical action is a single bare
   keypress: mode switch, preset recall, layer toggle, blackout, panic. If it
   needs the mouse or two hands, it is an *edit* feature, not a *perform* feature.
2. **The Output Is Sacred.** The rendered frame never shows errors, dialogs,
   cursors, stalls, or accidental UI. HIDDEN means provably clean (the OBS
   capture contract). Failures degrade to intentional images. Recording captures
   exactly what the audience sees, minus panel/HUD (today's invariant, kept).
3. **Perform/Edit Split.** Two surfaces, one engine: PANEL is the edit surface
   (mouse-friendly, full controls); HIDDEN/HUD is the perform surface (keys +
   big transient feedback). The instrument boots to the home **menu** with the
   last-used mode already running behind it and pre-selected — one key (Enter,
   or Esc) is on stage. *(Amended 2026-08-15 — the user's call: "it should boot
   into the menu not straight into particle." The rig is still playing at boot,
   it just asks which instrument first; see §3.)*
4. **Feedback Is Momentary, State Is Legible.** Every keystroke echoes as a
   large auto-fading toast; a compact always-current status line (derived from
   the panel spec, so it can never drift) answers "what is running" at a glance.
   Silence-on-input is a bug. Unknown key → gentle `? for keys` toast.
5. **Never Dead, Always Recoverable.** `Esc` always walks one step toward
   HIDDEN/safety. One panic key restores the mode's known-good look in <1 s from
   any state. No keyboard-reachable state requires a restart. State autosaves.
6. **Presets Are an Instrument.** Numbered bank slots (1–9) recalled blind, plus
   setlist order (`[`/`]`), per mode, stored as diffable JSON. Playing presets
   is performing, not file management.
7. **Every Action Is a Command.** Named, listable, bindable
   (`mode.dithergirl`, `output.blackout`, `preset.recall.3`). Keyboard now;
   MIDI/OSC/pedals later are bindings, not rewrites. `?` shows the live map.
8. **Scale With the Frame.** All geometry in the u-unit (frame-relative);
   worst-case contrast guaranteed (outline + scrim); the center of frame is kept
   clear of persistent UI; authored once, legible 720p→4K.

**Continuity rider:** the Particles panel keeps its shipped identity — right
sidebar, same palette constants, same section names (TEMPLATES / SOURCE / LOOK /
MOTION / SIGNAL), same widget shapes, same labels. The overhaul is additive
around it. Slider *labels* stay as shipped; human-language explanations live in
the `i` tooltips (graft from Cartridges, reconciled with continuity).

---

## 2. The mode-shell contract

### 2.1 Split of responsibilities

**Shell (host) owns:** window (`WINDOW_AUTOSIZE`, recreate on res change),
camera capture + loss degradation, still-image sources, mouse routing, key
routing through the command registry, audio (`LiveMic`), recorder (post-FX,
pre-panel — unchanged invariant), preset store CRUD + bank slots, overlay state
machine (HIDDEN/HUD/PANEL), OSD/toast/status renderer, home menu, blackout and
panic, the shared SIGNAL post-FX rack (CircuitBent), fps/status, mode lifecycle
(stop → start with a boot card), quit.

**A mode owns:** its engine objects (sim + renderer), its per-frame image, its
panel sections, its mode-local commands, its defaults and built-in looks.

**A mode does NOT own preset serialization.** Capture/apply are derived
generically from the panel spec (graft from Contract): each widget declares
`save`/`apply` flags (`apply="reset"` merges over defaults, `apply="keep"`
leaves live toggles alone — today's asymmetric semantics, now encoded as data).
This kills the `presets.KEYS` vs save-dict vs `_apply_fx` three-owner drift by
construction; hand-written per-mode preset code is how KEYS drifted the first time.

### 2.2 Mode interface

```python
class Mode(Protocol):
    id: str              # "particles", "dithergirl" — preset + command namespace
    title: str           # "Particles", "Dither Girl" — menu + toast display
    accent: tuple        # per-mode accent hue (graft: Cartridges) — see §5
    accepts_still: bool  # Dither Girl: True

    def start(self, host: "Host") -> None: ...   # build engines; may raise — shell
                                                 # catches, toasts, reinstates previous mode
    def stop(self) -> None: ...                  # release GL/matte/etc; idempotent
    def panel_spec(self) -> list[Section]: ...   # declared controls; shell renders + serializes
    def commands(self) -> dict[str, Command]: ...
    def step(self, frame_bgr, audio, dt) -> np.ndarray: ...  # sim+render → RGB at host.res
    def safe_look(self) -> str | dict: ...       # the panic target
    def on_resize(self, w: int, h: int) -> None: ...
```

The host feeds `step()` a frame from its active **Source** (camera, or a still
image whose `read()` returns the same frame every tick — graft from Contract:
modes never special-case stills). On camera loss the host passes the last good
frame and shows the amber HUD note; modes never see `None`.

The per-frame engine sync does not disappear — it relocates *inside*
`mode.step()` (the honest accounting; a params namespace does not remove the
need to copy values onto `pf`/`glow` each frame).

Modes register in a `REGISTRY` list (graft from Contract): one import + one
line adds a mode to the menu, key table, and preset store.

### 2.3 Panel-spec data model

Generalizes the shipped `_SLIDERS`/`_RANGES` pattern to whole sections. One
declaration per control (today: 4–5 touch-points per control).

```python
@dataclass
class Slider:  label; attr; lo; hi; fmt=".2f"; tip=""; save=True; apply="reset"
@dataclass
class Toggle:  label; attr; tip=""; save=True; apply="keep"
@dataclass
class Cycle:   label; attr; options; save=True; apply="reset"
@dataclass
class Action:  label; command
@dataclass
class Readout: render                      # draws into its row rect (swatch strip)
@dataclass
class PresetList: pass                     # TEMPLATES rows + Save + slots
@dataclass
class Section: title; widgets; open=True; key_hint=None
```

The widget draw methods (`_row/_slider/_section/_cycle`, rename box, armed
delete, tooltip, scroll) lift out of `OverlayUI` into `dtouch/imgui.py`;
`OverlayUI` becomes a generic walker over `panel_spec()` plus the shell's global
rows. The home menu draws with the same toolkit. The HUD status line renders
from the same spec (mode title + the widgets marked `status=True`), e.g.
`DITHER GIRL · blue noise · 3-bit · bias auto`.

### 2.4 The SIGNAL rack (where Glitch lives)

**Glitch is not a mode — it is a shell-owned post-FX layer** in every mode,
applied after `mode.step()` and video composite, before recorder and overlay.
CircuitBent bends *any* picture; a glitch mode would orphan glitched-particles,
the most-used combo. The rack contributes a `Section("SIGNAL", …)` appended to
every mode's panel; its state saves *inside each mode's looks* under
`"signal": {…}` so a glitched look recalls on one keystroke. `G` toggles it.

**In Dither Girl the rack's dither row is suppressed** — Dither Girl owns
dithering as the primary image; two visible dither subsystems in one panel is
exactly the bolted-features incoherence this overhaul exists to kill. (Judge
finding, fixed by rule: the rack hides any control the active mode claims.)

In Particles, SIGNAL keeps the shipped dither cycle and grows the audit's
quality controls (Bits, Gamma default-ON, Bias) — removing a shipped live
control is a regression, not a cleanup.

### 2.5 What happens to Flocking

**Flocking stays a layer of Particles in v1** — MOTION section + `F`, exactly
the live integration that exists (`flock_forces` on the particle cloud; zero
gains = solver skipped). Issue #7 lists Flocking as a mode, so this is a
deliberate, visible deviation, not a silent one: the only standalone flocking
engine is frozen experiment 06 (O(n²), offline, no camera) — promotion would
mean shipping a new engine, not a UI change. The home menu reserves a dimmed
"Flocking — coming soon" card as the landing site for issue #4's body-driven
flock; a future real-time boids-primary engine registers as a mode with zero
shell changes. The rationale goes on issue #7 for sign-off.

---

## 3. Home menu

**Form: in-app switcher screen, not a launcher.** One window; the menu is a
shell overlay state, not a Mode.

**Boot behavior (opens on the menu, still playing).** *(Amended 2026-08-15 —
the user's call, and the doc previously said the opposite: "launch goes
straight into the last-used mode … no menu".)*

Launch with no mode-implying flag opens the **home menu**. It is not a blank
canvas and not a launcher screen: the last-used mode (first run: Particles) has
already been started and is stepping live behind it, the camera runs through the
1-bit blue-noise backdrop, and that mode's card is **pre-selected**, so `Enter`
is a one-key resume. `Esc` at boot **enters the selection** rather than
"returning to the running mode" — at boot there is nothing behind the menu to
return to, and a dismiss that dropped you onto a mode you never chose is the
dead end principle 5 forbids. `q` still closes the menu and arms the quit
confirm.

CLI flags still mean "boot into" and **skip the menu entirely**: `--mode
dithergirl`, `--still PATH`, `--flock`/`--glitch`. A flag that had to wait
through a menu would be a worse launch than we had. `start.command` unchanged.

The 4 s boot hint follows where the launch landed, because the menu already
prints its own key line: in the menu it names the resume (`enter resumes
PARTICLES`), and the three-doors hint (`m menu - TAB panel - ? keys`) is posted
as the menu is left — on the mode, where it is true. It is never posted *under*
the menu (issue #18: the menu's hint line and the HUD's hint toasts were drawn
centered on the same baseline and rendered as one unreadable smear; the menu's
line now sits clear of the whole 3-deep toast stack, at a fixed offset so it
does not jump as toasts come and go).

**Layout:** live camera behind the menu, rendered through 1-bit blue-noise
dither and dimmed under a 65% scrim (graft from Cartridges: the menu proves the
camera works and demos the newest feature before a single click). Title
`dtouch` top-left. Centered row of mode cards from the REGISTRY, each card in
its mode's accent:

```
┌──────────────┐   ┌──────────────┐   ┌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┐
│  PARTICLES   │   │ DITHER GIRL  │   ┆   FLOCKING   ┆
│ webcam-matte │   │ live + still │   ┆ coming soon  ┆
│  instrument  │   │  dithering   │   ┆              ┆
│     [P]      │   │     [D]      │   └╌╌╌╌╌╌╌╌╌╌╌╌╌╌┘
└──────────────┘   └──────────────┘
```

**Navigation:** the mode's letter or digit selects-and-enters; `,`/`.` move
selection, `Enter` commits, `Esc` returns to the running mode untouched (at
boot, where there is no running mode, `Esc` commits the selection instead).
Mouse: click a card. Committing the card that is *already* running **at boot**
is an entry, not a switch — it gets the mode-title flash, never the `already in
X` scold, because that mode was started behind the menu so the screen would be
live and it is what the selection defaulted to. Mid-set the scold stands: `m`,
then `Enter` on the card you are already in, is a request for a mode you have,
and saying so is more useful than a flash that looks like something happened.

**Mode switch:** shell draws a **static boot card** (mode glyph + name in the
mode's accent, one frame, no animation — graft from Cartridges, feasibility-
corrected: the cv2 loop is single-threaded, so nothing can animate during GL
teardown), then `old.stop()` → `new.start(host)`, then a 3u center toast. The
recorder captures the card, not a gray flash. Mid-set switching never requires
the menu: `P` and `D` switch directly from any overlay state.

While blackout is armed, the boot card is NOT shown bright — the switch happens
under black (tick stays); the card is only for un-blacked-out switches. Inside
the menu, unknown keys get the `? for keys` hint (silence-on-input is a bug
there too) and `q` closes the menu and arms the quit confirm. The menu's bottom
hint line is part of the layout — `, . move - enter select - esc back`, or
`, . move - enter select - q quit` at boot, where `esc back` would be a lie —
and the real mode cards render their titles in CAPS per the sketch.

*(Amended 2026-08-15.)* No key in the menu is silent. `0` and `3`–`9` used to
do nothing at all: the digit branch mapped a key to `i = int(ch) - 1` and
returned nothing to hint on when that index was not an enabled card, while `s`,
`r`, TAB and every other unbound printable key hinted correctly. A digit that
names the **reserved card** now says so in words (`flocking - coming soon`) —
that card is on screen and dashed, so deflecting to a key map that cannot
explain it either was the wrong answer; a digit that names no card at all hints
`? for keys` like everything else. The **arrow keys** also move the selection,
because the menu is a row of cards and the first screen anyone meets, and a
child reaches for arrows long before `,`/`.`. That does not make them
load-bearing (§6.2): `,`/`.` stay the documented navigation and stay on the
hint line, the arrow codes here are macOS's masked values, and nothing in the
menu is reachable only by arrow.

Saving a look (`+ Save current look` row, or the `s` key): creates an
auto-named look and immediately opens its rename box — naming is one flow —
and two saves in one second must not collide (suffix on collision). Save,
rename, and delete all confirm with toasts, never stdout.

---

## 4. Per-mode panels

Panel stays a **right sidebar**, alpha scrim behind, single column, collapsible
sections, scroll when tall. The shell appends the same global block everywhere.

### 4.1 Particles (continuity mode)

```
dtouch · PARTICLES
- TEMPLATES        ← PresetList: rows, slot badges [1]..[9], Save current look
- SOURCE           ← matte cycle · output res · Video bg · Vid mix
- LOOK             ← color cycle · Trails Glow Spark Flow Size Count Glide Pull Reseed
- MOTION   (F)     ← Flock toggle · Cohere · Align · Separate      [starts closed]
- SIGNAL   (G)     ← Glitch toggle · dither cycle (bayer·blue·fs·riemersma·off) ·
                     Bits · Gamma (ON) · Bias (auto/light/dark) · Chroma · Drift ·
                     Crush · Scanlines                             [starts closed]
──────────────────
Sound react (A) · Sens · Record (R) · Mirror · Menu (M) · Quit
```

Identical section order, names, labels, and widget shapes as shipped. Visible
changes only: slot badges, key hints on section headers, SIGNAL's new dither-
quality controls. Boids sliders keep their labels; their tooltips say the human
thing ("how tightly the swarm pulls together", etc.).

### 4.2 Dither Girl

The dither pipeline as the *primary* image: camera or still → optional matte
gate → gamma-correct dither at chosen working scale → palette. SIGNAL rack
available on top (minus its dither row, §2.4).

```
dtouch · DITHER GIRL
- SOURCE           ← input cycle: camera / still… · matte cycle (dither the
                     subject only) · output res · Mirror
- ALGORITHM        ← big active-algorithm label (1.4u, accent) · cycle: Bayer /
                     Blue noise / Floyd-Steinberg / Riemersma / ASCII ·
                     live/slow badge · swatch strip: a gradient ramp rendered
                     through the current algorithm+bits+gamma+bias+palette,
                     re-rendered only when one of those changes (graft from
                     Cartridges — instant "what am I hearing" for the eyes).
                     Under ASCII the strip becomes one row of the live glyph
                     ramp, the most direct answer to "what am I looking at"
- TONE             ← Bits (1–4, int) · Gamma toggle (default ON) · Bias cycle
                     auto/light/dark · Contrast · Scale (30–720, default 72) ·
                     grid readout (ASCII only)
- PALETTE          ← ten two-colour pairs, each shipped only if its own off/on
                     measures ≥ 4.5:1 (§5): mono · amber · green phosphor ·
                     cyan · magenta · ice · blood · gameboy · sepia · hi-vis —
                     plus an **Invert** toggle, and Hue and Tint
- SIGNAL   (G)     ← shell rack minus dither row
──────────────────
Sound react (A) · Sens · Record (R) · Menu (M) · Quit
```

Built-ins include stream-tuned looks (bigger cells, higher contrast — the
compression-survivable presets streamers otherwise learn as folklore).

**ASCII is the fifth quantiser, not a second engine** (amended 2026-08-15). It
answers the question the four dithers answer — *given an output alphabet
smaller than the input's tonal range, how do I spend it to fake continuous
tone?* — with glyphs instead of grey levels, so it belongs in the same cycle
and reuses the same blue-noise texture, transfer functions and bias semantics
one level up, on the glyph index. It measures in the ordered class (0.6–1.2 ms
at 720p) and the badge reads `live`. See `dtouch/ascii_art.py` for the three
load-bearing choices (measured ramp, stroke weight as part of the ramp, tone
normalised to the ink colour), each arrived at by rendering the alternative.

Two TONE controls change **meaning** under ASCII, and the panel says so rather
than hiding it:

- **Bits** is the ramp LENGTH (n = 2/4/8/16 glyphs), not an output bit depth.
  The stroke-weight search narrows with n: a heavy stroke buys tonal reach, and
  spent on a two- or four-step ramp it is a cell-filling blob rather than a
  glyph, so short ramps get letters instead of a dot halftone.
- **Scale** is character ROWS, not working pixels — which is why its floor is
  30 rather than the 45 working pixels the four pixel dithers shipped with. A
  DIM readout under the slider gives the live grid (`grid 160 x 45 chars - cell
  8x16`), and says `cell floor - characters stay legible` once the legibility
  floor is what is setting the cell size and the slider has run out of room.

**Invert** (amended 2026-08-15 — the user's call: "instead of having white on
black or black on white couldn't any of the pallets just have an invert
switcher?"). `white-on-black` and `black-on-white` were one palette entered
twice: which end carries the ink is orthogonal to the hue, so spending two of
eleven slots on it bought a flip for one palette and denied it to the other
nine. They collapse to `mono` plus a toggle that flips ANY palette, composing
with Hue/Tint, the swatch strip, ASCII (where it flips ink and ground for
free — the glyph ramp is measured against whatever pair it is handed) and the
matte gate.

It saves and recalls like every other control, and it is `apply="reset"`
rather than a Toggle's default `apply="keep"`: Invert is part of the picture,
not a live rig switch, so a look that does not name it must land un-inverted.

The retired names are accepted on load forever, mapping to `mono` + invert
off/on, and the migration is bit-for-bit: `mono` inverted is the AUTHORED pair
`black-on-white` shipped with (245 paper, 16 ink, 17.45:1), not the plain swap
of `mono`'s own 0/255 — the two monochrome palettes were never each other's
mirror, and a naive swap would have silently re-toned every look that named
`black-on-white`, `newsprint` included.

**Whole-number sliders** (same amendment — "maybe some sliders are actually
only whole numbers?"). A control whose engine cannot use a fraction declares
its quantum in the spec (`Slider.step`, and `Slider.snap` because Bits and
Scale round while Crush truncates), and the one number it holds is what the
panel prints, what the OSD and the spec-derived HUD line print, and what the
engine consumes. Quantised: **Bits** (both racks, 4 settings), **Scale**
(whole working pixels / whole character rows), **Crush** (whole output bits,
`int()`), **Hue** (whole degrees — finer than the track can resolve anyway).
Deliberately NOT quantised: Chroma and Drift, which look like pixel counts but
are the amplitudes of a per-frame draw that then lands on a whole pixel — the
fraction is used, so the tooltip says where the whole numbers come in instead
of the slider pretending to be coarser than it is.

**Hue / Tint** (delivering §4.2's "two-color ramps later"): two sliders rather
than a longer list, because a list of named pairs cannot be dialled during a
set and a per-colour RGB editor is six sliders of fiddling in a panel meant to
be driven at arm's length in the dark. Tint is an amount, Hue a direction; at
Tint 0 every named palette is bit-identical to its shipped values, so the pair
COMPOSES with the palettes rather than replacing them. Both ends are steered,
so a duotone stays a duotone. The steering target is lifted toward white until
its relative luminance clears 65% of the source's — without that floor a hue
near blue costs ~14x luminance and Tint could quietly turn any palette into an
unreadable navy-on-black; with it, every shipped palette stays over 4.5:1 at
every hue and every tint (measured, 72 hues × 4 tints).

**Perf honesty:** default working height 72 px keeps all five algorithms fast;
ordered dithers (Bayer/blue/ASCII) are permitted full-res; if Scale is dragged
toward full-res while FS/Riemersma is active, an inline amber note reads
`slow — ordered dither recommended live`. ASCII carries a measured note rather
than a guessed threshold — it times its own whole step, rebuild included, and
reads `ascii N.N ms/frame - lower Scale or output` once the EMA passes 8 ms
(cleared again at 6, so it cannot blink while an operator hovers the line).
Never silent frame drops.

### 4.3 Dispositions

- **Flocking:** layer of Particles (§2.5), reserved menu card.
- **Glitch:** shell SIGNAL rack in every mode (§2.4), never a mode.
- **gui.py (Tk panel):** deleted — a drifted duplicate implementation of the
  instrument; the overlay panel is the one UI. (Flagged for sign-off.)

---

## 5. Visual style

**u-unit geometry.** `u = frame_height / 45` (≈16 px @720p, 24 @1080p,
48 @4K). All *new* geometry (HUD, toasts, menu, boot cards) is authored in u.
The shipped panel keeps its `_S()` scale factor, redefined internally as
`u/24 ×` so both systems are one formula; identical pixels at 720/1080p is a
pinned regression target. Hit targets ≥ 2.75u whole-row.

**Type scale** (Hershey, always double-drawn — black offset under near-white):

| Role | Size |
|---|---|
| Mode toast (center flash) | 3.0u, fade 1.2 s |
| OSD param readout + bar | 1.5u, bottom-left, fade 1.5 s |
| Panel title / active algorithm | 1.4u, accent |
| Row labels, values | 0.9u |
| Section headers | 0.8u CAPS |
| Hints, tooltips, status line | 0.75u |

**Color = state, never decoration:**

- Grayscale is the default chrome; panel scrim 86% as shipped; menu scrim 65%.
- **Per-mode accent** (graft from Cartridges): exactly one accent on screen,
  owned by the active mode — Particles keeps ACC green (continuity); Dither
  Girl gets a magenta-leaning accent, contrast-checked against the scrim (the
  Cartridge doc's near-white accent failed its own legibility rule; pick by
  measured contrast, not vibes). Accent = selected/active/ON + mode identity
  readable from 2 m.
- `RED` — record dot and destructive-armed only (delete-armed is RED: it is a
  destructive arm, not a warning).
- **Amber** (one addition) — armed/warning/transient: panic flash,
  perf warnings, camera-lost note, **and a persistent 2u corner tick while
  blackout is armed** (judge finding: a fading toast alone leaves an operator
  believing the app died; the tick is the one persistent element allowed on a
  blacked-out frame, and blackout output is intentionally not clean-capture).

  *Re-examined 2026-08-15 and kept.* Blackout in HIDDEN lights 1,225 px of
  2,073,600 at 1080p (the tick), which reads as a broken app rather than an
  armed one. Two things make it not a dead end, both measured on that exact
  frame: the tick is the **only** thing separating "armed" from "the app
  died", so removing it makes the state genuinely indistinguishable from a
  crash — the opposite of the fix; and the screen is black only while nobody
  is touching it. Arming it flashes `BLACKOUT` (10,608 px). Any unbound key
  paints the hint (1,963). A param nudge paints the OSD (4,356). One `TAB`
  brings the whole HUD back (2,807). Ten seconds hands-off returns to 1,225 —
  the clean-capture contract intact. The keys that used to answer with nothing
  here were the arrows and Enter, and that was §6.2's hint window, now fixed.
- Every state redundantly coded (color + text + shape). Contrast target:
  legible against a white wall. Cursor auto-hides after 2 s idle.

  *Measured 2026-08-15, after the Invert toggle raised the worry that ten
  palettes can now put a bright ground under the HUD.* The double-draw holds,
  and the worry is backwards: the black under-stroke carries the glyph, so the
  brighter the ground the better it reads. Status line — white 21.0:1, 235-grey
  17.6:1, mid-grey 5.32:1, black 14.7:1. Bottom hint (DIM, the smaller and
  weaker of the two) — white 21.0:1, 220-grey 15.3:1, mid-grey 5.32:1, dark
  40-grey 4.74:1. The weakest ground is a flat mid-grey, which still clears
  WCAG AA, and it is the ground **Invert moves away from**: over the inverted
  1-bit picture the toggle actually produces, both lines render as heavy black
  outline type. No scrim added — it would put permanent chrome on the picture
  in HUD, which is a state used for composing shots, for no measured gain.

**Layout discipline:** persistent UI only in the right sidebar and corners;
title-safe 3.5% inset; frame center reserved for transient toasts. Overlay
budget ≤ 1 ms/frame: panel rendered to a cached RGBA buffer, re-rendered only
on dirty, blitted as a small-ROI composite (fallback: today's full redraw is
shipped and acceptable — caching is an optimization gate, not a blocker).

---

## 6. Interaction model

### 6.1 Overlay states

**HIDDEN** (provably clean output) → **HUD** (status line top-left + momentary
toasts/OSD) → **PANEL** (full sidebar; HUD still active).

- `TAB` cycles HIDDEN → HUD → PANEL → HIDDEN.
- `Esc` always steps toward HIDDEN; in HIDDEN it does nothing. Esc never quits.
  (The shipped code ignores ESC deliberately; this retires "ignored" in favor of
  "safe by design" — flagged for sign-off with the Flocking item.)
- In HIDDEN, keys still work and each shows its toast, then fades. Entering
  HIDDEN shows one final toast `overlay hidden — TAB to show`, then pixels are
  untouched.
- Boot state: HUD.

### 6.2 Keyboard (perform layer — all bare keys)

All bindings are **case-folded**: the table binds lowercase codes; shifted
variants alias unshifted (judge finding — `cv2.waitKey` returns unshifted
lowercase; an uppercase-only table would silently demand Shift).

| Key | Command | Feedback |
|---|---|---|
| `1`–`9` | `preset.recall.N` (current mode's bank) | toast `3 - embers` (ASCII hyphen: Hershey has no middle dot) |
| `[` / `]` | `preset.prev` / `preset.next` (setlist) | same toast |
| `p` / `d` | `mode.particles` / `mode.dithergirl` | boot card + center flash |
| `m` | `menu.open` | menu |
| `f` | `layer.flock` (Particles) | toast |
| `g` | `layer.glitch` (rack) | toast |
| `Space` | `output.blackout` toggle (recorded too; UI responsive) | toast + amber corner tick while armed |
| `0` | `preset.panic` — mode's `safe_look()` in <1 s; disarms blackout AND the SIGNAL rack (glitch off) — panic must restore a known-good *picture*, not just known-good params | amber `RESET` |
| `a` | `audio.toggle` | toast |
| `r` | `record.toggle` | red dot; filename toast on stop |
| `v` | `video_bg.toggle` (Particles) | toast |
| `i` | `debug.toggle` — fps/frame-time/res HUD line (graft: Contract) | HUD line |
| `?` | `help.overlay` — live key map on a plate over the scrim; any key closes | — |
| `q` | quit: first press toasts `q again to quit`, second within 2 s quits | toast |
| `s` | `preset.save` (PANEL state only — edit action) | name toast |

Param nudging without the panel: `,`/`.` select prev/next control in spec
order (OSD shows name + value + bar), `-`/`=` nudge by 1/40 of range
(`_`/`+` = ×5). Works in every overlay state. Arrow keys are deliberately not
load-bearing (`waitKey` platform codes) — but not load-bearing means they must
say so, not vanish. The `? for keys` hint used to fire only for codes 32–126,
and macOS masks the arrows to 0–3 (Enter 13, Backspace 8, Delete 127), so those
seven keys did nothing anywhere with no feedback at all while every printable
unbound key answered correctly (amended 2026-08-15). They hint now. `TAB` and
`Esc` stay out of that window on purpose — the overlay stepper consumes them
before dispatch — and so does `waitKey`'s idle code, which is no keypress.

The walk skips any control the perform layer cannot take back (amended
2026-08-15). Today that is exactly one: **output resolution**. It is a `Cycle`,
so it was simply the 2nd of 23 stops in Particles and the 3rd of 15 in Dither
Girl — `.` `.` `=` with no panel open recreated the window at 4K, took the
frame rate with it, and left `0` with no answer, because panic restores the
mode's *look* and the window is not in the look. The test is not "is this a
system setting" and not `save=False` — `Mirror` and `input` are unsaved and
stay reachable, because pressing the key again undoes them. Resolution lives
on the edit surface, where changing it is a deliberate decision.

**A rename box holds the keyboard only while it is on screen** (amended
2026-08-15; scroll route closed 2026-08-19). The consumption rule below is
right and stays, but it was not tied to visibility, and four routes left
`renaming` set with nothing drawn: collapse the sidebar with the chevron, open
the menu from the panel's own `Menu (M)` row, hide the overlay — or scroll the
box off the top of the column, which needs no second control at all (the panel
column is ~1200 px in a 720 px window; TAB, `s`, a few wheel notches). The
screen then showed a completely normal instrument — bottom hint still reading
`m menu - TAB panel - ? keys`, all three dead — while TAB, `m`, `?`, space,
`0` and both presses of `q` were typed into a field nobody could see. Only
`Esc` got out, and nothing said so. The shell now cancels any rename box that
was not painted on the frame just composed, and *painted means pixels*: cv2
clips off-frame draws silently, so the draw walk reaching the row does not
count — the box's rect has to intersect the frame. Visibility is the rule, not
a list of routes, so any future way to take the panel off screen is covered.
One grace note keeps the rule from eating its own flow: a box that *opens*
below the fold (save appends the new look's row) is scrolled into view on the
next frame rather than silently expired — cancelling a rename the operator
just asked for would send the name they type to the global keys, `q q`
included.

**The key map sits on a plate** (amended 2026-08-15). The 65% scrim is a
multiply, so it darkens the picture without flattening it: a 1-bit output is 0
vs 255, and 65% of that is 0 vs 89 — still hard-edged, still full-contrast, and
at the same spatial scale as the glyph strokes. Measured over pure 1-bit noise,
the ground under the table swung 0–89 (std 44.5) and gave white ink only 2.9×
contrast against the brightest pixel it sat on. The block now gets the panel's
own ground (PANEL at 0.86) sized to the table: ground 28–40 (std 6.0), worst
case 6.4×. The scrim stays, so the picture is still visibly there around the
plate and help never reads as the instrument stopping. It costs 2.2 → 2.9 ms
per frame at 1080p, and only on the frames the modal is open: help is a modal,
explicitly outside the ≤1 ms steady-state overlay budget (§5). HIDDEN (0.002
ms), HUD (0.141) and PANEL (0.74) are unchanged.

**Rename typing consumes every key except `Esc`** (which cancels the rename and
nothing else). This is the explicit carve-out rule (judge finding: "panic works
mid-rename" contradicted the consumption contract — resolved: while typing, you
are on the edit surface; `Esc` then `0` is the two-press escape hatch, and
names may contain any character).

### 6.3 Mouse (edit layer — fallback, never required)

As shipped: hover highlight, click rows/toggles, whole-row slider drag, wheel +
drag scroll, hover rename/delete with two-click arm, `i` tooltips. New:
clicking a preset row's slot badge assigns the next free bank number. No
critical action is mouse-only; the mouse cannot reach anything keys cannot.

**Scrollbar** (amended 2026-08-15): a real scrollbar in the panel's left
gutter, drawn only when the content overflows — up arrow, draggable thumb,
clickable track, down arrow. It exists because the wheel is the least portable
input a cv2 window has (on macOS it never scrolled up at all), so a panel
taller than the frame was mouse-unreachable at the bottom: the mouse layer is a
fallback, and a fallback with a hole in it is not one. The arrows are drawn at
the shipped chrome's button scale but their HIT rects are the full gutter width
and 2.75u tall, flush against the top and bottom edges of the frame — an edge
target is infinitely tall to a mouse (Fitts). Drawn small, hit large. Its hits
are front-inserted, so a row that scrolled under the gutter cannot steal them.

### 6.4 Failure behavior (shell-owned)

- Camera loss / black-streak: hold last good frame; amber HUD note with the
  human fix (Continuity-Camera message kept verbatim).
- `mode.start()` raises: toast the human summary, reinstate the previous mode.
- Camera-permission denial: plain-language on-canvas explanation, not a trace.
- Autosave: bank + active mode + active preset → `state.json`; crash restart
  resumes the same look.

---

## 7. Preset model

**Per-mode presets, one file, spec-derived schema.**

Implementation note: `state.json` (not `presets.json` meta) is the single
last-mode/last-preset authority — one file, one writer, crash-safe to lose.

```json
{
  "version": 2,
  "meta": {},
  "modes": {
    "particles": {
      "looks":   { "embers": { "…": "spec-captured, incl. \"signal\": {…}" } },
      "bank":    { "1": "abstract", "3": "embers" },
      "setlist": ["abstract", "embers", "sigil"]
    },
    "dithergirl": { "looks": {}, "bank": {}, "setlist": [] }
  }
}
```

- Built-ins stay in code per mode, immune to rename/delete, bankable.
- Capture/apply derive from the panel spec's `save`/`apply` flags (§2.1) — the
  single schema authority. Asymmetric apply kept: `apply="keep"` widgets are
  untouched by look-switching; `apply="reset"` merge over defaults.
- Bank slots are explicit assignments (not list position) — rehearsal parity
  requires slot stability across saves/deletes.

**Migration:** `load()` detects a v1 file (no `"version"`), migrates in memory
(every v1 look → `modes.particles.looks`; v1 never persisted MOTION/SIGNAL keys
— verified against `presets.KEYS` filtering — so nothing changes meaning), and
writes v2 only after copying the original to `presets.v1.bak.json`. Round-trip
tests pin: every v1 fixture applies to identical engine state before and after.

---

## 8. Migration plan (ordered)

0. **Golden pins first** (graft from Contract): screenshot fixtures of today's
   panel at 720p/1080p/4K over bright + dark frames, compared with a defined
   tolerance (mean-abs-diff threshold — cv2 text rasterization varies across
   builds; exact-pixel goldens flake).
1. **Command registry + key routing** (`dtouch/commands.py`), wired behind
   current behavior. Table-driven tests: every key × every state.
2. **Overlay state machine + HUD/OSD/toast layer** (`dtouch/hud.py`), u-units,
   outlined text, status line, TAB/Esc cycling around the existing panel.
3. **Perform keys + safety:** blackout + tick, panic, quit-confirm, `?` help,
   camera-loss hold, digits against the existing preset list (temporary global
   bank — becomes per-mode at step 7; muscle-memory churn accepted since 3–7
   land in one release).
4. **imgui toolkit extraction** (`dtouch/imgui.py`); OverlayUI walks a spec
   reproducing today's panel exactly (goldens hold).
5. **Dither correctness pack** (parallel track, per the audit): gamma default,
   bias flip, blue-noise asset, Riemersma, GPU docstrings, DITHERS grows.
6. **Host/Mode split** (`dtouch/shell.py`, `dtouch/modes/particles.py`); SIGNAL
   rack to shell; `run.py` thin; `gui.py` deleted; `live_flow` kept as a shim.
   GL lifecycle: idempotent `stop()`, 100-swap soak test.
7. **Preset v2** (spec-derived schema, migration + backup + round-trip pins,
   banks/setlist, slot badges).
8. **Home menu + mode switching** (menu state, cards, boot card, `p`/`d`/`m`).
9. **Dither Girl mode** (`dtouch/modes/dithergirl.py`) per §4.2.
10. **Dark-room test + polish:** lights off, projector/TV across the room, one
    hand, 10-minute scripted set including a yanked camera. Any traceback, lost
    mode, needed mouse, or needed second hand = release blocker.

## 9. Risks (carried from the direction reviews)

- cv2 keyboard: focus-dependent, no key-up; all bindings printable ASCII.
- GL context lifecycle across swaps — soak test before the menu ships.
- Preset migration is the one user-data-loss surface — backup + pins mandatory.
- `q` gains a confirm; call out in release notes.
- FS/Riemersma at full res: amber inline warning, never silent drops.
- Deviations needing explicit sign-off: Flocking-as-layer (vs issue #7's mode
  list), Esc repurposing, `gui.py` deletion.
