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
Mouse: click a card. Committing the card that is *already* running is an entry,
not a switch — it gets the mode-title flash, never the `already in X` scold.

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
                     Blue noise / Floyd-Steinberg / Riemersma · live/slow badge ·
                     swatch strip: a gradient ramp rendered through the current
                     algorithm+bits+gamma+bias, re-rendered only when TONE
                     changes (graft from Cartridges — instant "what am I hearing"
                     for the eyes)
- TONE             ← Bits (1–4, int) · Gamma toggle (default ON) · Bias cycle
                     auto/light/dark · Contrast · Scale (working height 45–720,
                     default 72)
- PALETTE          ← mono: white-on-black · black-on-white · amber · green
                     phosphor (two-color ramps later)
- SIGNAL   (G)     ← shell rack minus dither row
──────────────────
Sound react (A) · Sens · Record (R) · Menu (M) · Quit
```

Built-ins include stream-tuned looks (bigger cells, higher contrast — the
compression-survivable presets streamers otherwise learn as folklore).

**Perf honesty:** default working height 72 px keeps all four algorithms fast;
ordered dithers (Bayer/blue) are permitted full-res; if Scale is dragged toward
full-res while FS/Riemersma is active, an inline amber note reads
`slow — ordered dither recommended live`. Never silent frame drops.

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
- Every state redundantly coded (color + text + shape). Contrast target:
  legible against a white wall. Cursor auto-hides after 2 s idle.

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
| `?` | `help.overlay` — live key map over scrim; any key closes | — |
| `q` | quit: first press toasts `q again to quit`, second within 2 s quits | toast |
| `s` | `preset.save` (PANEL state only — edit action) | name toast |

Param nudging without the panel: `,`/`.` select prev/next control in spec
order (OSD shows name + value + bar), `-`/`=` nudge by 1/40 of range
(`_`/`+` = ×5). Works in every overlay state. Arrow keys are deliberately not
load-bearing (`waitKey` platform codes).

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
