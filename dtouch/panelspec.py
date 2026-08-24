"""Panel-spec data model — controls declared as data (DESIGN.md §2.3).

Generalizes the shipped `_SLIDERS`/`_RANGES` pattern to whole sections: one
declaration per control. A panel is a list of `Section`s (plus trailing bare
widgets for the shell's global rows); a generic walker (OverlayUI today, the
shell for every mode later) renders it with the imgui toolkit and derives
preset capture/apply from the `save`/`apply` flags — the single schema
authority (DESIGN.md §2.1: `apply="reset"` merges over defaults,
`apply="keep"` leaves live toggles alone — today's asymmetric semantics,
encoded as data).

`gap` fields carry the shipped panel's exact vertical rhythm (baseline px at
1080p, scaled at draw time): for Slider/Cycle it is extra space after the
widget's own advance; for Toggle/Action it is added to the standard 28 px row
advance in one rounding (`S(28 + gap)`) — matching the shipped literals.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence


@dataclass
class Slider:
    label: str
    attr: str
    lo: float
    hi: float
    fmt: str = ".2f"
    tip: str = ""
    save: bool = True
    apply: str = "reset"
    gap: int = 0
    save_key: Optional[str] = None   # serialized name; defaults to attr
    # status (DESIGN.md §2.3): the HUD status line renders from the spec —
    # a format string ("{:.0f}-bit") or callable applied to the widget's
    # value; None = not part of the status line.
    status: Any = None
    # ----- quantisation (the whole-number sliders) -----
    # `step` is the control's quantum in value units — what the panel's INPUT
    # surfaces (the drag, the nudge keys) land on. It comes in two honesties,
    # told apart by `engine_snaps`:
    #
    # * `engine_snaps=True` (default): the engine downstream genuinely cannot
    #   use anything finer — Bits is `int(round(v))`, Crush is `int(v)`. Every
    #   write lands on the grid (OverlayUI.__setattr__), stored looks
    #   included, because a fractional value would only be a lie in the
    #   readout: the picture already rendered at the snapped value.
    #
    # * `engine_snaps=False`: the engine consumes the value CONTINUOUSLY and
    #   the step is a deliberate control-feel choice — Scale under ASCII is
    #   `frame_h / rows_req` before any rounding (a stored 45.55 and 46 are
    #   different character grids), Hue feeds `tint_rgb` as a float. The drag
    #   and the nudge still snap (that is the display-honesty win), but a
    #   STORED look is the authority on its own value (the reseed/`portrait`
    #   rule: the control adapts, never the look), so `apply_look` passes it
    #   through clamped and un-snapped, and the readout shows it faithfully
    #   (`display_fmt` widens to one decimal for an off-grid legacy value).
    #
    # None (the default step) means genuinely continuous, and a continuous
    # slider is left alone — faking precision and faking coarseness are the
    # same lie in opposite directions.
    #
    # `snap` is how a raw value lands on the quantum, and it exists because
    # the engines' own rounding is not uniform: Bits rounds, Crush TRUNCATES
    # (`int(ui.crush)`). A control must land where its engine lands or the
    # readout goes back to lying — with the extra sting that `"{:.0f}"`
    # rounds, so an unsnapped Crush of 2.92 drew "3" while the picture was
    # crushed to 2. Truncating here also makes the migration free: every
    # already-saved fractional Crush keeps the exact bit depth it rendered at.
    step: Optional[float] = None
    snap: str = "round"              # "round" | "floor"
    engine_snaps: bool = True        # engine lands on the grid itself (see above)
    nudge: bool = True               # reachable by ','/'.' — see `nudgeable`

    @property
    def store_key(self):
        return self.save_key if self.save_key is not None else self.attr


@dataclass
class Toggle:
    label: str
    attr: str
    tip: str = ""
    save: bool = True
    apply: str = "keep"
    on_text: str = "ON"          # shipped rows disagree ("ON" vs "on") — per instance
    off_text: str = "off"
    label_fn: Optional[Callable[[bool], str]] = None   # e.g. Record / Stop recording
    gap: int = 0
    save_key: Optional[str] = None
    status: Any = None           # see Slider.status (DESIGN.md §2.3)

    @property
    def store_key(self):
        return self.save_key if self.save_key is not None else self.attr


@dataclass
class Cycle:
    label: str
    attr: str                    # index attribute on the state object
    options: Sequence[Any]       # displayed value is options[index]
    key: Optional[str] = None    # hit-test key; defaults to label
    save: bool = True
    apply: str = "reset"
    gap: int = 2
    save_key: Optional[str] = None   # serialized name (stores the VALUE, not the index)
    status: Any = None           # see Slider.status (DESIGN.md §2.3)
    # Retired stored values, and what they mean now: {old_value: {store_key:
    # value, ...}}. A look that names a retired option is rewritten to the
    # current spelling before it is applied (see `migrate_legacy`), which is
    # what lets an option be REPLACED rather than merely renamed — Dither
    # Girl's "black-on-white" became the mono palette plus Invert, two keys,
    # and only the retired name knows that. Old names stay accepted forever:
    # this is the one user-data-loss surface (DESIGN.md §9), and a Cycle
    # silently ignores a value it does not recognise.
    legacy: Optional[dict] = None
    nudge: bool = True               # reachable by ','/'.' — see `nudgeable`
    tip: str = ""                    # `i` tooltip, same contract as Slider.tip

    @property
    def hit_key(self):
        return self.key if self.key is not None else self.label

    @property
    def store_key(self):
        return self.save_key if self.save_key is not None else self.attr


@dataclass
class Param:
    """A captured-but-undrawn engine parameter (no row, no pixels). It exists
    so the spec stays the single schema authority even for values without a
    panel control yet (e.g. Particles' attract_speed, saved since v1)."""
    attr: str
    save_key: Optional[str] = None
    save: bool = True
    apply: str = "reset"

    @property
    def store_key(self):
        return self.save_key if self.save_key is not None else self.attr


@dataclass
class Action:
    label: str
    command: str                 # hit-test key; posted to the walker's mailbox/handler
    gap: int = 0


@dataclass
class Readout:
    render: Callable             # draws into its row rect (e.g. swatch strip); returns next y


@dataclass
class PresetList:
    """TEMPLATES rows + Save (+ slot badges at DESIGN.md §8 step 7).
    The walker owns the row/rename/manage drawing; nothing to declare yet."""
    pass


@dataclass
class Section:
    title: str
    widgets: list = field(default_factory=list)
    open: bool = True
    key_hint: Optional[str] = None
    gap: int = 4                 # space after the section (open or closed)
    store: Optional[str] = None  # nest this section's captured state under
                                 # cfg[store] (the shell's SIGNAL rack → "signal",
                                 # DESIGN.md §2.4/§7)


# ----- spec-derived preset capture/apply (DESIGN.md §2.1/§7 — the single
# ----- schema authority; kills the KEYS vs save-dict vs _apply_fx drift) -----

_VALUE_WIDGETS = (Slider, Toggle, Cycle, Param)


def walk_spec(spec):
    """Yield (section_or_None, widget) over a panel spec, in spec order."""
    for item in spec:
        if isinstance(item, Section):
            for w in item.widgets:
                yield item, w
        else:
            yield None, item


def nudgeable(w):
    """Can `,`/`.` select this widget, and `-`/`=` move it (DESIGN.md §6.2)?

    Every Slider and Cycle, minus the ones that declare `nudge=False`.

    The rule for opting out is NOT "this widget is a system setting" and it is
    not `save=False` either — Dither Girl's `input` cycle is unsaved and stays
    reachable, because stepping it again steps it back and the picture is the
    only thing that moved. (Mirror used to be the second example here, but
    Mirror is a Toggle and was never in the walk — only Sliders and Cycles
    are.) The rule is narrower: **a control the perform layer cannot take back
    does not belong on a bare key.**

    Output resolution is the one that fails it. It was the 3rd of 15 stops in
    Dither Girl and the 2nd of 23 in Particles — `.` `.` `=`, with no panel
    open — and one press recreated the window at 4K, dropped the frame rate
    through the floor, and left `0` (panic) with no answer, because panic
    restores the mode's look and the window is not in the look. So the show
    goes slow, the escape hatch does nothing, and the only way back is to find
    the control on the edit surface. That is where it belongs: a deliberate
    two-second decision, not two keys from the default selection mid-set.
    """
    return isinstance(w, (Slider, Cycle)) and w.nudge


# ----- quantised sliders (the whole-number sliders) -----

def quantise(w, value):
    """Land a raw value on `w`'s quantum, clamped to the slider's range.

    Exact: this is what a writer that MEANS a particular value calls (the
    panel's drag, `apply_look` below), so that what the panel prints, what the
    look stores, and what the engine consumes are one number. A continuous
    slider (`step is None`) is returned untouched — including its full
    precision, because faking coarseness is the same lie as faking precision.

    The grid is anchored at ZERO, not at `w.lo`, because the engines round the
    raw value itself — `int(round(v))`, `int(v)` — never `v - lo`. Anchoring
    at `lo` disagreed with them on the half-steps: Bits runs 1-4, so
    `round((2.5 - 1))` said 3 while the engine's `round(2.5)` (banker's, like
    ours) rendered 2 — a stored 2.5 would have changed bit depth the moment it
    passed through here. Every stepped slider's `lo` sits on the zero grid
    (asserted in tests), so the anchor changes nothing else.
    """
    step = getattr(w, "step", None)
    if not step:
        return value
    try:
        v = float(value)
    except (TypeError, ValueError):
        return value
    if not math.isfinite(v):
        return value
    n = v / step
    n = math.floor(n) if getattr(w, "snap", "round") == "floor" else round(n)
    return min(max(n * step, w.lo), w.hi)


def display_fmt(w, value):
    """The fmt to draw `value` with: the widget's own, widened to one decimal
    when the value sits OFF a stepped slider's grid. Only an
    `engine_snaps=False` slider can hold such a value — a stored legacy
    fraction that `apply_look` passed through because the engine renders it
    (Scale 45.55 is a real character grid) — and `"{:.0f}"` would print "46"
    while 45.55 is on screen, which is the readout lie quantisation exists to
    kill, pointed the other way. One decimal for a legacy value is honest."""
    fmt = getattr(w, "fmt", ".2f")
    step = getattr(w, "step", None)
    if not step:
        return fmt
    try:
        v = float(value)
    except (TypeError, ValueError):
        return fmt
    if math.isfinite(v) and quantise(w, v) != v:
        return ".1f"
    return fmt


# The perform-layer nudge, as DESIGN.md §6.2 defines it: `-`/`=` move 1/40 of
# a slider's range, `_`/`+` move five of those. Both numbers live in the shell,
# and both are quoted here because `nudge_to` below is derived from them.
NUDGE_DIVISOR = 40.0
NUDGE_BIG = 5.0


def nudge_unreachable(w):
    """True when this control's quantum is COARSER than one nudge press.

    On Bits (1-4) a press is 0.075 against a smallest real step of 1, and on
    Crush (0-8) it is 0.2 — so quantised naively, every press lands back on
    the value it started from and the key does nothing at all. On Scale
    (30-720) a press is 17.25 whole pixels and on Hue 9 whole degrees: those
    move on their own and need no help, which is why the rule below is gated
    rather than applied to every quantised slider.
    """
    step = getattr(w, "step", None)
    return bool(step) and step > (w.hi - w.lo) / NUDGE_DIVISOR


def nudge_to(w, value, current):
    """`quantise`, except that a nudge-sized write to a control the nudge
    cannot otherwise move takes ONE whole step in the direction it asked for.

    Without it, `-`/`=` on Bits and Crush would be silence-on-input, which
    DESIGN.md §4 calls a bug. (Before quantisation they "worked" only in the
    sense that the thirteenth press finally moved the picture.)

    Three conditions, and each one is doing work:

    1. the quantum is coarser than a nudge press (`nudge_unreachable`) — so
       Scale and Hue, which nudge fine, keep plain snapping;
    2. the write would not move the control at all;
    3. the write is nudge-SIZED — no bigger than the `_`/`+` step, which is
       five presses. A larger write is somebody setting a value, and setting
       Bits to 3.47 must land on 3, not ratchet to 4.

    Together those say: the only writes that can reach the second branch are
    the ones the nudge keys make. Every writer that means an exact value — the
    panel drag, `apply_look`, a mode's seeded defaults — also quantises before
    writing, so it never even reaches condition 2.
    """
    snapped = quantise(w, value)
    step = getattr(w, "step", None)
    if not step or current is None or snapped != current \
            or not nudge_unreachable(w):
        return snapped
    try:
        raw, cur = float(value), float(current)
    except (TypeError, ValueError):
        return snapped
    delta = raw - cur
    if not math.isfinite(raw) or delta == 0.0:
        return snapped
    if abs(delta) > (w.hi - w.lo) * NUDGE_BIG / NUDGE_DIVISOR:
        return snapped
    return min(max(cur + math.copysign(step, delta), w.lo), w.hi)


# ----- retired stored values (DESIGN.md §9: the data-loss surface) -----

_ATOMS = (str, int, float, bool, type(None))


def migrate_legacy(spec, cfg: dict) -> dict:
    """Rewrite a look's retired option names into their current spelling.

    Declared per Cycle (`Cycle.legacy`), because a retired option can mean
    more than a rename: Dither Girl's "black-on-white" is now the mono palette
    with Invert ON, so the substitution writes TWO store keys and only the
    retired name knows to. Substitutions land in the same nesting scope the
    cycle serializes into (the SIGNAL rack's block, or the top level).

    The retired name is treated as the complete specification of what it used
    to mean, so it overwrites keys already present — a file that says
    `"palette": "black-on-white", "invert": false` was hand-edited into a
    contradiction, and the palette name is the half that has a defined meaning.

    Returns `cfg` itself when nothing matched (the overwhelmingly common
    case); otherwise a shallow copy, so the caller's stored look is never
    mutated under it.
    """
    hits = []
    for section, w in walk_spec(spec):
        legacy = getattr(w, "legacy", None)
        if not legacy:
            continue
        store = section.store if section is not None else None
        src = cfg.get(store) if store else cfg
        if not isinstance(src, dict):
            continue
        val = src.get(w.store_key)
        if not isinstance(val, _ATOMS):
            continue            # hand-edited to a dict/list: not a legacy name
        sub = legacy.get(val)
        if sub:
            hits.append((store, sub))
    if not hits:
        return cfg
    out = dict(cfg)
    for store, sub in hits:
        if store:
            nested = out.get(store)
            out[store] = dict(nested) if isinstance(nested, dict) else {}
            out[store].update(sub)
        else:
            out.update(sub)
    return out


def capture_look(state, spec) -> dict:
    """Read every save=True widget's value off `state` into a look dict.
    Cycles store the option VALUE (not the index); sections with a `store`
    key nest their block (the SIGNAL rack serializes under "signal")."""
    cfg = {}
    for section, w in walk_spec(spec):
        if not isinstance(w, _VALUE_WIDGETS) or not w.save:
            continue
        target = cfg
        if section is not None and section.store:
            target = cfg.setdefault(section.store, {})
        if isinstance(w, Cycle):
            opts = list(w.options)
            target[w.store_key] = opts[getattr(state, w.attr) % len(opts)]
        elif isinstance(w, Toggle):
            target[w.store_key] = bool(getattr(state, w.attr))
        else:
            target[w.store_key] = float(getattr(state, w.attr))
    return cfg


def apply_look(state, spec, cfg, defaults=None):
    """Write a look onto `state` per the spec's apply flags (DESIGN.md §7):

    - key present in the look → applied (both flags);
    - key absent + apply="reset" → reset to the mode's default, when the mode
      declares one — a reset-flagged widget with no declared default keeps its
      live value (today's asymmetric semantics, encoded as data);
    - key absent + apply="keep" → live value untouched (template-hopping never
      resets your toggles).

    Cycles map a stored VALUE back to its index; unknown values are ignored.
    Engines pick the new state up on the next step() sync.

    Values are coerced DEFENSIVELY and unusable ones are skipped, not raised:
    the store validates a preset file's shape, never its values, so a
    hand-edited (or half-merged) `presets.json` carrying `"contrast": "high"`
    is well-shaped and used to raise ValueError straight out of `float()` —
    at boot, before any window existed. A look that is 90% loadable loads,
    and the caller is told which keys went (returned) so it can say so. NaN
    and infinity are unusable too: they propagate silently through the render
    instead of failing where they were introduced.

    A look naming a RETIRED option value is rewritten first (`migrate_legacy`)
    — old names are accepted forever, so an option can be replaced rather than
    only renamed.

    Returns the list of skipped store keys (empty when the look applied whole).
    """
    defaults = defaults or {}
    skipped = []
    cfg = migrate_legacy(spec, cfg)
    for section, w in walk_spec(spec):
        if not isinstance(w, _VALUE_WIDGETS) or not w.save:
            continue
        src = cfg
        if section is not None and section.store:
            src = cfg.get(section.store) or {}
        key = w.store_key
        if key in src:
            val = src[key]
        elif w.apply == "reset" and key in defaults:
            val = defaults[key]
        else:
            continue
        if isinstance(w, Cycle):
            opts = list(w.options)
            if val in opts:
                setattr(state, w.attr, opts.index(val))
        elif isinstance(w, Toggle):
            setattr(state, w.attr, bool(val))
        else:
            try:
                num = float(val)
            except (TypeError, ValueError):
                skipped.append(key)
                continue
            if not math.isfinite(num):
                skipped.append(key)
                continue
            # A Slider's declared range is the whole truth about what values
            # that control can hold, so a look may not land outside it. It
            # used to be able to: `ascii stream` shipped with scale=30 under a
            # 45-720 slider, and recalling it drew the handle 3 px outside its
            # own track and then snapped 30 -> 45 on the first click anywhere
            # on it — a value the operator could see, could not restore, and
            # destroyed by touching the control. Clamping here (rather than in
            # each mode's built-ins) means no future look can do it either.
            if isinstance(w, Slider):
                # quantise() clamps too — and a look is an exact request, so
                # it goes through the exact path, never `nudge_to`. But only
                # where the engine itself lands on the grid: when it consumes
                # the value continuously (`engine_snaps=False` — Scale under
                # ASCII, Hue), a stored fraction is a real picture the look
                # has always rendered, and re-quantising it here would re-grid
                # a saved look (45.55 rows -> 46 rows moved 45.7% of the
                # pixels). The reseed/`portrait` rule: a stored look is the
                # authority; the control adapts, never the look.
                num = (quantise(w, num) if w.step and w.engine_snaps
                       else min(max(num, w.lo), w.hi))
            setattr(state, w.attr, num)
    return skipped
