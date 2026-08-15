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

    Returns the list of skipped store keys (empty when the look applied whole).
    """
    defaults = defaults or {}
    skipped = []
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
                num = min(max(num, w.lo), w.hi)
            setattr(state, w.attr, num)
    return skipped
