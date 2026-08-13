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


@dataclass
class Cycle:
    label: str
    attr: str                    # index attribute on the state object
    options: Sequence[Any]       # displayed value is options[index]
    key: Optional[str] = None    # hit-test key; defaults to label
    save: bool = True
    apply: str = "reset"
    gap: int = 2

    @property
    def hit_key(self):
        return self.key if self.key is not None else self.label


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
