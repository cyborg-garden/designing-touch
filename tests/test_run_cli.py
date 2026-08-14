"""Boot-precedence pins for the launcher (experiments/05-live-webcam/run.py).

DESIGN.md §3: CLI flags mean "boot into". The launcher's own docstring
advertises `--matte person`, so an engine flag must reach the constructed mode
instead of being swallowed by the state.json resume path (which builds the mode
with DEFAULT args). Precedence: --mode > --still > engine flags > state.json.
"""
import importlib.util
import os
import sys

import pytest

RUN_PY = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "experiments", "05-live-webcam", "run.py")


def _load_run():
    spec = importlib.util.spec_from_file_location("dtouch_run_cli", RUN_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


run = _load_run()


class FakeMode:
    """Records the args a boot would have constructed the mode with."""

    def __init__(self, **kw):
        self.kw = kw


class FakeHost:
    made = None

    def __init__(self, mode, **kw):
        self.mode, self.kw = mode, kw
        FakeHost.made = self

    def run(self):
        self.ran = True


@pytest.fixture
def launch(monkeypatch):
    """main() with the heavy bits stubbed; returns the FakeHost it built."""
    monkeypatch.setattr(run, "Host", FakeHost)
    monkeypatch.setattr(run, "ParticlesMode", FakeMode)
    monkeypatch.setattr(run, "DitherGirlMode", FakeMode)
    monkeypatch.setattr(run, "live", lambda **kw: None)

    def go(*argv):
        FakeHost.made = None
        monkeypatch.setattr(sys, "argv", ["run.py", *argv])
        run.main()
        return FakeHost.made

    return go


# ---------- engine flags reach the mode ----------

def test_matte_flag_reaches_the_mode_without_a_mode_flag(launch):
    """`python run.py --matte person` — the launcher's own documented example."""
    host = launch("--matte", "person")
    assert host.mode.kw["matte"] == "person"


def test_grid_and_particles_flags_reach_the_mode(launch):
    host = launch("--grid", "200x100", "--particles", "1234")
    assert host.mode.kw["grid"] == (200, 100)
    assert host.mode.kw["n"] == 1234


def test_flock_and_glitch_still_boot_flow(launch):
    host = launch("--flock", "--glitch")
    assert host.mode.kw["flock"] is True
    assert host.mode.kw["glitch"] is True


# ---------- state.json is only consulted with no mode-implying flag ----------

def test_engine_flags_skip_the_state_json_resume(launch):
    """A constructed mode (not None) means the shell never resolves from
    state.json — the resume path is what dropped the flags."""
    assert launch("--matte", "motion").mode is not None
    assert run.boot_mode_name(None, None, particles_flags=True) == "flow"


def test_no_flags_at_all_still_resumes_from_state_json(launch):
    """Defaults must NOT count as 'given' — bare `run.py` still resumes."""
    assert launch().mode is None
    assert run.boot_mode_name(None, None, particles_flags=False) is None


def test_engine_flags_given_reports_only_non_default_values():
    ap_args = type("A", (), dict(run.ENGINE_DEFAULTS))()
    assert run.engine_flags_given(ap_args) == []
    ap_args.matte = "person"
    ap_args.flock = True
    assert run.engine_flags_given(ap_args) == ["--matte", "--flock"]


# ---------- precedence: --mode > --still > engine flags ----------

def test_still_wins_over_an_engine_flag_and_says_so(launch, capsys, tmp_path):
    still = str(tmp_path / "x.jpg")
    host = launch("--still", still, "--flock")
    out = capsys.readouterr().out
    assert "--flock" in out and "ignored" in out and "dithergirl" in out
    assert host.mode.kw == {"still": True}          # DitherGirlMode, not flow
    assert host.kw["still"] == still


def test_mode_flag_wins_over_still_and_engine_flags(launch, tmp_path):
    still = str(tmp_path / "x.jpg")
    host = launch("--mode", "flow", "--still", still, "--matte", "person")
    assert host.mode.kw["matte"] == "person"


def test_explicit_dithergirl_mode_warns_about_dropped_engine_flags(launch,
                                                                  capsys):
    launch("--mode", "dithergirl", "--matte", "person")
    assert "ignored" in capsys.readouterr().out
