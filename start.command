#!/bin/bash
# Double-click this file (Finder) to launch the lighteater live preview.
# It sets up the Python environment on first run, then opens the window.

cd "$(dirname "$0")" || exit 1

# The [person] extra is wanted on this path. Two shipped templates —
# `portrait` and `sigil` — select the `person` matte, so a plain `pip install
# -e .` leaves the one cohort that never opens a terminal one click away from a
# fallback matte and an amber toast. It also carries the opencv-contrib-python
# `<5` bound (see pyproject.toml): installing `.` now and adding mediapipe
# later is exactly how an environment ends up on cv2 5.
#
# Wanted, but never REQUIRED: mediapipe 1.0.0 ships an arm64 macOS wheel only,
# so behind a hard `|| exit 1` this whole launcher died on an Intel Mac — the
# window never opened at all, for exactly the cohort the extra is here to
# protect, on a machine that launched fine with the fallback matte before. It
# degrades instead: the extra, then the engine alone, and only then a failure.
EXTRA=".[person]"

if [ ! -d .venv ]; then
  echo "First run — setting up (this takes a minute)…"
  python3 -m venv .venv || { echo "Could not create venv. Is python3 installed?"; read -r; exit 1; }
  ./.venv/bin/pip install --upgrade pip >/dev/null 2>&1
  if ! ./.venv/bin/pip install -e "$EXTRA"; then
    echo "The person matte would not install here (it needs an arm64 Mac)."
    echo "Setting up without it — portrait and sigil fall back to another matte."
    ./.venv/bin/pip install -e . \
      || { echo "Install failed — see messages above."; read -r; exit 1; }
  fi
elif ! ls .venv/lib/python*/site-packages/mediapipe >/dev/null 2>&1; then
  # A .venv built by an older start.command, which installed the engine alone.
  # Checked by looking for the directory rather than importing it: this runs on
  # every launch, and importing mediapipe costs a second of startup.
  echo "Adding the person matte (one-off, takes a minute)…"
  ./.venv/bin/pip install -e "$EXTRA" \
    || echo "Could not add it — portrait and sigil will fall back to another matte."
fi

echo "Launching… close the window (red X) to quit."
exec ./.venv/bin/python experiments/05-live-webcam/run.py "$@"
