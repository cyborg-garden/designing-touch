# 09 — SIGNAL rack on the GPU

PR #26 measured the CPU rack at 4K (~36–45 ms: scanlines+quantisation 22.6,
scan-drift 10.7, chroma 8.3, dither 2.5) and listed the port order. This
experiment benches the port (`dtouch/rack_gl.py` + `dtouch/shaders/rack/`):
per-stage CPU vs GPU cost at 3840x2160, the GL output path per phase and
quality tier (including THE one uint8 readback of the composed frame), and
full `PhysarumMode.step` fps rack-on, GPU path vs classic CPU path.

    python bench.py
    python bench.py --frames 90

Headless: standalone GL contexts only — no camera, no window, no audio.
Best-of-N is the code's cost (the repo's convention); the printed load
average and medians say what the machine was doing at the time. Writes
`out/rack_gl_<tier>.png` (the last GPU-racked 4K frame) for eyeballing.
