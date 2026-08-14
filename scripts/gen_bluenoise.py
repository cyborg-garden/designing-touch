"""Generate dtouch/assets/bluenoise64.npy — a 64×64 blue-noise threshold map.

Void-and-cluster method (Ulichney 1993), independently reimplemented from the
published description. Run once, offline:

    python scripts/gen_bluenoise.py

Output: 64×64 float32, values rank/4096 ∈ [0, 1 - 1/4096] — every rank
appears exactly once, so the threshold histogram is uniform by construction.
float32 was chosen over uint8 so the map drops straight into the ordered
dither core (which expects normalised thresholds) with no rescale; 16 KB.

Method
------
"Energy" at a pixel is the sum of toroidally wrapped Gaussians (σ=1.5)
centred on every minority pixel. The tightest cluster is the minority pixel
with the highest energy; the largest void is the empty pixel with the lowest.

1. Prototype: start from a random ~10% pattern; repeatedly move the tightest
   cluster pixel into the largest void until they coincide (stable).
2. Rank 0 … ones-1: from the prototype, repeatedly remove the tightest
   cluster, assigning descending ranks.
3. Rank ones … N-1: from the prototype, repeatedly insert into the largest
   void. (On a torus the energy field of the zeros is a constant minus the
   energy field of the ones, so "largest void of ones" ≡ "tightest cluster
   of zeros" — phases 2 and 3 of the published method collapse into one rule.)

The energy field is maintained incrementally: inserting/removing a pixel
adds/subtracts one rolled copy of the wrapped kernel.
"""
from __future__ import annotations

import os

import numpy as np

SIZE = 64
SIGMA = 1.5
SEED = 20260813
INITIAL_FRACTION = 0.1

OUT_PATH = os.path.join(os.path.dirname(__file__), os.pardir,
                        "dtouch", "assets", "bluenoise64.npy")


def _wrapped_kernel(size: int, sigma: float) -> np.ndarray:
    """Toroidally wrapped Gaussian centred at (0, 0)."""
    ax = np.arange(size)
    d = np.minimum(ax, size - ax).astype(np.float64)
    return np.exp(-(d[None, :] ** 2 + d[:, None] ** 2) / (2.0 * sigma * sigma))


def _energy_of(pattern: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Energy field of a binary pattern (FFT convolution, toroidal wrap)."""
    return np.real(np.fft.ifft2(np.fft.fft2(pattern) * np.fft.fft2(kernel)))


def _apply(energy: np.ndarray, kernel: np.ndarray, y: int, x: int, sign: float):
    energy += sign * np.roll(np.roll(kernel, y, axis=0), x, axis=1)


def _tightest_cluster(pattern, energy):
    """Minority (1) pixel with the highest energy."""
    masked = np.where(pattern, energy, -np.inf)
    return np.unravel_index(np.argmax(masked), masked.shape)


def _largest_void(pattern, energy):
    """Empty (0) pixel with the lowest energy."""
    masked = np.where(pattern, np.inf, energy)
    return np.unravel_index(np.argmin(masked), masked.shape)


def _prototype(rng: np.random.Generator, kernel: np.ndarray) -> np.ndarray:
    """Phase 1 relaxation: swap tightest cluster into largest void until stable."""
    n = SIZE * SIZE
    ones = int(n * INITIAL_FRACTION)
    flat = np.zeros(n, dtype=bool)
    flat[rng.choice(n, size=ones, replace=False)] = True
    pattern = flat.reshape(SIZE, SIZE)
    energy = _energy_of(pattern, kernel)
    for _ in range(10 * n):   # generous guard; converges long before this
        cy, cx = _tightest_cluster(pattern, energy)
        pattern[cy, cx] = False
        _apply(energy, kernel, cy, cx, -1.0)
        vy, vx = _largest_void(pattern, energy)
        pattern[vy, vx] = True
        _apply(energy, kernel, vy, vx, +1.0)
        if (vy, vx) == (cy, cx):   # removing it re-created the largest void
            break
    else:
        raise RuntimeError("void-and-cluster relaxation did not converge")
    return pattern


def generate() -> np.ndarray:
    rng = np.random.default_rng(SEED)
    kernel = _wrapped_kernel(SIZE, SIGMA)
    proto = _prototype(rng, kernel)
    n = SIZE * SIZE
    ones = int(proto.sum())
    rank = np.full((SIZE, SIZE), -1, dtype=np.int32)

    # Phase 2: rank the prototype's minority pixels by removing the tightest
    # cluster, descending.
    pattern = proto.copy()
    energy = _energy_of(pattern, kernel)
    for r in range(ones - 1, -1, -1):
        cy, cx = _tightest_cluster(pattern, energy)
        pattern[cy, cx] = False
        _apply(energy, kernel, cy, cx, -1.0)
        rank[cy, cx] = r

    # Phase 3 (and, on a torus, phase 4 — see module docstring): fill the
    # largest void, ascending, until every pixel is ranked.
    pattern = proto.copy()
    energy = _energy_of(pattern, kernel)
    for r in range(ones, n):
        vy, vx = _largest_void(pattern, energy)
        pattern[vy, vx] = True
        _apply(energy, kernel, vy, vx, +1.0)
        rank[vy, vx] = r

    assert rank.min() == 0 and rank.max() == n - 1
    assert len(np.unique(rank)) == n, "ranks must be a permutation"
    return (rank.astype(np.float32) / n)


def verify(tex: np.ndarray) -> None:
    """Report uniformity + spectral (clumping) checks."""
    n = tex.size
    ranks = np.sort(tex.flatten()) * n
    assert np.array_equal(ranks, np.arange(n)), "histogram not uniform"
    print(f"histogram: uniform — every rank 0..{n - 1} appears exactly once")

    # Spectral check on the 50% binary pattern: blue noise should have far
    # less low-frequency energy than a white-noise threshold map.
    half = (tex < 0.5).astype(np.float64)
    rng = np.random.default_rng(0)
    white = (rng.random(tex.shape) < 0.5).astype(np.float64)

    def low_freq_power(p):
        f = np.abs(np.fft.fftshift(np.fft.fft2(p - p.mean()))) ** 2
        cy, cx = np.array(p.shape) // 2
        yy, xx = np.indices(p.shape)
        r = np.hypot(yy - cy, xx - cx)
        low = f[(r > 0) & (r <= 8)].mean()
        high = f[r > 8].mean()
        return low, high

    lo_b, hi_b = low_freq_power(half)
    lo_w, hi_w = low_freq_power(white)
    print(f"blue  noise 50% pattern: low-freq power {lo_b:9.2f}, "
          f"high-freq {hi_b:9.2f}, ratio {lo_b / hi_b:.4f}")
    print(f"white noise 50% pattern: low-freq power {lo_w:9.2f}, "
          f"high-freq {hi_w:9.2f}, ratio {lo_w / hi_w:.4f}")
    assert lo_b / hi_b < 0.25 * (lo_w / hi_w), "low-frequency energy too high"
    print("spectral: OK — low-frequency energy well below white noise")


if __name__ == "__main__":
    tex = generate()
    verify(tex)
    out = os.path.abspath(OUT_PATH)
    np.save(out, tex)
    print(f"saved {out} ({tex.dtype}, {tex.shape}, {tex.nbytes} bytes)")
