"""Tests for dtouch.dither — ordered (Bayer, blue-noise) and error-diffusion
(Floyd-Steinberg, Riemersma) dithering, plus the sRGB gamma helpers."""
import time

import numpy as np
import pytest

from dtouch.dither import (
    bayer_dither,
    blue_noise_dither,
    floyd_steinberg,
    linear_to_srgb,
    riemersma_dither,
    srgb_to_linear,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ramp(h=32, w=32, channels=None):
    """Horizontal luminance ramp [0, 1] as float32."""
    r = np.tile(np.linspace(0.0, 1.0, w, dtype=np.float32), (h, 1))
    if channels is not None:
        r = np.stack([r] * channels, axis=2)
    return r


def _uniform(h=16, w=16, v=0.5, channels=None):
    arr = np.full((h, w), v, dtype=np.float32)
    if channels is not None:
        arr = np.stack([arr] * channels, axis=2)
    return arr


# ---------------------------------------------------------------------------
# Bayer dither
# ---------------------------------------------------------------------------

class TestBayerDither:
    def test_output_shape_2d(self):
        img = _ramp(32, 32)
        out = bayer_dither(img, bits=2)
        assert out.shape == img.shape

    def test_output_shape_3d(self):
        img = _ramp(32, 32, channels=3)
        out = bayer_dither(img, bits=2)
        assert out.shape == img.shape

    def test_output_dtype(self):
        out = bayer_dither(_ramp(), bits=2)
        assert out.dtype == np.float32

    def test_output_range(self):
        img = _ramp(32, 32)
        out = bayer_dither(img, bits=2)
        assert out.min() >= 0.0 and out.max() <= 1.0

    def test_quantised_levels(self):
        """With bits=2, output should have at most 4 distinct values (0, 1/3, 2/3, 1)."""
        img = _ramp(32, 64)
        out = bayer_dither(img, bits=2)
        unique = np.unique(np.round(out, 5))
        assert len(unique) <= 4

    def test_uniform_black_stays_black(self):
        img = _uniform(v=0.0)
        out = bayer_dither(img, bits=2)
        assert np.allclose(out, 0.0)

    def test_uniform_white_stays_white(self):
        img = _uniform(v=1.0)
        out = bayer_dither(img, bits=2)
        assert np.allclose(out, 1.0)

    def test_higher_bits_finer_levels(self):
        img = _ramp(32, 64)
        out2 = bayer_dither(img, bits=2)
        out4 = bayer_dither(img, bits=4)
        levels2 = len(np.unique(np.round(out2, 5)))
        levels4 = len(np.unique(np.round(out4, 5)))
        assert levels4 > levels2

    def test_different_matrix_sizes(self):
        img = _ramp(32, 32)
        for sz in (2, 4, 8):
            out = bayer_dither(img, bits=2, matrix_size=sz)
            assert out.shape == img.shape
            assert out.min() >= 0.0 and out.max() <= 1.0

    def test_invalid_bits_raises(self):
        with pytest.raises(ValueError):
            bayer_dither(_ramp(), bits=0)
        with pytest.raises(ValueError):
            bayer_dither(_ramp(), bits=9)

    def test_invalid_matrix_size_raises(self):
        with pytest.raises(ValueError):
            bayer_dither(_ramp(), bits=2, matrix_size=3)

    def test_average_of_uniform_mid_grey(self):
        """With gamma off (historical behaviour), dithered mid-grey should
        average close to 0.5 over a large area."""
        img = _uniform(h=64, w=64, v=0.5)
        out = bayer_dither(img, bits=1, gamma=False)   # only 0 and 1 possible
        assert abs(out.mean() - 0.5) < 0.1


# ---------------------------------------------------------------------------
# Floyd-Steinberg dither
# ---------------------------------------------------------------------------

class TestFloydSteinberg:
    def test_output_shape_2d(self):
        img = _ramp(16, 32)
        out = floyd_steinberg(img, bits=2)
        assert out.shape == img.shape

    def test_output_shape_3d(self):
        img = _ramp(16, 32, channels=3)
        out = floyd_steinberg(img, bits=2)
        assert out.shape == img.shape

    def test_output_dtype(self):
        out = floyd_steinberg(_ramp(8, 16), bits=2)
        assert out.dtype == np.float32

    def test_output_range(self):
        img = _ramp(16, 32)
        out = floyd_steinberg(img, bits=2)
        assert out.min() >= 0.0 and out.max() <= 1.0

    def test_uniform_black_stays_black(self):
        img = _uniform(8, 16, v=0.0)
        out = floyd_steinberg(img, bits=2)
        assert np.allclose(out, 0.0)

    def test_uniform_white_stays_white(self):
        img = _uniform(8, 16, v=1.0)
        out = floyd_steinberg(img, bits=2)
        assert np.allclose(out, 1.0)

    def test_average_close_to_input_grey(self):
        """Error diffusion preserves average luminance: mean of output ≈ mean
        of input (in the working domain, so gamma off here)."""
        img = _uniform(16, 32, v=0.5)
        out = floyd_steinberg(img, bits=1, gamma=False)  # 0/1 only
        assert abs(out.mean() - 0.5) < 0.1

    def test_quantised_levels(self):
        """With bits=2, output values come from {0, 1/3, 2/3, 1}."""
        img = _ramp(8, 16)
        out = floyd_steinberg(img, bits=2)
        unique = np.unique(np.round(out, 5))
        assert len(unique) <= 4

    def test_does_not_mutate_input(self):
        img = _ramp(8, 16)
        orig = img.copy()
        floyd_steinberg(img, bits=2)
        assert np.array_equal(img, orig)

    def test_invalid_bits_raises(self):
        with pytest.raises(ValueError):
            floyd_steinberg(_ramp(4, 4), bits=0)

    def test_3d_per_channel_independence(self):
        """Channels should be dithered independently — red ramp vs blue ramp."""
        h, w = 8, 16
        img = np.zeros((h, w, 3), np.float32)
        img[:, :, 0] = np.linspace(0.0, 1.0, w)   # red ramp
        img[:, :, 2] = np.linspace(1.0, 0.0, w)   # blue ramp (reversed)
        out = floyd_steinberg(img, bits=2)
        # first and last pixel of red/blue channels should differ
        assert not np.allclose(out[:, 0, 0], out[:, -1, 0])
        assert not np.allclose(out[:, 0, 2], out[:, -1, 2])


# ---------------------------------------------------------------------------
# sRGB gamma correction
# ---------------------------------------------------------------------------

class TestGamma:
    def test_roundtrip(self):
        x = np.linspace(0.0, 1.0, 1024, dtype=np.float32)
        assert np.allclose(linear_to_srgb(srgb_to_linear(x)), x, atol=1e-5)

    def test_known_values(self):
        assert srgb_to_linear(np.float32(0.0)) == 0.0
        assert abs(float(srgb_to_linear(np.float32(1.0))) - 1.0) < 1e-6
        # sRGB mid-grey is ~21.4% linear light — the whole point of dithering
        # in linear light rather than on the encoded values.
        assert abs(float(srgb_to_linear(np.float32(0.5))) - 0.214) < 2e-3

    def test_endpoints_preserved_with_gamma(self):
        for fn in (bayer_dither, floyd_steinberg,
                   blue_noise_dither, riemersma_dither):
            assert np.allclose(fn(_uniform(v=0.0), bits=1), 0.0)
            assert np.allclose(fn(_uniform(v=1.0), bits=1), 1.0)

    def test_gamma_default_on_changes_output(self):
        img = _uniform(h=64, w=64, v=0.5)
        assert not np.array_equal(bayer_dither(img, bits=1),
                                  bayer_dither(img, bits=1, gamma=False))

    def test_gamma_off_matches_historical_bayer(self):
        """gamma=False + invert=False must reproduce the original pipeline:
        floor(img * levels + threshold) / levels."""
        img = _ramp(32, 32)
        from dtouch.dither import _bayer_matrix
        mat = _bayer_matrix(4)
        threshold = np.tile(mat, (8, 8))[:32, :32]
        expected = np.clip(np.floor(img * 3.0 + threshold) / 3.0, 0.0, 1.0)
        out = bayer_dither(img, bits=2, invert=False, gamma=False)
        assert np.allclose(out, expected)

    def test_mid_grey_density_gamma_on(self):
        """sRGB 0.5 is ~21.4% linear light: 1-bit dithering in linear light
        should light ~21% of pixels, not 50%."""
        img = _uniform(h=64, w=64, v=0.5)
        out = bayer_dither(img, bits=1, invert=False, gamma=True)
        white = float((out > 0.5).mean())
        assert abs(white - 0.216) < 0.05

    def test_mid_grey_density_gamma_off(self):
        img = _uniform(h=64, w=64, v=0.5)
        out = bayer_dither(img, bits=1, invert=False, gamma=False)
        white = float((out > 0.5).mean())
        assert abs(white - 0.5) < 0.05

    def test_mid_grey_density_fs_gamma_on(self):
        """Error diffusion preserves the linear-light mean tightly."""
        img = _uniform(h=32, w=32, v=0.5)
        out = floyd_steinberg(img, bits=1, gamma=True)
        white = float((out > 0.5).mean())
        assert abs(white - 0.216) < 0.03


# ---------------------------------------------------------------------------
# Bayer bias-flip (invert)
# ---------------------------------------------------------------------------

class TestBayerInvert:
    def test_invert_raises_dark_frame_density(self):
        """The standard floor comparison rounds dark values down; the flipped
        comparison rounds them up — density must increase."""
        img = _uniform(h=64, w=64, v=0.2)
        d_std = float(bayer_dither(img, bits=1, invert=False, gamma=False).mean())
        d_inv = float(bayer_dither(img, bits=1, invert=True, gamma=False).mean())
        assert d_inv > d_std

    def test_auto_flips_for_dark_frame(self):
        img = _uniform(h=64, w=64, v=0.1)
        auto = bayer_dither(img, bits=1, invert="auto")
        assert np.array_equal(auto, bayer_dither(img, bits=1, invert=True))

    def test_auto_no_flip_for_light_frame(self):
        img = _uniform(h=64, w=64, v=0.9)
        auto = bayer_dither(img, bits=1, invert="auto")
        assert np.array_equal(auto, bayer_dither(img, bits=1, invert=False))

    def test_invert_preserves_endpoints(self):
        assert np.allclose(bayer_dither(_uniform(v=0.0), bits=2, invert=True), 0.0)
        assert np.allclose(bayer_dither(_uniform(v=1.0), bits=2, invert=True), 1.0)

    def test_invert_mirrors_pattern(self):
        """Inverting the input and the comparison must mirror the output."""
        img = _uniform(h=16, w=16, v=0.3)
        std = bayer_dither(img, bits=1, invert=False, gamma=False)
        inv = bayer_dither(1.0 - img, bits=1, invert=True, gamma=False)
        assert np.allclose(std, 1.0 - inv)


# ---------------------------------------------------------------------------
# Blue-noise ordered dithering
# ---------------------------------------------------------------------------

class TestBlueNoise:
    def test_asset_loads(self):
        from dtouch.dither import _blue_noise_matrix
        mat = _blue_noise_matrix()
        assert mat.shape == (64, 64)
        assert mat.dtype == np.float32

    def test_asset_threshold_histogram_uniform(self):
        """The texture is a rank matrix: every rank k/4096 exactly once."""
        from dtouch.dither import _blue_noise_matrix
        mat = _blue_noise_matrix()
        ranks = np.sort(mat.flatten()) * 4096.0
        assert np.allclose(ranks, np.arange(4096), atol=1e-3)

    def test_output_shape_2d_and_3d(self):
        for img in (_ramp(32, 32), _ramp(32, 32, channels=3)):
            out = blue_noise_dither(img, bits=2)
            assert out.shape == img.shape
            assert out.dtype == np.float32

    def test_output_binary_at_1_bit(self):
        out = blue_noise_dither(_ramp(64, 64), bits=1)
        assert set(np.unique(out)) <= {0.0, 1.0}

    def test_mid_grey_density(self):
        img = _uniform(h=64, w=64, v=0.5)
        out = blue_noise_dither(img, bits=1, invert=False, gamma=False)
        assert abs(out.mean() - 0.5) < 0.05

    def test_differs_from_bayer(self):
        img = _uniform(h=64, w=64, v=0.5)
        assert not np.array_equal(blue_noise_dither(img, bits=1),
                                  bayer_dither(img, bits=1))

    def test_less_low_frequency_energy_than_white_noise(self):
        """The defining property: a 50% blue-noise pattern has almost no
        low-frequency energy; a white-noise threshold pattern has lots."""
        from dtouch.dither import _blue_noise_matrix
        blue = (_blue_noise_matrix() < 0.5).astype(np.float64)
        rng = np.random.default_rng(0)
        white = (rng.random((64, 64)) < 0.5).astype(np.float64)

        def low_ratio(p):
            f = np.abs(np.fft.fftshift(np.fft.fft2(p - p.mean()))) ** 2
            yy, xx = np.indices(p.shape)
            r = np.hypot(yy - 32, xx - 32)
            return f[(r > 0) & (r <= 8)].mean() / f[r > 8].mean()

        assert low_ratio(blue) < 0.5 * low_ratio(white)


# ---------------------------------------------------------------------------
# Riemersma (Hilbert-curve) dithering
# ---------------------------------------------------------------------------

class TestRiemersma:
    def test_output_shape_2d(self):
        img = _ramp(16, 32)
        out = riemersma_dither(img, bits=2)
        assert out.shape == img.shape
        assert out.dtype == np.float32

    def test_output_shape_3d(self):
        img = _ramp(16, 32, channels=3)
        out = riemersma_dither(img, bits=2)
        assert out.shape == img.shape

    def test_output_binary_at_1_bit(self):
        out = riemersma_dither(_ramp(32, 32), bits=1)
        assert set(np.unique(out)) <= {0.0, 1.0}

    def test_uniform_black_stays_black(self):
        assert np.allclose(riemersma_dither(_uniform(8, 16, v=0.0), bits=2), 0.0)

    def test_uniform_white_stays_white(self):
        assert np.allclose(riemersma_dither(_uniform(8, 16, v=1.0), bits=2), 1.0)

    def test_density_tracks_input(self):
        """Error diffusion preserves the working-domain mean."""
        for v in (0.2, 0.5, 0.8):
            img = _uniform(h=32, w=32, v=v)
            out = riemersma_dither(img, bits=1, gamma=False)
            assert abs(out.mean() - v) < 0.05

    def test_differs_from_floyd_steinberg(self):
        img = _ramp(16, 32)
        assert not np.array_equal(riemersma_dither(img, bits=1, gamma=False),
                                  floyd_steinberg(img, bits=1, gamma=False))

    def test_does_not_mutate_input(self):
        img = _ramp(8, 16)
        orig = img.copy()
        riemersma_dither(img, bits=2)
        assert np.array_equal(img, orig)

    def test_invalid_args_raise(self):
        with pytest.raises(ValueError):
            riemersma_dither(_ramp(4, 4), bits=0)
        with pytest.raises(ValueError):
            riemersma_dither(_ramp(4, 4), ratio=0.0)
        with pytest.raises(ValueError):
            riemersma_dither(_ramp(4, 4), history=1)

    def test_hilbert_order_visits_every_pixel_once(self):
        from dtouch.dither import _hilbert_order
        for h, w in ((4, 4), (5, 7), (16, 16), (3, 9)):
            ys, xs = _hilbert_order(h, w)
            assert len(ys) == h * w
            flat = np.unique(ys * w + xs)
            assert len(flat) == h * w

    def test_hilbert_order_is_spatially_local(self):
        """Consecutive pixels on the curve are grid neighbours (the Hilbert
        property, for power-of-two squares)."""
        from dtouch.dither import _hilbert_order
        ys, xs = _hilbert_order(16, 16)
        step = np.abs(np.diff(ys.astype(int))) + np.abs(np.diff(xs.astype(int)))
        assert step.max() == 1


# ---------------------------------------------------------------------------
# Performance smoke — ordered paths must stay in the real-time budget
# ---------------------------------------------------------------------------

class TestPerformance:
    @staticmethod
    def _best_ms(fn, img, runs=5):
        best = np.inf
        for _ in range(runs):
            t0 = time.perf_counter()
            fn(img)
            best = min(best, time.perf_counter() - t0)
        return best * 1000.0

    def test_ordered_dither_720p_within_budget(self):
        """Bayer and blue-noise (gamma on, auto invert) on a 720p colour
        frame. Budget is ~6 ms in the live post-FX chain; assert a loose
        2× margin so CI noise can't flake the suite."""
        rng = np.random.default_rng(0)
        img = rng.random((720, 1280, 3), dtype=np.float32)
        for fn in (lambda a: bayer_dither(a, bits=3),
                   lambda a: blue_noise_dither(a, bits=3)):
            fn(img)   # warm caches / LUTs
            assert self._best_ms(fn, img) < 12.0
