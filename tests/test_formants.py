"""Sanity checks for formant extraction (Praat/Burg and IAIF)."""

import numpy as np

from tests.context import WAHOO4, WahooTestCase
from baboon_acoustics import (
    extract_praat_formants,
    get_frame_formants,
    interpolate_formants,
    lpc_levinson,
    lpc_spectrum_db,
)

FS = 44100


class TestPraatFormants(WahooTestCase):

    def test_shape_and_ranges(self):
        times, formants = extract_praat_formants(WAHOO4, n_formants=4)
        self.assertEqual(formants.shape, (4, len(times)))
        # Formants live between the pitch range and the analysis ceiling.
        self.assertGreaterEqual(float(np.nanmin(formants)), 100.0)
        self.assertLessEqual(float(np.nanmax(formants)), 5500.0)
        # F1 below F2 on the large majority of frames.
        f1, f2 = formants[0], formants[1]
        valid = ~np.isnan(f1) & ~np.isnan(f2)
        self.assertGreater(float(np.mean(f1[valid] < f2[valid])), 0.9)

    def test_interpolation_fills_grid_and_nan_outside(self):
        times, formants = extract_praat_formants(WAHOO4, n_formants=4)
        grid = np.linspace(times[0], times[-1], 25)
        interp = interpolate_formants(times, formants, grid)
        self.assertEqual(interp.shape, (4, 25))
        # Inside the source range most values must be defined.
        self.assertGreater(np.sum(~np.isnan(interp)), 0.5 * interp.size)
        # Outside the source range values are NaN.
        outside = interpolate_formants(
            times, formants, np.array([times[0] - 1.0]))
        self.assertTrue(np.isnan(outside).all())


class TestLpc(WahooTestCase):

    def test_lpc_coefficients_shape(self):
        rng = np.random.default_rng(1)
        a, energy = lpc_levinson(rng.standard_normal(2048), order=10)
        self.assertEqual(len(a), 11)
        self.assertEqual(a[0], 1.0)
        self.assertGreater(energy, 0.0)

    def test_lpc_spectrum_finds_resonance(self):
        # A pure tone at 500 Hz must produce an envelope peak near 500 Hz.
        tone = np.sin(2 * np.pi * 500 * np.arange(4096) / FS)
        a, energy = lpc_levinson(tone, order=8)
        freqs, env_db = lpc_spectrum_db(a, energy, n_fft=4096, fs=FS)
        peak = freqs[np.argmax(env_db)]
        self.assertAlmostEqual(peak, 500.0, delta=20.0)


class TestIaifFormants(WahooTestCase):

    def test_frame_formants_count_and_range(self):
        frame = self.x[:4096] * np.hanning(4096)
        formants = get_frame_formants(frame, self.fs, n_formants=3)
        self.assertEqual(formants.shape, (3,))
        self.assertTrue(np.all(formants >= 0))
        self.assertTrue(np.all(formants <= 3500.0))
        # Ascending order for the detected values.
        detected = formants[~np.isnan(formants)]
        self.assertTrue(np.all(np.diff(detected) > 0))

    def test_fewer_peaks_than_requested_pads_with_nan(self):
        # A near-pure tone has a single strong resonance below 3.5 kHz.
        tone = np.sin(2 * np.pi * 500 * np.arange(4096) / FS) * np.hanning(4096)
        formants = get_frame_formants(
            tone, FS, n_formants=4,
            min_prominence=40.0, min_dist_hz=800.0)
        self.assertTrue(np.any(np.isnan(formants)))


if __name__ == "__main__":
    unittest.main()
