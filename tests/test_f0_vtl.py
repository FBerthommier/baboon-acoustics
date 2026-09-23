"""Sanity checks for F0 estimation and VTL estimation."""

import numpy as np

from tests.context import WAHOO4, WahooTestCase
from baboon_acoustics import (
    estimate_f0_from_gn,
    estimate_f0_praat,
    reby_mccomb_vtl,
)

FS = 44100


class TestF0Autocorrelation(WahooTestCase):

    def test_recovers_synthetic_pulse_rate(self):
        # Impulse train at 150 Hz: one spike every fs/150 samples.
        t = np.arange(FS) / FS
        train = np.zeros(FS)
        train[:: int(FS / 150.0)] = 1.0
        train *= np.hanning(FS)  # taper to avoid edge bias
        f0 = estimate_f0_from_gn(train, FS)
        self.assertAlmostEqual(f0, 150.0, delta=2.0)

    def test_noise_returns_nan(self):
        rng = np.random.default_rng(2)
        self.assertTrue(np.isnan(estimate_f0_from_gn(rng.standard_normal(FS), FS)))

    def test_constant_returns_nan(self):
        self.assertTrue(np.isnan(estimate_f0_from_gn(np.ones(FS), FS)))


class TestF0Praat(WahooTestCase):

    def test_wahoo_contour_in_search_range(self):
        times, f0, strength = estimate_f0_praat(WAHOO4)
        self.assertEqual(len(times), len(f0))
        self.assertTrue(np.all(f0 > 0))
        self.assertTrue(np.all(f0 < 800.0))
        # The wahoo is a high-pitched call: expect several hundred Hz.
        self.assertGreater(f0.mean(), 200.0)
        self.assertTrue(np.all((strength >= 0) & (strength <= 1)))


class TestVtl(WahooTestCase):

    def test_recovers_known_tube_length(self):
        # Quarter-wave resonances of a 17.5 cm tube at c = 350 m/s.
        formants = np.array([[500.0, 1500.0, 2500.0, 3500.0]]).T
        vtl = reby_mccomb_vtl(formants)
        self.assertGreater(vtl, 0.0)
        self.assertAlmostEqual(vtl, 17.5, delta=0.1)

    def test_ignores_nan_entries(self):
        perfect = np.array([[500.0, 1500.0, 2500.0]]).T
        with_nan = np.array([[500.0, np.nan], [1500.0, np.nan], [2500.0, np.nan]])
        self.assertAlmostEqual(
            reby_mccomb_vtl(perfect), reby_mccomb_vtl(with_nan), places=6)

    def test_too_few_values_returns_nan(self):
        self.assertTrue(np.isnan(reby_mccomb_vtl(np.array([[500.0]]))))
        self.assertTrue(np.isnan(reby_mccomb_vtl(
            np.array([[np.nan, np.nan], [np.nan, np.nan]]))))

    def test_wahoo_vtl_is_anatomically_plausible(self):
        # Chacma baboon vocal tract: expect a positive VTL of a few tens
        # of cm at most; grossly larger values would flag mis-tracking.
        from baboon_acoustics import extract_praat_formants
        _, formants = extract_praat_formants(WAHOO4, n_formants=4)
        vtl = reby_mccomb_vtl(formants)
        self.assertGreater(vtl, 5.0)
        self.assertLess(vtl, 50.0)


if __name__ == "__main__":
    unittest.main()
