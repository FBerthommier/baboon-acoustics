"""Sanity checks for glottal-source extraction."""

import numpy as np

from tests.context import WAHOO4, WahooTestCase
from baboon_acoustics import (
    estimate_f0_from_gn,
    extract_glottal_pulse,
    extract_glottal_train,
    load_audio,
    normalize_amplitude,
)


class TestGlottalPulse(WahooTestCase):

    @classmethod
    def setUpClass(cls):
        # Native amplitudes, as recommended for glottal analysis.
        cls.x, cls.fs = load_audio(WAHOO4, normalize=False)

    def test_pulse_is_one_period_long(self):
        t_pulse, pulse = extract_glottal_pulse(self.x, self.fs, 0.26, 0.34)
        self.assertIsNotNone(pulse)
        period_s = len(pulse) / self.fs
        # Period consistent with the 200-500 Hz F0 of the wahoo.
        self.assertGreater(period_s, 1.0 / 500.0)
        self.assertLess(period_s, 1.0 / 150.0)
        self.assertAlmostEqual(t_pulse[0], 0.0, places=9)

    def test_short_segment_returns_none(self):
        t_pulse, pulse = extract_glottal_pulse(self.x, self.fs, 0.0, 0.01)
        self.assertIsNone(pulse)
        self.assertIsNone(t_pulse)


class TestGlottalTrain(WahooTestCase):

    @classmethod
    def setUpClass(cls):
        cls.x, cls.fs = load_audio(WAHOO4, normalize=False)

    def test_train_is_trimmed_but_complete(self):
        t_train, train = extract_glottal_train(self.x, self.fs)
        self.assertGreater(len(train), 0.5 * len(self.x))
        self.assertLess(len(train), len(self.x))
        self.assertTrue(np.all(np.isfinite(train)))

    def test_train_yields_plausible_f0(self):
        _, train = extract_glottal_train(self.x, self.fs)
        f0 = estimate_f0_from_gn(train, self.fs)
        self.assertFalse(np.isnan(f0))
        self.assertGreater(f0, 100.0)
        self.assertLess(f0, 600.0)


class TestNormalizeAmplitude(WahooTestCase):

    def test_flattens_slow_envelope(self):
        fs = self.fs
        t = np.arange(4096) / fs
        # 300 Hz oscillation with amplitude growing from 0.2 to 1.
        drifting = np.sin(2 * np.pi * 300 * t) * np.linspace(0.2, 1.0, 4096)
        flat = normalize_amplitude(drifting, fs)
        first = flat[512:1024]
        last = flat[-512:]
        # Before normalization the last block is ~5x larger; after, the
        # two blocks must have comparable amplitudes.
        ratio_before = np.std(drifting[-512:]) / np.std(drifting[512:1024])
        ratio_after = np.std(last) / np.std(first)
        self.assertLess(ratio_after, 0.5 * ratio_before)


if __name__ == "__main__":
    unittest.main()
