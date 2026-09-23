"""Sanity checks for baboon_acoustics.audio_io."""

from tests.context import BARK, WAHOO4, WahooTestCase
from baboon_acoustics import load_audio


class TestLoadAudio(WahooTestCase):

    def test_returns_float_mono_normalized(self):
        x, fs = load_audio(WAHOO4)
        self.assertEqual(x.dtype.name, "float64")
        self.assertEqual(x.ndim, 1)
        self.assertAlmostEqual(float(abs(x).max()), 1.0, places=6)
        self.assertEqual(fs, 44100)

    def test_unnormalized_keeps_native_amplitudes(self):
        x_norm, _ = load_audio(WAHOO4, normalize=True)
        x_raw, _ = load_audio(WAHOO4, normalize=False)
        # The raw waveform keeps the int16 scale (peak in the thousands).
        self.assertGreater(float(abs(x_raw).max()), 1000.0)
        self.assertGreater(float(abs(x_raw).max()),
                           float(abs(x_norm).max()))

    def test_stereo_file_reduced_to_first_channel(self):
        x, fs = load_audio(BARK)
        self.assertEqual(x.ndim, 1)
        self.assertAlmostEqual(len(x) / fs, 16.161, delta=0.01)

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_audio("does_not_exist.wav")


if __name__ == "__main__":
    unittest.main()
