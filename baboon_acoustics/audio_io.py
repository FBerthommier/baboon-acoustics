"""Audio input/output helpers.

This module provides a single entry point, :func:`load_audio`, that reads a
WAV file and returns a mono, floating-point waveform ready for acoustic
analysis (formant extraction, F0 estimation, inverse filtering).

typical use
-----------
>>> from baboon_acoustics.audio_io import load_audio
>>> samples, fs = load_audio("wahoo_4.wav")
>>> samples.shape, fs
(29315,) 44100
"""

from pathlib import Path

import numpy as np
from scipy import signal
from scipy.io import wavfile

#: Default analysis sampling frequency (Hz). Files recorded at another rate
#: are resampled to this value so that all downstream parameters (window
#: lengths in samples, LPC order, F0 search range) keep their usual meaning.
DEFAULT_FS = 44100


def load_audio(path, target_fs=DEFAULT_FS, normalize=True):
    """Load a WAV file as a mono, float64 waveform.

    The file is read with :func:`scipy.io.wavfile.read`. Multi-channel
    recordings are reduced to their first channel, integer samples are
    converted to ``float64``, and the waveform is optionally peak-normalized
    to the range [-1, 1]. If the file's sampling frequency differs from
    ``target_fs``, the signal is resampled with a polyphase filter.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the WAV file.
    target_fs : float, optional
        Desired sampling frequency in Hz. Default 44100.
    normalize : bool, optional
        If True (default), scale the waveform so that its maximum absolute
        value is 1.0. Set to False to keep the native sample amplitudes
        (useful when absolute amplitudes matter, e.g. for glottal-waveform
        displays).

    Returns
    -------
    samples : numpy.ndarray
        One-dimensional float64 waveform.
    fs : float
        Sampling frequency of the returned waveform (Hz); equal to
        ``target_fs`` when resampling occurred.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not point to an existing file.

    Examples
    --------
    >>> samples, fs = load_audio("data/wahoo_4.wav")
    >>> print(f"{len(samples)} samples at {fs} Hz "
    ...       f"({len(samples) / fs:.2f} s)")
    29316 samples at 44100 Hz (0.66 s)
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")

    fs, data = wavfile.read(str(path))
    if data.ndim > 1:
        data = data[:, 0]
    samples = data.astype(np.float64)

    if normalize:
        peak = np.max(np.abs(samples))
        if peak > 0:
            samples = samples / peak

    if fs != target_fs:
        # Rational polyphase resampling: upsample by target_fs/gcd, then
        # downsample by fs/gcd, which keeps the original spectral content
        # as intact as possible.
        gcd = np.gcd(int(target_fs), int(fs))
        up = int(target_fs) // gcd
        down = int(fs) // gcd
        samples = signal.resample_poly(samples, up, down)
        fs = target_fs

    return samples, fs
