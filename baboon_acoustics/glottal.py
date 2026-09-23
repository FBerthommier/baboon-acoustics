"""Glottal-source waveform extraction by IAIF.

Two granularities are provided:

- :func:`extract_glottal_pulse` isolates a *single* glottal cycle from a
  short, quasi-stationary segment (the pulse shape is the fingerprint of
  the phonation type);
- :func:`extract_glottal_train` reconstructs the *full* glottal pulse
  train over an entire vocalization by overlap-add of windowed IAIF
  estimates.

:func:`normalize_amplitude` removes slow amplitude drifts from a glottal
recording so that consecutive pulses can be compared (e.g. in phase
portraits).

typical use
-----------
>>> from baboon_acoustics.audio_io import load_audio
>>> from baboon_acoustics.glottal import extract_glottal_train
>>> x, fs = load_audio("wahoo_4.wav", normalize=False)
>>> t, glottal = extract_glottal_train(x, fs)
>>> len(glottal)
24585
"""

import numpy as np
from scipy import signal

from .formants_iaif import (
    DEFAULT_VT_ORDER,
    DEFAULT_WIN_LEN,
    DEFAULT_HOP,
    iaif_frame,
)
from .f0 import DEFAULT_F0_MAX, DEFAULT_F0_MIN, estimate_f0_from_gn


def extract_glottal_pulse(
    x,
    fs,
    t_start,
    t_end,
    vt_order=DEFAULT_VT_ORDER,
    f0_min=DEFAULT_F0_MIN,
    f0_max=DEFAULT_F0_MAX,
):
    """Extract a single glottal pulse from a quasi-stationary segment.

    The segment ``[t_start, t_end]`` (in seconds) is windowed with a
    Hamming window and decomposed with :func:`iaif_frame
    <baboon_acoustics.formants_iaif.iaif_frame>`. The fundamental period
    is measured from the autocorrelation of the estimated glottal
    waveform, and the pulse centred on the largest absolute excursion is
    returned (one period long).

    Parameters
    ----------
    x : numpy.ndarray
        Input waveform.
    fs : float
        Sampling frequency in Hz.
    t_start, t_end : float
        Boundaries of the analysis segment in seconds. Choose a segment
        where phonation is stable (100-200 ms is a good length).
    vt_order : int, optional
        LPC order of the IAIF vocal-tract filter. Default 60.
    f0_min, f0_max : float, optional
        Search range for the fundamental period (Hz). Defaults 50 and 500.

    Returns
    -------
    t_pulse : numpy.ndarray
        Time axis of the pulse, starting at 0 (s).
    pulse : numpy.ndarray
        One-period glottal-flow derivative waveform, or ``(None, None)``
        when the segment is too short (< 1024 samples) for the analysis.

    Examples
    --------
    >>> x, fs = load_audio("data/wahoo_4.wav", normalize=False)  # doctest: +SKIP
    >>> t, pulse = extract_glottal_pulse(x, fs, t_start=0.26, t_end=0.34)
    >>> len(pulse) / fs * 1000   # pulse duration in ms (~ one period)
    3.7
    """
    segment = np.asarray(x[int(t_start * fs) : int(t_end * fs)], dtype=float).copy()
    if len(segment) < 1024:
        return None, None

    windowed = segment * np.hamming(len(segment))
    _, _, glottal = iaif_frame(windowed, fs, vt_order)

    # Fundamental period from the autocorrelation of the glottal estimate.
    ac = np.correlate(glottal, glottal, mode="full")
    ac = ac[len(ac) // 2 :]
    ac = ac / (ac[0] + 1e-10)
    min_lag = int(np.ceil(fs / f0_max))
    max_lag = min(int(np.floor(fs / f0_min)), len(ac) - 1)
    peak_lag = int(np.argmax(ac[min_lag:max_lag])) + min_lag

    # Cut one period around the largest absolute excursion.
    centre = int(np.argmax(np.abs(glottal)))
    half_period = peak_lag // 2
    start = max(centre - half_period, 0)
    end = min(centre + half_period, len(glottal))
    pulse = glottal[start:end]
    t_pulse = np.arange(len(pulse)) / fs
    return t_pulse, pulse


def extract_glottal_train(
    x,
    fs,
    vt_order=DEFAULT_VT_ORDER,
    win_len=DEFAULT_WIN_LEN,
    hop=DEFAULT_HOP,
):
    """Reconstruct the full glottal pulse train of a vocalization.

    The signal is sliced into Hann windows (periodic form,
    ``fftbins=True``), each window is decomposed with IAIF, and the
    windowed glottal estimates are summed and normalized by the window
    sum (overlap-add). The result is trimmed to the region where windows
    overlap by at least half a window, avoiding edge artefacts.

    Parameters
    ----------
    x : numpy.ndarray
        Input waveform (use ``normalize=False`` in
        :func:`load_audio <baboon_acoustics.audio_io.load_audio>` to keep
        native amplitudes).
    fs : float
        Sampling frequency in Hz.
    vt_order : int, optional
        LPC order of the IAIF vocal-tract filter. Default 60.
    win_len : int, optional
        Window length in samples. Default 4096.
    hop : int, optional
        Hop between window onsets in samples. Default 1024.

    Returns
    -------
    t_glot : numpy.ndarray
        Time axis of the trimmed reconstruction (s).
    glottal : numpy.ndarray
        Glottal-flow derivative waveform, same length as ``t_glot``.

    Examples
    --------
    >>> x, fs = load_audio("data/wahoo_4.wav", normalize=False)  # doctest: +SKIP
    >>> t, glottal = extract_glottal_train(x, fs)
    >>> f0 = estimate_f0_from_gn(glottal, fs)
    >>> f"{f0:.0f} Hz"
    '269 Hz'
    """
    x = np.asarray(x, dtype=float)
    window = signal.get_window("hann", win_len, fftbins=False)

    n_frames = 1 + max((len(x) - win_len) // hop, 0)
    reconstruction = np.zeros(len(x))
    window_sum = np.zeros(len(x))
    for i in range(n_frames):
        start = i * hop
        frame = x[start : start + win_len]
        if len(frame) < win_len:
            frame = np.pad(frame, (0, win_len - len(frame)))
        frame_w = frame * window
        try:
            _, _, glottal = iaif_frame(frame_w, fs, vt_order)
            reconstruction[start : start + win_len] += glottal * window
            window_sum[start : start + win_len] += window
        except Exception:
            continue

    glottal_signal = reconstruction / np.maximum(window_sum, 1e-9)

    # Trim the edges, where too few windows overlap.
    valid = window_sum > 0.5
    if np.any(valid):
        first = int(np.argmax(valid))
        last = len(valid) - int(np.argmax(valid[::-1])) - 1
        glottal_signal = glottal_signal[first : last + 1]

    t_glot = np.arange(len(glottal_signal)) / fs
    return t_glot, glottal_signal


def normalize_amplitude(signal_in, fs, cutoff=50.0, order=4, eps=1e-6):
    """Remove slow amplitude drifts from a signal (centering + envelope flattening).

    The signal is high-pass corrected by subtracting a low-pass version of
    itself (removes floating DC), then divided by a low-pass version of
    its envelope, so that the local amplitude becomes ~1 everywhere. This
    is useful before phase-portrait analysis of glottal pulse trains,
    where slow amplitude changes would otherwise blur the trajectory.

    Parameters
    ----------
    signal_in : numpy.ndarray
        Input signal.
    fs : float
        Sampling frequency in Hz.
    cutoff : float, optional
        Cutoff of both Butterworth low-pass filters in Hz. Default 50.
    order : int, optional
        Butterworth filter order. Default 4.
    eps : float, optional
        Constant added to the envelope to avoid division by zero.
        Default 1e-6.

    Returns
    -------
    normalized : numpy.ndarray
        Centered, envelope-normalized signal. When the input is longer
        than 1024 samples, the first and last 512 samples are trimmed to
        remove filter edge effects.

    Examples
    --------
    >>> fs = 44100
    >>> t = np.arange(4096) / fs
    >>> drift = np.sin(2 * np.pi * 300 * t) * np.linspace(0.2, 1.0, 4096)
    >>> flat = normalize_amplitude(drift, fs)
    >>> np.max(np.abs(flat[-200:])) > 0.5   # trailing amplitude preserved
    True
    """
    b, a = signal.butter(order, cutoff / (0.5 * fs), btype="low")

    slow = signal.filtfilt(b, a, signal_in)
    envelope = np.abs(signal_in)
    slow_envelope = signal.filtfilt(b, a, envelope)

    centered = signal_in - slow
    normalized = centered / (slow_envelope + eps)

    if len(normalized) > 1024:
        normalized = normalized[512:-512]
    return normalized
