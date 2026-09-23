"""Iterative Adaptive Inverse Filtering (IAIF) analysis.

IAIF (Alku, 1991) separates a recorded vocalization into a glottal source
waveform and a vocal-tract filter. Unlike plain LPC (as used by Praat's
Burg method), IAIF first estimates and removes the contribution of the
glottal source, which reduces the influence of the source spectrum on the
estimated formant positions. The price is a higher computational cost and
a longer analysis window.

The module provides, from bottom to top:

- :func:`lpc_levinson` -- raw linear predictive coding of one frame;
- :func:`lpc_spectrum_db` -- frequency response of an LPC filter;
- :func:`iaif_frame` -- IAIF source/tract decomposition of one frame;
- :func:`get_frame_formants` -- formants of one frame as spectral peaks;
- :func:`iaif_sliding_analysis` -- full sliding-window analysis of a
  signal (spectrograms, F0 track, formant tracks).

typical use
-----------
>>> from baboon_acoustics.audio_io import load_audio
>>> from baboon_acoustics.formants_iaif import iaif_sliding_analysis
>>> x, fs = load_audio("wahoo_4.wav")
>>> result = iaif_sliding_analysis(x, fs, n_formants=4)
>>> result.formants.shape  # (n_formants, n_frames), frequencies in Hz
(4, 29)
"""

from typing import NamedTuple

import numpy as np
from scipy import fft, signal

from .audio_io import DEFAULT_FS
from .f0 import estimate_f0_from_gn

#: Default LPC order for the vocal-tract filter. IAIF needs a much higher
#: order than conventional LPC because the glottal source has been removed.
DEFAULT_VT_ORDER = 60

#: Default analysis window length in samples (4096 samples ~ 93 ms at
#: 44.1 kHz): long enough for stable high-order LPC, short enough to
#: track formant movement within a ~0.7 s vocalization.
DEFAULT_WIN_LEN = 4 * 1024

#: Default hop between successive windows (25 % overlap of the window
#: length).
DEFAULT_HOP = DEFAULT_WIN_LEN // 4

#: Default FFT length used for spectrograms and LPC spectra (zero-padded
#: to twice the window length).
DEFAULT_N_FFT = DEFAULT_WIN_LEN * 2

#: Default highest frequency considered for formant detection (Hz).
DEFAULT_FORMANT_MAX = 3500.0

#: Default minimum prominence of a spectral peak to count as a formant
#: candidate (dB).
DEFAULT_PEAK_PROMINENCE = 2.5

#: Default minimum spacing between two formant candidates (Hz).
DEFAULT_PEAK_MIN_DIST_HZ = 150.0


class IaifResult(NamedTuple):
    """Container for the outputs of :func:`iaif_sliding_analysis`.

    Attributes
    ----------
    times : numpy.ndarray
        Frame times (s), at the centre of each analysis window.
    freqs : numpy.ndarray
        Frequency axis of the spectrograms (Hz), from 0 to fs/2.
    spectrogram_db : numpy.ndarray
        Short-term magnitude spectrogram of the input signal (dB),
        shape ``(len(freqs), len(times))``.
    glottal_db : numpy.ndarray
        Magnitude spectrogram of the IAIF glottal-source estimate (dB),
        same shape as ``spectrogram_db``.
    envelope_db : numpy.ndarray
        Magnitude of the estimated vocal-tract transfer function
        ``|H_vt|`` (dB), same shape as ``spectrogram_db``.
    f0_track : numpy.ndarray
        Fundamental-frequency track (Hz), ``NaN`` where unvoiced.
    formants : numpy.ndarray
        Formant tracks (Hz), shape ``(n_formants, len(times))``.
    """

    times: np.ndarray
    freqs: np.ndarray
    spectrogram_db: np.ndarray
    glottal_db: np.ndarray
    envelope_db: np.ndarray
    f0_track: np.ndarray
    formants: np.ndarray


def lpc_levinson(x, order):
    """Compute linear predictive coefficients with the Levinson-Durbin recursion.

    The autocorrelation of ``x`` is computed efficiently through the FFT,
    then solved by the Levinson-Durbin algorithm to obtain the coefficients
    of the all-pole filter ``A(z) = 1 - a1 z^-1 - ... - aN z^-N`` that
    whitens the input.

    Parameters
    ----------
    x : numpy.ndarray
        Input frame (any real one-dimensional array). Its mean is removed
        internally.
    order : int
        LPC order (number of poles).

    Returns
    -------
    a : numpy.ndarray
        LPC coefficient vector of length ``order + 1`` (denominator of the
        all-pole filter, ``a[0] = 1``).
    energy : float
        Prediction-error variance of the fitted model. For a degenerate
        frame (fewer than ``order + 2`` samples), returns a coefficient
        vector of ones and a near-zero energy.

    Examples
    --------
    >>> rng = np.random.default_rng(0)
    >>> a, energy = lpc_levinson(rng.standard_normal(2048), order=10)
    >>> len(a)
    11
    """
    x = np.asarray(x, dtype=float)
    x = x - np.mean(x)
    n = len(x)
    if n < order + 2:
        return np.ones(order + 1), 1e-12

    # Autocorrelation via FFT: r = IFFT(|FFT(x)|^2).
    n_fft = 2 ** int(np.ceil(np.log2(2 * n - 1)))
    X = np.fft.rfft(x, n=n_fft)
    r = np.fft.irfft(np.abs(X) ** 2, n=n_fft)[: order + 1] / n

    a = np.zeros(order)
    energy = r[0] + 1e-12
    for i in range(order):
        lam = r[i + 1]
        if i > 0:
            lam -= np.dot(a[:i], r[i : 0 : -1])
        k = lam / energy
        a_new = a.copy()
        a_new[i] = k
        if i > 0:
            a_new[:i] -= k * a[i - 1 :: -1]
        a = a_new
        energy *= max(1 - k**2, 1e-10)

    return np.r_[1.0, -a], float(energy)


def lpc_spectrum_db(lpc_coeffs, error_variance, n_fft=4096, fs=DEFAULT_FS):
    """Evaluate the magnitude response of an LPC all-pole filter in dB.

    The LPC filter ``1 / A(z)``, scaled by the square root of the
    prediction-error variance, approximates the spectral envelope of the
    analysed frame. Spectral peaks of this envelope are the formant
    candidates.

    Parameters
    ----------
    lpc_coeffs : numpy.ndarray
        LPC denominator coefficients (as returned by :func:`lpc_levinson`).
    error_variance : float
        Prediction-error variance of the LPC fit.
    n_fft : int, optional
        Number of frequency samples. Default 4096.
    fs : float, optional
        Sampling frequency in Hz. Default 44100.

    Returns
    -------
    freqs : numpy.ndarray
        Frequency axis in Hz, from 0 to fs/2.
    spectrum_db : numpy.ndarray
        Envelope magnitude in dB at each frequency.

    Examples
    --------
    >>> a, e = lpc_levinson(np.sin(2 * np.pi * 500 * np.arange(2048) / 44100), order=8)
    >>> freqs, env_db = lpc_spectrum_db(a, e)
    >>> freqs[np.argmax(env_db)]  # resonance near 500 Hz
    500.2...
    """
    gain = np.sqrt(max(error_variance, 1e-20))
    freqs, H = signal.freqz([gain], lpc_coeffs, worN=n_fft, fs=fs)
    return freqs, 20 * np.log10(np.abs(H) + 1e-10)


def iaif_frame(frame, fs, vt_order=DEFAULT_VT_ORDER):
    """Decompose one frame into glottal source and vocal-tract filter (IAIF).

    The method alternates low-order LPC estimates of the glottal source
    (orders 1 then 2) and high-order LPC estimates of the vocal tract,
    each iteration removing the current source estimate from the signal
    (Alku, 1991). Two passes are performed.

    Parameters
    ----------
    frame : numpy.ndarray
        Analysis frame (preferably windowed, e.g. Hann).
    fs : float
        Sampling frequency in Hz (unused numerically, kept for a uniform
        calling interface across the package).
    vt_order : int, optional
        LPC order of the vocal-tract filter. Default 60.

    Returns
    -------
    lpc_coeffs : numpy.ndarray
        LPC coefficients of the final vocal-tract filter estimate.
    error_variance : float
        Prediction-error variance of the final vocal-tract fit.
    glottal : numpy.ndarray
        Estimated glottal-flow derivative waveform (same length as
        ``frame``).

    Examples
    --------
    >>> x, fs = load_audio("data/wahoo_4.wav")   # doctest: +SKIP
    >>> frame = x[:4096] * np.hanning(4096)
    >>> a, e, glottal = iaif_frame(frame, fs)
    >>> len(glottal)
    4096
    """
    # First pass: crude glottal estimate with a first-order source model.
    Hg1, _ = lpc_levinson(frame, 1)
    s1 = signal.lfilter(Hg1, [1.0], frame)
    Hvt1, _ = lpc_levinson(s1, vt_order)
    g1 = np.cumsum(signal.lfilter(Hvt1, [1.0], frame))
    g1 -= np.mean(g1)

    # Second pass: refine the source with a second-order model.
    Hg2, _ = lpc_levinson(g1, 2)
    s2 = signal.lfilter(Hg2, [1.0], frame)
    Hvt2, error_variance = lpc_levinson(s2, vt_order)
    glottal = np.cumsum(signal.lfilter(Hvt2, [1.0], frame))
    glottal -= np.mean(glottal)

    return Hvt2, error_variance, glottal


def get_frame_formants(
    frame,
    fs,
    vt_order=DEFAULT_VT_ORDER,
    n_formants=4,
    fmax=DEFAULT_FORMANT_MAX,
    min_prominence=DEFAULT_PEAK_PROMINENCE,
    min_dist_hz=DEFAULT_PEAK_MIN_DIST_HZ,
):
    """Estimate the first ``n_formants`` formants of one frame from its IAIF envelope.

    The frame is decomposed with :func:`iaif_frame`, the vocal-tract
    envelope is evaluated with :func:`lpc_spectrum_db`, and spectral peaks
    below ``fmax`` are detected with :func:`scipy.signal.find_peaks`. The
    ``n_formants + 2`` most prominent peaks are kept as candidates and the
    ``n_formants`` lowest in frequency are returned, so that a very
    prominent high-frequency peak cannot mask a genuine low formant.

    Parameters
    ----------
    frame : numpy.ndarray
        Analysis frame (preferably windowed).
    fs : float
        Sampling frequency in Hz.
    vt_order : int, optional
        LPC order of the vocal-tract filter. Default 60.
    n_formants : int, optional
        Number of formants to return. Default 4.
    fmax : float, optional
        Highest frequency considered for peak detection (Hz). Default 3500.
    min_prominence : float, optional
        Minimum peak prominence (dB). Default 2.5.
    min_dist_hz : float, optional
        Minimum spacing between peaks (Hz). Default 150.

    Returns
    -------
    formants : numpy.ndarray
        Array of length ``n_formants`` with formant frequencies in Hz,
        ascending; padded with ``NaN`` when fewer peaks than requested are
        found.

    Examples
    --------
    >>> x, fs = load_audio("data/wahoo_4.wav")   # doctest: +SKIP
    >>> f = get_frame_formants(x[:4096] * np.hanning(4096), fs, n_formants=3)
    >>> f.shape
    (3,)
    """
    lpc_coeffs, error_variance, _ = iaif_frame(frame, fs, vt_order)
    freqs, envelope_db = lpc_spectrum_db(lpc_coeffs, error_variance, n_fft=4096, fs=fs)

    band = freqs <= fmax
    freqs_band = freqs[band]
    envelope_band = envelope_db[band]

    bin_width = freqs[1] - freqs[0]
    min_dist = max(1, int(min_dist_hz / bin_width))
    peaks, properties = signal.find_peaks(
        envelope_band, prominence=min_prominence, distance=min_dist
    )
    if len(peaks) == 0:
        return np.full(n_formants, np.nan)

    # Keep the most prominent candidates, then the lowest in frequency.
    by_prominence = np.argsort(properties["prominences"])[::-1]
    candidates = np.sort(peaks[by_prominence[: n_formants + 2]])
    formants = freqs_band[candidates[:n_formants]]
    if len(formants) < n_formants:
        formants = np.append(formants, np.full(n_formants - len(formants), np.nan))
    return formants


def iaif_sliding_analysis(
    x,
    fs,
    n_formants=4,
    vt_order=DEFAULT_VT_ORDER,
    win_len=DEFAULT_WIN_LEN,
    hop=DEFAULT_HOP,
    n_fft=DEFAULT_N_FFT,
    fmax=DEFAULT_FORMANT_MAX,
    min_prominence=DEFAULT_PEAK_PROMINENCE,
    min_dist_hz=DEFAULT_PEAK_MIN_DIST_HZ,
    f0_min=50.0,
    f0_max=500.0,
    ac_threshold=0.3,
):
    """Sliding-window IAIF analysis of a whole signal.

    The signal is sliced into overlapping Hann windows. For each window,
    the routine computes (i) the short-term spectrogram of the input,
    (ii) the spectrogram of the IAIF glottal-source estimate, (iii) the
    IAIF vocal-tract envelope ``|H_vt|``, (iv) an F0 estimate from the
    autocorrelation of the glottal estimate, and (v) ``n_formants`` formant
    frequencies from the envelope peaks. Windows where the IAIF
    decomposition fails fall back to the input spectrogram for the source
    panel and ``NaN`` everywhere else.

    Parameters
    ----------
    x : numpy.ndarray
        Input waveform.
    fs : float
        Sampling frequency in Hz.
    n_formants : int, optional
        Number of formants to track. Default 4.
    vt_order : int, optional
        LPC order of the vocal-tract filter. Default 60.
    win_len : int, optional
        Window length in samples. Default 4096.
    hop : int, optional
        Hop between window onsets in samples. Default 1024.
    n_fft : int, optional
        FFT length for the spectrograms. Default 8192.
    fmax : float, optional
        Highest frequency for formant detection (Hz). Default 3500.
    min_prominence : float, optional
        Minimum formant peak prominence (dB). Default 2.5.
    min_dist_hz : float, optional
        Minimum spacing between formant peaks (Hz). Default 150.
    f0_min, f0_max : float, optional
        Search range for F0 (Hz). Defaults 50 and 500.
    ac_threshold : float, optional
        Minimum normalized autocorrelation for a frame to be considered
        voiced. Default 0.3.

    Returns
    -------
    result : IaifResult
        Named tuple with fields ``times``, ``freqs``, ``spectrogram_db``,
        ``glottal_db``, ``envelope_db``, ``f0_track`` and ``formants``
        (see :class:`IaifResult`).

    Examples
    --------
    >>> x, fs = load_audio("data/wahoo_4.wav")   # doctest: +SKIP
    >>> result = iaif_sliding_analysis(x, fs, n_formants=4)
    >>> result.f0_track[np.argwhere(~np.isnan(result.f0_track))[0]]
    array([268.7...])
    """
    x = np.asarray(x, dtype=float)
    window = signal.get_window("hann", win_len)

    n_frames = 1 + max((len(x) - win_len) // hop, 0)
    n_bins = n_fft // 2 + 1
    freqs = np.linspace(0, fs / 2, n_bins)
    times = (np.arange(n_frames) * hop + win_len / 2) / fs

    spectrogram_db = np.full((n_bins, n_frames), np.nan)
    glottal_db = np.full((n_bins, n_frames), np.nan)
    envelope_db = np.full((n_bins, n_frames), np.nan)
    f0_track = np.full(n_frames, np.nan)
    formants = np.full((n_formants, n_frames), np.nan)

    for i in range(n_frames):
        frame = x[i * hop : i * hop + win_len]
        if len(frame) < win_len:
            frame = np.pad(frame, (0, win_len - len(frame)))
        frame_w = frame * window

        spectrum = fft.rfft(frame_w, n=n_fft)
        spectrogram_db[:, i] = 20 * np.log10(np.abs(spectrum) + 1e-10)

        try:
            lpc_coeffs, error_variance, glottal = iaif_frame(frame_w, fs, vt_order)
            glottal_spectrum = fft.rfft(glottal * window, n=n_fft)
            glottal_db[:, i] = 20 * np.log10(np.abs(glottal_spectrum) + 1e-10)

            f_lpc, env_db = lpc_spectrum_db(
                lpc_coeffs, error_variance, n_fft=n_fft, fs=fs
            )
            envelope_db[:, i] = np.interp(
                freqs, f_lpc, env_db, left=np.nan, right=np.nan
            )

            f0_track[i] = estimate_f0_from_gn(
                glottal, fs, f0_min=f0_min, f0_max=f0_max, ac_threshold=ac_threshold
            )
            formants[:, i] = get_frame_formants(
                frame_w,
                fs,
                vt_order=vt_order,
                n_formants=n_formants,
                fmax=fmax,
                min_prominence=min_prominence,
                min_dist_hz=min_dist_hz,
            )
        except Exception:
            glottal_db[:, i] = spectrogram_db[:, i]
            envelope_db[:, i] = np.nan

    return IaifResult(
        times=times,
        freqs=freqs,
        spectrogram_db=spectrogram_db,
        glottal_db=glottal_db,
        envelope_db=envelope_db,
        f0_track=f0_track,
        formants=formants,
    )
