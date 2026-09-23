"""Fundamental-frequency (F0) estimation.

Two complementary estimators are provided:

- :func:`estimate_f0_from_gn` derives F0 from the autocorrelation of a
  (possibly IAIF-estimated) glottal waveform. It is the estimator used
  inside the sliding IAIF analysis.
- :func:`estimate_f0_praat` reproduces Praat's autocorrelation pitch
  analysis ("to_pitch_ac"), optionally preceded by a Gaussian low-pass
  pre-filter that mimics the *filtered autocorrelation* strategy of
  recent Praat versions and suppresses subharmonic errors.

typical use
-----------
>>> from baboon_acoustics.f0 import estimate_f0_praat
>>> times, f0, strength = estimate_f0_praat("data/wahoo_4.wav")
>>> f0.mean()   # mean F0 over voiced frames, Hz
323.6...
"""

import numpy as np
import parselmouth
from scipy.fft import fft, fftfreq, ifft

#: Default F0 search range (Hz). Mammal vocalizations commonly lie between
#: 50 and 500 Hz; widen the range for very low-pitched or ultrasonic calls.
DEFAULT_F0_MIN = 50.0
DEFAULT_F0_MAX = 500.0

#: Default autocorrelation threshold above which a frame counts as voiced
#: (normalized units, 0 to 1).
DEFAULT_AC_THRESHOLD = 0.3


def estimate_f0_from_gn(
    glottal,
    fs,
    f0_min=DEFAULT_F0_MIN,
    f0_max=DEFAULT_F0_MAX,
    ac_threshold=DEFAULT_AC_THRESHOLD,
):
    """Estimate F0 from the autocorrelation of a glottal waveform.

    The normalized autocorrelation of ``glottal`` is searched for its
    maximum between the lags corresponding to ``f0_max`` and ``f0_min``.
    The reciprocal of the best lag gives F0. Frames whose peak
    autocorrelation falls below ``ac_threshold`` (or peak lags touching
    the search boundaries) are declared unvoiced and return ``NaN``.

    Parameters
    ----------
    glottal : numpy.ndarray
        Glottal-flow waveform (e.g. the ``glottal`` output of
        :func:`baboon_acoustics.formants_iaif.iaif_frame`).
    fs : float
        Sampling frequency in Hz.
    f0_min, f0_max : float, optional
        Search range in Hz. Defaults 50 and 500.
    ac_threshold : float, optional
        Minimum normalized autocorrelation peak for a voiced frame.
        Default 0.3.

    Returns
    -------
    f0 : float
        Estimated fundamental frequency in Hz, or ``NaN`` for unvoiced or
        unreliable frames.

    Examples
    --------
    >>> fs = 44100
    >>> t = np.arange(fs) / fs                # 1 second
    >>> pulses = np.zeros(fs)
    >>> pulses[:: int(fs / 150.0)] = 1.0      # pulse train at 150 Hz
    >>> estimate_f0_from_gn(pulses, fs)
    150.0
    """
    glottal = np.asarray(glottal, dtype=float)
    glottal = glottal - np.mean(glottal)

    ac = np.correlate(glottal, glottal, mode="full")
    ac = ac[len(ac) // 2 :]
    if len(ac) == 0:
        return np.nan
    ac = ac / (ac[0] + 1e-10)

    min_lag = int(np.ceil(fs / f0_max))
    max_lag = int(np.floor(fs / f0_min))
    max_lag = min(max_lag, len(ac) - 1)
    if min_lag >= max_lag:
        return np.nan

    segment = ac[min_lag:max_lag]
    peak_rel = int(np.argmax(segment))
    if segment[peak_rel] < ac_threshold:
        return np.nan
    peak_lag = peak_rel + min_lag
    if peak_lag == min_lag or peak_lag == max_lag - 1:
        return np.nan

    f0 = fs / peak_lag
    if f0 <= f0_min or f0 >= f0_max:
        return np.nan
    return f0


def gaussian_lowpass_filter(samples, fs, cutoff=800.0, attenuation=0.03):
    """Apply a Gaussian low-pass filter in the frequency domain.

    This mimics the pre-filtering step of Praat's "filtered
    autocorrelation" pitch method: attenuating spectral content above the
    expected F0 range suppresses strong harmonics and subharmonic lures
    before the autocorrelation stage.

    Parameters
    ----------
    samples : numpy.ndarray
        Input waveform (mono, one-dimensional).
    fs : float
        Sampling frequency in Hz.
    cutoff : float, optional
        Cutoff frequency in Hz. Default 800.
    attenuation : float, optional
        Linear gain of the filter at ``cutoff`` (0.03 ~ -30.5 dB).
        Default 0.03.

    Returns
    -------
    filtered : numpy.ndarray
        Filtered waveform (same length as ``samples``).

    Notes
    -----
    The Gaussian width is chosen so that the gain equals ``attenuation``
    at ``cutoff``: sigma = cutoff / sqrt(-2 ln(attenuation)).

    Examples
    --------
    >>> fs = 44100
    >>> t = np.arange(44100) / fs
    >>> noisy = np.sin(2 * np.pi * 200 * t) + 0.5 * np.sin(2 * np.pi * 4000 * t)
    >>> clean = gaussian_lowpass_filter(noisy, fs)
    >>> float(np.max(np.abs(clean))) < float(np.max(np.abs(noisy)))
    True
    """
    sigma = cutoff / np.sqrt(-2.0 * np.log(attenuation))
    spectrum = fft(samples)
    freqs = fftfreq(len(samples), 1.0 / fs)
    filtered_spectrum = spectrum * np.exp(-((freqs / sigma) ** 2) / 2)
    return np.real(ifft(filtered_spectrum))


def estimate_f0_praat(
    audio_path,
    use_prefilter=True,
    time_step=0.01,
    pitch_floor=30.0,
    pitch_ceiling=800.0,
    voicing_threshold=0.10,
    octave_cost=0.055,
    octave_jump_cost=0.35,
    voiced_unvoiced_cost=0.14,
    very_accurate=False,
    filter_cutoff=800.0,
    filter_attenuation=0.03,
):
    """Estimate an F0 contour with Praat's autocorrelation method (Parselmouth).

    The audio file is loaded with Parselmouth and, when ``use_prefilter``
    is True, pre-filtered with :func:`gaussian_lowpass_filter` to simulate
    the filtered-autocorrelation strategy of Praat >= 6.4, which is more
    robust against subharmonic errors on animal vocalizations. Pitch is
    then analysed with Praat's ``to_pitch_ac`` command and voiced frames
    (F0 > 0) are returned.

    Parameters
    ----------
    audio_path : str or pathlib.Path
        Path to the audio file.
    use_prefilter : bool, optional
        Apply the Gaussian pre-filter (default True). Set to False to get
        the classic raw-autocorrelation behaviour of older Praat versions.
    time_step : float, optional
        Frame spacing in seconds. Default 0.01.
    pitch_floor, pitch_ceiling : float, optional
        F0 search range in Hz. Defaults 30 and 800 (adjusted for the
        baboon wahoo; Praat's own defaults are 75 and 600).
    voicing_threshold : float, optional
        Praat voicing threshold (0 to 1). Default 0.10.
    octave_cost : float, optional
        Praat octave cost. Default 0.055.
    octave_jump_cost : float, optional
        Praat octave-jump cost. Default 0.35.
    voiced_unvoiced_cost : float, optional
        Praat voiced/unvoiced cost. Default 0.14.
    very_accurate : bool, optional
        Praat "very accurate" (slower) mode. Default False.
    filter_cutoff : float, optional
        Pre-filter cutoff frequency in Hz. Default 800.
    filter_attenuation : float, optional
        Pre-filter gain at cutoff (linear). Default 0.03.

    Returns
    -------
    times : numpy.ndarray
        Frame times (s) of the voiced frames.
    f0 : numpy.ndarray
        F0 values (Hz) of the voiced frames.
    strength : numpy.ndarray
        Pitch strength of the voiced frames (0 to 1); low values indicate
        weak periodicity.

    Examples
    --------
    >>> times, f0, strength = estimate_f0_praat("data/wahoo_4.wav")
    >>> f0.min() > 0 and f0.max() < 800
    True
    """
    sound = parselmouth.Sound(str(audio_path))
    if use_prefilter:
        filtered = gaussian_lowpass_filter(
            sound.values[0], sound.sampling_frequency,
            cutoff=filter_cutoff, attenuation=filter_attenuation,
        )
        sound = parselmouth.Sound(filtered, sound.sampling_frequency)

    pitch = sound.to_pitch_ac(
        time_step=time_step,
        pitch_floor=pitch_floor,
        pitch_ceiling=pitch_ceiling,
        voicing_threshold=voicing_threshold,
        octave_cost=octave_cost,
        octave_jump_cost=octave_jump_cost,
        voiced_unvoiced_cost=voiced_unvoiced_cost,
        very_accurate=very_accurate,
    )

    selected = pitch.selected_array
    voiced = selected["frequency"] > 0
    times = np.asarray(pitch.ts())[voiced]
    f0 = selected["frequency"][voiced]
    strength = selected["strength"][voiced]
    return times, f0, strength
