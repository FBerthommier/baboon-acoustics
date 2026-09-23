"""Formant extraction with Praat's Burg method (via Parselmouth).

This module wraps Praat's "To Formant (burg)" command through the
`Parselmouth <https://parselmouth.readthedocs.io>`_ interface. Linear
predictive coding (LPC) with the Burg algorithm fits an all-pole filter to
short analysis windows; the resonances of that filter are reported as
formant candidates. The number of poles ``n_formants`` is the main knob:
for a given vocal tract, too small a value merges formants together, while
too large a value spreads spurious poles across the spectrum.

typical use
-----------
>>> from baboon_acoustics.formants_praat import extract_praat_formants
>>> times, formants = extract_praat_formants("wahoo_4.wav", n_formants=4)
>>> formants.shape  # (n_formants, n_frames), frequencies in Hz
(4, 17)
"""

import numpy as np
import parselmouth
from parselmouth.praat import call

from .audio_io import DEFAULT_FS

#: Default formant ceiling (Hz). Frequencies above this value are not
#: modelled by the LPC fit. 5500 Hz is the value used in the article for
#: the baboon wahoo analysis.
DEFAULT_FORMANT_CEILING = 5500.0

#: Default analysis window length (s). 40 ms balances temporal resolution
#: against the need for enough samples per window at low sampling rates.
DEFAULT_WINDOW = 0.040

#: Default time step between consecutive analysis windows (s).
DEFAULT_TIME_STEP = 0.040

#: Default pre-emphasis frequency (Hz): frequencies above this value are
#: boosted (+6 dB/octave) before LPC to counteract the spectral roll-off
#: of the glottal source.
DEFAULT_PRE_EMPHASIS = 50.0


def extract_praat_formants(
    audio_path,
    n_formants,
    fs_target=DEFAULT_FS,
    formant_ceiling=DEFAULT_FORMANT_CEILING,
    window=DEFAULT_WINDOW,
    time_step=DEFAULT_TIME_STEP,
    pre_emphasis=DEFAULT_PRE_EMPHASIS,
):
    """Extract formant tracks with Praat's Burg LPC method.

    The sound file is loaded with Parselmouth, resampled to ``fs_target``
    if needed, and analysed with Praat's "To Formant (burg)" command. Each
    analysis window yields up to ``n_formants`` formant frequencies;
    windows where Praat cannot find a given formant produce ``NaN``.

    Parameters
    ----------
    audio_path : str or pathlib.Path
        Path to the audio file (WAV, or any format Praat can read).
    n_formants : int
        Number of formants (LPC pole pairs) requested from the Burg
        algorithm. Values between 3 and 6 are typical for mammal
        vocalizations; the article compares ``n_formants=4`` and
        ``n_formants=6``.
    fs_target : float, optional
        Sampling frequency the sound is resampled to before analysis (Hz).
        Default 44100.
    formant_ceiling : float, optional
        Highest frequency modelled by LPC (Hz). Default 5500.
    window : float, optional
        Analysis window length in seconds. Default 0.040.
    time_step : float, optional
        Time step between window centres in seconds. Default 0.040.
    pre_emphasis : float, optional
        Pre-emphasis cutoff frequency in Hz. Default 50.

    Returns
    -------
    times : numpy.ndarray
        One-dimensional array of frame times (s), centre of each analysis
        window.
    formants : numpy.ndarray
        Array of shape ``(n_formants, len(times))`` with formant
        frequencies in Hz; entries are ``NaN`` where Praat found no value.

    Examples
    --------
    >>> times, F = extract_praat_formants("data/wahoo_4.wav", n_formants=4)
    >>> F1_hz = F[0]           # first-formant track
    >>> F1_khz = F[0] / 1000.0 # same track in kHz, as plotted in the article
    """
    sound = parselmouth.Sound(str(audio_path))
    if sound.sampling_frequency != fs_target:
        sound = sound.resample(fs_target)

    formant_object = call(
        sound,
        "To Formant (burg)",
        time_step,
        n_formants,
        formant_ceiling,
        window,
        pre_emphasis,
    )

    times = np.asarray(formant_object.xs())
    formants = np.full((n_formants, len(times)), np.nan)
    for i in range(n_formants):
        for j, t in enumerate(times):
            formants[i, j] = formant_object.get_value_at_time(i + 1, t)

    return times, formants


def interpolate_formants(source_times, formants, target_times):
    """Resample formant tracks onto a new time grid by linear interpolation.

    Frames whose value is ``NaN`` are excluded from the interpolation, and
    target times outside the valid source range receive ``NaN``. This is
    used, for instance, to overlay Praat formant tracks on the time grid of
    a sliding IAIF analysis.

    Parameters
    ----------
    source_times : numpy.ndarray
        One-dimensional array of the original frame times (s).
    formants : numpy.ndarray
        Formant tracks of shape ``(n_formants, len(source_times))`` in Hz.
    target_times : numpy.ndarray
        One-dimensional array of the desired frame times (s).

    Returns
    -------
    interpolated : numpy.ndarray
        Array of shape ``(n_formants, len(target_times))`` in Hz.

    Examples
    --------
    >>> times, F = extract_praat_formants("wahoo_4.wav", n_formants=4)
    >>> grid = np.linspace(times[0], times[-1], 20)
    >>> F_on_grid = interpolate_formants(times, F, grid)
    """
    source_times = np.asarray(source_times, dtype=float)
    target_times = np.asarray(target_times, dtype=float)
    interpolated = np.full((formants.shape[0], len(target_times)), np.nan)
    for i in range(formants.shape[0]):
        valid = ~np.isnan(formants[i])
        if np.sum(valid) >= 2:
            interpolated[i] = np.interp(
                target_times,
                source_times[valid],
                formants[i][valid],
                left=np.nan,
                right=np.nan,
            )
    return interpolated


def fill_leading_nan(track):
    """Replace a leading ``NaN`` in a formant track with its first valid value.

    Praat occasionally returns ``NaN`` for the very first frame (the first
    window is only half-filled). Because a missing first point breaks
    line drawing in plots, it is replaced by the next available value.

    Parameters
    ----------
    track : numpy.ndarray
        One-dimensional formant track (any unit).

    Returns
    -------
    track : numpy.ndarray
        Copy of the input with a leading ``NaN`` replaced, if possible.

    Examples
    --------
    >>> fill_leading_nan(np.array([np.nan, 700.0, 710.0]))
    array([700., 700., 710.])
    """
    track = track.copy()
    if len(track) > 0 and np.isnan(track[0]):
        valid = ~np.isnan(track)
        if np.any(valid):
            track[0] = track[np.argmax(valid)]
    return track
