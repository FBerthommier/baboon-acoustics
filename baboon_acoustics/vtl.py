"""Vocal tract length (VTL) estimation from formant frequencies.

Implements the method of Reby & McComb (2003): formant frequencies of a
tube closed at one end follow ``F_i = (2i-1) c / (4 VTL)``, so the odd
harmonics (1, 3, 5, ...) of the formant pattern are regressed through the
origin against measured formant frequencies, and the slope of that fit
yields the vocal tract length.

typical use
-----------
>>> from baboon_acoustics.formants_praat import extract_praat_formants
>>> from baboon_acoustics.vtl import reby_mccomb_vtl
>>> _, F = extract_praat_formants("wahoo_4.wav", n_formants=4)
>>> vtl_cm = reby_mccomb_vtl(F)
>>> print(f"{vtl_cm:.1f} cm")
17.9 cm
"""

import numpy as np

#: Speed of sound in warm air within the vocal tract (m/s), following
#: Reby & McComb (2003). The value is slightly lower than the 343 m/s of
#: free air at 20 C because of the warmth and humidity of exhaled air.
SPEED_OF_SOUND = 350.0


def reby_mccomb_vtl(formants, c=SPEED_OF_SOUND):
    """Estimate the vocal tract length from formant tracks (Reby & McComb, 2003).

    Every valid (non-NaN) formant measurement contributes one point
    ``(order, frequency)`` with ``order = 1, 3, 5, ...`` for
    F1, F2, F3, ...  A least-squares line through the origin,
    ``F = b * order``, is fitted across all points of all frames, and the
    VTL follows as ``VTL = c / (4 b)``.

    Parameters
    ----------
    formants : array_like
        Formant frequencies in Hz, either as a two-dimensional array of
        shape ``(n_formants, n_frames)`` (as produced by
        :mod:`baboon_acoustics.formants_praat` and
        :mod:`baboon_acoustics.formants_iaif`) or as a sequence of
        one-dimensional tracks. ``NaN`` entries are ignored, so tracks of
        different lengths and partially undefined estimates are accepted.
        Any number of formants (>= 2 valid values in total) can be given.
    c : float, optional
        Speed of sound in m/s. Default 350.

    Returns
    -------
    vtl : float
        Estimated vocal tract length in centimetres, or ``NaN`` when
        fewer than two valid formant values are available.

    Examples
    --------
    Formants of a 17.5 cm tube (c = 350 m/s): F_i = (2i-1) * 500 Hz.

    >>> F = np.array([[500.0, 1500.0, 2500.0, 3500.0]]).T
    >>> f"{reby_mccomb_vtl(F):.1f} cm"
    '17.5 cm'

    Works with NaN values and partial tracks:

    >>> F = np.array([[500.0, np.nan], [1500.0, np.nan]])
    >>> f"{reby_mccomb_vtl(F):.1f} cm"
    '17.5 cm'
    """
    try:
        matrix = np.asarray(formants, dtype=float)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        tracks = list(matrix)
    except (ValueError, TypeError):
        # Ragged input: a sequence of tracks of unequal length.
        tracks = [np.asarray(track, dtype=float).ravel() for track in formants]

    orders = []
    frequencies = []
    for i, track in enumerate(tracks):
        valid = ~np.isnan(track)
        orders.append(np.full(np.sum(valid), 2 * i + 1))
        frequencies.append(track[valid])

    x = np.concatenate(orders) if orders else np.array([])
    y = np.concatenate(frequencies) if frequencies else np.array([])
    if len(x) < 2:
        return np.nan

    slope = np.sum(x * y) / np.sum(x**2)
    return (c * 100.0) / (4.0 * slope)
