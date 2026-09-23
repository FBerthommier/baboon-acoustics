"""Plotting utilities for journal-style figures.

This module centralizes the visual conventions shared by the article
figures: compact Arial typography, thin axes, bold panel letters,
frequency/time axes in kHz and seconds, and the F1-F2 trajectory plot
with the Maximal Acoustic Space (MAS) contour.

typical use
-----------
>>> import matplotlib.pyplot as plt
>>> from baboon_acoustics.plotting import set_journal_style, add_panel_label
>>> set_journal_style()
>>> fig, ax = plt.subplots()
>>> add_panel_label(ax, "(A)")
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

#: Font size of the bold panel letters, e.g. "(A)".
PANEL_LABEL_SIZE = 11


def set_journal_style(
    base_size=8,
    tick_size=7,
    legend_size=6,
    linewidth=0.8,
    titlesize=None,
    embed_fonts=False,
):
    """Apply the compact journal figure style used throughout the article.

    Sets Arial fonts, small point sizes and thin lines suitable for
    multi-panel figures printed at column width.

    Parameters
    ----------
    base_size : float, optional
        Base font and axis-label size in points. Default 8.
    tick_size : float, optional
        Tick-label size in points. Default 7.
    legend_size : float, optional
        Legend font size in points. Default 6.
    linewidth : float, optional
        Axis spine and tick line width in points. Default 0.8.
    titlesize : float or None, optional
        Axis-title size in points; ``None`` (default) keeps the
        matplotlib default (``"large"``, i.e. 1.2 x ``base_size``).
    embed_fonts : bool, optional
        Embed TrueType fonts in PDF/PS output (fonttype 42). Default
        False.

    Returns
    -------
    None
        The function modifies the matplotlib rcParams in place.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> set_journal_style()
    >>> fig, ax = plt.subplots()
    >>> plt.close(fig)
    """
    rc = {
        "font.family": "Arial",
        "font.size": base_size,
        "axes.labelsize": base_size,
        "xtick.labelsize": tick_size,
        "ytick.labelsize": tick_size,
        "legend.fontsize": legend_size,
        "axes.linewidth": linewidth,
        "xtick.major.width": linewidth,
        "ytick.major.width": linewidth,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
    }
    if titlesize is not None:
        rc["axes.titlesize"] = titlesize
    if embed_fonts:
        rc["pdf.fonttype"] = 42
        rc["ps.fonttype"] = 42
    plt.rcParams.update(rc)


def add_panel_label(ax, label, dx=-0.05, dy=1.02, fontsize=PANEL_LABEL_SIZE):
    """Add a bold panel letter (e.g. "(A)") to an axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axis.
    label : str
        Panel label, e.g. ``"(A)"``.
    dx, dy : float, optional
        Position in axis coordinates (0-1). Defaults place the label just
        above the left edge of the axis.
    fontsize : float, optional
        Font size in points. Default 11.

    Returns
    -------
    None

    Examples
    --------
    >>> fig, ax = plt.subplots()
    >>> add_panel_label(ax, "(A)")
    """
    ax.text(
        dx, dy, label,
        transform=ax.transAxes,
        fontsize=fontsize,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def setup_frequency_axis(ax):
    """Format an axis as a time-frequency panel (kHz vs. s, 0-3.5 kHz).

    Applies the limits, tick spacing and labels shared by the
    spectrogram-like panels of the article figures.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axis.

    Returns
    -------
    None

    Examples
    --------
    >>> fig, ax = plt.subplots()
    >>> setup_frequency_axis(ax)
    """
    ax.set_ylim(0, 3.5)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1.0))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax.tick_params(labelsize=9, width=0.8, length=4)
    ax.set_xlabel("Time (s)", fontsize=10)
    ax.set_ylabel("Frequency (kHz)", fontsize=10)


def plot_f1f2_trajectory(ax, f1_khz, f2_khz, marker_types, marker_colors,
                          mas_f1_khz, mas_f2_khz):
    """Plot a formant trajectory in the F2-F1 plane, with the MAS contour.

    The Maximal Acoustic Space (MAS) contour of the species is drawn as a
    filled grey area with a thick outline; the trajectory is drawn on top
    as a thin connecting line with one marker per analysis frame. The
    x-axis (F2) is reversed, following the usual vowel-chart convention.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axis.
    f1_khz, f2_khz : array_like
        F1 and F2 values of the trajectory, in kHz.
    marker_types : sequence of str
        Matplotlib marker for each trajectory point (e.g. ``"o"``,
        ``"s"``, ``"D"``).
    marker_colors : sequence of str
        Face color for each trajectory point.
    mas_f1_khz, mas_f2_khz : array_like
        F1/F2 contour of the Maximal Acoustic Space in kHz. The contour
        is closed automatically if needed.

    Returns
    -------
    None

    Examples
    --------
    >>> fig, ax = plt.subplots()
    >>> plot_f1f2_trajectory(
    ...     ax, [0.5, 0.8], [1.5, 1.2], ["o", "s"], ["red", "blue"],
    ...     mas_f1_khz=[0.2, 1.1, 0.2], mas_f2_khz=[0.3, 1.6, 3.0])
    """
    mas_f1 = np.asarray(mas_f1_khz, dtype=float)
    mas_f2 = np.asarray(mas_f2_khz, dtype=float)

    ax.fill(mas_f2, mas_f1, alpha=0.10, color="gray")
    ax.plot(mas_f2, mas_f1, "k-", lw=2.5)

    ax.set_xlim(3.3, 0.0)
    ax.set_ylim(1.3, 0.1)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1.0))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.4))
    ax.tick_params(labelsize=9, width=0.8, length=4)
    ax.set_xlabel("F2 (kHz)", fontsize=10)
    ax.set_ylabel("F1 (kHz)", fontsize=10)

    f1 = np.asarray(f1_khz, dtype=float)
    f2 = np.asarray(f2_khz, dtype=float)
    ax.plot(f2, f1, "-", color="0.5", linewidth=0.4, zorder=1)
    for x, y, marker, color in zip(f2, f1, marker_types, marker_colors):
        size = 28 if marker == "o" else 22
        ax.scatter(
            x, y, s=size, marker=marker, c=color,
            edgecolors="black", linewidths=0.4, zorder=3,
        )
    ax.set_box_aspect(1)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)


def plot_phase_portrait(ax, signal_in, fs, title="Phase portrait (Hilbert)",
                         color="k", alpha=0.7, lw=0.5, normalize=True, cutoff=50.0):
    """Plot the analytic-signal (Hilbert) phase portrait of a waveform.

    The signal is transformed with the Hilbert transform; the real part is
    plotted against the imaginary part of the analytic signal. A purely
    periodic signal with stable amplitude traces a closed loop, so this
    representation reveals amplitude modulation, jitter and deterministic
    chaos in glottal pulse trains at a glance.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axis.
    signal_in : numpy.ndarray
        Input waveform (e.g. a glottal pulse train).
    fs : float
        Sampling frequency in Hz.
    title : str, optional
        Axis title.
    color, alpha, lw : optional
        Matplotlib line color, transparency and line width.
    normalize : bool, optional
        Apply :func:`baboon_acoustics.glottal.normalize_amplitude` before
        the transform to remove slow amplitude drifts. Default True.
    cutoff : float, optional
        Cutoff of the drift-removal filters (Hz). Default 50.

    Returns
    -------
    None

    Examples
    --------
    >>> from baboon_acoustics.glottal import extract_glottal_train  # doctest: +SKIP
    >>> fig, ax = plt.subplots()
    >>> plot_phase_portrait(ax, glottal_train, fs)
    """
    from scipy.signal import hilbert

    from .glottal import normalize_amplitude

    signal_in = signal_in - np.mean(signal_in)
    if normalize:
        signal_in = normalize_amplitude(signal_in, fs, cutoff=cutoff)
    if np.max(np.abs(signal_in)) < 1e-12:
        ax.text(0.5, 0.5, "Null signal", transform=ax.transAxes, ha="center")
        return

    analytic = hilbert(signal_in)
    scale = np.max(np.abs(analytic))
    ax.plot(np.real(analytic) / scale, np.imag(analytic) / scale,
            color=color, alpha=alpha, linewidth=lw)
    ax.set_xlabel("Real Part $\\Re(z)$")
    ax.set_ylabel("Imaginary Part $\\Im(z)$")
    ax.set_title(title)
    ax.axhline(y=0, color="grey", linestyle=":", linewidth=0.5, alpha=0.5)
    ax.axvline(x=0, color="grey", linestyle=":", linewidth=0.5, alpha=0.5)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)


def plot_formant_comparison_panel(
    ax,
    times,
    formants_khz,
    reference_khz,
    title,
    letter,
    vtl_value,
    vtl_reference=None,
    segment_boundaries=(1, 9),
):
    """Plot estimated vs. reference formant tracks on one panel.

    Draws each formant track as a black line with colour-coded markers on
    top of a grey dashed reference track with open markers. Marker shapes
    encode the position within the vocalization: circle for the initial
    frame, squares for the medial frames, diamonds for the final frames.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axis.
    times : array_like
        Frame times (s).
    formants_khz : numpy.ndarray
        Estimated formant tracks in kHz, shape ``(n_formants, n_frames)``.
    reference_khz : numpy.ndarray
        Reference tracks in kHz, same shape.
    title : str
        Panel title, e.g. ``"N=4"``.
    letter : str
        Bold panel letter, e.g. ``"A"``.
    vtl_value : float
        VTL estimate (cm) of the plotted data, shown in the title.
    vtl_reference : float, optional
        VTL of the reference data (cm); when given, it is appended to the
        title.
    segment_boundaries : tuple of int, optional
        ``(medial_start, final_start)`` frame indices separating the
        initial/medial/final marker groups. Default ``(1, 9)``.

    Returns
    -------
    None

    Examples
    --------
    >>> fig, ax = plt.subplots()
    >>> ref = np.full((4, 14), 1.0)
    >>> est = np.full((4, 14), 1.1)
    >>> plot_formant_comparison_panel(
    ...     ax, np.linspace(0.05, 0.5, 14), est, ref, "N=4", "A", 17.9)
    """
    times = np.asarray(times)
    medial_start, final_start = segment_boundaries

    # Marker shape encodes the position within the vocalization: circle for
    # the initial frame(s), squares for the medial frames, diamonds for the
    # final frames.
    segments = [
        (slice(0, medial_start), "o", "red"),
        (slice(medial_start, final_start), "s", "blue"),
        (slice(final_start, None), "D", "limegreen"),
    ]

    reference_khz = np.asarray(reference_khz, dtype=float)
    formants_khz = np.asarray(formants_khz, dtype=float)

    for ref_track in reference_khz:
        ax.plot(times, ref_track, "--", color="gray", lw=1)
        for seg, marker, _ in segments:
            ax.plot(
                times[seg], ref_track[seg], marker,
                color="gray", ms=4 if marker == "o" else 3.5,
                fillstyle="none",
            )

    for est_track in formants_khz:
        ax.plot(times, est_track, "-k", lw=1)
        for seg, marker, color in segments:
            ax.plot(
                times[seg], est_track[seg], marker,
                color=color, ms=4 if marker == "o" else 3.5,
            )

    ax.set_xlim(0, 0.55)
    ax.set_ylim(0, 6.5)
    ax.set_xlabel("Time (s)", fontsize=10)
    ax.set_ylabel("Frequency (kHz)", fontsize=10)
    ax.set_xticks(np.arange(0, 0.61, 0.2))
    ax.set_yticks(np.arange(0, 6.5, 2.0))
    ax.tick_params(axis="both", labelsize=8, width=0.8, length=3)
    ax.grid(True, linestyle=":", linewidth=0.3, color="0.8")
    ax.set_box_aspect(1)

    if vtl_reference is not None:
        full_title = f"{title} – VTL = {vtl_value:.1f} cm (ref = {vtl_reference:.1f} cm)"
    else:
        full_title = f"{title} – VTL = {vtl_value:.1f} cm"
    ax.set_title(full_title)

    ax.text(-0.15, 1.03, letter, transform=ax.transAxes,
            fontsize=10, fontweight="bold")
