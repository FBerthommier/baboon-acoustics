#!/usr/bin/env python3
"""Reproduce the six-panel analysis figure of the article.

Layout of the 3x2 grid:

(A) spectrogram + Parselmouth formants (N=4)
(B) spectrogram + Parselmouth formants (N=6)
(C) merged F1-F2 trajectory, extracted data
(D) reference F1-F2 trajectory from Boë et al. (2026)
(E) IAIF glottal-source spectrogram + F0 track
(F) IAIF vocal-tract envelope |Hvt2| + IAIF formant tracks

A single glottal pulse estimated by IAIF is saved alongside
(``glottal_pulse.png``).

Usage::

    python scripts/make_figure_multipanel.py [audio.wav] [--output-dir DIR]

By default the script analyses ``data/wahoo_4.wav`` and writes into
``figures/``.
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

# Make the package importable when the script is run from the repository.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from baboon_acoustics import (
    extract_glottal_pulse,
    extract_praat_formants,
    fill_leading_nan,
    iaif_sliding_analysis,
    interpolate_formants,
    load_audio,
    reby_mccomb_vtl,
    set_journal_style,
)
from baboon_acoustics.plotting import (
    add_panel_label,
    plot_f1f2_trajectory,
    setup_frequency_axis,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Formant/analysis parameters used in the article.
FMAX_IAIF = 3500.0          # highest frequency shown and searched (Hz)
N_FORMANTS_IAIF = 4         # formants tracked by the sliding IAIF analysis
TMASS_CSV = "vowel_space_mas_contour.csv"
REF_CSV = "reference_formants_boe2026.csv"

#: Segment boundaries of the wahoo (s): "wa" part, central part, "hoo" part.
#: They select which formant estimate (N=4 or N=6) is overlaid on each
#: segment in panels (A) and (B).
SEGMENT_BOUNDS = (0.08, 0.36)


def load_mas_contour():
    """Load the Maximal Acoustic Space (MAS) contour (kHz) from the data CSV."""
    data = np.genfromtxt(REPO_ROOT / "data" / TMASS_CSV, delimiter=",", names=True)
    return data["F1_kHz"], data["F2_kHz"]


def load_reference_khz():
    """Load the reference F1/F2 tracks of Boë et al. (2026) in kHz."""
    data = np.genfromtxt(REPO_ROOT / "data" / REF_CSV, delimiter=",", names=True)
    return data["F1_kHz"], data["F2_kHz"]


def marker_specs(n_frames):
    """Marker type and color for each frame of the F1-F2 trajectories.

    Circle for the initial frame, squares for the medial frames, diamonds
    for the final frames; red/blue/limegreen as in the article.
    """
    types = ["o"] + ["s"] * 8 + ["D"] * max(n_frames - 9, 0)
    colors = ["red"] + ["blue"] * 8 + ["limegreen"] * max(n_frames - 9, 0)
    return types[:n_frames], colors[:n_frames]


def draw_spectrogram_panel(ax, times, freqs_khz, spectrogram_db, title=None):
    """Draw one grey-scale spectrogram panel with standard axes."""
    ax.pcolormesh(times, freqs_khz, spectrogram_db, shading="nearest", cmap="Greys")
    ax.set_xlim(times[0], times[-1])
    setup_frequency_axis(ax)
    if title is not None:
        ax.text(0.85, 0.90, title, transform=ax.transAxes, fontsize=11,
                fontweight="bold",
                bbox=dict(facecolor="white", edgecolor="black",
                          boxstyle="round,pad=0.5"))


def overlay_formant_segments(ax, times, formant_hz, segment_styles):
    """Overlay a formant track on a spectrogram, styled per segment.

    ``segment_styles`` maps the three segments (before / between / after
    ``SEGMENT_BOUNDS``) to ``(color, linewidth)`` pairs.
    """
    before = times <= SEGMENT_BOUNDS[0]
    middle = (times > SEGMENT_BOUNDS[0]) & (times < SEGMENT_BOUNDS[1])
    after = times >= SEGMENT_BOUNDS[1]
    masks = (before, middle, after)
    for track in formant_hz:
        for mask, (color, lw) in zip(masks, segment_styles):
            if np.any(mask):
                ax.plot(times[mask], track[mask] / 1000, "-", color=color, lw=lw)
    for bound in SEGMENT_BOUNDS:
        ax.axvline(bound, color="white", linestyle="--", linewidth=1.5)


def save_glottal_pulse_figure(x, fs, t_start, t_end, out_path):
    """Extract and save one IAIF glottal pulse as a standalone figure."""
    from baboon_acoustics.formants_iaif import DEFAULT_VT_ORDER

    t_pulse, pulse = extract_glottal_pulse(
        x, fs, t_start, t_end, vt_order=DEFAULT_VT_ORDER
    )
    if pulse is None:
        print("Segment too short, cannot extract glottal pulse.")
        return
    plt.figure(figsize=(6, 3), dpi=300)
    plt.plot(t_pulse * 1000, pulse, "k-", lw=1.5)
    plt.xlabel("Time (ms)")
    plt.ylabel("Amplitude (a.u.)")
    plt.title("Glottal pulse estimated by IAIF")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Glottal pulse saved to: {out_path}")


def make_figure(x, fs, audio_path, out_path, pulse_path):
    """Assemble the six-panel figure and save it."""
    # --- Sliding IAIF analysis -------------------------------------------
    result = iaif_sliding_analysis(x, fs, n_formants=N_FORMANTS_IAIF)
    vtl_iaif = reby_mccomb_vtl(result.formants)
    print(f"VTL IAIF {N_FORMANTS_IAIF} formants: {vtl_iaif:.2f} cm")

    # --- Parselmouth tracks, interpolated on the IAIF time grid ---------
    _, raw4 = extract_praat_formants(audio_path, n_formants=4)
    times6, raw6 = extract_praat_formants(audio_path, n_formants=6)
    pars4 = interpolate_formants(times6, raw4, result.times)
    pars6 = interpolate_formants(times6, raw6, result.times)
    pars4 = np.array([fill_leading_nan(track) for track in pars4])
    pars6 = np.array([fill_leading_nan(track) for track in pars6])

    mas_f1, mas_f2 = load_mas_contour()
    ref_f1, ref_f2 = load_reference_khz()

    # --- Figure ------------------------------------------------------------
    fig = plt.figure(figsize=(7.0, 10.5), dpi=600)
    gs = gridspec.GridSpec(3, 2, hspace=0.18, wspace=0.18)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, 0])
    axD = fig.add_subplot(gs[1, 1])
    axE = fig.add_subplot(gs[2, 0])
    axF = fig.add_subplot(gs[2, 1])
    for ax in (axA, axB, axC, axD, axE, axF):
        ax.set_box_aspect(1)

    band = result.freqs <= FMAX_IAIF
    freqs_khz = result.freqs[band] / 1000.0
    times = result.times

    # Panel (A): spectrogram + N=4 tracks
    draw_spectrogram_panel(axA, times, freqs_khz, result.spectrogram_db[band], "N=4")
    overlay_formant_segments(
        axA, times, pars4,
        (("yellow", 2.0), ("blue", 3.0), ("yellow", 2.0)))
    add_panel_label(axA, "(A)", fontsize=14)

    # Panel (B): spectrogram + N=6 tracks
    draw_spectrogram_panel(axB, times, freqs_khz, result.spectrogram_db[band], "N=6")
    overlay_formant_segments(
        axB, times, pars6,
        (("red", 3.0), ("yellow", 2.0), ("green", 3.0)))
    add_panel_label(axB, "(B)", fontsize=14)

    # Panel (C): merged trajectory in the F1-F2 plane
    merged = np.concatenate(
        [raw6[:4, :1], raw4[:4, 1:9], raw6[:4, 9:14]], axis=1
    ) / 1000.0
    types, colors = marker_specs(merged.shape[1])
    plot_f1f2_trajectory(axC, merged[0], merged[1], types, colors, mas_f1, mas_f2)
    axC.text(0.07, 0.05, "Combined data from panels (A-B)",
             transform=axC.transAxes, fontsize=11, fontweight="bold",
             bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.5"))
    add_panel_label(axC, "(C)", fontsize=14)

    # Panel (D): reference trajectory from the previous study
    plot_f1f2_trajectory(axD, ref_f1, ref_f2, types, colors, mas_f1, mas_f2)
    axD.text(0.035, 0.05, "Data extracted from Boë et al. (2026)",
             transform=axD.transAxes, fontsize=11, fontweight="bold",
             bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.5"))
    add_panel_label(axD, "(D)", fontsize=14)

    # Panel (E): IAIF glottal source + F0 track
    axE.pcolormesh(times, freqs_khz, result.glottal_db[band],
                   shading="nearest", cmap="Greys")
    axE.set_xlim(times[0], times[-1])
    setup_frequency_axis(axE)
    axE.plot(times, result.f0_track / 1000, color="cyan", lw=2.0, label="F0")
    axE.legend(loc="upper right", fontsize=7)
    add_panel_label(axE, "(E)", fontsize=14)

    # Panel (F): IAIF vocal-tract envelope + formant tracks
    axF.pcolormesh(times, freqs_khz, result.envelope_db[band],
                   shading="nearest", cmap="Greys")
    axF.set_xlim(times[0], times[-1])
    setup_frequency_axis(axF)
    axF.plot(times, result.formants[0] / 1000, "r-", lw=2.0, label="F1")
    axF.plot(times, result.formants[1] / 1000, "g-", lw=2.0, label="F2")
    axF.plot(times, result.formants[2] / 1000, "b-", lw=2.0, label="F3")
    axF.legend(loc="upper right", fontsize=7)
    add_panel_label(axF, "(F)", fontsize=14)

    plt.subplots_adjust(left=0.09, right=0.98, bottom=0.06, top=0.98,
                        wspace=0.18, hspace=0.18)
    fig.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Main figure saved: {out_path}")

    save_glottal_pulse_figure(x, fs, t_start=0.26, t_end=0.34, out_path=pulse_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("audio", nargs="?", default=str(REPO_ROOT / "data" / "wahoo_4.wav"),
                        help="WAV file to analyse (default: data/wahoo_4.wav)")
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "figures"),
                        help="Directory for the output PNGs (default: figures/)")
    args = parser.parse_args()

    set_journal_style()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    x, fs = load_audio(args.audio, normalize=True)
    make_figure(
        x, fs, args.audio,
        out_path=output_dir / "analysis_3x2_panels.png",
        pulse_path=output_dir / "glottal_pulse.png",
    )


if __name__ == "__main__":
    main()
