#!/usr/bin/env python3
"""Reproduce Figure 1 of the article: Parselmouth formant re-analysis.

Three panels comparing formant tracks of the baboon wahoo vocalization
extracted with Praat's Burg method (N=4 poles, N=6 poles, and a merged
trajectory using the optimal N per segment), against the reference tracks
graphically extracted from Boë et al. (2026, their Figure 10h). Each
panel title shows the corresponding vocal-tract-length estimate.

Usage::

    python scripts/make_figure1.py [audio.wav] [--output-dir DIR]

By default the script analyses ``data/wahoo_4.wav`` and writes
``Figure1.png`` into ``figures/``.
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# Make the package importable when the script is run from the repository.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from baboon_acoustics import extract_praat_formants, reby_mccomb_vtl, set_journal_style
from baboon_acoustics.plotting import plot_formant_comparison_panel

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_reference_formants(csv_path):
    """Read the reference formant tracks (kHz) from the CSV data file.

    Returns a structured array with fields ``F1_kHz`` ... ``F4_kHz``,
    one entry per analysis frame.
    """
    return np.genfromtxt(csv_path, delimiter=",", names=True)


def build_merged_tracks(formants_n4, formants_n6):
    """Merge N=4 and N=6 tracks segment by segment.

    Follows the segment boundaries of the published analysis: the initial
    frame comes from N=6, frames 1-8 (central segment) from N=4, and
    frames 9+ (final segment) from N=6.

    Parameters
    ----------
    formants_n4, formants_n6 : numpy.ndarray
        Formant matrices of shape ``(n_formants, n_frames)`` in Hz.

    Returns
    -------
    merged : numpy.ndarray
        Merged tracks, shape ``(n_formants, 14)`` in Hz.
    """
    merged = np.concatenate(
        [formants_n6[:4, :1], formants_n4[:4, 1:9], formants_n6[:4, 9:15]],
        axis=1,
    )
    return merged[:, :14]


def add_comparison_legend(fig):
    """Add the shared legend describing markers and line styles."""
    handles = [
        Line2D([0], [0], color="gray", ls="--", lw=1, label="Graphical reference"),
        Line2D([0], [0], color="black", lw=1, label="Parselmouth estimate"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="red",
               markeredgecolor="black", markersize=5, label="Initial"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="blue",
               markeredgecolor="black", markersize=5, label="Medial"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="limegreen",
               markeredgecolor="black", markersize=5, label="Final"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, -0.03))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("audio", nargs="?", default=str(REPO_ROOT / "data" / "wahoo_4.wav"),
                        help="WAV file to analyse (default: data/wahoo_4.wav)")
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "figures"),
                        help="Directory for the output PNG (default: figures/)")
    args = parser.parse_args()

    set_journal_style(tick_size=7, legend_size=7, linewidth=0.6,
                      titlesize=8, embed_fonts=True)

    # --- Formant extraction with N=4 and N=6 LPC poles -----------------
    times4, formants4 = extract_praat_formants(args.audio, n_formants=4)
    times6, formants6 = extract_praat_formants(args.audio, n_formants=6)
    times = times6

    # --- Merged trajectory and reference data ---------------------------
    merged = build_merged_tracks(formants4, formants6)
    reference_khz = load_reference_formants(REPO_ROOT / "data" / "reference_formants_boe2026.csv")
    reference_hz = np.column_stack([
        reference_khz["F1_kHz"], reference_khz["F2_kHz"],
        reference_khz["F3_kHz"], reference_khz["F4_kHz"],
    ]).T * 1000.0

    # --- VTL estimates (first 14 frames, as in the article) -------------
    vtl_n4 = reby_mccomb_vtl(formants4[:, :14])
    vtl_n6 = reby_mccomb_vtl(formants6[:4, :14])
    vtl_merged = reby_mccomb_vtl(merged)
    vtl_reference = reby_mccomb_vtl(reference_hz)
    print(f"VTL Parselmouth (N=4)   : {vtl_n4:.2f} cm")
    print(f"VTL Parselmouth (N=6)   : {vtl_n6:.2f} cm")
    print(f"VTL Parselmouth (merged): {vtl_merged:.2f} cm")
    print(f"VTL reference           : {vtl_reference:.2f} cm")

    # --- Three-panel figure ----------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8), dpi=600)
    plt.subplots_adjust(left=0.08, right=0.99, bottom=0.22, top=0.85, wspace=0.35)

    t14 = times[:14]
    panels = [
        (formants4[:, :14], "N=4", "A", vtl_n4, None),
        (formants6[:4, :14], "N=6", "B", vtl_n6, None),
        (merged, "Merged", "C", vtl_merged, vtl_reference),
    ]
    for ax, (formants, title, letter, vtl, vtl_ref) in zip(axes, panels):
        plot_formant_comparison_panel(
            ax, t14, formants / 1000.0, reference_hz / 1000.0,
            title=title, letter=letter, vtl_value=vtl, vtl_reference=vtl_ref,
        )
    add_comparison_legend(fig)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "Figure1.png"
    fig.savefig(out_path, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Figure saved: {out_path}")


if __name__ == "__main__":
    main()
