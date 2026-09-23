#!/usr/bin/env python3
"""Reproduce the F0 contour analysis (filtered autocorrelation).

Estimates the fundamental-frequency contour of a vocalization with
Praat's autocorrelation method (via Parselmouth), preceded by a Gaussian
low-pass pre-filter that simulates the "filtered autocorrelation"
strategy of recent Praat versions and suppresses subharmonic errors.
Writes a CSV table (time, F0, strength) and a PNG plot of the contour.

Usage::

    python scripts/make_figure_f0.py [audio.wav] [--output-dir DIR]

By default the script analyses ``data/wahoo_4.wav`` and writes
``wahoo_4_f0_filtered_ac.csv`` / ``.png`` into ``figures/``.
"""

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

# Make the package importable when the script is run from the repository.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from baboon_acoustics import estimate_f0_praat

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Search range and frame step used for the wahoo analysis (Hz, s).
PITCH_FLOOR = 30.0
PITCH_CEILING = 800.0
TIME_STEP = 0.01


def save_f0_csv(times, f0, strength, csv_path):
    """Write the voiced F0 frames to a CSV file with columns time,f0,strength."""
    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["time", "f0", "strength"])
        for row in zip(times, f0, strength):
            writer.writerow([float(v) for v in row])
    print(f"Results saved to: {csv_path}")


def plot_f0_track(times, f0, png_path):
    """Plot the F0 contour with the analysis settings in the title."""
    plt.figure(figsize=(14, 6))
    plt.plot(times, f0, color="blue", linewidth=1.5)
    plt.xlabel("Time (s)")
    plt.ylabel("F0 (Hz)")
    plt.title(
        "F0 contour using autocorrelation - "
        "with Gaussian pre\u2011filter (filtered AC simulation)\n"
        f"Range {PITCH_FLOOR}\u2013{PITCH_CEILING} Hz | "
        f"Step {TIME_STEP * 1000:.0f} ms"
    )
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(png_path, dpi=300)
    plt.close()
    print(f"Plot saved to: {png_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("audio", nargs="?", default=str(REPO_ROOT / "data" / "wahoo_4.wav"),
                        help="WAV file to analyse (default: data/wahoo_4.wav)")
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "figures"),
                        help="Directory for the outputs (default: figures/)")
    args = parser.parse_args()

    times, f0, strength = estimate_f0_praat(
        args.audio,
        time_step=TIME_STEP,
        pitch_floor=PITCH_FLOOR,
        pitch_ceiling=PITCH_CEILING,
    )
    if len(f0) == 0:
        sys.exit("No voiced frames detected.")
    print(f"Voiced frames: {len(f0)}")
    print(f"Mean F0: {f0.mean():.1f} Hz (std: {f0.std():.1f})")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.audio).stem
    save_f0_csv(times, f0, strength, output_dir / f"{stem}_f0_filtered_ac.csv")
    plot_f0_track(times, f0, output_dir / f"{stem}_f0_filtered_ac.png")


if __name__ == "__main__":
    main()
