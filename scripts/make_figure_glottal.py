#!/usr/bin/env python3
"""Reproduce the glottal-source analysis figure (four panels).

The full glottal pulse train of the vocalization is reconstructed by
overlap-add of windowed IAIF estimates, then a 160 ms segment is analysed:

(A) glottal pulse train waveform
(B) Hilbert phase portrait (amplitude normalized)
(C) spectrogram of the glottal train
(D) long-term average spectrum (Welch) with F0 markers

Usage::

    python scripts/make_figure_glottal.py [audio.wav] [--segment {hoo,wa}]
                                          [--output-dir DIR]

``--segment hoo`` analyses 0.39-0.55 s (the "hoo" part of the wahoo,
default); ``--segment wa`` analyses 0.10-0.26 s (the "wa" part).
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import spectrogram, welch

# Make the package importable when the script is run from the repository.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from baboon_acoustics import estimate_f0_from_gn, extract_glottal_train, load_audio
from baboon_acoustics.plotting import plot_phase_portrait

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Analysis segments (s) for the two parts of the wahoo vocalization.
SEGMENTS = {
    "hoo": (0.39, 0.55),
    "wa": (0.10, 0.26),
}


def plot_glottal_train_analysis(x, fs, segment, out_png):
    """Assemble and save the four-panel glottal-source figure.

    Parameters
    ----------
    x : numpy.ndarray
        Input waveform (native amplitudes, i.e. ``normalize=False``).
    fs : float
        Sampling frequency in Hz.
    segment : tuple of float
        ``(t_start, t_end)`` of the analysed segment in seconds.
    out_png : str or pathlib.Path
        Output file path.
    """
    t_start, t_end = segment
    print("Reconstructing full glottal train...")
    t_full, glot_full = extract_glottal_train(x, fs)

    mask = (t_full >= t_start) & (t_full <= t_end)
    t_glot = t_full[mask]
    glot_signal = glot_full[mask]
    if len(glot_signal) == 0:
        raise RuntimeError(f"Empty segment: no samples between {t_start} and {t_end} s")
    print(f"Extracted segment: {len(glot_signal)} samples, "
          f"max amplitude = {np.max(np.abs(glot_signal)):.4f}")

    f_spec, t_spec_rel, Sxx = spectrogram(
        glot_signal, fs, window="hann", nperseg=1024, noverlap=512,
        nfft=2048, mode="psd",
    )
    Sxx_db = 10 * np.log10(Sxx + 1e-12)
    f_psd, psd = welch(
        glot_signal, fs, nperseg=4096, noverlap=2048, nfft=8192,
        window="hann", scaling="density",
    )
    psd_db = 10 * np.log10(psd + 1e-12)

    fig = plt.figure(figsize=(7.5, 5.5), dpi=300)
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1.2],
                           hspace=0.35, wspace=0.3)

    # (A) glottal pulse train
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t_glot * 1000, glot_signal, "k-", lw=0.8)
    ax1.set_xlabel("Time (ms)")
    ax1.set_ylabel("Amplitude (a.u.)")
    ax1.set_title(f"(A) Glottal pulse train ({t_start * 1000:.0f}-{t_end * 1000:.0f} ms)")
    ax1.grid(alpha=0.3)

    # (B) phase portrait
    ax2 = fig.add_subplot(gs[0, 1])
    plot_phase_portrait(ax2, glot_signal, fs,
                        title="(B) Phase Portrait (Amplitude Norm.)",
                        color="steelblue", alpha=0.6, lw=0.3, normalize=True)

    # (C) spectrogram of the glottal train
    ax3 = fig.add_subplot(gs[1, 0])
    im = ax3.pcolormesh(t_spec_rel + t_start, f_spec / 1000, Sxx_db,
                        shading="auto", cmap="inferno")
    ax3.set_ylim(0, 5)
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Frequency (kHz)")
    ax3.set_title("(C) Spectrogram of glottal train")
    plt.colorbar(im, ax=ax3, label="PSD (dB)")

    # (D) long-term average spectrum with F0 markers
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.semilogx(f_psd, psd_db, "k-", lw=1)
    ax4.set_xlim(50, 5000)
    ax4.set_ylim([np.max(psd_db) - 60, np.max(psd_db) + 5])
    ax4.set_xlabel("Frequency (Hz)")
    ax4.set_ylabel("Magnitude (dB)")
    ax4.set_title("(D) Long-term av. spectrum (Welch)")
    ax4.grid(True, alpha=0.3, which="both")

    f0 = estimate_f0_from_gn(glot_signal, fs)
    if not np.isnan(f0):
        ax4.axvline(f0, color="r", linestyle="--", alpha=0.7, label=f"F0 = {f0:.1f} Hz")
        ax4.axvline(2 * f0, color="green", linestyle="--", alpha=0.7, label="2×F0")
        ax4.axvline(f0 / 2, color="orange", linestyle=":", alpha=0.7, label="F0/2")
        if segment == SEGMENTS["wa"]:
            # Subharmonics of the "wa" part appear at odd multiples of F0/2.
            ax4.axvline(3 * f0 / 2, color="orange", linestyle=":", alpha=0.7,
                        label="3*F0/2")
            ax4.axvline(5 * f0 / 2, color="orange", linestyle=":", alpha=0.7,
                        label="5*F0/2")
        ax4.legend(fontsize=5)

    plt.suptitle(
        f"Glottal source analysis (IAIF original, amplitude normalized) – "
        f"{t_start:.2f}s to {t_end:.2f}s",
        fontsize=10,
    )
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Figure saved: {out_png}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("audio", nargs="?", default=str(REPO_ROOT / "data" / "wahoo_4.wav"),
                        help="WAV file to analyse (default: data/wahoo_4.wav)")
    parser.add_argument("--segment", choices=sorted(SEGMENTS), default="hoo",
                        help="part of the wahoo to analyse (default: hoo)")
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "figures"),
                        help="Directory for the output PNG (default: figures/)")
    args = parser.parse_args()

    # Native amplitudes: the glottal waveform is displayed in a.u. and the
    # IAIF decomposition is scale-invariant.
    x, fs = load_audio(args.audio, normalize=False)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_png = output_dir / f"glottal_train_analysis_{args.segment}.png"
    plot_glottal_train_analysis(x, fs, SEGMENTS[args.segment], out_png)


if __name__ == "__main__":
    main()
