#!/usr/bin/env python3
"""Guided tour of the ``baboon_acoustics`` package.

This script walks through every analysis step of the article, one section
per concept, on two example recordings:

- ``data/wahoo_4.wav`` -- a 0.67 s chacma baboon *wahoo* call (the
  vocalization re-analysed in the article);
- ``data/The_Baboon_Bark.wav`` -- a 16 s recording of baboon barks, to
  show that the same functions apply unchanged to other material.

Each section first explains, in plain words, what the functions do and
which parameters matter, then runs them. The script is meant to be read
top to bottom like a tutorial; run it with::

    python examples/example_analysis.py

Outputs (small figures and printed tables) are written to
``examples/output/``. Expect a few minutes of runtime: the sliding IAIF
analysis of the wahoo is the slowest step.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

# Make the package importable when the script is run from the repository.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from baboon_acoustics import (
    estimate_f0_from_gn,
    estimate_f0_praat,
    extract_glottal_pulse,
    extract_glottal_train,
    extract_praat_formants,
    iaif_sliding_analysis,
    load_audio,
    reby_mccomb_vtl,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
WAHOO = REPO_ROOT / "data" / "wahoo_4.wav"
BARK = REPO_ROOT / "data" / "The_Baboon_Bark.wav"


def section(title):
    """Print a visible section header so the console output stays readable."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ==========================================================================
# SECTION 1 -- Loading audio: load_audio()
# ==========================================================================
# ``load_audio(path, target_fs=44100, normalize=True)`` reads a WAV file and
# returns ``(samples, fs)``. It takes care of three chores for you:
#   1. multi-channel files are reduced to their first (left) channel;
#   2. integer samples are converted to floating-point numbers;
#   3. the waveform is resampled to ``target_fs`` (44.1 kHz by default) if
#      the file was recorded at another rate, so that all default analysis
#      parameters keep their usual meaning.
# With ``normalize=True`` the waveform is scaled to a maximum absolute
# value of 1, which is convenient for spectrograms; pass ``normalize=False``
# when absolute amplitudes matter (glottal-waveform displays).
def example_load_audio():
    section("1. Loading audio with load_audio()")

    x, fs = load_audio(WAHOO)
    print(f"wahoo_4.wav     : {len(x)} samples at {fs} Hz "
          f"({len(x) / fs:.2f} s), peak = {np.max(np.abs(x)):.3f}")

    x_bark, fs_bark = load_audio(BARK)
    print(f"The_Baboon_Bark: {len(x_bark)} samples at {fs_bark} Hz "
          f"({len(x_bark) / fs_bark:.2f} s), peak = {np.max(np.abs(x_bark)):.3f}")
    return x, fs, x_bark, fs_bark


# ==========================================================================
# SECTION 2 -- Formants with Praat: extract_praat_formants()
# ==========================================================================
# ``extract_praat_formants(path, n_formants, ...)`` runs Praat's
# "To Formant (burg)" command through the Parselmouth interface. The
# algorithm fits a linear-prediction (all-pole) filter to successive 40 ms
# windows; the resonances of that filter are reported as formants.
#
# The key parameter is ``n_formants``, the number of LPC pole pairs:
#   - too few poles: neighbouring formants merge into a single estimate;
#   - too many poles: extra poles land on harmonics or spurious peaks.
# For a ~17 cm vocal tract analysed up to 5.5 kHz, theory predicts about
# 5.5 poles; the article compares N=4 (the original study's setting) and
# N=6, and shows that N=4 only tracks the central part of the wahoo
# correctly.
#
# The function returns ``(times, formants)`` where ``formants`` has shape
# ``(n_formants, n_frames)`` in Hz, with NaN where Praat found nothing.
def example_praat_formants():
    section("2. Formant tracking with extract_praat_formants()")

    formants_by_n = {}
    for n in (4, 6):
        times, formants = extract_praat_formants(WAHOO, n_formants=n)
        formants_by_n[n] = formants
        f1 = formants[0]
        valid = ~np.isnan(f1)
        print(f"N={n}: {len(times)} frames, F1 range "
              f"{np.nanmin(f1):.0f}-{np.nanmax(f1):.0f} Hz "
              f"({np.sum(valid)} valid frames)")

    # The same call works on any recording -- here, on the bark. Note the
    # wider default settings still apply (5.5 kHz ceiling, 40 ms windows).
    times_bark, formants_bark = extract_praat_formants(BARK, n_formants=5)
    print(f"bark : {len(times_bark)} frames, mean F1 = "
          f"{np.nanmean(formants_bark[0]):.0f} Hz")
    return formants_by_n[4], formants_by_n[6], formants_bark


# ==========================================================================
# SECTION 3 -- Vocal tract length: reby_mccomb_vtl()
# ==========================================================================
# ``reby_mccomb_vtl(formants, c=350.0)`` turns formant tracks into a single
# vocal-tract-length (VTL) estimate, in centimetres, using the method of
# Reby & McComb (2003). For a tube closed at one end, formants fall near
# the odd quarter-wavelength resonances F_i = (2i-1) * c / (4 * VTL). The
# function collects every valid formant measurement as a point
# (order, frequency) -- orders 1, 3, 5, 7 for F1..F4 -- fits a line
# through the origin, and converts the slope into a length.
#
# Input can be the ``(n_formants, n_frames)`` matrix returned above; NaN
# entries are ignored automatically. Fewer than two valid values in total
# returns NaN.
def example_vtl(formants_n4, formants_n6, formants_bark):
    section("3. Vocal tract length with reby_mccomb_vtl()")

    vtl_n4 = reby_mccomb_vtl(formants_n4)
    vtl_n6 = reby_mccomb_vtl(formants_n6)
    vtl_bark = reby_mccomb_vtl(formants_bark)
    print(f"wahoo_4, N=4 poles : VTL = {vtl_n4:.1f} cm")
    print(f"wahoo_4, N=6 poles : VTL = {vtl_n6:.1f} cm")
    print(f"bark,    N=5 poles : VTL = {vtl_bark:.1f} cm")
    print("(anatomy check: an adult chacma baboon's oral tract is roughly "
          "8-12 cm; inflated values flag formant mis-tracking)")


# ==========================================================================
# SECTION 4 -- F0: estimate_f0_praat() and estimate_f0_from_gn()
# ==========================================================================
# Two F0 estimators are available:
#
# ``estimate_f0_praat(path, ...)`` runs Praat's autocorrelation pitch
# analysis on the file. With ``use_prefilter=True`` (default) the signal
# first passes through a Gaussian low-pass filter that mimics the
# "filtered autocorrelation" of recent Praat versions -- this suppresses
# the subharmonic (half-F0) errors that older versions make on calls with
# strong subharmonics, like the "wa" part of the wahoo. It returns three
# arrays (times, f0, strength) containing only the voiced frames.
#
# ``estimate_f0_from_gn(glottal, fs, ...)`` estimates F0 from the
# autocorrelation of a glottal waveform (Section 5). It is the estimator
# embedded in the sliding IAIF analysis; frames whose periodicity is too
# weak return NaN.
def example_f0(x, fs):
    section("4. F0 estimation: estimate_f0_praat() and estimate_f0_from_gn()")

    times, f0, strength = estimate_f0_praat(WAHOO)
    print(f"filtered AC  : {len(f0)} voiced frames, F0 {f0.min():.0f}-"
          f"{f0.max():.0f} Hz, mean {f0.mean():.0f} Hz")

    times_raw, f0_raw, _ = estimate_f0_praat(WAHOO, use_prefilter=False)
    print(f"raw AC       : {len(f0_raw)} voiced frames, F0 {f0_raw.min():.0f}-"
          f"{f0_raw.max():.0f} Hz, mean {f0_raw.mean():.0f} Hz")
    print("(compare the two: large discrepancies reveal nonlinear source "
          "dynamics, see the article)")

    # ... and the same analysis on the bark:
    t_bark, f0_bark, s_bark = estimate_f0_praat(BARK)
    print(f"bark         : {len(f0_bark)} voiced frames, mean F0 "
          f"{f0_bark.mean():.0f} Hz")

    # Plot the two wahoo contours side by side.
    fig, ax = plt.subplots(figsize=(7, 3), dpi=150)
    ax.plot(times, f0, "b-", lw=1.5, label="filtered AC")
    ax.plot(times_raw, f0_raw, "r-", lw=0.8, alpha=0.7, label="raw AC")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("F0 (Hz)")
    ax.set_title("wahoo_4.wav: F0 with and without Gaussian pre-filter")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "example_f0.png", dpi=150)
    plt.close(fig)
    print(f"saved {OUTPUT_DIR / 'example_f0.png'}")


# ==========================================================================
# SECTION 5 -- Glottal source: extract_glottal_pulse() / extract_glottal_train()
# ==========================================================================
# Iterative adaptive inverse filtering (IAIF; Alku, 1991) inverts the
# source-filter model: it estimates and removes the vocal-tract resonances
# so that what remains is (an approximation of) the glottal flow
# derivative. Two granularities are provided:
#
# ``extract_glottal_pulse(x, fs, t_start, t_end)`` isolates ONE glottal
# cycle from a short, stable segment. The pulse shape (skew, ripple)
# characterizes the phonation type.
#
# ``extract_glottal_train(x, fs)`` reconstructs the FULL pulse train over
# the whole recording by overlap-add of windowed IAIF estimates. Use
# ``normalize=False`` in load_audio to keep native amplitudes here.
#
# From the reconstructed train, ``estimate_f0_from_gn()`` (Section 4)
# gives a source-based F0, and slow amplitude drifts can be removed with
# ``normalize_amplitude()`` before drawing phase portraits.
def example_glottal():
    section("5. Glottal source: extract_glottal_pulse() / extract_glottal_train()")

    x, fs = load_audio(WAHOO, normalize=False)

    t_pulse, pulse = extract_glottal_pulse(x, fs, t_start=0.26, t_end=0.34)
    period_ms = len(pulse) / fs * 1000
    print(f"single pulse : {len(pulse)} samples ({period_ms:.2f} ms, i.e. "
          f"one period near F0 = {1000 / period_ms:.0f} Hz)")

    t_train, glottal_train = extract_glottal_train(x, fs)
    print(f"pulse train  : {len(glottal_train)} samples reconstructed by "
          "overlap-add")

    f0_source = estimate_f0_from_gn(glottal_train, fs)
    print(f"source F0    : {f0_source:.1f} Hz")

    # Show the pulse inside the reconstructed train.
    fig, axes = plt.subplots(2, 1, figsize=(7, 4), dpi=150)
    axes[0].plot(t_train, glottal_train, "k-", lw=0.5)
    axes[0].set_xlim(0.39, 0.55)
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude (a.u.)")
    axes[0].set_title("reconstructed glottal train (hoo segment)")
    axes[1].plot(t_pulse * 1000, pulse, "k-", lw=1.5)
    axes[1].set_xlabel("Time (ms)")
    axes[1].set_ylabel("Amplitude (a.u.)")
    axes[1].set_title("single glottal pulse")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "example_glottal.png", dpi=150)
    plt.close(fig)
    print(f"saved {OUTPUT_DIR / 'example_glottal.png'}")


# ==========================================================================
# SECTION 6 -- Full IAIF analysis: iaif_sliding_analysis()
# ==========================================================================
# ``iaif_sliding_analysis(x, fs, n_formants=4, ...)`` is the all-in-one
# analysis behind panels (E) and (F) of the article figure. It slides a
# 4096-sample Hann window over the signal (1024-sample hop) and, for each
# window, computes:
#   - the short-term spectrogram of the signal,
#   - the spectrogram of the IAIF glottal estimate,
#   - the vocal-tract envelope |Hvt| in dB,
#   - an F0 estimate from the glottal estimate,
#   - ``n_formants`` formants (peaks of the envelope below 3.5 kHz).
#
# The result is a named tuple: ``result.times``, ``result.freqs``,
# ``result.spectrogram_db``, ``result.glottal_db``, ``result.envelope_db``,
# ``result.f0_track``, ``result.formants`` (Hz, shape
# ``(n_formants, n_frames)``).
#
# Warning: cost scales with the number of windows, so for long recordings
# analyse a selected time span first (slicing ``x`` before calling).
def example_iaif(x, fs):
    section("6. Sliding IAIF analysis with iaif_sliding_analysis()")

    result = iaif_sliding_analysis(x, fs, n_formants=4)
    voiced = result.f0_track[~np.isnan(result.f0_track)]
    print(f"{len(result.times)} frames of "
          f"{len(result.freqs)} frequency bins")
    print(f"F0 track     : {len(voiced)} voiced frames, "
          f"{voiced.min():.0f}-{voiced.max():.0f} Hz")
    print(f"VTL from IAIF formants: {reby_mccomb_vtl(result.formants):.1f} cm")

    fig, axes = plt.subplots(2, 1, figsize=(7, 5), dpi=150, sharex=True)
    band = result.freqs <= 3500
    axes[0].pcolormesh(result.times, result.freqs[band] / 1000,
                       result.glottal_db[band], shading="nearest", cmap="Greys")
    axes[0].plot(result.times, result.f0_track / 1000, "c-", lw=2)
    axes[0].set_ylabel("Frequency (kHz)")
    axes[0].set_title("IAIF glottal source + F0 track")
    axes[1].pcolormesh(result.times, result.freqs[band] / 1000,
                       result.envelope_db[band], shading="nearest", cmap="Greys")
    for i, color in enumerate(("r", "g", "b")):
        axes[1].plot(result.times, result.formants[i] / 1000,
                     color=color, lw=1.5, label=f"F{i + 1}")
    axes[1].set_ylabel("Frequency (kHz)")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_title("IAIF vocal-tract envelope + formant tracks")
    axes[1].legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "example_iaif.png", dpi=150)
    plt.close(fig)
    print(f"saved {OUTPUT_DIR / 'example_iaif.png'}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    x, fs, _, _ = example_load_audio()

    formants_n4, formants_n6, formants_bark = example_praat_formants()
    example_vtl(formants_n4, formants_n6, formants_bark)
    example_f0(x, fs)
    example_glottal()
    example_iaif(x, fs)

    section("Done. All example outputs are in examples/output/")


if __name__ == "__main__":
    main()
