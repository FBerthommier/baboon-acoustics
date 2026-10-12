#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
f0_analysis_parselmouth.py
F0 analysis using Parselmouth (Praat) with optional Gaussian pre-filtering
to simulate Praat 6.4.65 "filtered autocorrelation" and suppress subharmonics.

Usage: python f0_analysis_parselmouth.py <audio.wav>
"""

import sys
import parselmouth
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from scipy.fft import fft, ifft, fftfreq
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# Analysis parameters (standard for intonation)
# ============================================================================
TIME_STEP = 0.01                # Time step (s) [10 ms]
PITCH_FLOOR = 30.0      # 75.0        # Minimum pitch (Hz)
PITCH_CEILING = 800.0           # Maximum pitch (Hz)
VOICING_THRESHOLD = 0.10  # 0.45      # Voicing threshold
OCTAVE_COST = 0.055             # Octave cost (adjusted for filtered AC)
OCTAVE_JUMP_COST = 0.35         # Octave jump cost
VOICED_UNVOICED_COST = 0.14     # Voiced/unvoiced cost
VERY_ACCURATE = False           # Very accurate mode (slower)

# ============================================================================
# Gaussian filter parameters to simulate Praat's filtered AC
# ============================================================================
USE_FILTER = True               # Enable/disable Gaussian pre-filtering
PITCH_TOP = 800                 # Pitch top (cutoff frequency in Hz)
ATTENUATION_AT_TOP = 0.03       # Attenuation at cutoff (linear, 0.03 ≈ -30.5 dB)
CUTOFF_FREQUENCY = PITCH_TOP    # Cutoff frequency (Hz)
# Sigma for Gaussian filter: H(f) = exp(-(f/sigma)^2/2)
# Solve for sigma such that H(CUTOFF) = ATTENUATION_AT_TOP
GAUSSIAN_SIGMA = CUTOFF_FREQUENCY / np.sqrt(-2 * np.log(ATTENUATION_AT_TOP))

# ============================================================================
# Modular functions
# ============================================================================

def load_sound(file_path: Path) -> parselmouth.Sound:
    """Load an audio file using Parselmouth."""
    return parselmouth.Sound(str(file_path))

def gaussian_lowpass_filter(sound: parselmouth.Sound) -> parselmouth.Sound:
    """
    Apply a Gaussian low-pass filter in the frequency domain.
    This mimics the pre‑filtering step of Praat's "filtered autocorrelation"
    and suppresses subharmonics.
    """
    # Extract mono samples as numpy array
    samples = sound.values[0]
    sampling_freq = sound.sampling_frequency

    # FFT
    N = len(samples)
    yf = fft(samples)
    xf = fftfreq(N, 1 / sampling_freq)

    # Gaussian filter for positive and negative frequencies
    gauss_filter = np.exp(-(xf / GAUSSIAN_SIGMA) ** 2 / 2)
    yf_filtered = yf * gauss_filter

    # Inverse FFT
    filtered_samples = np.real(ifft(yf_filtered))

    # Return a new Sound object
    return parselmouth.Sound(filtered_samples, sampling_freq)

def analyze_pitch_ac(sound: parselmouth.Sound) -> parselmouth.Pitch:
    """
    Pitch analysis using 'to_pitch_ac' (autocorrelation).
    Parameters match Praat's default for intonation analysis.
    """
    pitch = sound.to_pitch_ac(
        time_step=TIME_STEP,
        pitch_floor=PITCH_FLOOR,
        pitch_ceiling=PITCH_CEILING,
        voicing_threshold=VOICING_THRESHOLD,
        octave_cost=OCTAVE_COST,
        octave_jump_cost=OCTAVE_JUMP_COST,
        voiced_unvoiced_cost=VOICED_UNVOICED_COST,
        very_accurate=VERY_ACCURATE
    )
    return pitch

def extract_pitch_data(pitch_object: parselmouth.Pitch) -> pd.DataFrame:
    """
    Extract time, F0, and strength from a Pitch object.
    Keep only voiced frames (F0 > 0).
    """
    times = pitch_object.ts()
    freq = pitch_object.selected_array['frequency']
    strength = pitch_object.selected_array['strength']
    df = pd.DataFrame({
        'time': times,
        'f0': freq,
        'strength': strength
    })
    return df[df['f0'] > 0].copy()

def save_results(df: pd.DataFrame, wav_path: Path) -> Path:
    """Save the DataFrame as CSV."""
    suffix = "_f0_filtered_ac" if USE_FILTER else "_f0_raw_ac"
    output_csv = wav_path.with_name(wav_path.stem + suffix + ".csv")
    df.to_csv(output_csv, index=False)
    print(f"Results saved to: {output_csv}")
    return output_csv

def plot_f0_track(df: pd.DataFrame, wav_path: Path) -> Path:
    """Generate and save the F0 contour plot with English labels/legend."""
    plt.figure(figsize=(14, 6))
    plt.plot(df['time'], df['f0'], color='blue', linewidth=1.5)
    tabf0 = df['f0']
    print('F1 < 1.5*f0 Limit')
    print(1.5 * tabf0[::4])
    plt.xlabel('Time (s)')
    plt.ylabel('F0 (Hz)')
    title_type = "with Gaussian pre‑filter (filtered AC simulation)" if USE_FILTER else "without pre‑filter (raw AC)"
    plt.title(f'F0 contour using autocorrelation - {title_type}\n'
              f'Range {PITCH_FLOOR}–{PITCH_CEILING} Hz | Step {TIME_STEP*1000:.0f} ms')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    suffix = "_f0_filtered_ac" if USE_FILTER else "_f0_raw_ac"
    plot_path = wav_path.with_name(wav_path.stem + suffix + ".png")
    plt.savefig(plot_path, dpi=300)
    print(f"Plot saved to: {plot_path}")
    plt.close()
    return plot_path

# ============================================================
# MAIN – corrected argument handling (single, clean)
# ============================================================
if __name__ == "__main__":
    # Get audio file from command line, or use default
    if len(sys.argv) > 1:
        wav_path = Path(sys.argv[1])
    else:
        wav_path = Path('wahoo_4.wav')   # default

    if not wav_path.exists():
        print(f"File {wav_path} not found.")
        sys.exit(1)

    print(f"Loading {wav_path}...")
    sound = load_sound(wav_path)

    if USE_FILTER:
        print("Applying Gaussian low‑pass filter (simulates Praat's filtered AC)...")
        print(f"  Cutoff frequency: {CUTOFF_FREQUENCY} Hz")
        print(f"  Attenuation at cutoff: {ATTENUATION_AT_TOP:.2f} ({20*np.log10(ATTENUATION_AT_TOP):.1f} dB)")
        sound = gaussian_lowpass_filter(sound)

    print("Analyzing pitch using 'to_pitch_ac' (autocorrelation)...")
    print(f"Parameters: floor={PITCH_FLOOR} Hz, ceiling={PITCH_CEILING} Hz, time_step={TIME_STEP} s")

    pitch_obj = analyze_pitch_ac(sound)
    df = extract_pitch_data(pitch_obj)

    if df.empty:
        print("No voiced frames detected.")
        sys.exit(1)

    print(f"Voiced frames: {len(df)}")
    print(f"Mean F0: {df['f0'].mean():.1f} Hz (std: {df['f0'].std():.1f})")

    save_results(df, wav_path)
    plot_f0_track(df, wav_path)

    print("Analysis complete.")