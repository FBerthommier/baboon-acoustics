#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.signal import (
    hilbert, get_window, spectrogram, welch, lfilter, correlate,
    butter, filtfilt
)
import scipy.io.wavfile as wavfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os, sys, warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
FS_TARGET = 44100
IAIF_ORDER = 60
FFT_WIN_LEN = 4 * 1024
HOP = FFT_WIN_LEN // 4
F0_MIN, F0_MAX = 50, 500

WA = False
if WA:
    GLOT_START_TIME = 0.10
    GLOT_END_TIME   = 0.26
else:
    GLOT_START_TIME = 0.39
    GLOT_END_TIME   = 0.55

CUTENV = 50.0

# ============================================================
# LPC / IAIF ORIGINAL (LEVINSON-DURBIN)
# ============================================================
def lpc_levinson(x, order):
    x = x - np.mean(x)
    n = len(x)
    if n < order + 2:
        return np.ones(order + 1), 1e-12
    nf = 2 ** int(np.ceil(np.log2(2 * n - 1)))
    X = np.fft.rfft(x, n=nf)
    R = np.fft.irfft(np.abs(X) ** 2, n=nf)[:order + 1] / n
    a = np.zeros(order)
    e = R[0] + 1e-12
    for i in range(order):
        lam = R[i + 1]
        if i > 0:
            lam -= np.dot(a[:i], R[i:0:-1])
        k = lam / e
        anew = a.copy()
        anew[i] = k
        if i > 0:
            anew[:i] -= k * a[i - 1::-1]
        a = anew
        e *= max(1 - k ** 2, 1e-10)
    return np.r_[1., -a], float(e)

def iaif_frame(frame, fs, vt_order):
    Hg1, _ = lpc_levinson(frame, 1)
    s1 = lfilter(Hg1, [1.], frame)
    Hvt1, _ = lpc_levinson(s1, vt_order)
    g1 = np.cumsum(lfilter(Hvt1, [1.], frame))
    g1 -= np.mean(g1)
    Hg2, _ = lpc_levinson(g1, 2)
    s2 = lfilter(Hg2, [1.], frame)
    Hvt2, sigma2 = lpc_levinson(s2, vt_order)
    gn = np.cumsum(lfilter(Hvt2, [1.], frame))
    gn -= np.mean(gn)
    return Hvt2, sigma2, gn

def estimate_f0_from_gn(gn, fs, f0_min=F0_MIN, f0_max=F0_MAX, ac_thresh=0.3):
    gn = gn - np.mean(gn)
    r = correlate(gn, gn, mode='full')
    r = r[len(r)//2:]
    if len(r) == 0:
        return np.nan
    r = r / (r[0] + 1e-10)
    min_period = int(np.ceil(fs / f0_max))
    max_period = int(np.floor(fs / f0_min))
    if max_period >= len(r):
        max_period = len(r) - 1
    if min_period >= max_period:
        return np.nan
    sub_r = r[min_period:max_period]
    peak_idx_rel = np.argmax(sub_r)
    peak_val = sub_r[peak_idx_rel]
    if peak_val < ac_thresh:
        return np.nan
    peak_idx = peak_idx_rel + min_period
    if peak_idx == min_period or peak_idx == max_period - 1:
        return np.nan
    f0 = fs / peak_idx
    if f0 <= f0_min or f0 >= f0_max:
        return np.nan
    return f0

# ============================================================
# AMPLITUDE NORMALISATION (removes slow variations)
# ============================================================

def normalize_amplitude(signal, fs, cutoff=CUTENV, order=4, eps=1e-6):
    """
    Centers and normalises a glottal signal using a low‑pass filter
    for both the slow component (floating DC) and the slow envelope.

    Parameters
    ----------
    signal : array
        Input signal (glottal train).
    fs : float
        Sampling frequency (Hz).
    cutoff : float, optional
        Cutoff frequency of the filter (Hz). Default 30 Hz.
    order : int, optional
        Order of the Butterworth filter. Default 4.
    eps : float, optional
        Added to denominator to avoid division by zero.

    Returns
    -------
    signal_norm : array
        Centred and normalised signal (envelope amplitude ~1).
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low')
    
    signal_low = filtfilt(b, a, signal)
    envelope_raw = np.abs(signal)
    envelope_low = filtfilt(b, a, envelope_raw)
    
    signal_centered = signal - signal_low
    signal_norm = signal_centered / (envelope_low + eps)
    
    if len(signal_norm) > 1024:
        signal_trimmed = signal_norm[512:-512]
    else:
        signal_trimmed = signal_norm  
    
    return signal_trimmed  
    
# ============================================================
# PHASE PORTRAIT (with amplitude normalisation)
# ============================================================
def plot_phase_portrait(signal, fs, title="Phase Portrait (Hilbert)", ax=None,
                        color='k', alpha=0.7, lw=0.5, normalize=True):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    signal = signal - np.mean(signal)
    if normalize:
        signal = normalize_amplitude(signal, fs, cutoff=CUTENV)
    if np.max(np.abs(signal)) < 1e-12:
        ax.text(0.5, 0.5, 'Null signal', transform=ax.transAxes, ha='center')
        return ax
    analytic_signal = hilbert(signal)
    re = np.real(analytic_signal)
    im = np.imag(analytic_signal)
    norm_factor = np.max(np.abs(analytic_signal))
    re /= norm_factor
    im /= norm_factor
    ax.plot(re, im, color=color, alpha=alpha, linewidth=lw)
    ax.set_xlabel('Real Part $\\Re(z)$')
    ax.set_ylabel('Imaginary Part $\\Im(z)$')
    ax.set_title(title)
    ax.axhline(y=0, color='grey', linestyle=':', linewidth=0.5, alpha=0.5)
    ax.axvline(x=0, color='grey', linestyle=':', linewidth=0.5, alpha=0.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    return ax

# ============================================================
# OVERLAP-ADD
# ============================================================
def extract_full_glottal_train(x, fs, vt_order=IAIF_ORDER, win_len=FFT_WIN_LEN, hop=HOP):
    win = get_window('hann', win_len, fftbins=False)
    n_samples = len(x)
    glot_reconstructed = np.zeros(n_samples)
    window_sum = np.zeros(n_samples)
    n_frames = 1 + max((n_samples - win_len) // hop, 0)
    success_count = 0
    for i in range(n_frames):
        start = i * hop
        end = start + win_len
        frame = x[start:end]
        if len(frame) < win_len:
            frame = np.pad(frame, (0, win_len - len(frame)))
        frame_w = frame * win
        try:
            _, _, gn = iaif_frame(frame_w, fs, vt_order)
            gn_w = gn * win
            glot_reconstructed[start:end] += gn_w
            window_sum[start:end] += win
            success_count += 1
        except Exception as e:
            print(f"Frame {i}: IAIF failed ({e})")
    print(f"[IAIF reconstruction] {success_count}/{n_frames} frames successfully processed.")
    window_sum = np.maximum(window_sum, 1e-9)
    glot_signal = glot_reconstructed / window_sum
    valid = (window_sum > 0.5)
    if np.any(valid):
        first = np.argmax(valid)
        last = len(valid) - np.argmax(valid[::-1]) - 1
        glot_signal = glot_signal[first:last+1]
        t_glot = np.arange(len(glot_signal)) / fs
    else:
        t_glot = np.arange(len(glot_signal)) / fs
    return t_glot, glot_signal

def load_audio(path, target_fs=FS_TARGET):
    fs, x = wavfile.read(path)
    if x.ndim > 1:
        x = x[:, 0]
    if fs != target_fs:
        try:
            import parselmouth
            snd = parselmouth.Sound(x, fs)
            snd = snd.resample(target_fs)
            x = snd.values[0, :]
            fs = target_fs
        except ImportError:
            from scipy.signal import resample_poly
            from fractions import Fraction
            frac = Fraction(target_fs, fs)
            x = resample_poly(x, frac.numerator, frac.denominator)
            fs = target_fs
    return x, fs

# ============================================================
# MAIN FIGURE (A, B=normalised portrait, C, D)
# ============================================================

def plot_glottal_train_analysis(x, fs, out_png='glottal_train_analysis.png'):
    print("Reconstructing full glottal train...")
    t_full, glot_full = extract_full_glottal_train(x, fs)

    mask = (t_full >= GLOT_START_TIME) & (t_full <= GLOT_END_TIME)
    t_glot = t_full[mask]
    glot_signal = glot_full[mask]

    if len(glot_signal) == 0:
        print("Error: empty segment.")
        return
    print(f"Extracted segment: {len(glot_signal)} samples, max amplitude = {np.max(np.abs(glot_signal)):.4f}")

    f_glot, t_glot_spec_rel, Sxx_glot = spectrogram(
        glot_signal, fs, window='hann', nperseg=1024, noverlap=512,
        nfft=2048, mode='psd'
    )
    t_glot_spec_abs = t_glot_spec_rel + GLOT_START_TIME
    Sxx_glot_db = 10 * np.log10(Sxx_glot + 1e-12)

    f_welch, psd_welch = welch(
        glot_signal, fs, nperseg=4096, noverlap=2048,
        nfft=8192, window='hann', scaling='density'
    )
    psd_welch_db = 10 * np.log10(psd_welch + 1e-12)

    fig = plt.figure(figsize=(7.5, 5.5), dpi=300)
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1.2],
                           hspace=0.35, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t_glot * 1000, glot_signal, 'k-', lw=0.8)
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Amplitude (a.u.)')
    ax1.set_title(f'(A) Glottal pulse train ({GLOT_START_TIME*1000:.0f}-{GLOT_END_TIME*1000:.0f} ms)')
    ax1.grid(alpha=0.3)
    
    ax2 = fig.add_subplot(gs[0, 1])
    plot_phase_portrait(glot_signal, fs, title='(B) Phase Portrait (Amplitude Norm.)', ax=ax2,
                        color='steelblue', alpha=0.6, lw=0.3, normalize=True)
    
    ax3 = fig.add_subplot(gs[1, 0])
    im = ax3.pcolormesh(t_glot_spec_abs, f_glot/1000, Sxx_glot_db,
                        shading='auto', cmap='inferno')
    ax3.set_ylim(0, 5)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Frequency (kHz)')
    ax3.set_title('(C) Spectrogram of glottal train')
    plt.colorbar(im, ax=ax3, label='PSD (dB)')

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.semilogx(f_welch, psd_welch_db, 'k-', lw=1)
    ax4.set_xlim(50, 5000)
    ax4.set_ylim([np.max(psd_welch_db)-60, np.max(psd_welch_db)+5])
    ax4.set_xlabel('Frequency (Hz)')
    ax4.set_ylabel('Magnitude (dB)')
    ax4.set_title('(D) Long-term av. spectrum (Welch)')
    ax4.grid(True, alpha=0.3, which='both')

    f0_est = estimate_f0_from_gn(glot_signal, fs)
    if not np.isnan(f0_est):
        ax4.axvline(f0_est, color='r', linestyle='--', alpha=0.7, label=f'F0 = {f0_est:.1f} Hz')
        ax4.axvline(2*f0_est, color='green', linestyle='--', alpha=0.7, label='2×F0')
        ax4.axvline(f0_est/2, color='orange', linestyle=':', alpha=0.7, label='F0/2')
        if WA:
            ax4.axvline(3*f0_est/2, color='orange', linestyle=':', alpha=0.7, label='3*F0/2')
            ax4.axvline(5*f0_est/2, color='orange', linestyle=':', alpha=0.7, label='5*F0/2')
        
        ax4.legend(fontsize=5)

    plt.suptitle(f'Glottal source analysis (IAIF original, amplitude normalized) – {GLOT_START_TIME:.2f}s to {GLOT_END_TIME:.2f}s',
                 fontsize=10)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Figure saved: {out_png}")

# ============================================================
# EXECUTION – CORRECTED ARGUMENT HANDLING
# ============================================================
if __name__ == "__main__":
    # One single, correct argument check
    if len(sys.argv) > 1:
        wav_path = sys.argv[1]
    else:
        wav_path = 'wahoo_4.wav'   # default
    if not os.path.exists(wav_path):
        print(f"File {wav_path} not found.")
        sys.exit(1)

    x, fs = load_audio(wav_path)
    print(f"Loaded file: {len(x)} samples, duration = {len(x)/fs:.2f} s")
    if WA:
        plot_glottal_train_analysis(x, fs, 'glottal_train_analysis_wa.png')
    else:
        plot_glottal_train_analysis(x, fs, 'glottal_train_analysis_hoo.png')
        