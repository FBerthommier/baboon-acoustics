#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Acoustic analysis for a reply article: 3x2 panels
(A) spectrogram + Parselmouth formants (N=4)
(B) spectrogram + Parselmouth formants (N=6)
(C) merged trajectories (extracted data)
(D) reference data from a previous study
(E) IAIF source spectrogram + fundamental frequency
(F) |Hvt2| envelope + IAIF formants
"""

import numpy as np
import scipy.signal as sig
import scipy.io.wavfile as wavfile
import scipy.fft as fft
import parselmouth
from parselmouth.praat import call
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
import sys, os, warnings
from pathlib import Path
warnings.filterwarnings('ignore')

print('Parselmouth version:', parselmouth.__version__)
print('Praat version:', parselmouth.PRAAT_VERSION)

# ============================================================
# JOURNAL STYLE (IJP / Springer)
# ============================================================
plt.rcParams.update({
    'font.family'      : 'Arial',
    'font.size'        : 8,
    'axes.labelsize'   : 8,
    'xtick.labelsize'  : 7,
    'ytick.labelsize'  : 7,
    'legend.fontsize'  : 6,
    'axes.linewidth'   : 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.major.size' : 3,
    'ytick.major.size' : 3
})

PANEL_LABEL_SIZE = 11

# ============================================================
# IAIF CONFIGURATION
# ============================================================
FS_TARGET = 44100
IAIF_ORDER = 60
FFT_WIN_LEN = 4*1024           # 4096
HOP = FFT_WIN_LEN // 4
N_FFT_SG = FFT_WIN_LEN * 2
FMAX = 3500
FIG_DPI = 600
SPEED_OF_SOUND = 350.0         # m/s

# Peak detection for formants
PEAK_PROMINENCE = 2.5   # dB
PEAK_MIN_DIST_HZ = 150  # min distance between peaks (Hz)

# F0 estimation range
F0_MIN = 50
F0_MAX = 500

# ============================================================
# GLOTTAL TRAIN ANALYSIS PARAMETERS
# ============================================================
WA = False
if WA:
    GLOT_START_TIME = 0.10         # début de l'analyse (secondes)
    GLOT_END_TIME   = 0.26         # fin de l'analyse (secondes) – 160 ms
else:
    GLOT_START_TIME = 0.39         # début de l'analyse (secondes)
    GLOT_END_TIME   = 0.55         # fin de l'analyse (secondes) – 160 ms

GLOT_ZOOM_PERIODS = 8          # nombre de périodes affichées dans le zoom

# ============================================================
# PARSELMOUTH FORMANT CONFIGURATION (Burg method)
# ============================================================
FMAX_PAR = 5500.0         # Hz
WINDOW_PAR = 0.040        # 40 ms
STEP_PAR = 0.040          # 40 ms
PRE_EMPH_PAR = 50.0       # Hz

# ============================================================
# FIXED MAS CONTOUR (from published data, in Hz)
# ============================================================
MAS_F2_kHz = np.array([
    2.62, 2.40, 2.00, 1.50, 1.00, 0.60, 0.35, 0.30, 0.38, 0.50,
    0.62, 0.75, 0.88, 1.00, 1.12, 1.22, 1.32, 1.42, 1.55, 1.70,
    1.88, 2.05, 2.22, 2.40, 2.55, 2.62
])
MAS_F1_kHz = np.array([
    0.210, 0.195, 0.185, 0.180, 0.180, 0.182, 0.190, 0.225, 0.320, 0.430,
    0.540, 0.660, 0.790, 0.900, 0.990, 1.050, 1.090, 1.105, 1.080, 1.020,
    0.930, 0.820, 0.650, 0.480, 0.320, 0.230
])
MAS_F2_Hz = MAS_F2_kHz * 1000
MAS_F1_Hz = MAS_F1_kHz * 1000
MAS_F2_closed = np.append(MAS_F2_Hz, MAS_F2_Hz[0])
MAS_F1_closed = np.append(MAS_F1_Hz, MAS_F1_Hz[0])

# ============================================================
# HARDCODED REFERENCE DATA FOR PANEL (D)
# (Extracted from a previous study, see main text)
# ============================================================
F1_ref = np.array([0.510, 0.795, 0.710, 0.771, 0.799, 0.846, 0.927, 0.842, 0.852, 0.70, 0.525, 0.435, 0.410, 0.395])
F2_ref = np.array([1.150, 1.859, 1.746, 1.788, 1.802, 1.817, 1.746, 1.746, 1.802, 1.20, 1.00, 1.02, 0.95, 0.87])

ref_marker_type = [
    'o',
    's','s','s','s','s','s','s','s',
    'D','D','D','D','D'
]
ref_marker_color = [
    'red',
    'blue','blue','blue','blue','blue','blue','blue','blue',
    'limegreen','limegreen','limegreen','limegreen','limegreen'
]

def plot_f1f2_trajectory(ax, F1, F2, marker_type, marker_color, xlim=(3.3, 0.0), ylim=(1.3, 0.1)):
    """
    Plot a formant trajectory in the F2-F1 plane.
    """
    ax.fill(
        MAS_F2_closed/1000,
        MAS_F1_closed/1000,
        alpha=0.10,
        color='gray'
    )
    ax.plot(
        MAS_F2_closed/1000,
        MAS_F1_closed/1000,
        'k-',
        lw=2.5
    )
    ax.set_xlim(3.3, 0.0)
    ax.set_ylim(1.3, 0.1)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1.0))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.4))
    ax.tick_params(labelsize=9, width=0.8, length=4)
    ax.set_xlabel('F2 (kHz)', fontsize=10)
    ax.set_ylabel('F1 (kHz)', fontsize=10)

    F1 = np.asarray(F1)
    F2 = np.asarray(F2)
    ax.plot(F2, F1, '-', color='0.5', linewidth=0.4, zorder=1)
    for x, y, m, c in zip(F2, F1, marker_type, marker_color):
        size = 28 if m == 'o' else 22
        ax.scatter(x, y, s=size, marker=m, c=c,
                   edgecolors='black', linewidths=0.4, zorder=3)
    ax.set_box_aspect(1)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)


# ----------------------------------------------------------------------
# VTL
# ----------------------------------------------------------------------

def reby_mccomb_vtl_global_3(F1, F2, F3, SPEED_OF_SOUND):
    """
    Compute a single VTL estimate (cm) from all frames.
    Parameters:
        F1, F2, F3, F4 : 1D arrays (same length) containing frequencies in Hz.
        c : speed of sound (m/s), default 350.
    Returns:
        VTL (cm) or NaN if fewer than two valid formants in total.
    """
    x_vals = []
    y_vals = []
    for f1, f2, f3 in zip(F1, F2, F3):
        for order, f in zip([1, 3, 5], [f1, f2, f3]):
            if not np.isnan(f):
                x_vals.append(order)
                y_vals.append(f)
    if len(x_vals) < 2:
        return np.nan
    x = np.array(x_vals)
    y = np.array(y_vals)
    b = np.sum(x * y) / np.sum(x**2)
    c_cm = c * 100.0
    vtl = c_cm / (4.0 * b)
    return vtl

def reby_mccomb_vtl_global(F1, F2, F3, F4, c=SPEED_OF_SOUND):
    """
    Compute a single VTL estimate (cm) from all frames.
    Parameters:
        F1, F2, F3, F4 : 1D arrays (same length) containing frequencies in Hz.
        c : speed of sound (m/s), default 350.
    Returns:
        VTL (cm) or NaN if fewer than two valid formants in total.
    """
    x_vals = []
    y_vals = []
    for f1, f2, f3, f4 in zip(F1, F2, F3, F4):
        for order, f in zip([1, 3, 5, 7], [f1, f2, f3, f4]):
            if not np.isnan(f):
                x_vals.append(order)
                y_vals.append(f)
    if len(x_vals) < 2:
        return np.nan
    x = np.array(x_vals)
    y = np.array(y_vals)
    b = np.sum(x * y) / np.sum(x**2)
    c_cm = c * 100.0
    vtl = c_cm / (4.0 * b)
    return vtl

# ============================================================
# AUDIO LOADING
# ============================================================
def load_audio(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Audio file not found: {path}")
    fs, d = wavfile.read(path)
    if d.ndim > 1:
        d = d[:, 0]
    d = d.astype(np.float64)
    d /= (np.max(np.abs(d)) + 1e-9)
    if fs != FS_TARGET:
        d = sig.resample(d, int(len(d) * FS_TARGET / fs))
        fs = FS_TARGET
    print(f"Loaded: {path}, fs={fs} Hz")
    return d, fs

# ============================================================
# LPC / IAIF FUNCTIONS
# ============================================================
def lpc_levinson(x, order):
    x = x - np.mean(x)
    n = len(x)
    if n < order + 2:
        return np.ones(order+1), 1e-12
    nf = 2**int(np.ceil(np.log2(2*n - 1)))
    X = np.fft.rfft(x, n=nf)
    R = np.fft.irfft(np.abs(X)**2, n=nf)[:order+1] / n
    a = np.zeros(order)
    e = R[0] + 1e-12
    for i in range(order):
        lam = R[i+1]
        if i > 0:
            lam -= np.dot(a[:i], R[i:0:-1])
        k = lam / e
        anew = a.copy()
        anew[i] = k
        if i > 0:
            anew[:i] -= k * a[i-1::-1]
        a = anew
        e *= max(1 - k**2, 1e-10)
    return np.r_[1., -a], float(e)

def lpc_spectrum_db(a, sigma2, n_fft=4096, fs=FS_TARGET):
    w, H = sig.freqz([np.sqrt(max(sigma2, 1e-20))], a, worN=n_fft, fs=fs)
    return w, 20*np.log10(np.abs(H) + 1e-10)

def iaif_frame(frame, fs, vt_order):
    Hg1, _ = lpc_levinson(frame, 1)
    s1 = sig.lfilter(Hg1, [1.], frame)
    Hvt1, _ = lpc_levinson(s1, vt_order)
    g1 = np.cumsum(sig.lfilter(Hvt1, [1.], frame))
    g1 -= np.mean(g1)
    Hg2, _ = lpc_levinson(g1, 2)
    s2 = sig.lfilter(Hg2, [1.], frame)
    Hvt2, sigma2 = lpc_levinson(s2, vt_order)
    gn = np.cumsum(sig.lfilter(Hvt2, [1.], frame))
    gn -= np.mean(gn)
    return Hvt2, sigma2, gn

def estimate_f0_from_gn(gn, fs, f0_min=F0_MIN, f0_max=F0_MAX, ac_thresh=0.3):
    gn = gn - np.mean(gn)
    r = np.correlate(gn, gn, mode='full')
    r = r[len(r)//2:]
    if len(r) == 0:
        return np.nan
    r = r / (r[0] + 1e-10)
    min_period = int(np.ceil(fs / f0_max))
    max_period = int(np.floor(fs / f0_min))
    if max_period > len(r):
        max_period = len(r)
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

def get_frame_formants_3(frame, fs, vt_order, fmax, min_prominence, min_dist_hz):
    Hvt2, sigma2, _ = iaif_frame(frame, fs, vt_order)
    f_lpc, lpc_db = lpc_spectrum_db(Hvt2, sigma2, n_fft=4096, fs=fs)
    mask = f_lpc <= fmax
    f_sub = f_lpc[mask]
    s_sub = lpc_db[mask]
    min_dist = max(1, int(min_dist_hz / (f_lpc[1] - f_lpc[0])))
    pks, props = sig.find_peaks(s_sub, prominence=min_prominence, distance=min_dist)
    if len(pks) == 0:
        return np.array([np.nan, np.nan, np.nan])
    order = np.argsort(props['prominences'])[::-1]
    top = np.sort(pks[order[:5]])
    f_peaks = f_sub[top]
    if len(f_peaks) >= 3:
        return f_peaks[:3]
    else:
        return np.append(f_peaks, [np.nan]*(3-len(f_peaks)))
        
def get_frame_formants(frame, fs, vt_order, fmax, min_prominence, min_dist_hz):
    Hvt2, sigma2, _ = iaif_frame(frame, fs, vt_order)
    f_lpc, lpc_db = lpc_spectrum_db(Hvt2, sigma2, n_fft=4096, fs=fs)
    mask = f_lpc <= fmax
    f_sub = f_lpc[mask]
    s_sub = lpc_db[mask]
    min_dist = max(1, int(min_dist_hz / (f_lpc[1] - f_lpc[0])))
    pks, props = sig.find_peaks(s_sub, prominence=min_prominence, distance=min_dist)
    if len(pks) == 0:
        return np.array([np.nan, np.nan, np.nan, np.nan])
    
    # Sort peaks by descending prominence to keep the most significant ones
    order = np.argsort(props['prominences'])[::-1]
    # Keep up to 6 candidates (more than 4 to avoid missing lower frequency peaks)
    top_candidates = pks[order[:6]]
    top_candidates = np.sort(top_candidates)  # sort by frequency
    # Take the first 4 in frequency
    if len(top_candidates) >= 4:
        f_peaks = f_sub[top_candidates[:4]]
    else:
        f_peaks = f_sub[top_candidates]
    # Pad with NaN if fewer than 4 formants
    if len(f_peaks) < 4:
        f_peaks = np.append(f_peaks, [np.nan] * (4 - len(f_peaks)))
    return f_peaks       

# ============================================================
# SLIDING IAIF ANALYSIS
# ============================================================
def iaif_sliding_analysis_3(x, fs):
    win = sig.get_window('hann', FFT_WIN_LEN)
    n_bins = N_FFT_SG // 2 + 1
    n_fr = 1 + max((len(x)-FFT_WIN_LEN)//HOP, 0)
    freqs = np.linspace(0, fs/2, n_bins)
    times = (np.arange(n_fr) * HOP + FFT_WIN_LEN/2) / fs

    Sxx = np.full((n_bins, n_fr), np.nan)
    Sgn = np.full((n_bins, n_fr), np.nan)
    SHvt2 = np.full((n_bins, n_fr), np.nan)
    f0_track = np.full(n_fr, np.nan)
    iaif_formants = np.full((3, n_fr), np.nan)

    print("Computing sliding IAIF (spectrograms, F0, formants, |Hvt2|)...")
    for i in range(n_fr):
        frame = x[i*HOP:i*HOP+FFT_WIN_LEN]
        if len(frame) < FFT_WIN_LEN:
            frame = np.pad(frame, (0, FFT_WIN_LEN-len(frame)))
        frame_w = frame * win

        Fx = fft.rfft(frame_w, n=N_FFT_SG)
        Sxx[:, i] = 20*np.log10(np.abs(Fx) + 1e-10)

        try:
            Hvt2, sigma2, gn = iaif_frame(frame_w, fs, IAIF_ORDER)
            gn_w = gn * win
            Fg = fft.rfft(gn_w, n=N_FFT_SG)
            Sgn[:, i] = 20*np.log10(np.abs(Fg) + 1e-10)

            f_lpc, env_db = lpc_spectrum_db(Hvt2, sigma2, n_fft=N_FFT_SG, fs=fs)
            env_interp = np.interp(freqs, f_lpc, env_db, left=np.nan, right=np.nan)
            SHvt2[:, i] = env_interp

            f0_track[i] = estimate_f0_from_gn(gn, fs)
            f1, f2, f3 = get_frame_formants(frame_w, fs, IAIF_ORDER,
                                            FMAX, PEAK_PROMINENCE, PEAK_MIN_DIST_HZ)
            iaif_formants[0, i] = f1
            iaif_formants[1, i] = f2
            iaif_formants[2, i] = f3
        except Exception:
            Sgn[:, i] = Sxx[:, i]
            SHvt2[:, i] = np.nan

    return times, freqs, Sxx, Sgn, SHvt2, f0_track, iaif_formants

def iaif_sliding_analysis(x, fs):
    win = sig.get_window('hann', FFT_WIN_LEN)
    n_bins = N_FFT_SG // 2 + 1
    n_fr = 1 + max((len(x)-FFT_WIN_LEN)//HOP, 0)
    freqs = np.linspace(0, fs/2, n_bins)
    times = (np.arange(n_fr) * HOP + FFT_WIN_LEN/2) / fs

    Sxx = np.full((n_bins, n_fr), np.nan)
    Sgn = np.full((n_bins, n_fr), np.nan)
    SHvt2 = np.full((n_bins, n_fr), np.nan)
    f0_track = np.full(n_fr, np.nan)
    iaif_formants = np.full((4, n_fr), np.nan)  #  4 formants

    print("Computing sliding IAIF (spectrograms, F0, formants, |Hvt2|)...")
    for i in range(n_fr):
        frame = x[i*HOP:i*HOP+FFT_WIN_LEN]
        if len(frame) < FFT_WIN_LEN:
            frame = np.pad(frame, (0, FFT_WIN_LEN-len(frame)))
        frame_w = frame * win

        Fx = fft.rfft(frame_w, n=N_FFT_SG)
        Sxx[:, i] = 20*np.log10(np.abs(Fx) + 1e-10)

        try:
            Hvt2, sigma2, gn = iaif_frame(frame_w, fs, IAIF_ORDER)
            gn_w = gn * win
            Fg = fft.rfft(gn_w, n=N_FFT_SG)
            Sgn[:, i] = 20*np.log10(np.abs(Fg) + 1e-10)

            f_lpc, env_db = lpc_spectrum_db(Hvt2, sigma2, n_fft=N_FFT_SG, fs=fs)
            env_interp = np.interp(freqs, f_lpc, env_db, left=np.nan, right=np.nan)
            SHvt2[:, i] = env_interp

            f0_track[i] = estimate_f0_from_gn(gn, fs)
            # get_frame_formants 4 values (F1..F4)
            f1, f2, f3, f4 = get_frame_formants(frame_w, fs, IAIF_ORDER,
                                                FMAX, PEAK_PROMINENCE, PEAK_MIN_DIST_HZ)
            iaif_formants[0, i] = f1
            iaif_formants[1, i] = f2
            iaif_formants[2, i] = f3
            iaif_formants[3, i] = f4            
        except Exception:
            Sgn[:, i] = Sxx[:, i]
            SHvt2[:, i] = np.nan
    # print(iaif_formants)      
    return times, freqs, Sxx, Sgn, SHvt2, f0_track, iaif_formants

def extract_and_save_glottal_pulse(x, fs, png_path):
    """
    Extract a single glottal pulse from the IAIF source estimate
    using a stable segment (e.g., vowel [a]).
    """
    t_start = 0.26
    t_end = 0.34
    ns = int(t_start * fs)
    ne = int(t_end * fs)
    segment = x[ns:ne].copy()
    if len(segment) < 1024:
        print("Segment too short, cannot extract glottal pulse.")
        return
    window = np.hamming(len(segment))
    windowed_seg = segment * window
    try:
        Hvt2, sigma2, gn = iaif_frame(windowed_seg, fs, IAIF_ORDER)
    except Exception as e:
        print(f"IAIF failed on segment: {e}")
        return
    # Autocorrelation to estimate fundamental period
    ac = np.correlate(gn, gn, mode='full')
    ac = ac[len(ac)//2:]
    ac = ac / (ac[0] + 1e-10)
    min_lag = int(np.ceil(fs / F0_MAX))
    max_lag = int(np.floor(fs / F0_MIN))
    if max_lag > len(ac):
        max_lag = len(ac)
    peak_idx = np.argmax(ac[min_lag:max_lag]) + min_lag
    period = peak_idx / fs
    abs_gn = np.abs(gn)
    max_idx = np.argmax(abs_gn)
    half_period = int(peak_idx // 2)
    start = max(max_idx - half_period, 0)
    end = min(max_idx + half_period, len(gn))
    pulse = gn[start:end]
    pulse_time = np.arange(len(pulse)) / fs
    plt.figure(figsize=(6, 3), dpi=300)
    plt.plot(pulse_time * 1000, pulse, 'k-', lw=1.5)
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude (a.u.)')
    plt.title('Glottal pulse estimated by IAIF')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Glottal pulse saved to: {png_path}")
    
# ============================================================
# PARSELMOUTH FORMANT EXTRACTION (generic n_form)
# ============================================================
def extract_parselmouth_formants_n(audio_path, n_form, target_times, fs_target=FS_TARGET):
    snd = parselmouth.Sound(audio_path)
    if snd.sampling_frequency != fs_target:
        snd = snd.resample(fs_target)
    formant = call(snd, "To Formant (burg)",
                   STEP_PAR, n_form, FMAX_PAR, WINDOW_PAR, PRE_EMPH_PAR)
    times_pars = formant.xs()
    F1_raw = np.array([formant.get_value_at_time(1, t) if t >= times_pars[0] and t <= times_pars[-1] else np.nan for t in times_pars])
    F2_raw = np.array([formant.get_value_at_time(2, t) if t >= times_pars[0] and t <= times_pars[-1] else np.nan for t in times_pars])
    F3_raw = np.array([formant.get_value_at_time(3, t) if t >= times_pars[0] and t <= times_pars[-1] else np.nan for t in times_pars])
    ok = ~np.isnan(F1_raw) & ~np.isnan(F2_raw) & ~np.isnan(F3_raw)
    if np.sum(ok) < 2:
        F1_interp = np.full_like(target_times, np.nan)
        F2_interp = np.full_like(target_times, np.nan)
        F3_interp = np.full_like(target_times, np.nan)
    else:
        F1_interp = np.interp(target_times, times_pars[ok], F1_raw[ok], left=np.nan, right=np.nan)
        F2_interp = np.interp(target_times, times_pars[ok], F2_raw[ok], left=np.nan, right=np.nan)
        F3_interp = np.interp(target_times, times_pars[ok], F3_raw[ok], left=np.nan, right=np.nan)
    return F1_interp, F2_interp, F3_interp, F1_raw, F2_raw, F3_raw

# ============================================================
# FIGURE UTILITIES
# ============================================================
def setup_frequency_axis(ax):
    ax.set_ylim(0, 3.5)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1.0))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax.tick_params(labelsize=9, width=0.8, length=4)
    ax.set_xlabel('Time (s)', fontsize=10)
    ax.set_ylabel('Frequency (kHz)', fontsize=10)

def add_panel_label(ax, label):
    ax.text(-0.05, 1.02, label, transform=ax.transAxes,
            fontsize=14, fontweight='bold', va='bottom', ha='left')

def fix_initial_nan(arr):
    """Replace the first element if NaN with the next valid value."""
    arr = arr.copy()
    if np.isnan(arr[0]):
        first_valid = np.argmax(~np.isnan(arr))
        if first_valid > 0:
            arr[0] = arr[first_valid]
    return arr

def make_figure(x, fs, audio_path, out_path):
    # IAIF analysis
    times, freqs, Sxx, Sgn, SHvt2, f0_track, iaif_formants = iaif_sliding_analysis(x, fs)
    vtl_iaif = reby_mccomb_vtl_global(iaif_formants[0,:], iaif_formants[1,:], iaif_formants[2,:], iaif_formants[3,:])
    print(f"VTL IAIF 4 formants : {vtl_iaif:.2f} cm")

    # Parselmouth formants for N=4 and N=6
    F1_pars4, F2_pars4, F3_pars4, F1_raw4, F2_raw4, _ = extract_parselmouth_formants_n(audio_path, 4, times, fs)
    F1_pars6, F2_pars6, F3_pars6, F1_raw8, F2_raw8, _ = extract_parselmouth_formants_n(audio_path, 6, times, fs)

    # Create figure: 3 rows, 2 columns
    fig = plt.figure(figsize=(7.0, 10.5), dpi=FIG_DPI)
    gs = gridspec.GridSpec(3, 2, hspace=0.18, wspace=0.18)

    axA = fig.add_subplot(gs[0, 0])  # (A) spectrogram + N=4 formants
    axB = fig.add_subplot(gs[0, 1])  # (B) spectrogram + N=6 formants
    axC = fig.add_subplot(gs[1, 0])  # (C) merged trajectories
    axD = fig.add_subplot(gs[1, 1])  # (D) reference data
    axE = fig.add_subplot(gs[2, 0])  # (E) IAIF source + F0
    axF = fig.add_subplot(gs[2, 1])  # (F) |Hvt2| envelope + IAIF formants

    for ax in [axA, axB, axC, axD, axE, axF]:
        ax.set_box_aspect(1)

    mask = freqs <= FMAX
    freqs_kHz = freqs[mask] / 1000.0
    tmin = times[0]
    tmax = times[-1]

    # ---------- Panel A: N=4 ----------
    imA = axA.pcolormesh(times, freqs_kHz, Sxx[mask], shading='nearest', cmap='Greys')
    axA.set_xlim(tmin, tmax)
    setup_frequency_axis(axA)
    F1_pars4 = fix_initial_nan(F1_pars4)
    F2_pars4 = fix_initial_nan(F2_pars4)
    F3_pars4 = fix_initial_nan(F3_pars4)
    mask_before = times <= 0.08
    mask_middle = (times > 0.08) & (times < 0.36)
    mask_after = times >= 0.36
    for formant in [F1_pars4, F2_pars4, F3_pars4]:
        if np.any(mask_before):
            axA.plot(times[mask_before], formant[mask_before]/1000, 'y-', lw=2.0)
        if np.any(mask_middle):
            axA.plot(times[mask_middle], formant[mask_middle]/1000, 'b-', lw=3.0)
        if np.any(mask_after):
            axA.plot(times[mask_after], formant[mask_after]/1000, 'y-', lw=2.0)
    axA.axvline(0.08, color='white', linestyle='--', linewidth=1.5)
    axA.axvline(0.36, color='white', linestyle='--', linewidth=1.5)
    add_panel_label(axA, '(A)')
    axA.text(0.85, 0.90, "N=4", transform=axA.transAxes, fontsize=11, fontweight='bold',
             bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))

    # ---------- Panel B: N=6 ----------
    imB = axB.pcolormesh(times, freqs_kHz, Sxx[mask], shading='nearest', cmap='Greys')
    axB.set_xlim(tmin, tmax)
    setup_frequency_axis(axB)
    F1_pars6 = fix_initial_nan(F1_pars6)
    F2_pars6 = fix_initial_nan(F2_pars6)
    F3_pars6 = fix_initial_nan(F3_pars6)
    for formant in [F1_pars6, F2_pars6, F3_pars6]:
        if np.any(mask_before):
            axB.plot(times[mask_before], formant[mask_before]/1000, 'r-', lw=3.0)
        if np.any(mask_middle):
            axB.plot(times[mask_middle], formant[mask_middle]/1000, 'y-', lw=2.0)
        if np.any(mask_after):
            axB.plot(times[mask_after], formant[mask_after]/1000, 'g-', lw=3.0)
    axB.axvline(0.08, color='white', linestyle='--', linewidth=1.5)
    axB.axvline(0.36, color='white', linestyle='--', linewidth=1.5)
    add_panel_label(axB, '(B)')
    axB.text(0.85, 0.90, "N=6", transform=axB.transAxes, fontsize=11, fontweight='bold',
             bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))

    # ---------- Panel C: merged trajectories ----------
    # Merge formants from raw data (first frame of N=6, frames 1..8 of N=4, then frames 9..13 of N=6)
    # Adjust indices: F1_raw8[0], F1_raw4[1:9] (8 frames), F1_raw8[9:14] (5 frames) -> total 14 points
    F1_merge = np.concatenate(([F1_raw8[0]], F1_raw4[1:9], F1_raw8[9:14])) / 1000
    F2_merge = np.concatenate(([F2_raw8[0]], F2_raw4[1:9], F2_raw8[9:14])) / 1000
    plot_f1f2_trajectory(axC, F1_merge, F2_merge, ref_marker_type, ref_marker_color,
                         xlim=(2.5, 0.5), ylim=(1.2, 0.2))
    axC.text(0.07, 0.05, "Combined data from panels (A-B)",
             transform=axC.transAxes, fontsize=11, fontweight='bold',
             bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
    add_panel_label(axC, '(C)')

    # ---------- Panel D: reference data (previous study) ----------
    plot_f1f2_trajectory(axD, F1_ref, F2_ref, ref_marker_type, ref_marker_color,
                         xlim=(2.5, 0.5), ylim=(1.2, 0.2))
    axD.text(0.035, 0.05, "Data extracted from Boë et al. (2026)",
             transform=axD.transAxes, fontsize=11, fontweight='bold',
             bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
    add_panel_label(axD, '(D)')

    # ---------- Panel E: IAIF source + F0 ----------
    imE = axE.pcolormesh(times, freqs_kHz, Sgn[mask], shading='nearest', cmap='Greys')
    axE.set_xlim(tmin, tmax)
    setup_frequency_axis(axE)
    axE.plot(times, f0_track/1000, color='cyan', lw=2.0, label='F0')
    axE.legend(loc='upper right', fontsize=7)
    add_panel_label(axE, '(E)')

    # ---------- Panel F: |Hvt2| envelope + IAIF formants ----------
    imF = axF.pcolormesh(times, freqs_kHz, SHvt2[mask], shading='nearest', cmap='Greys')
    axF.set_xlim(tmin, tmax)
    setup_frequency_axis(axF)
    axF.plot(times, iaif_formants[0,:]/1000, 'r-', lw=2.0, label='F1')
    axF.plot(times, iaif_formants[1,:]/1000, 'g-', lw=2.0, label='F2')
    axF.plot(times, iaif_formants[2,:]/1000, 'b-', lw=2.0, label='F3')
    axF.legend(loc='upper right', fontsize=7)
    add_panel_label(axF, '(F)')

    plt.subplots_adjust(left=0.09, right=0.98, bottom=0.06, top=0.98, wspace=0.18, hspace=0.18)
    plt.savefig(out_path, dpi=FIG_DPI, bbox_inches='tight')
    plt.close()
    print(f"Main figure saved: {out_path}")

# ============================================================
# MAIN - CORRECTED ARGUMENT HANDLING
# ============================================================
if __name__ == '__main__':
    # Read command-line argument or use default (anonymised)
    if len(sys.argv) > 1:
        wav_path = sys.argv[1]
    else:
        wav_path = 'wahoo_4.wav'
    
    # Convert to Path for existence check
    wav_path = Path(wav_path)
    if not wav_path.exists():
        print(f"File {wav_path} not found.")
        sys.exit(1)
    
    # Convert to string for functions that expect a path
    wav_str = str(wav_path)
    x, fs = load_audio(wav_str)
    
    # Output filenames remain fixed (as in original script)
    extract_and_save_glottal_pulse(x, fs, 'glottal_pulse.png')
    make_figure(x, fs, wav_str, 'analysis_3x2_panels.png')
