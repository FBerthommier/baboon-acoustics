import numpy as np
import matplotlib.pyplot as plt
import parselmouth
from parselmouth.praat import call
from pathlib import Path
from matplotlib.lines import Line2D

# ============================================================
# IJP FORMAT
# ============================================================

plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 8,
    'axes.labelsize': 8,
    'axes.titlesize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'pdf.fonttype': 42,
    'ps.fonttype': 42
})

FS_TARGET = 44100

# ============================================================
# PARSELMOUTH FORMANT CONFIGURATION (Burg method)
# ============================================================
FMAX_PAR = 5500.0         # Hz
WINDOW_PAR = 0.040        # 40 ms
STEP_PAR = 0.040          # 40 ms
PRE_EMPH_PAR = 50.0       # Hz

# ----------------------------------------------------------------------
# Formant extraction functions (Parselmouth)
# ----------------------------------------------------------------------
def extract_parselmouth_formants_n(audio_path, n_formants, fs_target=FS_TARGET):
    """
    Extract the first n formants using Parselmouth (Burg method).
    Returns (times, F1, F2, F3, F4) where F4 = None if n_formants < 4.
    """
    snd = parselmouth.Sound(audio_path)
    if snd.sampling_frequency != fs_target:
        snd = snd.resample(fs_target)
    formants = call(snd, "To Formant (burg)",
                   STEP_PAR, n_formants, FMAX_PAR, WINDOW_PAR, PRE_EMPH_PAR)
    times = formants.xs()
    F1 = np.array([formants.get_value_at_time(1, t) for t in times])
    F2 = np.array([formants.get_value_at_time(2, t) for t in times])
    F3 = np.array([formants.get_value_at_time(3, t) for t in times]) if n_formants >= 3 else None
    F4 = np.array([formants.get_value_at_time(4, t) for t in times]) if n_formants >= 4 else None
    return times, F1/1000.0, F2/1000.0, F3/1000.0, F4/1000.0

def reby_mccomb_vtl_global(F1, F2, F3, F4, c=0.350):
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

# ----------------------------------------------------------------------
# Plotting function on a given axis (for panel)
# ----------------------------------------------------------------------
def plot_formants_on_ax(ax, times, F1, F2, F3, F4, F1_ref, F2_ref, F3_ref, F4_ref,
                        title, letter, n_label, vtl_value, vtl_ref=None):
    """
    Plot the 4 formants (points and lines) on axis `ax`.
    References are in gray with empty markers.
    title : string (e.g., "First 4 formants", "Mixed")
    letter : string (e.g., "A", "B", "C") displayed bold top-left
    n_label : string like "N=4", "N=6", "Mixed" for legend
    vtl_value : float, VTL for the Parselmouth data
    vtl_ref : float or None, VTL for reference data (only used for figure C)
    """
    # Plot references (gray)
    for F in [F1_ref, F2_ref, F3_ref, F4_ref]:
        ax.plot(times, F, '--', color='gray', lw=1)
        ax.plot(times[0], F[0], 'o', color='gray', ms=4, fillstyle='none')
        ax.plot(times[1:9], F[1:9], 's', color='gray', ms=3.5, fillstyle='none')
        ax.plot(times[9:14], F[9:14], 'D', color='gray', ms=3.5, fillstyle='none')

    # Plot real data
    for F in [F1, F2, F3, F4]:
        ax.plot(times, F, '-k', lw=1)
        ax.plot(times[0], F[0], 'or', ms=4)
        ax.plot(times[1:9], F[1:9], 's', color='blue', ms=3.5)
        ax.plot(times[9:14], F[9:14], 'D', color='limegreen', ms=3.5)

    ax.set_xlim(0, 0.55)
    ax.set_ylim(0, 6.5)

    ax.set_xlabel("Time (s)", fontsize=10)
    ax.set_ylabel("Frequency (kHz)", fontsize=10)

    ax.set_xticks(np.arange(0, 0.61, 0.1))
    ax.set_yticks(np.arange(0, 6.6, 1.0))

    ax.tick_params(
        axis='both',
        labelsize=8,
        width=0.8,
        length=3)

    ax.grid(
        True,
        linestyle=':',
        linewidth=0.3,
        color='0.8')

    ax.set_box_aspect(1)
    
    # Build title with VTL info
    if vtl_ref is not None:
        full_title = f"{title} – VTL = {vtl_value:.1f} cm (ref = {vtl_ref:.1f} cm)"
    else:
        full_title = f"{title} – VTL = {vtl_value:.1f} cm"
    ax.set_title(full_title)
    ax.grid(
        True,
        linestyle=':',
        linewidth=0.3,
        color='0.8')
    ax.set_xticks(np.arange(0, 0.61, 0.2))
    ax.set_yticks(np.arange(0, 6.5, 2.0))
    ax.set_box_aspect(1)

    # Add bold letter in top-left corner (inside axes)
    ax.text(
        -0.15, 1.03, letter,
        transform=ax.transAxes,
        fontsize=10,
        fontweight='bold'
    )
    # Add custom legend: dashed gray = ref, solid black = Parselmouth (n_label)
    # legend_elements = [Line2D([0], [0], color='gray', linestyle='--', label='ref'),
    #                    Line2D([0], [0], color='black', linestyle='-', label=f'Parselmouth ({n_label})')]
    # ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
    handles = [
        Line2D([0], [0], color='gray',
               ls='--', lw=1,
               label='Graphical reference'),

        Line2D([0], [0], color='black',
               lw=1,
               label='Parselmouth estimate'),

        Line2D([0], [0],
               marker='o',
               color='w',
               markerfacecolor='red',
               markeredgecolor='black',
               markersize=5,
               label='Initial'),

        Line2D([0], [0],
               marker='s',
               color='w',
               markerfacecolor='blue',
               markeredgecolor='black',
               markersize=5,
               label='Medial'),

        Line2D([0], [0],
               marker='D',
               color='w',
               markerfacecolor='limegreen',
               markeredgecolor='black',
               markersize=5,
               label='Final')]

    fig.legend(
        handles=handles,
        loc='lower center',
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, -0.03)
    )
# ----------------------------------------------------------------------
# Data (adjust according to your file)
# ----------------------------------------------------------------------
audio_path = 'wahoo_4.wav'  # anonymised

# Extract for N=4 and N=6
N=6
times4, F1a, F2a, F3a, F4a = extract_parselmouth_formants_n(audio_path, 4)
times6, F1b, F2b, F3b, F4b = extract_parselmouth_formants_n(audio_path, N)

# Use the times from the N=6 extraction as time reference
times = times6  # time vector (0.0527, 0.0927, ...)

# Build the merged dataset (Figure C)
# Assume the first 14 points are used
F1 = np.concatenate(([F1b[0]], F1a[1:9], F1b[9:15]))
F2 = np.concatenate(([F2b[0]], F2a[1:9], F2b[9:15]))
F3 = np.concatenate(([F3b[0]], F3a[1:9], F3b[9:15]))
F4 = np.concatenate(([F4b[0]], F4a[1:9], F4b[9:15]))

# Reference data (graphical capture)
F1_ref = np.array([0.510, 0.795, 0.710, 0.771, 0.799, 0.846, 0.927, 0.842, 0.852, 0.70, 0.525, 0.435, 0.410, 0.395])
F2_ref = np.array([1.150, 1.859, 1.746, 1.788, 1.802, 1.817, 1.746, 1.746, 1.802, 1.20, 1.00, 1.02, 0.95, 0.87])

F3_ref = np.array([1.930, 2.640, 2.484, 2.413, 2.526, 2.583, 2.597, 2.455, 2.597, 1.916, 1.902, 1.888, 1.873, 1.817])
F4_ref = np.array([2.782, 4.598, 5.109, 5.081, 5.024, 4.541, 4.400, 4.414, 4.754, 2.650, 2.750, 2.500, 2.400, 2.750])

# ----------------------------------------------------------------------
# VTL calculation (with NaN cleaning)
# ----------------------------------------------------------------------
# For each dataset, take the first 14 points
F1a_14 = F1a[:14]
F2a_14 = F2a[:14]
F3a_14 = F3a[:14]
F4a_14 = F4a[:14]

F1b_14 = F1b[:14]
F2b_14 = F2b[:14]
F3b_14 = F3b[:14]
F4b_14 = F4b[:14]

F1_merged = F1[:14]
F2_merged = F2[:14]
F3_merged = F3[:14]
F4_merged = F4[:14]

vtla = reby_mccomb_vtl_global(F1a_14, F2a_14, F3a_14, F4a_14)
vtlb = reby_mccomb_vtl_global(F1b_14, F2b_14, F3b_14, F4b_14)
vtlm = reby_mccomb_vtl_global(F1_merged, F2_merged, F3_merged, F4_merged)
vtlref = reby_mccomb_vtl_global(F1_ref, F2_ref, F3_ref, F4_ref)

print(f"VTL Parselmouth (N=4) : {vtla:.2f} cm")
print(f"VTL Parselmouth (N={N}) : {vtlb:.2f} cm")
print(f"VTL Parselmouth (merged) : {vtlm:.2f} cm")
print(f"VTL reference : {vtlref:.2f} cm")

# ----------------------------------------------------------------------
# Create three figures in a single panel (1 row, 3 columns)
# ----------------------------------------------------------------------
fig, axes = plt.subplots(
    1, 3,
    figsize=(7.2, 2.8),
    dpi=600
)

plt.subplots_adjust(
    left=0.08,
    right=0.99,
    bottom=0.22,
    top=0.85,
    wspace=0.35
)
# Figure A: N=4
plot_formants_on_ax(axes[0], times[:14], F1a_14, F2a_14, F3a_14, F4a_14,
                    F1_ref, F2_ref, F3_ref, F4_ref,
                    title="N=4", letter="A", n_label="N=4", vtl_value=vtla)

# Figure B: N=6
plot_formants_on_ax(axes[1], times[:14], F1b_14, F2b_14, F3b_14, F4b_14,
                    F1_ref, F2_ref, F3_ref, F4_ref,
                    title="N=6", letter="B", n_label=f"N={N}", vtl_value=vtlb)

# Figure C: Merged
plot_formants_on_ax(axes[2], times[:14], F1_merged, F2_merged, F3_merged, F4_merged,
                    F1_ref, F2_ref, F3_ref, F4_ref,
                    title="Merged", letter="C", n_label="Mixed",
                    vtl_value=vtlm, vtl_ref=vtlref)
F1test = F1_merged
print('Test F1 > 1.5*F0')
print(F1test.reshape(-1, 1))
plt.savefig(
    "Figure1.png",   # anonymised output name
    dpi=600,
    bbox_inches="tight",
    facecolor="white"
)