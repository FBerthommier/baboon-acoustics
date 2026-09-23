"""Baboon acoustics analysis package.

Python tools for the acoustic analysis of baboon vocalizations: formant
extraction (Praat's Burg method and IAIF inverse filtering),
fundamental-frequency estimation, vocal-tract-length estimation and
glottal-source analysis. Developed as supplementary material for the
article "Re-analysis of formant trajectories in a chacma baboon wahoo
vocalization" (anonymized for review).

The main entry points are re-exported at the package level::

    from baboon_acoustics import load_audio, extract_praat_formants

Submodules
----------
audio_io
    WAV loading, mono conversion, normalization, resampling.
formants_praat
    Formant tracking with Praat's Burg LPC method (via Parselmouth).
formants_iaif
    Iterative adaptive inverse filtering (LPC, glottal source, formant
    peaks, sliding analysis).
f0
    Fundamental-frequency estimation (autocorrelation, Praat
    filtered-AC).
vtl
    Vocal-tract-length estimation (Reby & McComb, 2003).
glottal
    Glottal pulse and glottal train extraction.
plotting
    Journal-style plotting helpers.
"""

from .audio_io import DEFAULT_FS, load_audio
from .f0 import estimate_f0_from_gn, estimate_f0_praat
from .formants_iaif import (
    get_frame_formants,
    iaif_frame,
    iaif_sliding_analysis,
    lpc_levinson,
    lpc_spectrum_db,
)
from .formants_praat import (
    extract_praat_formants,
    fill_leading_nan,
    interpolate_formants,
)
from .glottal import extract_glottal_pulse, extract_glottal_train, normalize_amplitude
from .plotting import add_panel_label, set_journal_style, setup_frequency_axis
from .vtl import reby_mccomb_vtl

__version__ = "1.0.0"

__all__ = [
    "DEFAULT_FS",
    "load_audio",
    "extract_praat_formants",
    "interpolate_formants",
    "fill_leading_nan",
    "lpc_levinson",
    "lpc_spectrum_db",
    "iaif_frame",
    "get_frame_formants",
    "iaif_sliding_analysis",
    "estimate_f0_from_gn",
    "estimate_f0_praat",
    "reby_mccomb_vtl",
    "extract_glottal_pulse",
    "extract_glottal_train",
    "normalize_amplitude",
    "set_journal_style",
    "add_panel_label",
    "setup_frequency_axis",
]
