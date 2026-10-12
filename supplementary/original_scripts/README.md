# Original analysis scripts (as archived with the article)

These four Python scripts are the **verbatim, unmodified versions** that
produced every figure of the article and of
`../Supplementary_materials.pdf`. They are archived exactly as used for
the study so that the supplement's function-by-function descriptions
remain accurate:

| Script | Produces |
|--------|----------|
| `make_figure1.py` | Figure 1 of the article (N=4 / N=6 / merged formant re-analysis + VTL) |
| `make_figure_multipanel.py` | the six-panel figure (`analysis_3x2_panels.png`) and `glottal_pulse.png` |
| `make_figure_ac.py` | the F0 contour with Gaussian pre-filtered autocorrelation (`wahoo_4_f0_filtered_ac.csv` / `.png`) |
| `make_figure_glottal.py` | the glottal-source figures (`glottal_train_analysis_wa.png`, `..._hoo.png`) |

Run them from this folder with a WAV file of the `../../data/` folder,
e.g. `python make_figure_multipanel.py ../../data/wahoo_4.wav`
(dependencies: `numpy`, `scipy`, `matplotlib`, `praat-parselmouth`;
see the repository root `requirements.txt`).

These are *frozen* research code, kept for provenance. For new work,
use the refactored, documented `baboon_acoustics` package at the
repository root instead: it implements the same algorithms
(function-by-function mapping is given in the repository README and in
the user documentation, `../docs/`), plus tests, examples and
interactive demos.
