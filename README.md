# baboon-acoustics

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
[![License: MIT](https://img.shields.io/badge/License-MIT-informational.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org)

**Frédéric Berthommier** — CNRS, Grenoble INP, GIPSA-Lab, Université
Grenoble Alpes, 11 Rue des Mathématiques – BP 46, 38402 Saint Martin
d'Hères Cedex, Grenoble, France —
<frederic.berthommier@gipsa-lab.grenoble-inp.fr>

## 📖 Read and interact — no installation needed

| | |
|---|---|
| 📄 **Supplementary materials of the article** | [`supplementary/Supplementary_materials.pdf`](supplementary/Supplementary_materials.pdf) — opens directly in the browser |
| 📄 **Article figure set** | [`supplementary/figures/`](supplementary/figures) — all figures of the article and of the supplement |
| 📄 **Original analysis scripts** (as archived) | [`supplementary/original_scripts/`](supplementary/original_scripts) |
| 🎛 **Interactive demos** (audio + dynamic plots) | [`demos/index.html`](https://FBerthommier.github.io/baboon-acoustics/demos/) — served by GitHub Pages, or open any `demos/*.html` locally: the pages are fully self-contained |
| 📄 **User documentation of the package** | [`docs/baboon-acoustics-documentation.pdf`](docs/baboon-acoustics-documentation.pdf) — concepts, tutorial, API reference, FAQ |

> The demo pages work offline: each one embeds its audio (base64) and the
> precomputed analysis results. To publish them in one click, enable
> **Settings → Pages → Source: GitHub Actions** after pushing — the
> included workflow (`.github/workflows/pages.yml`) deploys the
> repository automatically; the links above then become live.

Python tools for the acoustic analysis of baboon vocalizations: formant
tracking (Praat's Burg method and IAIF inverse filtering),
fundamental-frequency (F0) estimation, vocal-tract-length (VTL) estimation,
and glottal-source analysis. The functions are designed to be reused as-is
on your own recordings — no advanced Python skills required.

This repository is the complete, archived supplementary material of the
article *On the Reliability of Wahoo Formant Measurements: A Response to
Boë et al. (2026)* — Frédéric Berthommier, *International Journal of
Primatology* (accepted 1 October 2026) — a comment on Boë et al. 2026,
"Stalking the wild formant: How to reliably measure formants in primate
vocalizations", *International Journal of Primatology*, 47(3), 799–841.
It contains the supplementary document and figures, the original analysis
scripts, the audio material, and the refactored `baboon_acoustics` package
with its documentation, tests and interactive demos. This GitHub
repository is imported by Zenodo so that **one single DOI** archives and
badges everything cited by the article (Open Data badge).

## Highlights

- `extract_praat_formants()` — formant tracks in two lines of code, via
  Praat's Burg LPC method (`n_formants` configurable, 40 ms windows).
- `reby_mccomb_vtl()` — one-call vocal-tract-length estimate (Reby &
  McComb 2003), working with any number of formants and NaN-tolerant.
- `estimate_f0_praat()` — F0 contour with the Gaussian pre-filter that
  emulates Praat's *filtered autocorrelation* (robust to subharmonics).
- `iaif_sliding_analysis()` — full sliding IAIF decomposition: glottal
  source, vocal-tract envelope, F0 and formant tracks.
- The scripts in `scripts/` reproduce every figure of the article exactly
  and regenerate the interactive demos.

## Installation

Requirements: **Python 3.9+** and the four packages listed in
`requirements.txt` (`numpy`, `scipy`, `matplotlib`, `praat-parselmouth`).
[Praat](https://www.fon.hum.uva.nl/praat/) itself does not need to be
installed — `praat-parselmouth` ships it.

```bash
git clone https://github.com/FBerthommier/baboon-acoustics.git
cd baboon-acoustics
python -m pip install -r requirements.txt
```

Tested with: Python 3.14, numpy 2.5, scipy 1.18, matplotlib 3.11,
praat-parselmouth 0.4.5 (Windows 10); the code has no version-sensitive
constructs and runs on any recent stack.

To verify your installation, run the test suite (about 30 s):

```bash
python -m unittest discover -s tests
```

## Quick start

Extract formants and estimate the vocal tract length of a recording:

```python
from baboon_acoustics import load_audio, extract_praat_formants, reby_mccomb_vtl

# 1. Load any WAV file (mono/stereo, any sample rate; resampled to 44.1 kHz)
samples, fs = load_audio("my_recording.wav")

# 2. Track the first 4 formants with Praat's Burg method (values in Hz)
times, formants = extract_praat_formants("my_recording.wav", n_formants=4)
print(formants.shape)        # (4, n_frames); NaN where a formant is undefined

# 3. Estimate the vocal tract length (cm) from all valid measurements
vtl_cm = reby_mccomb_vtl(formants)
print(f"VTL = {vtl_cm:.1f} cm")
```

Estimate an F0 contour with the subharmonic-robust filtered
autocorrelation:

```python
from baboon_acoustics import estimate_f0_praat

times, f0, strength = estimate_f0_praat("my_recording.wav")
print(f"mean F0 = {f0.mean():.1f} Hz over {len(f0)} voiced frames")
```

A complete guided tour — every function commented, step by step, on two
example recordings — is provided in
[`examples/example_analysis.py`](examples/example_analysis.py). Run it
with:

```bash
python examples/example_analysis.py
```

## Interactive demos (no installation needed)

The [`demos/`](demos/index.html) folder contains self-contained HTML
pages: the audio (base64) and the precomputed analysis results are
embedded in each file, so you can **double-click them in any browser,
fully offline**. Plots are dynamic — track toggles, hover readouts, a
playback cursor synchronized with the sound, click-to-seek, mouse-wheel
zoom:

| Demo | Content |
|------|---------|
| [`demo_formants.html`](demos/demo_formants.html) | Praat/Burg N=4 vs N=6 vs merged trajectories against the reference, VTL badges, synchronized audio |
| [`demo_spectrogram.html`](demos/demo_spectrogram.html) | spectrogram with Praat, IAIF-formant and F0 overlays; playback cursor and seek |
| [`demo_f0.html`](demos/demo_f0.html) | filtered vs raw autocorrelation F0 for the four wahoos (subharmonic diagnostic) |
| [`demo_glottal.html`](demos/demo_glottal.html) | IAIF pulse train (wheel zoom), single glottal cycle, Hilbert phase portrait |

The pages are generated by `scripts/make_demos.py` (edit it to build a
demo on your own recordings):

```bash
python scripts/make_demos.py     # rewrites demos/*.html
```

## Repository structure

```
baboon-acoustics/
├── baboon_acoustics/          # the reusable analysis package
│   ├── audio_io.py            #   load WAV files: mono conversion, normalization, resampling
│   ├── formants_praat.py      #   formant tracking with Praat's Burg LPC method
│   ├── formants_iaif.py       #   IAIF inverse filtering: LPC, glottal source, formant peaks
│   ├── f0.py                  #   F0 estimation: autocorrelation and Praat filtered-AC
│   ├── vtl.py                 #   vocal-tract-length estimation (Reby & McComb 2003)
│   ├── glottal.py             #   glottal pulse and glottal train extraction
│   └── plotting.py            #   journal-style figures, panel labels, F1-F2 trajectories
├── examples/
│   └── example_analysis.py    # guided tour: every function explained, on real recordings
├── demos/                     # self-contained interactive HTML demos (open in any browser)
│   ├── index.html             # demo gallery
│   ├── demo_formants.html     # N=4 / N=6 / merged tracks + VTL, with audio playback
│   ├── demo_spectrogram.html  # spectrogram with Praat/IAIF/F0 overlays + playback cursor
│   ├── demo_f0.html           # filtered vs raw AC contours for the four wahoos
│   └── demo_glottal.html      # pulse train (wheel zoom), glottal cycle, phase portrait
├── scripts/                   # reproduction of the article figures (thin wrappers)
│   ├── make_figure1.py        #   Figure 1: N=4 vs N=6 vs merged formant tracks + VTL
│   ├── make_figure_multipanel.py  # six-panel analysis figure + single glottal pulse
│   ├── make_figure_glottal.py #   glottal train: waveform, phase portrait, spectra
│   ├── make_figure_f0.py      #   F0 contour (filtered autocorrelation) -> CSV + PNG
│   └── make_demos.py          #   regenerates the demos/ HTML pages
├── data/                      # example audio and reference data
│   ├── wahoo_1..4.wav         #   chacma baboon wahoo calls (44.1 kHz)
│   ├── The_Baboon_Bark.wav    #   longer bark recording (16 s)
│   ├── wahoo_4_f0_filtered_ac.csv       # archived reference F0 table (filtered AC)
│   ├── reference_formants_boe2026.csv   # reference tracks digitized from Boë et al. (2026)
│   └── vowel_space_mas_contour.csv      # Maximal Acoustic Space contour (kHz)
├── supplementary/             # the article's supplementary material (as archived)
│   ├── Supplementary_materials.pdf  # corrected & completed supplement document
│   ├── figures/               #   all figures of the article and of the supplement
│   ├── original_scripts/      #   the four analysis scripts described in the supplement (verbatim)
│   └── source/                #   LaTeX source of the supplement document
├── tests/                     # unit tests (python -m unittest discover -s tests)
├── docs/
│   ├── documentation.tex      # source of the user documentation
│   └── baboon-acoustics-documentation.pdf   # compiled documentation
├── README.md
├── LICENSE                    # MIT
├── requirements.txt
├── CITATION.cff               # citation metadata (GitHub "Cite this repository")
├── .zenodo.json               # metadata pushed to Zenodo at each release
└── .github/workflows/pages.yml  # deploys the repository to GitHub Pages on push
```

Outputs of the scripts are written to `figures/` (not tracked by git).

## Reproducing the article figures

Each script reproduces one figure of the article (and its supplementary
figures) from the audio in `data/`. They accept an optional WAV path and
output directory:

```bash
python scripts/make_figure1.py                 # Figure1.png
python scripts/make_figure_multipanel.py       # analysis_3x2_panels.png + glottal_pulse.png
python scripts/make_figure_glottal.py          # glottal_train_analysis_hoo.png
python scripts/make_figure_glottal.py --segment wa   # ..._wa.png ("wa" part)
python scripts/make_figure_f0.py               # wahoo_4_f0_filtered_ac.csv + .png
```

On the provided data, the scripts reproduce the analysis exactly (the F0
table matches the archived reference to a relative difference below 1e-9)
and the figures are visually identical to the originals.

## Documentation

A complete user guide (concepts, installation, tutorial, API reference,
FAQ) is provided as a PDF:
[`docs/baboon-acoustics-documentation.pdf`](docs/baboon-acoustics-documentation.pdf)
(LaTeX source in `docs/documentation.tex`; rebuild with
`latexmk -pdf documentation.tex`).

## Citation

If you use this code, please cite the software release:

```bibtex
@software{baboon_acoustics_2026,
  author  = {Berthommier, Fr{\'e}d{\'e}ric},
  title   = {baboon-acoustics: supplementary code and materials of
             "On the Reliability of Wahoo Formant Measurements:
             A Response to Bo{\"e} et al. (2026)"},
  year    = {2026},
  url     = {https://github.com/FBerthommier/baboon-acoustics},
  doi     = {10.5281/zenodo.XXXXXXX},
  license = {MIT}
}
```

… and the article: Berthommier, F. (2026). On the Reliability of Wahoo
Formant Measurements: A Response to Boë et al. (2026). *International
Journal of Primatology* (DOI to be inserted upon online publication).

Key methods implemented here: IAIF inverse filtering (Alku 1991);
formant-based VTL estimation (Reby & McComb 2003); Praat/Parselmouth Burg
formant analysis (Boersma & Weenink; Jadoul et al. 2018).

## License

Released under the [MIT License](LICENSE).

## References

- Berthommier, F. (2026). On the Reliability of Wahoo Formant
  Measurements: A Response to Boë et al. (2026). *International Journal
  of Primatology* (accepted 1 October 2026; DOI to be inserted upon
  online publication).
- Alku, P. (1991). Glottal wave analysis with pitch synchronous iterative
  adaptive inverse filtering. *Proceedings of Eurospeech 1991*, 1081–1084.
- Boë, L.-J., Laporte, M., Sawallis, T. R., Badin, P., & Schwartz, J.-L.
  (2026). Stalking the wild formant: How to reliably measure formants in
  primate vocalizations. *International Journal of Primatology*, 47(3),
  799–841. https://doi.org/10.1007/s10764-026-00551-4
- Fitch, W. T., Anikin, A., Pisanski, K., Valente, D., & Reby, D. (2025).
  Formant analysis of vertebrate vocalizations: Achievements, pitfalls,
  and promises. *BMC Biology*, 23, Article 92.
  https://doi.org/10.1186/s12915-025-02188-w
- Fitch, W. T., Neubauer, J., & Herzel, H. (2002). Calls out of chaos:
  The adaptive significance of nonlinear phenomena in mammalian vocal
  production. *Animal Behaviour*, 63(3), 407–418.
- Grawunder, S., Uomini, N., Samuni, L., Bortolato, T., Girard-Buttoz,
  C., Wittig, R. M., & Crockford, C. (2022). Chimpanzee vowel-like sounds
  and voice quality suggest formant space expansion through the hominoid
  lineage. *Philosophical Transactions of the Royal Society B*, 377(1841),
  20200455. https://doi.org/10.1098/rstb.2020.0455
- Jadoul, Y., Thompson, B., & de Boer, B. (2018). Introducing Parselmouth:
  A Python interface to Praat. *Journal of Phonetics*, 71, 1–15.
- Reby, D., & McComb, K. (2003). Anatomical constraints generate honesty:
  acoustic cues to age and weight in the roars of red deer. *Animal
  Behaviour*, 65(3), 519–530.
