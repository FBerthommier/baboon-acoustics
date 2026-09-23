# baboon-acoustics

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
[![License: MIT](https://img.shields.io/badge/License-MIT-informational.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org)

Python tools for the acoustic analysis of baboon vocalizations: formant
tracking (Praat's Burg method and IAIF inverse filtering),
fundamental-frequency (F0) estimation, vocal-tract-length (VTL) estimation,
and glottal-source analysis. The functions are designed to be reused as-is
on your own recordings — no advanced Python skills required.

This repository is the supplementary code of the article *Re-analysis of
formant trajectories in a chacma baboon wahoo vocalization* (**anonymized
for review**; the article is a comment on Boë et al. 2026, "Stalking the
wild formant: How to reliably measure formants in primate vocalizations",
*International Journal of Primatology*). The article's supplementary
materials (audio, reference figures, PDF documentation) are archived on
Zenodo under the concept DOI
[10.5281/zenodo.20764395](https://doi.org/10.5281/zenodo.20764395).

## Highlights

- `extract_praat_formants()` — formant tracks in two lines of code, via
  Praat's Burg LPC method (`n_formants` configurable, 40 ms windows).
- `reby_mccomb_vtl()` — one-call vocal-tract-length estimate (Reby &
  McComb 2003), working with any number of formants and NaN-tolerant.
- `estimate_f0_praat()` — F0 contour with the Gaussian pre-filter that
  emulates Praat's *filtered autocorrelation* (robust to subharmonics).
- `iaif_sliding_analysis()` — full sliding IAIF decomposition: glottal
  source, vocal-tract envelope, F0 and formant tracks.
- Four scripts reproduce every figure of the article exactly.

## Installation

Requirements: **Python 3.9+** and the four packages listed in
`requirements.txt` (`numpy`, `scipy`, `matplotlib`, `praat-parselmouth`).
[Praat](https://www.fon.hum.uva.nl/praat/) itself does not need to be
installed — `praat-parselmouth` ships it.

```bash
git clone https://github.com/<your-account>/baboon-acoustics.git
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
├── scripts/                   # reproduction of the article figures (thin wrappers)
│   ├── make_figure1.py        #   Figure 1: N=4 vs N=6 vs merged formant tracks + VTL
│   ├── make_figure_multipanel.py  # six-panel analysis figure + single glottal pulse
│   ├── make_figure_glottal.py #   glottal train: waveform, phase portrait, spectra
│   └── make_figure_f0.py      #   F0 contour (filtered autocorrelation) -> CSV + PNG
├── data/                      # example audio and reference data
│   ├── wahoo_1..4.wav         #   chacma baboon wahoo calls (44.1 kHz)
│   ├── The_Baboon_Bark.wav    #   longer bark recording (16 s)
│   ├── reference_formants_boe2026.csv   # reference tracks digitized from Boë et al. (2026)
│   └── vowel_space_mas_contour.csv      # Maximal Acoustic Space contour (kHz)
├── tests/                     # unit tests (python -m unittest discover -s tests)
├── docs/
│   ├── documentation.tex      # source of the user documentation
│   ├── baboon-acoustics-documentation.pdf   # compiled documentation
│   └── figures/               # reference figures used by the documentation
├── README.md
├── LICENSE                    # MIT
├── requirements.txt
├── CITATION.cff               # citation metadata (GitHub "Cite this repository")
└── .zenodo.json               # metadata pushed to Zenodo at each release
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

## Linking this repository to Zenodo (for the maintainer)

This repository is set up for automatic archiving on
[Zenodo](https://zenodo.org), which mints a DOI for every release:

1. Create a release on GitHub (`git tag v1.0.0 && git push --tags`, then
   "Draft new release" on the repository page).
2. Zenodo (once the repository is enabled in your Zenodo account under
   "GitHub") imports the tag, applies the metadata from `.zenodo.json`,
   and publishes a new versioned DOI.
3. After the first release, copy the **concept DOI** shown on the Zenodo
   record and:
   - replace `10.5281/zenodo.XXXXXXX` in the badge at the top of this
     README;
   - add the same identifier to `related_identifiers` in `.zenodo.json`
     (relation `isVersionOf`);
   - update `identifiers` in `CITATION.cff`.
4. Before publishing the first public release, also replace the
   placeholder author entries ("anonymized for review") in
   `.zenodo.json`, `CITATION.cff` and `LICENSE` with the real author
   list — the article is currently under double-blind review.

The article's supplementary deposit (audio, reference figures, method
PDF) remains available at
[10.5281/zenodo.20764395](https://doi.org/10.5281/zenodo.20764395).

## Citation

If you use this code, please cite the software release:

```bibtex
@software{baboon_acoustics_2026,
  author  = {{anonymized for review}},
  title   = {baboon-acoustics: formant, F0, glottal-source and vocal-tract-length analysis tools},
  year    = {2026},
  url     = {https://github.com/<your-account>/baboon-acoustics},
  doi     = {10.5281/zenodo.XXXXXXX},
  license = {MIT}
}
```

… and the article (reference to be added upon acceptance).

Key methods implemented here: IAIF inverse filtering (Alku 1991);
formant-based VTL estimation (Reby & McComb 2003); Praat/Parselmouth Burg
formant analysis (Boersma & Weenink; Jadoul et al. 2018).

## License

Released under the [MIT License](LICENSE).

## References

- Alku, P. (1991). Glottal wave analysis with pitch synchronous iterative
  adaptive inverse filtering. *Proceedings of Eurospeech 1991*, 1081–1084.
- Boë, L.-J., Laporte, M., Sawallis, T. R., Badin, P., & Schwartz, J.-L.
  (2026). Stalking the wild formant: How to reliably measure formants in
  primate vocalizations. *International Journal of Primatology*.
  https://doi.org/10.1007/s10764-026-00551-4
- Jadoul, Y., Thompson, B., & de Boer, B. (2018). Introducing Parselmouth:
  A Python interface to Praat. *Journal of Phonetics*, 71, 1–15.
- Reby, D., & McComb, K. (2003). Anatomical constraints generate honesty:
  acoustic cues to age and weight in the roars of red deer. *Animal
  Behaviour*, 65(3), 519–530.
