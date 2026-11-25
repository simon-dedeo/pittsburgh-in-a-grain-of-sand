# Pittsburgh in a Grain of Sand

**Bayesian radiometric modeling, viscoelastic mount metrology, and autonomous correction of a near-field window optical defect.**

- **Paper (PDF):** [`paper/SPOT_FIT.pdf`](paper/SPOT_FIT.pdf) — also hosted at https://sites.santafe.edu/~simon/spot_fit.pdf
- **Live sky monitor:** https://proofsandreasons.io
- **Parent skyline-monitor repository:** https://github.com/simon-dedeo/cmu-skyline-monitor

---

## Summary

A single ~0.23 mm speck of debris on a Porter Hall window at Carnegie Mellon — roughly 23–28 mm in front of a 4K sky monitor's $f/4$ entrance pupil — prints a flat-topped 101 px defocused shadow whose off-axis ellipticity (1.095, oriented within 9° of radial) matches rectilinear $\sec\theta = 1.12$ projection.

Over 125 days (73 accepted nightly fixes), the shadow traced a straight 310 px path following Andrade's $t^{1/3}$ viscoelastic creep law ($\alpha = 0.34$, $R^2 = 0.989$), turning an optical nuisance into a passive sub-millimetre mount strain gauge. Calibrating the camera's empirical Camera Response Function (CRF) from exposure brackets eliminates an apparent dark offset and reveals $>12\sigma$ direct-sun forward scattering in red ($k_{\mathrm{fac}} = +0.0072$) atop achromatic blocking ($t \approx 0.952$). A closed-loop Bayesian Stan M2 pipeline (`production/akdeniz/spotnight.py`) and conjugate MAP flat-field corrector (`production/pandr/flatfield.py`) reduce clean-sky linear dip from **+4.15% to −0.12%** across RGB (0/18 frames made worse; per-channel B/G/R residuals −0.08%, −0.02%, −0.01%).

---

## Repository Layout

```text
pittsburgh-in-a-grain-of-sand/
├── paper/                      # LaTeX source (SPOT_FIT.tex), compiled PDF, bibliography, and figures
├── data/                       # Complete paper datasets (see data/DATA.md)
├── production/
│   ├── akdeniz/                # Nightly tracker, CRF fitter, Stan M2 model, TMAP builder & publisher
│   └── pandr/                  # Live conjugate-MAP bracket corrector, HDR fusion, nightly A/B audit
├── offline_july/               # Original July 2026 offline Stan models and photometry scripts
├── lean_verification/          # Archived v11 Lean 4 formalization of the sRGB-era model
├── tests/                      # Synthetic end-to-end CRF+MAP test and dipole asymmetry metric
├── reproduce_figures.py        # Regenerates Figs. 2, 3, 5, 7 and prints Table 1 & 3 statistics from data/
└── PRODUCTION_STATUS.md        # Live deployment & verification summary
```

---

## Quick Reproduction

### 1. Reproduce all v12 figures and summary tables

Requires Python 3 with `numpy`, `scipy`, `matplotlib`, and `opencv-python` (optional: `cmdstanpy` with `--run-stan` to refit `spot_m2.stan` from scratch).

```bash
python3 reproduce_figures.py
```

This reads exclusively from `data/`, writes regenerated PDFs (`crf_and_m2.pdf`, `profile_v12.pdf`, `ab_longitudinal.pdf`, `creep_4month.pdf`) to `reproduced_figs/`, and prints the verified numerical checks (Andrade creep parameters, empirical CRF fit vs. sRGB exposure ratios, Stan M2 posterior summary, 2D TMAP FWHM & ellipticity, and Table 3 A/B statistics).

To refit the Bayesian Stan M2 model directly from `data/spot_photometry_14d.csv.gz` (requires `cmdstanpy` + CmdStan):

```bash
python3 reproduce_figures.py --run-stan
```

### 2. Run the synthetic end-to-end corrector test

Injects the published 2D transmittance map (`data/spot_tmap.npy`) into a synthetic 3-bracket HDR set in empirical-CRF linear radiance, runs `production/pandr/flatfield.py`, and verifies that all three channels are nulled to $< 0.5\%$ linear residual:

```bash
python3 tests/test_synthetic_end_to_end.py
```

### 3. Compile the paper

```bash
cd paper
pdflatex SPOT_FIT.tex && bibtex SPOT_FIT && pdflatex SPOT_FIT.tex && pdflatex SPOT_FIT.tex
```

---

## Citation

If you use the code, dataset, or methods from this repository, please cite:

```bibtex
@misc{dedeo2026grainofsand,
  author       = {DeDeo, Simon},
  title        = {Pittsburgh in a Grain of Sand: Bayesian Radiometric Modeling, Viscoelastic Mount Metrology, and Autonomous Correction of a Near-Field Window Optical Defect},
  year         = {2026},
  howpublished = {\url{https://sites.santafe.edu/~simon/spot_fit.pdf}},
  note         = {Code and data: \url{https://github.com/simon-dedeo/pittsburgh-in-a-grain-of-sand}}
}
```
