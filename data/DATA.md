# Data Manifest

All datasets needed to reproduce every figure, table, empirical Camera Response Function (CRF) fit, Bayesian Stan M2 fit, viscoelastic mount-creep analysis, and synthetic test in *Pittsburgh in a Grain of Sand* (v12, 2026-11-24).

| File | Size | Coverage | Reproduces |
|---|---:|---|---|
| `history_v1.csv` | ~7 KB | 2026-07-22 – 2026-11-05 | Fig. 7 (Andrade $t^{1/3}$ viscoelastic creep, v1 nightly tracker) |
| `history.csv` | ~2 KB | 2026-11-05 – 2026-11-24 | Fig. 7 (v2/v3 multi-bracket tracker continuation) |
| `state.json` | ~1 KB | 2026-11-24 | Current tracker state and rolling 10-night velocity window |
| `spot_photometry_14d.csv.gz` | ~4.6 MB | 14-day window ending 2026-11-24 (35 MB raw tail) | Fig. 2 (empirical CRF vs. sRGB exposure ratios; M2 scatter-forcing data), Table 1 (`spot_m2.stan` input) |
| `bayes_fit.json` | ~1 KB | 2026-11-24 (`N=2921`, $\hat{R}=1.0055$) | Table 1 & Fig. 2 right panel (live Stan M2 posterior) |
| `spot_model.json` | ~2 KB | 2026-11-24 (`2026-11-24-bayes3`, schema 3) | Live deployed model bundle (position, velocity, CRF, Stan M2 prior, TMAP metadata) |
| `spot_tmap.npy` | ~1.1 MB | 2026-11-24 (`300 x 300 x 3`, float32) | Fig. 3 (2D transmittance map, radial profile, ellipticity), `tests/test_synthetic_end_to_end.py` |
| `ab_log_v1.csv` | ~183 KB | 2026-08-17 – 2026-11-05 (`n=2016` clean) | Fig. 5 & Table 3 (prospective matched A/B, legacy v1) |
| `ab_log_v2.csv` | ~38 KB | 2026-11-09 – 2026-11-23 (`n=434` clean) | Fig. 5 & Table 3 (prospective matched A/B, post-Nov-5) |
| `ab_verify_20261124_bayes3.csv` | ~3 KB | 2026-11-24 (`n=18` clean) | Fig. 5 & Table 3 (live `2026-11-24-bayes3` verification, Mertens linear + enhanced) |
| `sweep2.csv` | ~95 KB | 2026-07-20 – 2026-07-27 (`n=495` clean) | Table 3 (July offline validation row) |
| `live_paired.csv` | ~8 KB | 2026-07-28 | July initial live paired A/B verification |
| `drift.csv` | ~1.5 MB | 2026-07-20 – 2026-11-24 | Far-field skyline aim drift (validating differential glass-vs.-camera motion) |
| `sample_bracket_20260725/` | ~30 MB | 2026-07-25 23:17:28 UTC (`br0/br1/br2` PNG + July schema-2 bundle) | Fig. 1 (`full_before/after.jpg`) & Fig. 6 (`before/after/diff.png`) |

> **Full archive note:** The complete 5-minute lossless bracket archive (~65 GB+ and growing since 2026-07-20) and full 4-month raw photometry log (~303 MB) exceed GitHub limits and are available on request (`sdedeo@andrew.cmu.edu`). `spot_photometry_14d.csv.gz` contains the exact 35 MB tail window used by the live `spotnight.py` CRF and Stan M2 pipeline.
