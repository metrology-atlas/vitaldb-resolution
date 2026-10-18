# Changelog

## 0.6.0 (2026-10-18)

Details and tables: `results/2026-10-18_refits_duration_zero.md`. Baseline (every model v1): `v0.5.0` = e2376f4.

### Models (v2 unless noted; v1 files in `data/models/archive/`)
- **Stored fit statistics equal what the library recomputes**, for every model and every archived version
  (`vitaldb_resolution.fitcheck`, `tests/test_fit_consistency.py`; the data checks run when `VITALDB_DATA` points
  to the VitalDB CSVs).
- `aneend`: 2 records outside the model range excluded (`event_outside_model_range`, not corrected); n 6254.
- `opstart`: refitted with the library likelihood (+33.2 loglik). The 10.6 gap came from the old fitter
  integrating the time-of-day prior on its own 0.1-min grid. Posteriors move by < 0.03 min.
- `anestart`, `opend`, `intraop_ppf`, `weight`: fit blocks re-evaluated; parameters unchanged.
- `intraop_uo`, `intraop_crystalloid`, `intraop_ebl`: anaesthesia duration is a latent location covariate
  (ΔBIC −926, −6254, −1751).
- `intraop_uo`: department on the heaping propensities (ΔBIC −87; not adopted for crystalloid, +22, or EBL, +51).
- `intraop_uo`: zero rule. A recorded 0 is a point mass, p0 = 23/3707, plus the latent mass recording as 0; it is
  flagged `zero_anuric` or `zero_unexplained`. Missing stays distinct from zero.
- `height`, `bmi`: still v1.
- References (Rietveld 2002, Healey 2011, Ko 2022 and others) added to every model.
- Floor rounding (H1) fitted for the three volume models. Nearest is preferred by BIC in all of them.

### Model versions
- Each model has an `id`, an integer `version`, a `fit_hash`, `fitted_on` and `supersedes`.
- `data/models.lock` lists the release's models and every published version.
- Pin a version with `model="aneend@1"`.
- The Monte Carlo cache records the model versions it was built with and warns when they are stale.
- `CONTRIBUTING.md` describes how to propose a new version.

### API
- `density` on every response: `latent` means the fitted latent truncated to each interval, computed at call time;
  `uniform` means rectangular over the interval. `model` carries the id, version and fit_hash.
- Latent components report `interval`, `mean` and `uniform_mean`. New method `Distribution.latent_uniform_gap()`.
- Optional covariates `duration=` (minutes) and `department=`. Results without them use the marginal and are
  flagged `duration_marginal` or `department_marginal`.
- New status `bottom_coded`.
- Lactate 15.0 and aPTT 400.0 (and `preop_aptt`) are now `top_coded`; ESR 2 is `bottom_coded`.
- ICU days: new flag `icu_exceeds_postop`. `order_violation` is now evaluated. `counting_rule_unknown` is renamed
  `counting_rule_assumed`.
- New modules `fitcheck` and `versions`.

### Registry (schema 0.5)
- `anchor_basis` and `anchor_note` for every variable with an anchor or a counting rule.
- `reporting_limits` for lac, aptt, preop_aptt and esr.
- New icu_days description (nights or completed days, assumed).
- `p`: "probably phosphorus (unconfirmed)".
- Volume variables list their optional covariates.
- New exclusion and flag rules: `uo_zero_anuric`, `uo_zero_unexplained`, `duration_unusable`,
  `icu_exceeds_postop`, `lactate_top_coded`, `aptt_top_coded` and `esr_bottom_coded`.

### Validation report
- Model versions, zero counts by flag, density counts, and the largest latent–uniform mean gap for each modelled
  variable.

### Tests
- `test_fit_consistency.py`, `test_versions.py`, `test_density.py`, `test_no_per_case_files.py`.
