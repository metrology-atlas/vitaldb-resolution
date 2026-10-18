# Release 0.6.0: refits, duration and department covariates, UO zero rule, model versions (2026-10-18)

This file holds aggregate results only, with no per-case data. It answers the metrology-atlas handovers of 2026-10-18:
handover 1 (items 1–4) and handover 2 (the UO zero rule, within-interval density, model versions, anchors, and the
new findings).

All log-likelihoods are the library's own evaluation (`vitaldb_resolution.fitcheck`), so stored and recomputed
values agree. `tests/test_fit_consistency.py` checks this for every current model and every archived version. The
raw BIC comparisons are in `results/volume_study.json`.

## 1. Anaesthesia end

The stored v1 fit (n 6256, loglik −13065.92) included 2 records whose end lies outside the model range. Each was
clipped to a constant −690.8, which accounts for the whole gap. Both records are now excluded in
`exclusion_rules.json` (`event_outside_model_range`, cases 3980 and 4959: *excluded, out of range, not corrected*).
On the remaining 6254 records the refit returned the same parameters, so only the fit block changed (v2): loglik
−11684.409.

## 2. Op-start log-likelihood gap (10.6) and the other timestamp models

**Cause.** The original fitter integrated the first-case clock (time-of-day) prior on its own 0.1-min grid. The
library integrates it on the model grid with cell-centre rounding. The stored parameters were optimal for the
fitter's likelihood, but the stored loglik (−18063.18) was not what the library computes (−18073.75).

**Fix.** Timestamp models are now fitted with `tools/fit_timestamps.py`, a JAX transcription of
`engine.TimestampModel` that reproduces the engine exactly. The fit block is the engine's own evaluation. If a refit
gains less than 0.01, the parameters are kept and only the fit block is re-evaluated.

| model | n | v1 stored | v1 recomputed | v2 (stored = recomputed) | parameters |
|---|---|---|---|---|---|
| opstart | 6387 | −18063.178 | −18073.745 | −18040.569 | refitted (+33.2) |
| anestart | 6387 | −21645.009 | −21644.900 | −21644.900 | kept |
| aneend | 6254 (v1: 6256) | −13065.915 | −11684.409 | −11684.409 | kept |
| opend | 5850 | −9411.359 | −9411.497 | −9411.497 | kept |
| intraop_ppf | 2328 | −3850.160 | −3847.549 | −3847.549 | kept (a grid refit gained only 0.04) |
| weight | 6041 | −38917.288 | −38917.244 | −38917.244 | kept |
| height | 6041 | −32378.343 | −32378.343 | (still v1) | unchanged |

**What the op-start refit changed.** The gain of 33 comes from two places: the latent's upper tail (the last spline
coefficient) and the clock prior at times with no first cases. Posteriors are practically unchanged. In 400 sampled
cases the posterior mean moved by at most 0.03 min, and the 95% interval width changed by at most ±0.05 min. The
propensities are unchanged (100-mode logit −2.835 → −2.833).

The previous UO fit block was also internally inconsistent: it stored n_params 9 but computed BIC with 8. Every BIC
is now −2·loglik + n_params·ln n.

## 3. Registry and metadata

* **`p`:** the description now says "probably phosphorus (unconfirmed)". The name is unchanged.
* **Kidney colloid:** the `unit_uncertain` flag is kept; the decision is still pending.
* **References:** every model file now has a top-level `references` list, with ids that match the atlas.
  - Every model cites Lee et al. 2022 (VitalDB).
  - All models except bmi cite Rietveld 2002 (rounding and heaping).
  - Height cites Healey et al. 2011, Leatt et al. 1986, Voss and Bailey 1997, and Ko et al. 2022
    (Epidemiol Health 44:e2022024).
  - Weight cites Khosla and Billewicz 1964, Vivanti et al. 2013, and Ko et al. 2022.

## 4. Anaesthesia duration as a latent location covariate

**Model.** f(x | d) ∝ g(log x − b·log(d/180)) / x, where d = aneend − anestart in charted minutes.

**Without a usable duration** the API returns the marginal latent (the average over 40 quantile shifts of the
fitted durations) and flags `duration_marginal`. A duration is unusable when it is missing, is the 1900 sentinel,
has a derived end (float noise), or ends ≥ 240 min after caseend. The duration can be passed as `duration=`
(minutes), or it is computed from anestart and aneend in the case row.

BIC on the same records (records with a usable duration; derived aneend excluded). Lower is better:

| model | n | marginal | duration | ΔBIC | b |
|---|---|---|---|---|---|
| intraop_uo | 3630 | 31013.2 | 30086.8 | −926 | 1.113 |
| intraop_crystalloid | 5863 | 46632.5 | 40378.8 | −6254 | 1.378 |
| intraop_ebl | 3882 | 24140.8 | 22389.7 | −1751 | 1.426 |

Colloid has no fitted value model in the library, so there was nothing to extend. It can follow once it is fitted.

**With derived aneend included** (UO 3666, crystalloid 5971 and EBL 3930 records), b changes by at most 0.004
(1.110, 1.375, 1.424) and the propensities by at most 0.05 on the logit scale. Derived ends are therefore left out
of the fit. They are used only at prediction time, with the flag.

**Heaping propensities, marginal → duration (logit scale):**

| model | mode | marginal | duration |
|---|---|---|---|
| UO | 100 mL | −2.940 | −2.855 |
| UO | 50 mL | −0.542 | −0.556 |
| UO | 10 mL | 1.073 | 1.074 |
| crystalloid | 100 / 50 mL | −0.087 / 3.192 | −0.088 / 3.192 |
| EBL | modes 0–3 | −0.358 / −4.021 / −1.150 / 1.966 | −0.318 / −3.964 / −1.163 / 1.966 |

The propensities barely change: the covariate sharpens the latent, not the recording behaviour.

**Records without a usable duration** are excluded from the fit: 39 UO, 111 crystalloid and 51 EBL records. The
v1 fits used them; the API gives them the marginal.

## 5. Department as a covariate on the heaping propensities (handover 2)

Department enters the logit of every stick-breaking propensity as indicator offsets against general surgery
(thoracic, gynaecology, urology). This is on top of the duration covariate, on the same records:

| model | BIC duration | BIC duration + department | ΔBIC | adopted |
|---|---|---|---|---|
| intraop_uo | 30086.8 (k 9) | 29999.5 (k 18) | −87 | **yes** |
| intraop_crystalloid | 40378.8 (k 8) | 40400.9 (k 14) | +22 | no |
| intraop_ebl | 22389.7 (k 13) | 22440.8 (k 25) | +51 | no |

**UO: posterior share of each recording grid, averaged over the fitted records.** The table attributes each value to
the grid that produced it. A value on the 100 mL grid is also on the 50 and 10 mL grids, so these shares are lower
for the coarse grids than the atlas's raw "on the grid" counts.

| department | n | 100 mL | 50 mL | 10 mL | 5 mL |
|---|---|---|---|---|---|
| General surgery | 2653 | 5% | 29% | 49% | 16% |
| Thoracic surgery | 783 | 2% | 13% | 59% | 26% |
| Gynaecology | 153 | 15% | 53% | 31% | 0% |
| Urology | 41 | 0% | 76% | 24% | 0% |

The direction matches the atlas finding. Gynaecology and urology record on coarse grids and never use 5 mL, while
thoracic surgery records finest. The gynaecology and urology offsets on the 10 mL logit, and the urology offset on
the 100 mL logit, sit at the boundary (propensity ≈ 1 or ≈ 0), so they should be read as "always" or "never". With
41 urology records, the urology split between 100 and 50 mL is weakly identified.

Without a department, or for one not in the fit, the propensities are averaged over the fitted department shares
(general 73.1%, thoracic 21.6%, gynaecology 4.2%, urology 1.1%). These results carry the flag
`department_marginal`.

## 6. Rounding anchor: floor (H1) against nearest (H3)

All three volume models were fitted with floor rounding, meaning the last graduation passed, on the same records.
The ΔBIC below is floor minus nearest, so a positive value favours nearest:

| model | ΔBIC, with duration | ΔBIC, marginal |
|---|---|---|
| intraop_uo | +383 | +317 |
| intraop_crystalloid | +47 | +12 |
| intraop_ebl | +52 | +25 |

Nearest is preferred everywhere. The three open anchors are now `anchor_basis: tested` in the registry.

## 7. UO zero rule

A recorded 0 is a point mass at 0, with p0 = 23/3707 = 0.0062, mixed with the latent mass that records as 0. The
latent is fitted on positive values only. This is the same fit as before: v1 already truncated recorded zeros.

The flags are defined in `exclusion_rules.json`:

| flag | rule | cases |
|---|---|---|
| `zero_anuric` | uo = 0 and (preop_cr > 2 or asa = 6) | 9 |
| `zero_unexplained` | any other recorded 0 | 14 |

Missing is never treated as zero.

**Caveat.** For a recorded 0, the latent component still takes about 54% of the posterior weight, because the
latent's mass below 2.5 mL is comparable to p0. A recorded 0 is therefore "a true zero or a small unmeasured
volume", not confirmed anuria, unless it is flagged `zero_anuric`.

Exact zeros in the other volume variables, left unchanged: EBL 11, crystalloid 0, colloid 5857 (mostly "not given").

## 8. Within-interval density (handover 2)

For a modelled variable, each component of a result is now the fitted latent truncated to that component's
rounding interval, computed at call time on the model grid. The latents are log-spline or log-t, not log-normal,
so this is numerical. With covariates it is the conditional latent.

The component weights come from the same integrals multiplied by the propensities. Each latent component reports
its `interval`, `mean` and `uniform_mean` (the interval midpoint). Every response carries
`density: "latent" | "uniform" | ...` and the model's id, version and fit_hash. Wherever no latent is fitted, the
result is uniform (GUM: midpoint, width/√12).

No posterior tables are stored in the model files. The Monte Carlo cache records the model id, version and fit_hash
it was built with, and warns when they no longer match the installed release.

**UO latent mean minus uniform mean, duration 180 min:**

| recorded | 100 mL | 50 mL | 10 mL |
|---|---|---|---|
| 150 mL | — | −1.1 | −0.1 |
| 300 mL | −6.8 | −1.0 | −0.1 |
| 1000 mL | −3.2 | −0.5 | −0.0 |
| 2000 mL | −1.8 | −0.3 | −0.0 |

`validation_report.json` → `latent_uniform_gap_max` gives, for each modelled variable, the largest gap over all
records, as a fraction of the interval width and with the recorded value where it occurs:

| variable | largest gap | at recorded value |
|---|---|---|
| intraop_ppf | 0.45 | 150 mg |
| height | 0.40 | 140 cm |
| intraop_crystalloid | 0.26 | 50 mL |
| intraop_uo | 0.24 | 5 mL |
| intraop_ebl | 0.22 | 3 mL |
| weight | 0.03 | 34 kg |

Near the median the gap is a few per cent of the interval width (UO table above). The largest gaps occur at small
volumes and for coarse components, where the latent is steep across the interval; there a uniform would be
noticeably biased.

## 9. Model versions (handover 2)

Every model file now has an `id`, an integer `version`, a `fit_hash`, `fitted_on` and `supersedes`. The fit_hash
is the sha256 of the fitted numbers only (loglik to 1e-4). `data/models.lock` lists the current models and every
published version.

* **Baseline:** commit e2376f4, tagged `v0.5.0`, is v1 of everything.
* **v2:** aneend, anestart, opstart, opend, intraop_uo, intraop_crystalloid, intraop_ebl, intraop_ppf and weight.
  The v1 files are in `data/models/archive/` and stay loadable, e.g. `model="aneend@1"`.
* **Still v1:** height and bmi. Height gained references only, and its re-evaluated loglik is equal to 1e-4.

Archived versions are immutable, so their recorded mismatches are pinned in the test:

| archived version | recomputed − stored loglik |
|---|---|
| ppf@1 | +2.611 |
| weight@1 | +0.043 |
| opend@1 | −0.138 |
| anestart@1 | +0.108 |
| opstart@1 | −10.567 |
| aneend@1 | n 6254 recomputed vs 6256 stored |

The contributor path is described in `CONTRIBUTING.md`.

**For vitaldb-bench:** labels or results built from a distribution should record the model id, version and
fit_hash (`Distribution.model`). That repository was not changed.

## 10. Registry: anchors, ICU days, reporting limits (handover 2)

* **`anchor_basis` and `anchor_note`** are set for every variable with an anchor or a counting rule:
  - adm, dis: tested;
  - caseend, lab_dt, lab analytes, pre-op copies, sparse drugs, colloid: convention;
  - age, icu_days: assumption;
  - the three volume models: tested (section 6).
* **icu_days:**
  - The counting rule is now read as nights or completed days, which is an assumption; the ±1-day uncertainty is
    kept.
  - New flag `icu_exceeds_postop`: 36 cases, 24 of them emergencies, where ICU days ≥ the days from the case-day
    midnight to the discharge midnight. The atlas counted 37 with its definition. The 36 include the 6
    `order_violation` cases.
  - The old flag `counting_rule_unknown` is renamed `counting_rule_assumed`.
* **Reporting limits:**

  | variable | limit | status | results | cases |
  |---|---|---|---|---|
  | lactate (`lac`) | 15.0 | `top_coded` | 317 | 28 |
  | aPTT | 400.0 | `top_coded` | 226 | 87 |
  | ESR | 2 | `bottom_coded` (new status) | 89 | 72 |

  `preop_aptt` has 2 values at 400.0 and is top-coded too.

## 11. Imputed op ends and derived aneend (no change; recorded for the atlas)

* **Imputed op end (537 cases), by anaesthesia type:** GA 7.0%, spinal 30%, sedation 41.7%.
* **By department:** general 6.9%, thoracic 16.6%, gynaecology 3.5%, urology 3.4%.
* **Derived aneend (130 cases):** about 2% in every department.
* There is no trend with caseid, and 14 cases are both.

## Questions for the VitalDB team

1. Were the missing op-end times imputed deliberately (op end = recording end)? Why is the rate so much higher for
   spinal anaesthesia, sedation and thoracic cases?
2. How was aneend derived for the 130 cases with float-noise values?
3. Is a recorded intraoperative urine output of 0 a measured zero or "not measured"?
4. Is `p` inorganic phosphorus (mg/dL)? What is the colloid unit for kidney cases?
5. Are 15.0 mmol/L (lactate), 400 s (aPTT) and 2 mm/h (ESR) analyser reporting limits?
