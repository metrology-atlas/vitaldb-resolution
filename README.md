# vitaldb-resolution 0.6

Every recorded VitalDB time and value as a probability distribution of the true value: native resolution,
or a mixture of resolutions deconvolved from heaped data. Models are JSON files evaluated by one generic engine.

## Install and use

```bash
pip install -e .            # numpy, scipy; pandas for the DataFrame helpers
pytest -q tests             # 25 tests, no data needed
```

```python
import vitaldb_resolution as vr
reg = vr.load()

d = reg.distribution('intraop_uo', 100)                 # heaped volume
d.mean(), d.std(), d.interval(0.95), d.sample(1000)
d.components                                             # one per rounding mode, with weights
d.to_dict()                                              # JSON response (include_grid=True for densities)

reg.distribution('height', 160.0, sex='F', age=60, emop=0)          # covariate-dependent latent
reg.distribution('weight', 60.3, sex='F', age=60, department='General surgery', variability=True)
reg.distribution('hct', 35)                              # source unknown -> mixture of the two analysers
reg.distribution('hct', 35, source='blood_gas')

row = clinical.iloc[0]                                   # a clinical_data.csv row
reg.case_distribution(row, 'opstart')                    # seconds from casestart, clock time in d.extra
reg.case_distribution(row, 'op_duration')                # derived interval (minutes)
reg.case_distribution(row, 'bmi')
reg.distributions('aneend', clinical)                    # one per row
reg.lab_distributions(lab)                               # lab_data.csv rows; infers haematocrit source
vr.Registry.summary_frame(dists)                         # mean, sd, 95% interval, status, flags
```

HTTP: `python -m vitaldb_resolution.serve --port 8000`, then
`GET /distribution?variable=intraop_uo&value=100`, `POST /distribution {"variable", "value", "covariates"}`,
`GET /registry/<variable>`.

## Monte Carlo cache (whole dataset)

```bash
python -m vitaldb_resolution.cache build --clinical clinical_data.csv --lab lab_data.csv \
       --out vitaldb_cache --dtype int16 --knots 1024        # defaults; ~2 min to build, 214 MB
```

```python
cache = vr.Cache.load('vitaldb_cache')                   # 0.7 s
s = cache.sampler(seed=1, variability=True)              # height/weight measurement variability at draw time
d = s.draw()        # dict: variable -> array over the 6,388 cases (NaN where status is not ok),
                    # derived 'bmi' and the four intervals from the same draws, 'lab' -> 928,448 rows
cache.status('opend'), cache.flags('opend')
```

Fitted-model posteriors are stored as quantile tables (`--knots`, `--dtype`); simple shapes are stored exactly.
Precision against the exact posteriors, worst case over all 66,180 tables (see `tools/precision_study.py`):

| Knots | dtype | RAM | W1 / SD (max) | SD error (max) | mean error / SD (max) |
| --- | --- | --- | --- | --- | --- |
| 256 | int16 | 112 MB | 3.2e-3 | 1.1 % | 2.9e-3 |
| 1024 | int16 (default) | 214 MB | 6.3e-4 | 0.42 % | 4.7e-4 |
| 1024 | float32 / float64 | 349 / 620 MB | 6.0e-4 | 0.42 % | 4.7e-4 |
| 4096 | int16 | 621 MB | 2.3e-4 | 0.11 % | 2.0e-4 |
| 4096 | float32 / float64 | 1.2 / 2.2 GB | 2.1e-4 | 0.11 % | 2.0e-4 |

The number of knots sets the precision; the storage type does not (int16 adds ~1e-5 SD).

## Response

`status`: `ok`, `missing`, `invalid` (1900 placeholder), `imputed` (op end set from recording end, anaesthesia end
derived from it), `not_applicable` (categorical, identifiers), `zero_or_not_recorded` (0 in a drug or fluid field),
`top_coded` (age ">89", lactate 15.0, aPTT 400.0: open interval above), `bottom_coded` (ESR 2), `error`
(missing covariates). Only `ok` carries components. A recorded urine output of 0 returns a point mass at 0 mixed
with the latent below half the base resolution, flagged `zero_anuric` or `zero_unexplained`; missing is never 0.

`density`: where the shape inside each interval comes from. `latent` means the fitted latent, truncated to each
component's interval and computed at call time; with the duration covariate it is the conditional latent, and
without it the marginal (flag `duration_marginal`). `uniform` means rectangular over the interval (GUM: midpoint,
width/√12), used wherever no latent is fitted. The other values are `exact`, `triangular` and `derived`. Latent
components report their `interval`, `mean` and `uniform_mean`.

`model`: `id`, `version` and `fit_hash` of the model used. Pin a version with `model="aneend@1"`.

`kind`: `single_resolution`, `heaped`, `mixed_source`, `computed_value`, `exact`, `count_uncertain`, `derived`.

`flags` (value still returned): `off_grid`, `outside_model_support`, `child_native_resolution`, `source_unknown`,
`possibly_albumin`, `unit_uncertain`, `off_unit_dose_grid`, `counting_rule_assumed`, `icu_exceeds_postop`, `order_violation`,
`same_time`, `clock_uncertain`, `clock_offset_mismatch`, `age_covariate_top_coded`, `recorded_bmi_mismatch`,
`duration_marginal`, `department_marginal`, `zero_anuric`, `zero_unexplained`.

## Model versions and contributing

Each model has an integer version and a `fit_hash`, and `data/models.lock` lists them for the release. Published
versions never change: superseded ones stay loadable from `data/models/archive/`. To propose a new version of a
model, see [CONTRIBUTING.md](CONTRIBUTING.md).

## Contents

| Path | What |
| --- | --- |
| `vitaldb_resolution/data/registry.json` | 114 entries: method, resolution, anchor, model file, covariates, rules, evidence |
| `vitaldb_resolution/data/models/*.model.json` | 11 models (current versions); `archive/` holds superseded versions |
| `vitaldb_resolution/data/models.lock` | id, version and fit_hash of every model; every published version |
| `vitaldb_resolution/data/exclusion_rules.json` | exclusion and flag rules with case counts |
| `vitaldb_resolution/engine.py` | generic evaluator (`rounding_engine` v0.4) |
| `vitaldb_resolution/registry.py` | the API |
| `vitaldb_resolution/distribution.py` | mixture distribution object |
| `tools/build_registry.py` | rebuilds the registry from the data and the recorded decisions |
| `tools/validate.py` | reproduces stored log-likelihoods; runs every clinical and lab value |
| `vitaldb_resolution/fitcheck.py`, `versions.py` | recompute stored fit statistics; model ids, versions, fit_hash |
| `tools/fit_*.py`, `tools/version_models.py` | fitters using the library likelihood; versioning |
| `vitaldb_resolution/cache.py` | whole-dataset cache: build (CLI) and vectorised sampler |
| `tools/precision_study.py` | precision, memory and speed of every knots x dtype option |

## Methods in the registry

| Method | Used for |
| --- | --- |
| `model` | urine output, blood loss, crystalloid, propofol bolus, height, weight |
| `timestamp_model` | anaesthesia and operation start/end |
| `uniform` | lab analytes, pre-op copies, sparse drugs, dates, device times |
| `source_dependent` / `decimal_dependent` | haematocrit (CBC 0.1 % or blood gas 1 %), pre-op PaO2 |
| `neighbours` | PT %, fibrinogen (computed values; only some values reachable) |
| `exact` | blood product units, unit-dose drugs |
| `colloid_rules`, `age`, `triangular_count` | colloid, age, ICU days |
| `derived` | BMI, length of stay, the four charted intervals |

Assumptions: inputs of derived variables are independent given their records and covariates; the haematocrit
mixture uses the share of whole-number results from each analyser, not the value's likelihood under each.

## Timing and availability

Time variables are distributions of when an event happened, not of when its value became known. VitalDB's lab
timestamp (`lab_dt`) is a per-specimen time: central-lab panels are logged about 10 min or more after the draw and
blood gases 3-12 min after it, and the result follows a turnaround the release does not record. Its registry entry
carries `available_at: lab_time_plus_turnaround`: a lab value must not be used as model input at `lab_dt` (the
availability rule), only after an assumed turnaround.

## Licence and data

Code: MIT (`LICENSE`). The fitted models and registry are derived from the VitalDB open dataset (CC BY 4.0);
see `NOTICE.md` for attribution. No VitalDB records are included.
