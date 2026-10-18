# Statuses, kinds and flags

## Status

Only `ok` responses carry components.

| Status | Meaning |
|---|---|
| `ok` | a distribution exists |
| `missing` | no value recorded |
| `invalid` | a placeholder (1900-01-01 dates) or a value outside the model's range |
| `imputed` | set by the system rather than recorded (operation end from the recording end, anaesthesia end derived from it) |
| `not_applicable` | categorical fields and identifiers |
| `zero_or_not_recorded` | 0 in a drug or fluid field: none given, or not charted |
| `top_coded` | at an upper reporting limit: age ">89", lactate 15.0, aPTT 400.0 (open interval above) |
| `bottom_coded` | at a lower reporting limit: ESR 2 (open interval below) |
| `error` | required covariates missing (named in `note`) |

## Kind

| Kind | How the distribution arises |
|---|---|
| `single_resolution` | one resolution step |
| `heaped` | a mixture of rounding modes |
| `mixed_source` | a mixture of sources (analysers) with different resolutions |
| `computed_value` | a computed value; interval to neighbouring reachable values |
| `exact` | a point mass |
| `count_uncertain` | a count with an uncertain counting rule |
| `derived` | sampled from its inputs |
| `zero_point_mass` | a recorded urine output of 0: a point mass at 0 mixed with the latent below half the base resolution |

## Density

Where the shape inside each interval comes from (`density`, on every response):

| Density | Meaning |
|---|---|
| `latent` | the fitted latent truncated to each component's interval, computed at call time. With the duration covariate it is the conditional latent; without it, the marginal (flag `duration_marginal`). Latent components report `interval`, `mean` and `uniform_mean`. |
| `uniform` | rectangular over the interval (GUM: midpoint, width/√12); used wherever no latent is fitted |
| `exact`, `triangular`, `derived` | a point mass, a ±1-day count, or sampled from the inputs |

`model` names the model's `id`, `version` and `fit_hash`; pin a version with `model="aneend@1"`.

## Flags

Flags never remove the distribution; they say what to be careful about.

`off_grid`, `outside_model_support`, `child_native_resolution`, `source_unknown`, `possibly_albumin`,
`unit_uncertain`, `off_unit_dose_grid`, `counting_rule_assumed`, `icu_exceeds_postop`, `order_violation`, `same_time`,
`clock_uncertain`, `clock_offset_mismatch`, `age_covariate_top_coded`, `recorded_bmi_mismatch`, `duration_marginal`,
`department_marginal`, `zero_anuric`, `zero_unexplained`.

The rules that set statuses and flags, and how many cases each affects, are in
`vitaldb_resolution/data/exclusion_rules.json` (counts; case IDs only where a specific case is discussed).
