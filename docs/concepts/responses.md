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
| `top_coded` | age recorded as ">89" |
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

## Flags

Flags never remove the distribution; they say what to be careful about.

`off_grid`, `outside_model_support`, `child_native_resolution`, `source_unknown`, `possibly_albumin`,
`unit_uncertain`, `off_unit_dose_grid`, `counting_rule_unknown`, `order_violation`, `same_time`,
`clock_uncertain`, `clock_offset_mismatch`, `age_covariate_top_coded`, `recorded_bmi_mismatch`.

The rules that set statuses and flags, and how many cases each affects, are in
`vitaldb_resolution/data/exclusion_rules.json` (counts only; no case IDs).
