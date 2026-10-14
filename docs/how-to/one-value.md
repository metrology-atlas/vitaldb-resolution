# Get the distribution of one value

## From a clinical row

`case_distribution` takes a whole `clinical_data.csv` row (a dict or a pandas Series) and picks the
covariates from it. This is the easiest way to handle timestamps and derived variables, which need several
fields of the same case.

```python
import vitaldb_resolution as vr

reg = vr.load()
row = {  # an illustrative row in VitalDB's format (times in seconds from the recording start)
    "caseend": 11542, "anestart": -552, "aneend": 10848, "opstart": 1668, "opend": 10368,
    "adm": -236220, "age": 77, "sex": "M", "height": 160.2, "weight": 67.5, "bmi": 26.3,
    "department": "General surgery", "emop": 0,
}
d = reg.case_distribution(row, "opstart")
print(d.summary())         # mean about 1666 s from the recording start, sd about 86 s
print(d.extra)             # the recorded and mean clock times, and the reference used
```

Derived variables are sampled from their inputs' posteriors:

```python
print(reg.case_distribution(row, "op_duration").summary())   # minutes, from opstart and opend
print(reg.case_distribution(row, "bmi").summary())           # from height and weight
```

## From a value and covariates

```python
d = reg.distribution("weight", 60.3, sex="F", age=60, department="General surgery")
dv = reg.distribution("weight", 60.3, sex="F", age=60, department="General surgery", variability=True)
print(d.std(), dv.std())   # with variability: written-in weights also carry measurement variability
```

## Read the response

| Attribute | What it is |
|---|---|
| `status` | `ok`, or why there is no distribution (see [Statuses, kinds and flags](../concepts/responses.md)) |
| `kind` | how the distribution arises: `heaped`, `single_resolution`, `mixed_source`, `derived`, ... |
| `components` | the mixture: one per rounding mode or source, each with a `label` and `weight` |
| `flags` | warnings that do not prevent a distribution, e.g. `off_grid` or `source_unknown` |
| `extra` | additional fields, e.g. clock times for timestamps |

Methods: `mean()`, `std()`, `var()`, `interval(level)`, `cdf(x)`, `pdf(x)`, `ppf(q)`, `sample(n, rng)`,
`summary()` and `to_dict(include_grid=False)`. The summary methods raise `ValueError` when `status` is not
`ok`; check `d.ok` first.
