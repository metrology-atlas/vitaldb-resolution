# Get started

The registry and its fitted models ship with the package, so everything on this page runs without the
VitalDB data.

## A heaped volume

```python
import vitaldb_resolution as vr

reg = vr.load()
d = reg.distribution("intraop_uo", 100)                  # urine output recorded as 100 mL
print(d.mean(), d.std(), d.interval(0.95))               # about 99.3, 15.7, (61.1, 135.9) mL
for c in d.components:
    print(c.label, round(c.weight, 3))                   # 100 mL 0.214, 50 mL 0.558, 10 mL 0.195, 5 mL (base) 0.033
```

A charted 100 mL was most likely rounded to 50 mL (56%), but could have come from rounding to the nearest
100, 10 or 5 mL. Each rounding mode is one component of the mixture; together they give the distribution
of the true volume.

## Covariates

Some models depend on the patient. Height and weight need sex, age and whether the operation was an
emergency; the registry entry says which covariates a variable needs.

```python
print(reg.entry("height")["required_covariates"])
print(reg.distribution("height", 160.0, sex="F", age=60, emop=0))      # mean 159.6, sd 1.74 cm
```

## A value whose source is unknown

```python
print(reg.distribution("hct", 35))                       # mixture of the two analysers, flag source_unknown
print(reg.distribution("hct", 35, source="blood_gas"))   # blood-gas analyser: 1% resolution
```

## No distribution

Only `status == "ok"` responses carry a distribution; the others say why there is none.

```python
d = reg.distribution("age", ">89")
print(d.status, d.ok)                                    # top_coded False
```

## Use it

```python
import numpy as np

d = reg.distribution("intraop_uo", 100)
draws = d.sample(10_000, rng=np.random.default_rng(0))  # true values, for Monte Carlo analyses
response = d.to_dict()                                   # JSON-ready
```

Next: [the distribution of one value](how-to/one-value.md) in more detail, or [whole tables](how-to/whole-tables.md).
