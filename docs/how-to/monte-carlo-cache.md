# Propagate uncertainty with the cache

For analyses over the whole dataset, evaluating each distribution separately is slow. The cache stores
every record's distribution as a compact quantile table and draws one value per record for the whole
dataset in tens of milliseconds.

## Build it once

```bash
python -m vitaldb_resolution.cache build --clinical clinical_data.csv --lab lab_data.csv \
       --out vitaldb_cache --dtype int16 --knots 1024      # defaults: about 2 minutes, 214 MB
```

The cache is derived from the data: keep it with the data and out of any repository (the package's
`.gitignore` excludes `vitaldb_cache*/`).

## Draw

```python
# requires the VitalDB data
import vitaldb_resolution as vr

cache = vr.Cache.load("vitaldb_cache")
sampler = cache.sampler(seed=1, variability=True)
estimates = []
for _ in range(200):                                   # one Monte Carlo iteration per draw
    d = sampler.draw()                                 # variable -> array over the 6,388 cases
    estimates.append(your_analysis(d))                 # BMI and the four intervals come from the same draws
```

Records whose status is not `ok` are NaN in every draw; `cache.status(v)` and `cache.flags(v)` give them
per record. Derived values (BMI, the intervals) are computed from the same draws as their inputs, so
shared inputs stay consistent.

## Precision

The number of knots sets the precision; the storage type barely matters (int16 adds about 1e-5 SD).

| Knots | dtype | RAM | W1 / SD (max) | SD error (max) |
|---|---|---|---|---|
| 256 | int16 | 112 MB | 3.2e-3 | 1.1 % |
| 1024 | int16 (default) | 214 MB | 6.3e-4 | 0.42 % |
| 4096 | int16 | 621 MB | 2.3e-4 | 0.11 % |

Worst case over all 66,180 tables, against the exact posteriors (`tools/precision_study.py`).
