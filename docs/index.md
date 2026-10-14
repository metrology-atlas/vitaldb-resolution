# vitaldb-resolution

**Every recorded VitalDB time and value as a probability distribution of the true value.** A urine output
charted as 100 mL was probably rounded to the nearest 50 or 100 mL; a height of 160 cm may have been
written in, not measured; an operation start charted at 18:05 was typed by a person to the minute. This
library turns each recorded value into the distribution of what it actually was: its native resolution, or
a mixture of rounding modes deconvolved from heaped data.

```python
import vitaldb_resolution as vr

reg = vr.load()
d = reg.distribution("intraop_uo", 100)       # urine output recorded as 100 mL
print(d)                                       # <Distribution intraop_uo=100 ok heaped: mean 99.26, sd 15.7, 95% [61.05, 135.9]>
```

## What it covers

| | |
|---|---|
| **Variables** | 114 entries: every field of VitalDB's `clinical_data.csv` and every analyte of `lab_data.csv` ([list](reference/variables.md)) |
| **Fitted models** | 11: urine output, blood loss, crystalloid, propofol bolus, height, weight, the four charted anaesthesia and operation times, BMI |
| **Interfaces** | a Python API, a whole-dataset Monte Carlo cache, a small HTTP service |
| **Dependencies** | numpy, scipy; pandas for the table helpers |

Each model is a JSON file evaluated by one generic engine, so the fitted parameters, the evidence for each
choice and the rules that exclude or flag values can all be read and checked.

## Install

```bash
pip install git+https://github.com/meshlab/vitaldb-resolution
```

The registry and models ship with the package. The VitalDB data themselves are not included: download
them from [vitaldb.net](https://vitaldb.net) to run whole tables or build the cache.

## Licence and data

Code: MIT, © 2026 Andrew J. Goodwin, University of Sydney. The fitted models, registry and rules are
derived from the VitalDB open dataset (CC BY 4.0) and contain fitted parameters, value lists and aggregate
counts only: no patient records and no case IDs. If you use them, cite VitalDB: Lee HC et al.,
*Scientific Data* 2022;9:279, [doi:10.1038/s41597-022-01411-5](https://doi.org/10.1038/s41597-022-01411-5).
