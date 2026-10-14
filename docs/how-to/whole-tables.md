# Work with whole tables

These need VitalDB's `clinical_data.csv` and `lab_data.csv` (download from vitaldb.net) and pandas.

```python
# requires the VitalDB data
import pandas as pd
import vitaldb_resolution as vr

reg = vr.load()
clinical = pd.read_csv("clinical_data.csv", encoding="utf-8-sig")
lab = pd.read_csv("lab_data.csv")

dists = reg.distributions("aneend", clinical)          # one per row
table = vr.Registry.summary_frame(dists)               # mean, sd, 95% interval, status, flags per row
labs = reg.lab_distributions(lab)                      # one per lab row; infers the haematocrit's analyser
```

`summary_frame` is the quickest way to see which records have no distribution (`status`) and which carry
warnings (`flags`). The exclusion and flag rules, with the number of cases each affects, are in
`reg.rules` (`exclusion_rules.json`).

!!! warning "Lab values and time"
    `lab_distributions` describes each result's value. It says nothing about when the value became known:
    VitalDB's lab timestamp marks the specimen, and the result follows an unrecorded turnaround. See
    [Timing and availability](../concepts/availability.md).
