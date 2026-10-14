# Heaping and rounding modes

A value written by a person is rarely recorded at the instrument's resolution. Urine output measured to
the nearest 5 mL is charted as 100, 150 or 200 far more often than as 105; heights cluster on whole and
half centimetres, and on tens. This is **heaping**: the recorded values pile up on round numbers.

Each recorded value is then the true value rounded in one of several ways, a **rounding mode** (to 5 mL,
10 mL, 50 mL, 100 mL...), and the choice of mode depends on the value itself (large volumes are rounded
more coarsely). A value recorded as 100 could have come from any mode that maps something to 100.

## The model

For a fitted variable, the model file holds:

- a **latent distribution** of true values (a log-spline, or a regression on covariates such as sex, age
  and department for height and weight);
- an ordered list of **rounding modes**, each with a probability of being used that can vary with the
  value (stick-breaking allocation: each mode takes a share of what the previous modes left);
- the **grid** on which the posterior is tabulated.

The parameters are fitted by maximum likelihood to the whole column of recorded values. Given one recorded
value, the posterior is the latent distribution restricted to the true values that each mode would record
as that value, weighted by how likely each mode was there. Each mode that can produce the value becomes one
component of the returned mixture.

## Other methods

Not every variable needs a fitted model. The registry gives each one a method, with the evidence for it:

| Method | Used for | Distribution |
|---|---|---|
| `model` | urine output, blood loss, crystalloid, propofol bolus, height, weight | fitted mixture of rounding modes |
| `timestamp_model` | anaesthesia and operation start and end | see [Charted times](timestamps.md) |
| `uniform` | lab analytes, pre-operative copies, sparse drugs, dates, device times | uniform over one resolution step |
| `source_dependent`, `decimal_dependent` | haematocrit (CBC 0.1% or blood gas 1%), pre-operative PaO2 | resolution chosen by the analyser or the decimal |
| `neighbours` | PT %, fibrinogen | computed values: only some are reachable, so the interval runs to the neighbouring reachable values |
| `exact` | blood product units, unit-dose drugs | point mass |
| `colloid_rules`, `age`, `triangular_count` | colloid, age, ICU days | container rules, completed years, a day count ±1 |
| `derived` | BMI, length of stay, the four charted intervals | sampled from the inputs' posteriors |

The anchor says how a single-resolution value relates to the truth: `floor` (completed years, whole days
since admission), `ceiling`, or `nearest`.

## Assumptions

Inputs of derived variables are treated as independent given their records and covariates. The
haematocrit mixture uses the share of whole-number results from each analyser, not the value's likelihood
under each.
