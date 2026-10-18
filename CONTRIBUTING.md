# Contributing

Every model in `vitaldb_resolution/data/models/` can be examined on its own, and improvements and corrections are
welcome. Model versions keep this safe: a proposed fit sits beside the current one, and nothing that someone has
already used changes under them.

## How models are versioned

- Each model file has an `id` (the variable) and an integer `version`.
- **Published versions never change.** A refit or a correction is a new version. The old file moves to
  `data/models/archive/<id>.v<N>.model.json` and stays loadable: `reg.distribution("aneend", t, model="aneend@1", ...)`.
- `fit_hash` is the sha256 of the canonical JSON of the fitted numbers only: parameters, propensities, loglik and n
  (`vitaldb_resolution.versions.fit_hash`). Editing a description, a note or a reference does not change it, and
  does not need a new version.
- `fitted_on` records the data release, the sha256 of `exclusion_rules.json` at fitting time, the fitter's commit
  and the seed. `supersedes` names the previous version and gives a one-line reason.
- Each release ships `data/models.lock`. It lists every model's id, version and fit_hash, and the fit_hash of
  every version ever published.
- Releases follow semantic versioning. A new model version is a minor bump; an API change is a major bump.
- Every result names the model id, version and fit_hash it came from (`Distribution.model`). Record these with any
  label or result that uses a distribution.

## Proposing a new version of a model

1. **Refit as a new version.** Use the fitting tools (`tools/fit_volume.py`, `tools/fit_timestamps.py`,
   `tools/fit_logt.py`). Each one evaluates the library's own likelihood. The stored fit block must be the engine's
   evaluation, not an approximation.
2. **Archive the current version.** Move it to `data/models/archive/<id>.v<N>.model.json` unchanged. Give the new
   file `version` N+1, its `fit_hash`, `fitted_on` and `supersedes`. `tools/version_models.py` shows how.
3. **Validate.**
   - Run `pytest`. With `VITALDB_DATA` set to the folder holding the VitalDB CSVs, `tests/test_fit_consistency.py`
     checks that the stored loglik equals the recomputed loglik.
   - Run `python tools/validate.py <clinical_data.csv> <lab_data.csv> --full`.
4. **Compare with the current version on the same records:**
   - loglik and BIC;
   - the heaping propensities;
   - one figure of the fitted against the observed distribution of recorded values.

   Put the aggregate results in `results/`. Do not add per-case files: `tests/test_no_per_case_files.py` enforces
   this.
5. **Update `models.lock`** with the new version and add it to `published`. Add a `CHANGELOG.md` entry.
6. **Open a pull request.** Say what changed, why, and what the comparison shows.

A correction that changes no fitted number (wording, references, a flag description) is an ordinary pull request
and needs no new version.
