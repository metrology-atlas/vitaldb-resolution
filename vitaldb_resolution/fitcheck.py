"""Recompute each fitted model's log-likelihood on its fitting records with the library's own evaluator.

Every ``*.model.json`` stores a ``fit`` block (n, loglik, BIC). ``recompute(variable, clinical)`` selects the
same records the fit used and evaluates them with ``engine``; ``tests/test_fit_consistency.py`` checks that
stored and recomputed values agree for every model (needs the VitalDB CSVs, see that test).
"""
import math

import numpy as np

from . import engine
from .registry import Registry, _ref_window, anaesthesia_duration

VALUE_MODELS = ('intraop_uo', 'intraop_ebl', 'intraop_crystalloid', 'intraop_ppf', 'height', 'weight')
TIMESTAMP_MODELS = ('anestart', 'opstart', 'opend', 'aneend')


def _adults_first_case(d):
    import pandas as pd
    a = d.assign(age=pd.to_numeric(d.age, errors='coerce'))
    return a[a.age >= 18].sort_values('caseid').drop_duplicates('subjectid')


def value_records(spec, d):
    """(recorded values, covariate dicts) used to fit a value model."""
    import pandas as pd
    v = spec['variable']
    if v in ('height', 'weight'):
        rows = _adults_first_case(d); rows = rows[rows[v].notna()]
        keys = spec.get('required_covariates', [])
        return rows[v].values.astype(float), [{k: r[k] for k in keys} for r in rows.to_dict('records')]
    res = spec['resolution']['effective']; x = d[v]
    keep = (x > 0) & (np.abs(x / res - np.round(x / res)) < 1e-9)
    if v == 'intraop_ppf':
        age = pd.to_numeric(d.age, errors='coerce'); keep &= age.notna() & d.weight.notna()
        rows = d[keep].assign(age=age[keep])
        return rows[v].values.astype(float), [{k: r[k] for k in ('weight', 'age', 'sex')} for r in rows.to_dict('records')]
    rows = d[keep]
    if spec['latent'].get('location_covariates'):
        keys = spec.get('required_covariates', [])
        covs = [{'duration': anaesthesia_duration(r), **{k: r[k] for k in keys}} for r in rows.to_dict('records')]
        ok = np.array([c['duration'] is not None for c in covs])
        return rows[v].values.astype(float)[ok], [c for c, o in zip(covs, ok) if o]
    return rows[v].values.astype(float), [None] * len(rows)


def value_loglik(spec, vals, covs):
    M = engine.Model(spec); tot = 0.0; cache = {}
    for v, c in zip(vals, covs):
        key = None if c is None else tuple(sorted((k, round(float(x), 9) if isinstance(x, (int, float, np.floating)) else x)
                                                 for k, x in c.items()))
        if key not in cache: cache[key] = M.pmf(spec, c)
        q = cache[key]
        tot += math.log(max(q[int(round(v / M.ru))], 1e-300))
    return tot


def timestamp_records(reg, var, d):
    rows = []
    for r in d.to_dict('records'):
        if not reg.case_distribution(r, var).ok: continue
        ctx = reg.timestamp_context(var, r[var], r)
        c = {'ref': ctx['ref']}
        if var == 'anestart': c.update(ref_window=_ref_window(ctx['recording_start_clock']), emop=int(r['emop']))
        rows.append((ctx['M'], c))
    return rows


def timestamp_loglik(spec, rows):
    TM = engine.TimestampModel(spec)
    return float(sum(math.log(max(TM.prob_recorded(M, c), 1e-300)) for M, c in rows))


def recompute(variable, clinical, reg=None, version=None):
    """Stored and recomputed fit statistics for one model: dict(n, n_stored, loglik, loglik_stored).

    Args:
        variable: Model id (the variable name).
        clinical: ``clinical_data.csv`` as a DataFrame.
        reg: A :class:`Registry` (default: the installed one).
        version: An archived version to check instead of the current one.
    """
    from . import versions
    reg = reg or Registry.load()
    fn = f'{variable}.model.json'
    if version is not None and version != reg.spec(fn).get('version'):
        fn = versions.archive_file(variable, version)
    spec = reg.spec(fn)
    if variable in TIMESTAMP_MODELS:
        rows = timestamp_records(reg, variable, clinical); n = len(rows); ll = timestamp_loglik(spec, rows)
    else:
        vals, covs = value_records(spec, clinical); n = len(vals); ll = value_loglik(spec, vals, covs)
    return {'variable': variable, 'version': spec.get('version'), 'n': n, 'n_stored': spec['fit']['n'],
            'loglik': ll, 'loglik_stored': spec['fit']['loglik']}
