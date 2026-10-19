# SPDX-License-Identifier: MIT
"""Model ids, versions and fingerprints.

Each model file carries an ``id`` and an integer ``version``. A published version never changes: a refit or a
correction writes a new version, and the old file moves to ``data/models/archive/<id>.v<N>.model.json``, where it
stays loadable (``model="aneend@1"``). ``fit_hash`` is the sha256 of the canonical JSON of the fitted numbers only
(parameters, propensities, loglik to 1e-4, n), so editing descriptions, notes or references does not change it.
``data/models.lock`` lists every model's current id, version and fit_hash, and every published version.
"""
import hashlib
import json
import os

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
MODELS = os.path.join(DATA, 'models')
ARCHIVE = os.path.join(MODELS, 'archive')
LOCK = os.path.join(DATA, 'models.lock')

FITTED_SECTIONS = ('latent', 'allocation', 'zero_model', 'measurement_variability', 'reference')
TEXT_KEYS = {'note', 'notes', 'description', 'meaning', 'evidence', 'references', 'citation', 'source', 'method',
             'label', 'data', 'date', 'basis'}


def _numbers(x):
    """The numeric skeleton of a JSON value: numbers (12 significant digits) and booleans, keyed by path;
    descriptive text is dropped.
    """
    if isinstance(x, bool) or x is None:
        return x
    if isinstance(x, (int, float)):
        return float(f'{float(x):.12g}')
    if isinstance(x, dict):
        out = {k: _numbers(v) for k, v in x.items() if k not in TEXT_KEYS}
        return {k: v for k, v in out.items() if v is not None and v != {} and v != []}
    if isinstance(x, list):
        return [_numbers(v) for v in x]
    return None          # strings: descriptive, or structure that the numbers already pin down


def fitted_numbers(spec):
    """The fitted content of a model file, as hashed by :func:`fit_hash`."""
    out = {s: _numbers(spec[s]) for s in FITTED_SECTIONS if s in spec}
    if 'fit' in spec:
        out['fit'] = {k: _numbers(spec['fit'][k]) for k in ('n',) if k in spec['fit']}
        if 'loglik' in spec['fit']:      # to 1e-4: re-evaluating the same parameters must not make a new version
            out['fit']['loglik'] = round(float(spec['fit']['loglik']), 4)
    if not out:                                  # derived models (bmi): the defining formula
        out = {k: spec.get(k) for k in ('formula', 'inputs')}
    return out


def fit_hash(spec):
    """sha256 (hex) of the canonical JSON of :func:`fitted_numbers`."""
    canon = json.dumps(fitted_numbers(spec), sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(canon.encode()).hexdigest()


def archive_file(model_id, version):
    """File name, relative to ``data/models``, of an archived version."""
    return f'archive/{model_id}.v{int(version)}.model.json'


def parse_pin(pin):
    """``"aneend@1"`` -> ``("aneend", 1)``; ``"aneend"`` -> ``("aneend", None)``."""
    if '@' not in pin:
        return pin, None
    i, v = pin.split('@', 1)
    return i, int(v.lstrip('v'))


def load_lock(path=LOCK):
    with open(path, encoding='utf-8') as fh:
        return json.load(fh)
