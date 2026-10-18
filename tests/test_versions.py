# SPDX-License-Identifier: MIT
"""Model versions: fingerprints match the files, models.lock matches the files, published versions never change,
and pinned versions load."""
import json
from pathlib import Path

import pytest

from vitaldb_resolution import load, versions as V

MODELS = Path(V.MODELS)
LOCK = V.load_lock()


def _spec(f):
    return json.loads((MODELS / f).read_text(encoding='utf-8'))


def _files():
    return sorted(p.name for p in MODELS.glob('*.model.json')) + sorted(f'archive/{p.name}' for p in (MODELS / 'archive').glob('*.model.json'))


@pytest.mark.parametrize('f', _files())
def test_fit_hash_matches_file(f):
    s = _spec(f)
    assert isinstance(s['id'], str) and isinstance(s['version'], int) and s['version'] >= 1
    assert s['fit_hash'] == V.fit_hash(s)
    assert {'data', 'exclusion_rules_sha256', 'code', 'seed'} <= set(s['fitted_on'])


def test_lock_matches_files():
    cur = {p.name for p in MODELS.glob('*.model.json')}
    assert {m['file'] for m in LOCK['models'].values()} == cur
    for mid, m in LOCK['models'].items():
        s = _spec(m['file'])
        assert (s['id'], s['version'], s['fit_hash']) == (mid, m['version'], m['fit_hash'])


@pytest.mark.parametrize('pin', sorted(LOCK['published']))
def test_published_versions_unchanged(pin):
    mid, ver = V.parse_pin(pin)
    f = LOCK['models'][mid]['file'] if LOCK['models'][mid]['version'] == ver else V.archive_file(mid, ver)
    assert V.fit_hash(_spec(f)) == LOCK['published'][pin]['fit_hash']


def test_superseded_versions_are_archived():
    for mid, m in LOCK['models'].items():
        s = _spec(m['file'])
        for v in range(1, m['version']):
            assert (MODELS / V.archive_file(mid, v)).exists()
        if m['version'] > 1:
            assert s['supersedes']['version'] == m['version'] - 1 and s['supersedes']['reason']


def test_editing_text_does_not_change_the_hash():
    s = _spec('intraop_uo.model.json')
    t = json.loads(json.dumps(s)); t['latent']['note'] = 'edited'; t.setdefault('references', []).append({'id': 'x'})
    assert V.fit_hash(t) == V.fit_hash(s)
    t['allocation'][0]['prob']['logit'] += 1e-6
    assert V.fit_hash(t) != V.fit_hash(s)


def test_pinned_version_loads_and_reports_provenance():
    reg = load()
    for mid, m in LOCK['models'].items():
        if m['version'] < 2 or mid not in ('aneend', 'intraop_uo'): continue
        cov = dict(adm=-86400 * 2, caseend=14000, emop=0, anestart=600, duration=180)
        val = 13000 if mid == 'aneend' else 300
        d_cur = reg.distribution(mid, val, **cov)
        d_v1 = reg.distribution(mid, val, model=f'{mid}@1', **cov)
        assert d_cur.model['version'] == m['version'] and d_v1.model['version'] == 1
        assert d_v1.model['fit_hash'] == LOCK['published'][f'{mid}@1']['fit_hash']
        assert d_cur.model['fit_hash'] == m['fit_hash']
    with pytest.raises(ValueError):
        reg.distribution('intraop_uo', 300, model='intraop_uo@99')
