# SPDX-License-Identifier: MIT
"""Stored fit statistics agree with what the library recomputes.

Data-free checks run always. The recomputation needs the VitalDB clinical table: set VITALDB_DATA to the folder
holding clinical_data.csv (it is skipped otherwise; it takes a few minutes).
"""
import json
import math
import os
from pathlib import Path

import pytest

MODELS = Path(__file__).resolve().parents[1] / 'vitaldb_resolution' / 'data' / 'models'
FITTED = ['intraop_uo', 'intraop_ebl', 'intraop_crystalloid', 'intraop_ppf', 'height', 'weight',
          'anestart', 'opstart', 'opend', 'aneend']


@pytest.mark.parametrize('var', FITTED)
def test_fit_block_is_self_consistent(var):
    fit = json.loads((MODELS / f'{var}.model.json').read_text())['fit']
    assert fit['n'] > 0 and fit['loglik'] < 0
    assert fit['bic'] == pytest.approx(-2 * fit['loglik'] + fit['n_params'] * math.log(fit['n']), abs=1e-6)


DATA = os.environ.get('VITALDB_DATA')


@pytest.fixture(scope='module')
def clinical():
    if not DATA or not (Path(DATA) / 'clinical_data.csv').exists():
        pytest.skip('set VITALDB_DATA to the folder with clinical_data.csv')
    import pandas as pd
    return pd.read_csv(Path(DATA) / 'clinical_data.csv', encoding='utf-8-sig')


@pytest.mark.parametrize('var', FITTED)
def test_stored_loglik_equals_recomputed(var, clinical):
    from vitaldb_resolution import fitcheck
    r = fitcheck.recompute(var, clinical)
    assert r['n'] == r['n_stored']
    assert r['loglik'] == pytest.approx(r['loglik_stored'], abs=1e-3)


# Archived versions are immutable, so a v1 whose stored statistics were wrong stays wrong; the size of each known
# mismatch is pinned here (recomputed - stored loglik, and n), so any other change still fails.
KNOWN_V1 = {
    'intraop_ppf': (2.611, None, 'original fit used continuous CDFs over rounding intervals, not the library grid'),
    'weight': (0.043, None, 'original Nelder-Mead fit block'),
    'opend': (-0.138, None, 'original fitter discretisation'),
    'anestart': (0.108, None, 'original fitter discretisation'),
    'opstart': (-10.567, None, 'time-of-day prior integrated on the fitter\'s own 0.1-min grid'),
    'aneend': (None, 6254, '2 records outside the model range were clipped and counted (n 6256)'),
}
ARCHIVED = sorted(p.name.split('.v')[0] for p in (MODELS / 'archive').glob('*.v1.model.json'))


@pytest.mark.parametrize('var', ARCHIVED)
def test_archived_v1_recomputes_as_recorded(var, clinical):
    from vitaldb_resolution import fitcheck
    r = fitcheck.recompute(var, clinical, version=1)
    diff, n, _ = KNOWN_V1.get(var, (0.0, None, ''))
    assert r['n'] == (n if n is not None else r['n_stored'])
    if diff is not None:
        assert r['loglik'] - r['loglik_stored'] == pytest.approx(diff, abs=2e-3)
