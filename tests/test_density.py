# SPDX-License-Identifier: MIT
"""Within-interval density: a fitted latent truncated to each component's interval, computed at call time;
uniform (GUM rectangular) exactly where no latent is fitted."""
import pytest

from vitaldb_resolution import load

reg = load()


def _uo(v):
    return reg.distribution('intraop_uo', v, duration=180)


def test_latent_and_uniform_agree_in_the_middle():
    d = _uo(150)                       # near the latent median
    assert d.density == 'latent'
    assert d.latent_uniform_gap() < 0.05
    for c in d.to_dict()['components']:
        assert abs(c['mean'] - c['uniform_mean']) < 0.05 * (c['interval'][1] - c['interval'][0])


@pytest.mark.parametrize('v', [1000, 2000])
def test_latent_mean_moves_towards_the_mode_in_the_tail(v):
    d = _uo(v)
    comps = d.to_dict()['components']
    assert all(c['mean'] < c['uniform_mean'] for c in comps)
    coarse = max(comps, key=lambda c: c['interval'][1] - c['interval'][0])
    assert coarse['uniform_mean'] - coarse['mean'] > 0.5


@pytest.mark.parametrize('var,v,cov', [('intraop_uo', 300, {'duration': 180}), ('intraop_uo', 300, {}),
                                       ('intraop_ebl', 500, {'duration': 120}), ('intraop_crystalloid', 1000, {}),
                                       ('height', 170, dict(sex='M', age=50, emop=0)),
                                       ('intraop_uo', 0, {'preop_cr': 3.1})])
def test_component_weights_sum_to_one(var, v, cov):
    d = reg.distribution(var, v, **cov)
    assert d.ok and sum(c.weight for c in d.components) == pytest.approx(1.0, abs=1e-12)


def test_marginal_is_flagged_without_duration():
    assert 'duration_marginal' in reg.distribution('intraop_uo', 300).flags
    assert 'duration_marginal' not in _uo(300).flags


def test_zero_returns_point_mass_and_flag():
    d = reg.distribution('intraop_uo', 0, preop_cr=3.1, duration=180)
    assert d.kind == 'zero_point_mass' and 'zero_anuric' in d.flags
    assert d.components[0].kind == 'point' and d.components[0].p['value'] == 0
    assert 'zero_unexplained' in reg.distribution('intraop_uo', 0, preop_cr=0.8, asa=2, duration=180).flags


CASES = {'na': 140, 'hct': 35.2, 'preop_hb': 12.3, 'intraop_mdz': 2.5, 'age': 54, 'caseend': 12000, 'adm': -86400}


@pytest.mark.parametrize('var', sorted(CASES))
def test_uniform_fallback_where_no_latent(var):
    d = reg.distribution(var, CASES[var])
    assert d.ok and d.density == 'uniform' and d.to_dict()['density'] == 'uniform'


def test_latent_exactly_where_a_model_is_fitted():
    for var, v, cov in [('intraop_uo', 300, {}), ('weight', 70, dict(sex='F', age=40, department='General surgery')),
                        ('intraop_ppf', 120, dict(weight=70, age=50, sex='M'))]:
        d = reg.distribution(var, v, **cov)
        assert reg.entry(var)['method'] == 'model' and d.density == 'latent' and d.model['density'] == 'latent'
        assert d.model['id'] == var and d.model['fit_hash']
    for var in reg.variables('uniform')[:40]:
        assert 'model' not in reg.entry(var)
