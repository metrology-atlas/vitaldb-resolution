import json

import numpy as np
import pytest

import vitaldb_resolution as vr
from vitaldb_resolution.distribution import Component, Distribution

reg = vr.load()
RNG = np.random.default_rng(1)

# A synthetic case: recording starts 08:00:20 (adm = midnight - 08:00 = -28800 s, sub-minute offset 40 s).
CASE = dict(adm=-28800, caseend=3 * 3600 + 40, emop=0, sex='F', age=60, department='General surgery',
            anestart=-1220, opstart=880, opend=7180, aneend=8380,     # clocks 07:40, 08:15, 10:00, 10:20
            height=160.0, weight=60.0, opname='Distal gastrectomy')


def mass(d):
    lo = min(c.lower for c in d.components); hi = max(c.upper for c in d.components)
    return float(d.cdf(hi + 1e-9) - d.cdf(lo - 1e-9))


# ---------------- Distribution object ----------------
@pytest.mark.parametrize('comp', [
    Component('uniform', lower=1, upper=2), Component('triangular', lower=0, mode=1, upper=2),
    Component('tabulated', x=np.arange(0.5, 10), p=np.ones(10), dx=1.0)])
def test_component_moments_match_samples(comp):
    d = Distribution('x', 0, components=[comp])
    s = d.sample(200000, RNG)
    assert abs(s.mean() - d.mean()) < 0.01 * max(1, abs(d.mean()))
    assert abs(s.std() - d.std()) < 0.02 * max(1, d.std())
    assert mass(d) == pytest.approx(1)


def test_interval_edges_nearest_floor_ceiling():
    from vitaldb_resolution.distribution import uniform_interval
    assert uniform_interval(10, 1, 'nearest') == (9.5, 10.5, 'lower')
    assert uniform_interval(10, 1, 'floor') == (10, 11, 'lower')
    assert uniform_interval(10, 1, 'ceiling') == (9, 10, 'upper')


# ---------------- statuses ----------------
def test_statuses():
    assert reg.distribution('intraop_uo', None).status == 'missing'
    z = reg.distribution('intraop_uo', 0)                                  # zero rule: point mass at 0, flagged
    assert z.status == 'ok' and z.kind == 'zero_point_mass' and 'zero_unexplained' in z.flags
    assert reg.distribution('intraop_ebl', 0).status == 'zero_or_not_recorded'
    assert reg.distribution('lac', 15.0).status == 'top_coded'
    assert reg.distribution('esr', 2).status == 'bottom_coded'
    assert reg.distribution('aptt', 400.0).status == 'top_coded' and reg.distribution('aptt', 399.9).ok
    assert reg.distribution('sex', 'F').status == 'not_applicable'
    assert reg.distribution('age', '>89').status == 'top_coded'
    assert reg.distribution('adm', -3685366320).status == 'invalid'
    assert reg.distribution('height', 160.0).status == 'error'          # covariates missing


# ---------------- single resolution and rule-based ----------------
def test_lab_uniform():
    d = reg.distribution('k', 4.1)
    assert d.components[0].lower == pytest.approx(4.05) and d.components[0].upper == pytest.approx(4.15)
    assert d.mean() == pytest.approx(4.1)


def test_hct_sources():
    assert reg.distribution('hct', 35.4).components[0].p['resolution'] == 0.1
    assert reg.distribution('hct', 35, source='blood_gas').components[0].p['resolution'] == 1
    d = reg.distribution('hct', 35)
    assert d.kind == 'mixed_source' and len(d.components) == 2


def test_neighbours_ptpct():
    d = reg.distribution('pt%', 100)
    assert d.kind == 'computed_value' and d.components[0].lower < 100 < d.components[0].upper


def test_age_rules():
    assert reg.distribution('age', 45).components[0].lower == 45
    assert reg.distribution('age', 0.5).components[0].upper == pytest.approx(0.6)


def test_colloid_rules():
    assert reg.distribution('intraop_colloid', 500, opname='x').kind == 'exact'
    assert 'possibly_albumin' in reg.distribution('intraop_colloid', 200, opname='x').flags
    assert 'unit_uncertain' in reg.distribution('intraop_colloid', 35, opname='Kidney transplantation').flags
    assert reg.distribution('intraop_colloid', 300, opname='Liver transplantation').kind == 'exact'


def test_exact_counts_and_flags():
    assert reg.distribution('intraop_eph', 15).std() == 0
    assert 'unit_uncertain' in reg.distribution('intraop_rbc', 400).flags


# ---------------- fitted models ----------------
def test_value_model_posterior_contains_recorded():
    d = reg.distribution('intraop_uo', 100)
    assert d.kind == 'heaped' and mass(d) == pytest.approx(1, abs=1e-6)
    lo, hi = d.interval(0.95)
    assert lo < 100 < hi and hi - lo > 5           # wider than the 5 mL base interval: heaping
    assert {c.label for c in d.components} >= {'100 mL', '50 mL', '10 mL', '5 mL (base)'}


def test_off_grid_snaps_and_flags():
    d = reg.distribution('intraop_uo', 123)
    assert 'off_grid' in d.flags and d.ok


def test_height_written_in_and_children():
    d = reg.distribution('height', 160.0, sex='F', age=60, emop=0)
    assert 0 < d.extra['possibly_written_in'] <= 1
    assert reg.distribution('height', 160.3, sex='F', age=60, emop=0).extra['possibly_written_in'] == 0
    c = reg.distribution('height', 95.0, sex='F', age=3, emop=0)
    assert 'child_native_resolution' in c.flags


def test_measurement_variability_widens():
    a = reg.distribution('weight', 60.3, sex='F', age=60, department='General surgery')
    b = reg.distribution('weight', 60.3, sex='F', age=60, department='General surgery', variability=True, rng=RNG)
    assert b.std() > a.std()


# ---------------- timestamps ----------------
@pytest.mark.parametrize('v', ['anestart', 'opstart', 'opend', 'aneend'])
def test_timestamp_scale_and_anchor(v):
    d = reg.case_distribution(CASE, v)
    assert d.ok and mass(d) == pytest.approx(1, abs=1e-6)
    t = CASE[v]
    if v == 'aneend': assert d.mean() <= t + 1e-6        # ceiling anchor: true end before the record
    if v == 'opend': assert d.mean() >= t - 1e-6         # floor anchor: true end after the record
    assert abs(d.mean() - t) < 15 * 60
    assert d.extra['recorded_clock'].count(':') == 2


def test_timestamp_exclusions():
    c = dict(CASE, opend=CASE['caseend'] - 601)
    assert reg.case_distribution(c, 'opend').status == 'imputed'
    c = dict(CASE, aneend=CASE['aneend'] + 0.25)
    assert reg.case_distribution(c, 'aneend').status == 'imputed'
    c = dict(CASE, opend=CASE['aneend'] + 60)
    assert 'order_violation' in reg.case_distribution(c, 'opend').flags


# ---------------- derived ----------------
def test_bmi_and_intervals():
    b = reg.case_distribution(CASE, 'bmi')
    assert b.mean() == pytest.approx(60 / 1.6 ** 2, rel=0.01)
    op = reg.case_distribution(CASE, 'op_duration')
    assert op.extra['charted'] == pytest.approx(105)
    assert op.mean() > 105                                # floor end: charted operation is short
    los = reg.distribution('los', adm=-86400 - 28800, dis=5 * 86400 - 28800)
    assert los.mean() == pytest.approx(6)


def test_json_round_trip():
    for v in ['intraop_uo', 'k', 'hct']:
        json.dumps(reg.distribution(v, 100 if v == 'intraop_uo' else 4.1).to_dict())
    json.dumps(reg.case_distribution(CASE, 'opstart').to_dict(include_grid=True))


def test_every_registry_entry_has_a_handler():
    for k, e in reg.vars.items():
        assert hasattr(reg, '_h_' + e['method']), k


# ---------------- Monte Carlo cache ----------------
def test_cache_matches_api(tmp_path):
    import pandas as pd

    from vitaldb_resolution import cache as C
    rows = pd.DataFrame([dict(CASE, caseid=1, subjectid=1, casestart=0, intraop_uo=100, intraop_ebl=300,
                              icu_days=1, dis=5 * 86400 - 28800, death_inhosp=0, preop_hb=12.3)])
    lab = pd.DataFrame({'caseid': [1, 1, 1], 'dt': [100, 100, 200], 'name': ['k', 'ph', 'hct'],
                        'result': [4.1, 7.40, 35.0]})
    cols, tables = C.collect(reg, rows, lab, log=lambda s: None)
    for dt in ('int16', 'float32', 'float64'):
        C.write(cols, tables, rows, lab, reg, str(tmp_path / dt), dt, 1024)
        s = C.Cache.load(str(tmp_path / dt)).sampler(seed=2)
        D = [s.draw() for _ in range(4000)]
        for v in ('intraop_uo', 'opstart', 'aneend', 'height', 'op_duration'):
            a = np.array([d[v][0] for d in D]); x = reg.case_distribution(rows.iloc[0].to_dict(), v)
            assert abs(a.mean() - x.mean()) < 4 * x.std() / np.sqrt(4000) + 1e-9, (dt, v)
            assert abs(a.std() / x.std() - 1) < 0.08, (dt, v)
        L = np.array([d['lab'] for d in D])
        assert abs(L[:, 0].mean() - 4.1) < 0.005 and L[:, 2].std() > 0.25    # stand-alone hct: mixture


def test_cache_table_precision():
    from vitaldb_resolution import cache as C
    x = reg.distribution('intraop_uo', 100)
    X, P = C.breakpoints(x.components)
    q = C.table_from(X, P, 1024)
    u = (np.arange(20000) + 0.5) / 20000
    pk = C.knot_levels(1024); i = np.clip(np.searchsorted(pk, u, side='right') - 1, 0, 1022)
    ap = q[i] + (u - pk[i]) / (pk[i + 1] - pk[i]) * (q[i + 1] - q[i])
    assert abs(ap.mean() - x.mean()) < 1e-3 * x.std() and abs(ap.std() / x.std() - 1) < 5e-3
