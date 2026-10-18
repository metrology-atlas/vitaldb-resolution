"""Build vitaldb_resolution/data/registry.json from the VitalDB parameter dictionaries, the data, and the
decisions recorded in the 'VitalDB Resolution and Heaping' doc. Re-run after any decision changes."""
import json, os, sys
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'vitaldb_resolution', 'data')
if len(sys.argv) < 3:
    sys.exit('usage: python tools/build_registry.py <folder with clinical_data.csv, clinical_parameters.csv, '
             'lab_parameters.csv> <lab_data.csv>')
CLIN, LAB = sys.argv[1], sys.argv[2]

cp = pd.read_csv(f'{CLIN}/clinical_parameters.csv', encoding='utf-8-sig').set_index('Parameter')
lp = pd.read_csv(f'{CLIN}/lab_parameters.csv', encoding='utf-8-sig').set_index('Parameter')
d = pd.read_csv(f'{CLIN}/clinical_data.csv', encoding='utf-8-sig')
lab = pd.read_csv(LAB)

def desc(name, src):
    t = cp if src == 'clinical' else lp
    if name in t.index:
        r = t.loc[name]
        u = r['Unit'] if isinstance(r['Unit'], str) else ''
        return r['Description'], u
    return '', ''

R = {}
def add(name, src, **kw):
    de, un = desc(name, src)
    e = {'variable': name, 'source': src, 'description': kw.pop('description', de), 'unit': kw.pop('unit', un)}
    e.update(kw)
    R[name] = e

# ---------------- clinical: identifiers, device times, dates ----------------
for v in ['caseid', 'subjectid']:
    add(v, 'clinical', type='identifier', method='not_applicable')
add('casestart', 'clinical', type='device_time', method='exact', resolution=1,
    evidence='Recording start; set to 0 for anonymisation.')
add('caseend', 'clinical', type='device_time', method='uniform', resolution=1, anchor='floor',
    evidence='Device clock, whole seconds.')
for v, ref in [('anestart', 'recording start'), ('opstart', 'recording start'),
               ('aneend', 'recording end'), ('opend', 'recording end')]:
    add(v, 'clinical', type='timestamp', method='timestamp_model', model=f'{v}.model.json',
        required_covariates=['adm', 'caseend'] + (['emop'] if v == 'anestart' else []),
        output_scale='seconds from casestart (same as the recorded value); clock minutes in summary',
        evidence=f'Charted clock minute modelled as an offset from the {ref} (device time). See the Timestamps section.',
        exclusion_rules=[r for r in ['date_1900', 'adm_not_whole_minute', 'opend_imputed' if v == 'opend' else None,
                                     'aneend_derived' if v == 'aneend' else None,
                                     'opend_after_aneend' if v in ('opend', 'aneend') else None,
                                     'opend_equals_aneend' if v in ('opend', 'aneend') else None] if r])
for v in ['adm', 'dis']:
    add(v, 'clinical', type='date', method='uniform', resolution=86400, anchor='floor',
        special_values=[{'test': 'value < -1e8', 'status': 'invalid', 'note': '1900-01-01 placeholder'}],
        evidence='Discharge minus admission is a whole number of days in every case; value marks midnight.')
add('icu_days', 'clinical', type='count', method='triangular_count', half_width=1, unit='days',
    counting_rule='nights or completed days (assumed)',
    evidence='Day count, read as nights or completed days: 815 one-day stays against 98 two-day stays argue against '
             'calendar days touched. The rule is an assumption, so ±1 day is kept.',
    optional_covariates=['adm', 'dis'],
    flag_rules=[{'test': 'icu_days > (dis - adm)/86400', 'flag': 'order_violation',
                 'note': 'ICU days exceed the whole hospital stay'},
                {'test': 'icu_days > 0 and icu_days >= days from the case-day midnight to the discharge midnight',
                 'flag': 'icu_exceeds_postop',
                 'note': 'probably includes preoperative ICU days (mostly emergencies with long preoperative stays); '
                         'do not use as postoperative ICU length'}])
add('death_inhosp', 'clinical', type='binary', method='not_applicable')

# ---------------- demographics and body size ----------------
add('age', 'clinical', type='count', method='age', resolution=1, anchor='floor',
    infant_resolution=0.1, top_code={'text': '>89', 'lower': 90},
    evidence='Completed years; 19.2% end in 0 or 5 (20% expected), so no heaping. 8 values top-coded ">89"; '
             '10 infants recorded in tenths of a year.')
add('sex', 'clinical', type='categorical', method='not_applicable')
add('height', 'clinical', type='continuous', method='model', model='height.model.json',
    required_covariates=['sex', 'age', 'emop'], record_unit=0.1)
add('weight', 'clinical', type='continuous', method='model', model='weight.model.json',
    required_covariates=['sex', 'age', 'department'], record_unit=0.05)
add('bmi', 'clinical', type='derived', method='derived', model='bmi.model.json', inputs=['height', 'weight'],
    required_covariates=['sex', 'age', 'emop', 'department'])
for v in ['asa', 'emop', 'department', 'optype', 'dx', 'opname', 'approach', 'position', 'ane_type', 'preop_htn',
          'preop_dm', 'preop_ecg', 'preop_pft', 'cormack', 'airway', 'tubesize', 'dltubesize', 'lmasize',
          'iv1', 'iv2', 'aline1', 'aline2', 'cline1', 'cline2']:
    if v in d.columns:
        add(v, 'clinical', type='categorical', method='not_applicable',
            evidence='Discrete category or device size; not a resolution question.')

# ---------------- fluids, blood products, drugs ----------------
DURATION_NOTE = {'duration': 'charted anaesthesia duration in minutes; taken from anestart and aneend in a case row, '
                 'or passed as duration=; without a usable one the latent is the marginal over fitted cases (flag duration_marginal)'}
for v in ['intraop_uo', 'intraop_ebl', 'intraop_crystalloid']:
    kw = dict(zero_status='zero_or_not_recorded') if v != 'intraop_uo' else dict(
        zero_rule='A recorded 0 is a point mass at 0 mixed with the latent below 2.5 mL (zero_model in the model file); '
                  'flag zero_anuric (preop_cr > 2 or asa == 6) or zero_unexplained. Missing means not measured, not 0.')
    notes = dict(DURATION_NOTE)
    if v == 'intraop_uo':
        notes['department'] = ('shifts the heaping propensities (100 / 50 / 10 mL grids); without it, or for another '
                               'department, propensities are averaged over the fitted department shares (flag department_marginal)')
    add(v, 'clinical', type='continuous', method='model', model=f'{v}.model.json', required_covariates=[],
        optional_covariates=['duration'] + (['department'] if v == 'intraop_uo' else []), covariate_notes=notes, **kw)
add('intraop_ppf', 'clinical', type='continuous', method='model', model='intraop_ppf.model.json',
    required_covariates=['weight', 'age', 'sex'], zero_status='zero_or_not_recorded',
    evidence='Single weight-based induction dose written as a preferred value. Zero in 3,719 of 6,043 GAs '
             '(TIVA is on the pump tracks, not here).')
add('intraop_colloid', 'clinical', type='continuous', method='colloid_rules', resolution=1, anchor='nearest',
    required_covariates=['opname'], zero_status='zero_or_not_recorded',
    rules=[
        {'when': 'kidney surgery (opname contains "kidney" or "nephrectomy") and value <= 50',
         'treat': 'native 1 mL', 'flag': 'unit_uncertain'},
        {'when': 'liver transplantation and value multiple of 100', 'treat': 'exact (100 mL vials)'},
        {'when': 'other surgery and value multiple of 500', 'treat': 'exact (500 mL bags)'},
        {'when': 'other surgery, value in 100..400 and multiple of 50', 'treat': 'nearest 100 mL (±50)',
         'flag': 'possibly_albumin'},
        {'when': 'otherwise', 'treat': 'native 1 mL'}],
    evidence='Container sizes are exact counts; partial amounts estimated. Kidney question pending.')
add('intraop_rbc', 'clinical', type='count', method='exact', zero_status='zero_or_not_recorded',
    flag_rules=[{'test': 'value >= 50 and intraop_ebl < 1000', 'flag': 'unit_uncertain', 'note': 'probably mL entered as units (5 cases)'}],
    evidence='Unit count; counts fall smoothly, no heaping.')
add('intraop_ffp', 'clinical', type='count', method='exact', zero_status='zero_or_not_recorded',
    evidence='Unit count; large values go with very large blood loss.')
for v, unit_dose in [('intraop_ftn', 50), ('intraop_eph', 5), ('intraop_phe', 10), ('intraop_ca', 150),
                     ('intraop_rocu', 10)]:
    add(v, 'clinical', type='count', method='exact', unit_dose=unit_dose, zero_status='zero_or_not_recorded',
        evidence='Recorded total is a count of standard doses or ampoules; treated as exact.')
for v, res in [('intraop_mdz', 0.5), ('intraop_vecu', 1), ('intraop_epi', 1)]:
    add(v, 'clinical', type='continuous', method='uniform', resolution=res, anchor='nearest',
        zero_status='zero_or_not_recorded', evidence='Sparse (80, 12 and 89 non-zero); native resolution only.')

# ---------------- lab analytes ----------------
def finest(v):
    v = v[np.isfinite(v)]
    for r in [100, 50, 10, 5, 1, 0.5, 0.1, 0.05, 0.01, 0.001]:
        if (np.abs(v / r - np.round(v / r)) > 1e-6).mean() < 0.002:
            return r
    return None

def reachable(v):
    return sorted(set(np.round(v[np.isfinite(v)], 3).tolist()))

def hct_prior():
    ph = set(map(tuple, lab.loc[lab.name == 'ph', ['caseid', 'dt']].values.tolist()))
    h = lab[lab.name == 'hct'].dropna(subset=['result'])
    bg = np.array([(c, t) in ph for c, t in h[['caseid', 'dt']].values.tolist()])
    whole = np.abs(h.result.values - np.round(h.result.values)) < 1e-9
    n_cbc, n_bg = int((whole & ~bg).sum()), int((whole & bg).sum())
    return {'whole_number': {'cbc': round(n_cbc / (n_cbc + n_bg), 4), 'blood_gas': round(n_bg / (n_cbc + n_bg), 4)},
            'note': f'P(source | whole-number value) from {n_cbc} CBC and {n_bg} blood-gas whole-number results; '
                    'a value with a non-zero decimal is always CBC'}

for a in sorted(lab.name.unique()):
    v = lab.result[lab.name == a].dropna().values
    if a == 'hct':
        add(a, 'lab', type='continuous', method='source_dependent', anchor='nearest', covariate='source',
            resolution_by_source={'cbc': 0.1, 'blood_gas': 1},
            infer_source='blood_gas if a pH result shares the timestamp, else cbc',
            unknown_source_prior=hct_prior(),
            flag_rules=[{'test': 'no other analyte at the same timestamp', 'flag': 'source_unknown',
                         'note': 'probably body fluid (9 results)'}],
            evidence='Two analysers distinguishable record by record; neither is heaped.')
    elif a in ('pt%', 'fib'):
        add(a, 'lab', type='continuous', method='neighbours', reachable_values=reachable(v),
            evidence='Computed from a clotting time, so only some values are reachable; the interval runs to the '
                     'midpoints of the neighbouring reachable values.')
    else:
        r = finest(v)
        add(a, 'lab', type='continuous', method='uniform', resolution=r, anchor='nearest',
            evidence='Single resolution; no heaping in the whole-number screen.' if r and r >= 1 else
                     'Single resolution; last digit flat.')
R['p']['description'] = ('Probably phosphorus (unconfirmed): not in lab_parameters.csv; median 3.2, IQR 2.7-3.8, '
                        '5th-95th percentile 1.9-5.0, consistent with inorganic phosphorus in mg/dL. Name kept as released.')
R['lab_dt'] = {'variable': 'lab_dt', 'source': 'lab', 'description': 'Lab timestamp (seconds from casestart)',
               'unit': 'sec', 'type': 'timestamp', 'method': 'uniform', 'resolution': 1, 'anchor': 'floor',
               'available_at': 'lab_time_plus_turnaround',
               'evidence': 'Machine-generated per-specimen time, not the result time: central-lab panels are logged about 10 min or more after the draw (specimen at the lab); blood gases 3-12 min after it (point-of-care entry or lab logging). The result is known only after a turnaround the release does not record, so a lab value must not be used at this time (the availability rule).'}

# ---------------- pre-operative copies in clinical_data ----------------
PREOP = {'preop_hb': 'hb', 'preop_plt': 'plt', 'preop_pt': 'pt%', 'preop_aptt': 'aptt', 'preop_na': 'na',
         'preop_k': 'k', 'preop_gluc': 'gluc', 'preop_alb': 'alb', 'preop_ast': 'ast', 'preop_alt': 'alt',
         'preop_bun': 'bun', 'preop_cr': 'cr', 'preop_ph': 'ph', 'preop_hco3': 'hco3', 'preop_be': 'be',
         'preop_pao2': 'po2', 'preop_paco2': 'pco2', 'preop_sao2': 'sao2'}
for v, a in PREOP.items():
    x = pd.to_numeric(d[v], errors='coerce').dropna().values
    if v == 'preop_pao2':
        add(v, 'clinical', type='continuous', method='decimal_dependent', anchor='nearest',
            resolution_if_decimal=0.1, resolution_if_whole=1, lab_analyte=a,
            evidence='41% end in .0 against 10% expected: two sources, decided by whether a decimal is present.')
    elif v == 'preop_pt':
        add(v, 'clinical', type='continuous', method='neighbours', lab_analyte=a,
            reachable_values=R['pt%']['reachable_values'], evidence='PT %, computed; uses the lab reachable values.')
    else:
        add(v, 'clinical', type='continuous', method='uniform', resolution=finest(x), anchor='nearest',
            lab_analyte=a, evidence='Pre-operative copy of a lab result; behaves like the lab data.')

# ---------------- reporting limits (open intervals, like age > 89) ----------------
LIMITS = {'lac': {'upper': 15.0, 'note': 'values above 15 mmol/L are reported as 15.0: 317 results at 15.0 (28 cases) '
                                          'against at most 16 at any nearby value'},
          'aptt': {'upper': 400.0, 'note': 'capped at 400.0 s: 226 results (87 cases)'},
          'esr': {'lower': 2.0, 'note': 'floor at 2 mm/h: no result below it; 89 results (72 cases) at 2'}}
for a, lim in LIMITS.items():
    R[a]['reporting_limits'] = lim
R['preop_aptt']['reporting_limits'] = dict(LIMITS['aptt'], note='pre-operative copy of aptt (2 values at 400.0); ' + LIMITS['aptt']['note'])

# ---------------- anchor basis ----------------
VOLUME_ANCHOR = json.load(open(os.path.join(ROOT, 'results', 'anchor_basis_volume.json')))
for k, e in R.items():
    if k in ('adm', 'dis'):
        e.update(anchor_basis='tested', anchor_note='Discharge minus admission is a whole number of days in every case, '
                                                    'so each value marks a midnight.')
    elif k in ('caseend', 'lab_dt'):
        e.update(anchor_basis='convention', anchor_note='Clock truncation to the whole second; at most 1 s, negligible.')
    elif k == 'age':
        e.update(anchor_basis='assumption', anchor_note='The release says only "years"; infants in tenths and the >89 cap '
                                                        'point to an exact age cut down to completed years.')
    elif k == 'icu_days':
        e.update(anchor_basis='assumption', anchor_note='815 one-day stays against 98 two-day stays argue against calendar '
                                                        'days touched; read as nights or completed days.')
    elif k in VOLUME_ANCHOR:
        e.update(anchor='nearest', **VOLUME_ANCHOR[k])
    elif e.get('anchor') and (e['source'] == 'lab' or e.get('lab_analyte')):
        e.update(anchor_basis='convention', anchor_note='Laboratory results rounded to the last reported digit; at most '
                                                        'half the last digit.')
    elif e.get('anchor'):
        e.update(anchor_basis='convention', anchor_note='Native recording unit, rounding to the nearest assumed; not '
                                                        'tested (sparse or exact-count data).')

# ---------------- derived ----------------
R['los'] = {'variable': 'los', 'source': 'derived', 'description': 'Length of hospital stay (dis - adm)',
            'unit': 'days', 'type': 'derived', 'method': 'derived', 'formula': '(dis - adm) / 86400',
            'inputs': ['adm', 'dis'], 'distribution': 'triangular over n ± 1 days',
            'evidence': 'Both ends are dates covering a whole day.'}
for k, a, b, desc_ in [('ind_to_incision', 'anestart', 'opstart', 'Anaesthesia start to incision'),
                       ('op_duration', 'opstart', 'opend', 'Operation duration'),
                       ('ane_duration', 'anestart', 'aneend', 'Anaesthesia duration'),
                       ('emergence', 'opend', 'aneend', 'Operation end to anaesthesia end')]:
    R[k] = {'variable': k, 'source': 'derived', 'description': desc_, 'unit': 'min', 'type': 'derived',
            'method': 'derived', 'formula': f'({b} - {a}) / 60', 'inputs': [a, b],
            'required_covariates': ['adm', 'caseend', 'emop'],
            'distribution': 'difference of the two event posteriors, sampled independently given the device references'}

reg = {'schema_version': '0.5', 'dataset': 'VitalDB (clinical_data.csv, lab_data.csv)',
       'built': pd.Timestamp.today().strftime('%Y-%m-%d'),
       'statuses': ['ok', 'missing', 'invalid', 'imputed', 'not_applicable', 'zero_or_not_recorded', 'special_value',
                    'top_coded', 'bottom_coded', 'error'],
       'anchor_bases': {'tested': 'settled by the data', 'convention': 'standard practice, negligible if wrong',
                        'assumption': 'a reasoned reading the release does not confirm', 'open': 'not yet settled'},
       'methods': {
           'model': 'fitted rounding model (value kind); posterior over the true value',
           'timestamp_model': 'fitted reference-offset model; posterior over the true event time',
           'uniform': 'single resolution; uniform over the rounding interval',
           'source_dependent': 'resolution chosen by a covariate (analyser)',
           'decimal_dependent': 'resolution chosen by whether the record has a decimal',
           'neighbours': 'computed value; interval to midpoints of neighbouring reachable values',
           'exact': 'exact count; point mass', 'triangular_count': 'day count with ±1 uncertainty',
           'age': 'completed years, top-coded and infant rules', 'colloid_rules': 'rule set by operation',
           'derived': 'computed from other variables\' posteriors', 'not_applicable': 'categorical or identifier'},
       'variables': R}
with open(os.path.join(DATA, 'registry.json'), 'w', encoding='utf-8') as fh:
    json.dump(reg, fh, indent=1, ensure_ascii=False); fh.write('\n')
print(len(R), 'variables;', pd.Series([e['method'] for e in R.values()]).value_counts().to_dict())
