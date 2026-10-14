"""Validation against the data: (1) each fitted model reproduces its stored log-likelihood through the package,
(2) every clinical and lab value returns a status without error, (3) summary of statuses and flags.
usage: python tools/validate.py <clinical_data.csv> <lab_data.csv> [--full]"""
import json, sys, time
import numpy as np, pandas as pd
import vitaldb_resolution as vr

if len(sys.argv) < 3: sys.exit(__doc__)
clin, labf = sys.argv[1], sys.argv[2]
FULL = '--full' in sys.argv
reg = vr.load()
d = pd.read_csv(clin, encoding='utf-8-sig')
report = {'loglik': {}, 'clinical': {}, 'lab': {}}

# ---------------- 1. log-likelihood reproduction ----------------
def ll_value(v):
    spec = reg.spec(f'{v}.model.json'); M = reg.model(f'{v}.model.json'); res = spec['resolution']['effective']
    x = d[v].dropna(); x = x[(x > 0) & (np.abs(x / res - np.round(x / res)) < 1e-9)]
    vals, cnt = np.unique(np.round(x / M.ru).astype(int), return_counts=True)
    return M.loglik(vals, cnt), spec['fit']['loglik'], len(x)

def ll_covariate(v, frame, cov_cols):
    spec = reg.spec(f'{v}.model.json'); M = reg.model(f'{v}.model.json')
    tot = 0.0
    for _, r in frame.iterrows():
        q = M.pmf(spec, {k: r[k] for k in cov_cols})
        tot += np.log(max(q[int(round(r[v] / M.ru))], 1e-300))
    return tot, spec['fit']['loglik'], len(frame)

def ll_timestamp(v):
    spec = reg.spec(f'{v}.model.json'); TM = reg.model(f'{v}.model.json')
    tot, n = 0.0, 0
    for _, r in d.iterrows():
        dist = reg.case_distribution(r, v)
        if not dist.ok: continue
        ctx = reg.timestamp_context(v, r[v], r)
        c = {'ref': ctx['ref']}
        if v == 'anestart':
            from vitaldb_resolution.registry import _ref_window
            c.update(ref_window=_ref_window(ctx['recording_start_clock']), emop=int(r['emop']))
        tot += np.log(TM.prob_recorded(ctx['M'], c)); n += 1
    return tot, spec['fit']['loglik'], n

for v in ['intraop_uo', 'intraop_ebl', 'intraop_crystalloid']:
    report['loglik'][v] = ll_value(v)
adults = d[pd.to_numeric(d.age, errors='coerce') >= 18].sort_values('caseid').drop_duplicates('subjectid')
adults = adults.assign(age=pd.to_numeric(adults.age))
if FULL:
    report['loglik']['height'] = ll_covariate('height', adults, ['sex', 'age', 'emop'])
    report['loglik']['weight'] = ll_covariate('weight', adults, ['sex', 'age', 'department'])
    for v in ['anestart', 'opstart', 'opend', 'aneend']:
        report['loglik'][v] = ll_timestamp(v)
for k, (got, stored, n) in report['loglik'].items():
    print(f'{k:22s} n={n:5d} loglik package {got:12.2f}  stored {stored:12.2f}  diff {got - stored:+.3f}')

# ---------------- 2. every clinical value ----------------
t0 = time.time()
rows = d if FULL else d.sample(500, random_state=0)
for v in list(reg.vars):
    e = reg.vars[v]
    if e['source'] == 'lab': continue
    if e['source'] == 'clinical' and v not in d.columns: continue
    st, fl = {}, {}
    for r in rows.to_dict('records'):
        x = reg.case_distribution(r, v, n_samples=2000)
        st[x.status] = st.get(x.status, 0) + 1
        for f in x.flags: fl[f] = fl.get(f, 0) + 1
        if x.ok:
            mu = x.mean()
            assert np.isfinite(mu), (v, r['caseid'])
            json.dumps(x.to_dict())
    report['clinical'][v] = {'status': st, 'flags': fl}
print(f'clinical: {len(report["clinical"])} variables x {len(rows)} cases in {time.time() - t0:.0f} s')

# ---------------- 3. lab values ----------------
lab = pd.read_csv(labf)
lab = lab if FULL else pd.concat([g.sample(min(len(g), 300), random_state=0) for _, g in lab.groupby('name')])
t0 = time.time()
ds = reg.lab_distributions(lab)
S = reg.summary_frame(ds)
for nm, g in S.groupby('variable'):
    report['lab'][nm] = {'status': g['status'].value_counts().to_dict(),
                         'flags': g['flags'][g['flags'] != ''].value_counts().to_dict()}
print(f'lab: {len(ds)} results in {time.time() - t0:.0f} s; statuses {S.status.value_counts().to_dict()}')

bad = {k: v for k, v in {**report['clinical'], **report['lab']}.items() if 'error' in v['status']}
print('variables with errors:', bad or 'none')
json.dump(report, open('validation_report.json', 'w'), indent=1, default=str)
