"""Validation against the data: (1) each fitted model reproduces its stored log-likelihood through the package,
(2) every clinical and lab value returns a status without error, (3) summary of statuses, flags and densities,
(4) for each modelled variable, the largest gap between the latent and uniform means within an interval, as a fraction
of the interval width.
usage: python tools/validate.py <clinical_data.csv> <lab_data.csv> [--full]"""
import json, sys, time
import numpy as np, pandas as pd
import vitaldb_resolution as vr

if len(sys.argv) < 3: sys.exit(__doc__)
clin, labf = sys.argv[1], sys.argv[2]
FULL = '--full' in sys.argv
reg = vr.load()
d = pd.read_csv(clin, encoding='utf-8-sig')
report = {'models': {}, 'loglik': {}, 'clinical': {}, 'lab': {}}
for mid, m in vr.versions.load_lock()['models'].items():
    report['models'][mid] = m

# ---------------- 1. log-likelihood reproduction (stored fit block vs the library) ----------------
from vitaldb_resolution import fitcheck
quick = ['intraop_uo', 'intraop_ebl', 'intraop_crystalloid', 'intraop_ppf']
for v in (quick + ['height', 'weight', 'anestart', 'opstart', 'opend', 'aneend']) if FULL else quick:
    r = fitcheck.recompute(v, d, reg)
    report['loglik'][v] = {'n': r['n'], 'n_stored': r['n_stored'], 'loglik': r['loglik'], 'loglik_stored': r['loglik_stored'],
                           'diff': r['loglik'] - r['loglik_stored']}
    print(f"{v:22s} n={r['n']:5d} (stored {r['n_stored']:5d}) loglik package {r['loglik']:12.3f}  stored "
          f"{r['loglik_stored']:12.3f}  diff {r['loglik'] - r['loglik_stored']:+.4f}")

# ---------------- 1b. zeros and missing values of the volume variables ----------------
zeros = {}
for v in ['intraop_uo', 'intraop_ebl', 'intraop_crystalloid', 'intraop_colloid']:
    x = d[v]; zeros[v] = {'recorded': int(x.notna().sum()), 'missing': int(x.isna().sum()), 'zero': int((x == 0).sum())}
uo0 = d[d.intraop_uo == 0]
an = (uo0.preop_cr > 2) | (uo0.asa == 6)
zeros['intraop_uo'].update(zero_anuric=int(an.sum()), zero_unexplained=int((~an).sum()),
                           p0=float((d.intraop_uo == 0).sum() / d.intraop_uo.notna().sum()))
report['zeros'] = zeros
print('zeros:', json.dumps(zeros))

# ---------------- 2. every clinical value ----------------
t0 = time.time()
rows = d if FULL else d.sample(500, random_state=0)
for v in list(reg.vars):
    e = reg.vars[v]
    if e['source'] == 'lab': continue
    if e['source'] == 'clinical' and v not in d.columns: continue
    st, fl, dn, gap = {}, {}, {}, None
    for r in rows.to_dict('records'):
        x = reg.case_distribution(r, v, n_samples=2000)
        st[x.status] = st.get(x.status, 0) + 1
        for f in x.flags: fl[f] = fl.get(f, 0) + 1
        dn[x.density] = dn.get(x.density, 0) + 1
        g = x.latent_uniform_gap() if x.ok else None
        if g is not None and (gap is None or g > gap[0]): gap = (g, r[v])
        if x.ok:
            mu = x.mean()
            assert np.isfinite(mu), (v, r['caseid'])
            json.dumps(x.to_dict())
    report['clinical'][v] = {'status': st, 'flags': fl, 'density': dn}
    if gap: report['clinical'][v]['latent_uniform_gap_max'] = {'fraction_of_width': round(gap[0], 4), 'recorded_value': gap[1]}
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
gaps = {k: v['latent_uniform_gap_max'] for k, v in report['clinical'].items() if 'latent_uniform_gap_max' in v}
report['latent_uniform_gap_max'] = gaps
print('largest latent-uniform mean gap (fraction of interval width):', json.dumps(gaps))
json.dump(report, open('validation_report.json', 'w'), indent=1, default=str)
