"""Precision and memory of the cache storage options (dtype x knots) against the exact posteriors.
usage: python tools/precision_study.py clinical_data.csv lab_data.csv outdir"""
import json
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

import vitaldb_resolution as vr
from vitaldb_resolution import cache as C

clin, labf, out = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(out, exist_ok=True)
reg = vr.load()
clinical = pd.read_csv(clin, encoding='utf-8-sig'); lab = pd.read_csv(labf)
pk = os.path.join(out, 'exact.pkl')
if os.path.exists(pk):
    cols, tables = pickle.load(open(pk, 'rb'))
else:
    t0 = time.time(); cols, tables = C.collect(reg, clinical, lab); print('collect', round(time.time() - t0), 's', flush=True)
    pickle.dump((cols, tables), open(pk, 'wb'))

owner = np.empty(len(tables), object)
for v, c in cols.items():
    for t in (c.t1, c.t2):
        owner[t[t >= 0]] = v
unit_res = {'intraop_uo': 5, 'intraop_ebl': 10, 'intraop_crystalloid': 10, 'intraop_ppf': 10, 'height': 0.1,
            'weight': 0.05, 'anestart': 60, 'opstart': 60, 'opend': 60, 'aneend': 60}

P_EVAL = (np.arange(8192) + 0.5) / 8192
OPTS = [(k, dt) for k in (256, 1024, 4096) for dt in ('int16', 'float32', 'float64')]
metrics = {o: {'w1_rel': [], 'max_rel': [], 'mean_rel': [], 'sd_rel': [], 'w1_abs': []} for o in OPTS}
n = len(tables); CH = 1000
tables_mb = {}
t0 = time.time()
done_f = os.path.join(out, 'metrics_partial.pkl')
if os.path.exists(done_f):
    metrics, tables_mb = pickle.load(open(done_f, 'rb'))
for k in (256, 1024, 4096):
    if all(len(metrics[(k, d)]['w1_rel']) for d in ('int16', 'float32', 'float64')): continue
    Qk = C.tables_matrix(tables, k)
    for dt in ('int16', 'float32', 'float64'):
        q, lo, sc = C.quantize(Qk, dt) if dt != 'float64' else (Qk, None, None)
        tables_mb[(k, dt)] = (q.nbytes + (0 if lo is None else lo.nbytes + sc.nbytes)) / 1e6
        m = metrics[(k, dt)]
        pk = C.knot_levels(k); i = np.clip(np.searchsorted(pk, P_EVAL, side='right') - 1, 0, k - 2)
        f = (P_EVAL - pk[i]) / (pk[i + 1] - pk[i])
        for s0 in range(0, n, CH):
            rows = np.arange(s0, min(n, s0 + CH))
            ex = np.vstack([np.interp(P_EVAL, tables[j][1], tables[j][0]) for j in rows])
            sd = ex.std(1); sd = np.where(sd > 0, sd, np.nan)
            Q = C.dequantize(q, lo, sc, rows).astype(np.float64)
            ap = Q[:, i] + f * (Q[:, i + 1] - Q[:, i])
            dd = ap - ex
            m['w1_abs'].append(np.abs(dd).mean(1)); m['w1_rel'].append(np.abs(dd).mean(1) / sd)
            m['max_rel'].append(np.abs(dd).max(1) / sd); m['mean_rel'].append(dd.mean(1) / sd)
            m['sd_rel'].append(ap.std(1) / sd - 1)
        del q
        pickle.dump((metrics, tables_mb), open(done_f, 'wb'))
        print('evaluated', k, dt, round(time.time() - t0), 's', flush=True)
    del Qk
for o in OPTS:
    for kk in metrics[o]: metrics[o][kk] = np.concatenate(metrics[o][kk])

groups = {'value models (uo, ebl, crystalloid, ppf)': ['intraop_uo', 'intraop_ebl', 'intraop_crystalloid', 'intraop_ppf'],
          'height and weight': ['height', 'weight'], 'timestamps': ['anestart', 'opstart', 'opend', 'aneend'],
          'other (PT %, fibrinogen, haematocrit mixtures)': None}
known = set(sum([g for g in groups.values() if g], []))
res = {'n_tables': int(n), 'tables_by_variable': pd.Series(owner).value_counts().to_dict(), 'options': []}
exact_bytes = int(sum(X.nbytes + P.nbytes for X, P in tables))
res['exact_ragged_MB'] = exact_bytes / 1e6
for (k, dt) in OPTS:
    mb = tables_mb[(k, dt)]
    m = metrics[(k, dt)]; row = {'knots': k, 'dtype': dt, 'tables_MB': round(mb, 1), 'groups': {}}
    for gname, vs in groups.items():
        sel = np.isin(owner, vs) if vs else ~np.isin(owner, list(known))
        if not sel.any(): continue
        w = m['w1_rel'][sel]; mx = m['max_rel'][sel]; mr = np.abs(m['mean_rel'][sel]); sr = np.abs(m['sd_rel'][sel])
        row['groups'][gname] = {'n': int(sel.sum()), 'w1_rel_median': float(np.nanmedian(w)),
                                'w1_rel_p99': float(np.nanpercentile(w, 99)), 'w1_rel_max': float(np.nanmax(w)),
                                'max_rel_p99': float(np.nanpercentile(mx, 99)),
                                'mean_rel_max': float(np.nanmax(mr)), 'sd_rel_max': float(np.nanmax(sr))}
    per = {}
    for v in sorted(set(owner)):
        sel = owner == v
        per[v] = {'w1_abs_median': float(np.nanmedian(m['w1_abs'][sel])), 'w1_abs_max': float(np.nanmax(m['w1_abs'][sel])),
                  'unit_resolution': unit_res.get(v)}
    row['per_variable_abs'] = per
    res['options'].append(row)

# memory of the full cache and draw speed for each option
for row in res['options']:
    d = os.path.join(out, f"cache_{row['knots']}_{row['dtype']}")
    C.write(cols, tables, clinical, lab, reg, d, row['dtype'], row['knots'])
    import gc; gc.collect()
    c = C.Cache.load(d); row['cache_total_MB'] = round(c.nbytes() / 1e6, 1)
    s = c.sampler(0, variability=True); s.draw()
    t0 = time.time(); [s.draw() for _ in range(20)]; row['draw_ms_full_dataset'] = round((time.time() - t0) / 20 * 1000, 1)
    t0 = time.time(); [s.draw(include_lab=False) for _ in range(50)]; row['draw_ms_clinical_only'] = round((time.time() - t0) / 50 * 1000, 2)
    print(row['knots'], row['dtype'], row['tables_MB'], row['cache_total_MB'], row['draw_ms_full_dataset'], flush=True)
    del c, s
json.dump(res, open(os.path.join(out, 'precision.json'), 'w'), indent=1)
print('done')
