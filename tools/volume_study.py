"""Duration covariate, department-on-propensity and rounding-anchor comparisons for the volume models (results/volume_study.json)."""
import json, math, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fit_volume as F

d = pd.read_csv(sys.argv[1], encoding='utf-8-sig')
out = sys.argv[2] if len(sys.argv) > 2 else 'results/volume_study.json'
res = json.load(open(out)) if os.path.exists(out) else {}
def run(var, tag, duration, anchor=None, with_derived=False, department=False):
    key = f'{var}:{tag}'
    if key in res: print('skip', key); return
    spec = json.load(open(os.path.join(F.MODELS, f'{var}.model.json')))
    if anchor: spec = F.set_anchor(spec, anchor)
    vals, durs, depts = F.records(d, var, spec, with_derived=with_derived)
    new, ll, k, p = F.fit(spec, vals, durs, duration, tag=key, depts=depts if department else None)
    res[key] = {'n': len(vals), 'k': k, 'loglik': ll, 'bic': -2 * ll + k * math.log(len(vals)), 'params': p}
    json.dump(new, open(f'/tmp/claude-0/vdb/volfits/{var}__{tag}.model.json', 'w'), indent=1)
    os.makedirs(os.path.dirname(out), exist_ok=True); json.dump(res, open(out, 'w'), indent=1)
os.makedirs('/tmp/claude-0/vdb/volfits', exist_ok=True)
for var in sys.argv[3:] or ['intraop_uo', 'intraop_crystalloid', 'intraop_ebl']:
    run(var, 'marginal', False)
    run(var, 'duration', True)
    run(var, 'duration_with_derived', True, with_derived=True)
    run(var, 'duration_floor', True, anchor='floor')
    run(var, 'marginal_floor', False, anchor='floor')
    run(var, 'duration_department', True, department=True)
print('done')
