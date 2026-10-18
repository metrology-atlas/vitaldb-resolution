"""Refit a log-Student-t regression value model (propofol bolus) with the library's own likelihood.

The original propofol fit used continuous normal/t CDFs over rounding intervals; the library evaluates the same
model on its grid, which gave a slightly different log-likelihood (2.6 on 2,328 records). This refits on the
grid, vectorised over records, so the stored and recomputed values agree.

usage: python tools/fit_logt.py <clinical_data.csv> intraop_ppf [--write]
"""
import copy, json, math, os, sys, time
import numpy as np, pandas as pd
from scipy import optimize, stats
from vitaldb_resolution import engine, fitcheck

MODELS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'vitaldb_resolution', 'data', 'models')
sig = lambda t: 1 / (1 + np.exp(-t))


def setup(spec, vals, covs):
    M = engine.Model(spec); L = spec['latent']; alloc = spec['allocation']
    Z = np.array([[engine.transform(c['transform'], cv[c['name']]) for c in L['covariates']] for cv in covs])
    recs = [M._round(c['mode']) for c in alloc]
    keyv = np.round(vals / M.ru).astype(int)
    hit = [np.round(r / M.ru).astype(int)[None, :] == keyv[:, None] for r in recs]       # n x G per mode
    hit0 = [np.round(r / M.ru).astype(int) == 0 for r in recs]
    stat = []
    for c, r in zip(alloc, recs):
        m = np.ones(len(M.X), bool)
        if c.get('prob'): m &= r > 0
        if c.get('applies_below') is not None: m &= M.X <= c['applies_below']
        stat.append(m)
    return M, Z, hit, hit0, stat


def pack(spec):
    L = spec['latent']; x = [L['intercept']] + [c['coef'] for c in L['covariates']] + [math.log(L['scale']), math.log(L['df'])]
    x += [c['prob']['logit'] for c in spec['allocation'] if c.get('prob')]
    return np.array(x, float)


def unpack(spec, x):
    s = copy.deepcopy(spec); L = s['latent']; nc = len(L['covariates'])
    L['intercept'] = float(x[0])
    for j, c in enumerate(L['covariates']): c['coef'] = float(x[1 + j])
    L['scale'] = float(math.exp(x[1 + nc])); L['df'] = float(math.exp(x[2 + nc])); i = 3 + nc
    for c in s['allocation']:
        if c.get('prob'): c['prob']['logit'] = float(x[i]); i += 1
    return s


def nll(x, spec, M, Z, hit, hit0, stat):
    L = spec['latent']; nc = len(L['covariates'])
    mu = x[0] + Z @ x[1:1 + nc]; sc = math.exp(x[1 + nc]); df = math.exp(x[2 + nc]); i = 3 + nc
    W = stats.t.pdf((M.LX[None, :] - mu[:, None]) / sc, df) / M.X[None, :]; W /= W.sum(1, keepdims=True)
    rem = np.ones(len(M.X)); q = np.zeros(len(mu)); q0 = np.zeros(len(mu))
    for k, c in enumerate(spec['allocation']):
        p = sig(x[i]) if c.get('prob') else 1.0
        if c.get('prob'): i += 1
        take = rem * p * stat[k]
        q += (W * (take[None, :] * hit[k])).sum(1); q0 += W @ (take * hit0[k])
        rem = rem - take
    ll = np.log(np.clip(q, 1e-300, None))
    if spec.get('truncation', {}).get('exclude_recorded_zero'): ll -= np.log1p(-q0)
    return -ll.sum()


if __name__ == '__main__':
    d = pd.read_csv(sys.argv[1], encoding='utf-8-sig'); var = sys.argv[2]
    p = os.path.join(MODELS, f'{var}.model.json'); spec = json.load(open(p))
    vals, covs = fitcheck.value_records(spec, d); t0 = time.time()
    M, Z, hit, hit0, stat = setup(spec, vals, covs); x0 = pack(spec)
    f = lambda x: nll(x, spec, M, Z, hit, hit0, stat)
    print(f'{var}: n={len(vals)} stored {spec["fit"]["loglik"]:.3f} engine {fitcheck.value_loglik(spec, vals, covs):.3f} vectorised {-f(x0):.3f}')
    r = optimize.minimize(f, x0, method='L-BFGS-B', options={'maxiter': 2000, 'ftol': 1e-13})
    r = optimize.minimize(f, r.x, method='Nelder-Mead', options={'maxiter': 6000, 'xatol': 1e-7, 'fatol': 1e-9})
    new = unpack(spec, r.x); ll = fitcheck.value_loglik(new, vals, covs); k = spec['fit'].get('n_params', len(x0))
    print(f'{var}: refitted engine loglik {ll:.3f} (vectorised {-r.fun:.3f}) [{time.time() - t0:.0f} s]; params {np.round(r.x, 4)} vs {np.round(x0, 4)}')
    new['fit'] = {'n': len(vals), 'loglik': ll, 'n_params': k, 'aic': -2 * ll + 2 * k, 'bic': -2 * ll + k * math.log(len(vals)),
                  'data': spec['fit'].get('data'), 'method': 'tools/fit_logt.py: library (grid) likelihood, L-BFGS-B then Nelder-Mead',
                  'date': time.strftime('%Y-%m-%d'), 'previous': {k_: spec['fit'][k_] for k_ in ('n', 'loglik', 'bic') if k_ in spec['fit']}}
    if '--write' in sys.argv: json.dump(new, open(p, 'w'), indent=1)
