"""Fit log-spline volume models (urine output, crystalloid, blood loss) with the library's own likelihood,
optionally with a location covariate (log anaesthesia duration) and alternative rounding anchors.

Likelihood: exactly `engine.Model` (grid, rounding modes, capture windows, truncation of recorded zero), except
that the cubic B-spline of the latent is evaluated by interpolation from a fine table during optimisation; the
stored log-likelihood is always the engine's own evaluation.

usage: python tools/fit_volume.py <clinical_data.csv> <variable> [--duration] [--department] [--anchor floor]
       [--with-derived] [--write]

--department adds department (indicator offsets against general surgery) to the logit of every stick-breaking
propensity.
"""
import copy
import json
import math
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from scipy import optimize
from scipy.interpolate import BSpline

jax.config.update('jax_enable_x64', True)
from vitaldb_resolution import engine

MODELS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'vitaldb_resolution', 'data', 'models')
DUR_REF = 180.0          # minutes; centre of the duration transform (median anaesthesia duration)
DEPTS = ['Thoracic surgery', 'Gynecology', 'Urology']      # offsets against General surgery (the largest)


def duration_minutes(d):
    """Charted anaesthesia duration (min) and whether it is usable: both times valid and aneend not derived from
    an imputed op end (float noise) or outside the model range."""
    dur = (d.aneend - d.anestart) / 60
    derived = np.abs(d.aneend - np.round(d.aneend)) > 1e-9
    usable = d.aneend.notna() & d.anestart.notna() & (d.aneend > -1e8) & (d.adm > -1e8) & (dur > 0) & \
             ((d.aneend - d.caseend) / 60 < 240) & ~derived
    return dur, usable, derived


def records(d, var, spec, with_derived=False, require_duration=True):
    res = spec['resolution']['effective']; x = d[var]
    pos = (x > 0) & (np.abs(x / res - np.round(x / res)) < 1e-9)
    dur, usable, derived = duration_minutes(d)
    keep = pos & (usable | (with_derived & derived & (dur > 0))) if require_duration else pos
    return x[keep].values.astype(float), dur[keep].values.astype(float), d.department[keep].values


def set_anchor(spec, anchor):
    s = copy.deepcopy(spec)
    for c in s['allocation']:
        if c['kind'] == 'rounding' and c['mode']['type'] == 'absolute':
            c['mode']['anchor'] = anchor; c.pop('anchor', None)
    return s


def build(spec, vals):
    """Static arrays for each record value: a window of grid cells, per mode the cells recording that value and
    the static part of each mode's probability (capture window, recorded > 0, applies_below)."""
    M = engine.Model(spec); X = M.X; nG = len(X); alloc = spec['allocation']
    recs = [M._round(c['mode']) for c in alloc]
    static = []
    for c, rec in zip(alloc, recs):
        m = np.ones(nG, bool)
        cw = c.get('capture_window')
        if isinstance(cw, dict): dd = rec - X; m &= (dd >= cw['lo'] - 1e-9) & (dd < cw['hi'] - 1e-9)
        elif cw is not None: m &= np.abs(X - rec) < cw
        if c.get('prob'): m &= rec > 0
        if c.get('applies_below') is not None: m &= X <= c['applies_below']
        static.append(m)
    uv = np.unique(np.r_[0.0, vals])
    spans = {}
    for v in uv:
        hit = np.zeros(nG, bool)
        for rec, st in zip(recs, static): hit |= (np.round(rec / M.ru) == round(v / M.ru)) & st
        idx = np.where(hit)[0]; spans[v] = (idx.min(), idx.max() + 1) if len(idx) else (0, 1)
    W = max(b - a for a, b in spans.values())
    g = np.zeros((len(uv), W), np.int64); ok = np.zeros((len(uv), W), bool)
    hitm = np.zeros((len(uv), len(alloc), W), bool); stat = np.zeros((len(uv), len(alloc), W), bool)
    for i, v in enumerate(uv):
        a, b = spans[v]; gi = np.arange(a, a + W); valid = gi < nG; gi = np.minimum(gi, nG - 1)
        g[i] = gi; ok[i] = valid
        for k, rec in enumerate(recs):
            hitm[i, k] = (np.round(rec[gi] / M.ru) == round(v / M.ru)) & valid
            stat[i, k] = static[k][gi] & valid
    vidx = np.searchsorted(uv, vals)
    return dict(M=M, uv=uv, vidx=vidx, g=g, ok=ok, hit=hitm, stat=stat, LX=M.LX, W=W)


def _dept_coefs(pr):
    have = {c['transform']['value']: c['coef'] for c in pr.get('covariates', []) if c['name'] == 'department'}
    return [have.get(dp, 0.0) for dp in DEPTS]


def pack(spec, duration, department=False):
    L = spec['latent']
    x = list(L['coefficients']); names = [f'c{j}' for j in range(len(x))]
    if duration:
        b = L['location_covariates'][0]['coef'] if L.get('location_covariates') else 0.0
        x.append(b); names.append('b_duration')
    for k, c in enumerate(spec['allocation']):
        pr = c.get('prob')
        if not pr: continue
        x.append(pr['logit']); names.append(f'logit{k}')
        if pr['form'] == 'logistic_logx': x.append(pr['slope']); names.append(f'slope{k}')
        if department:
            x += _dept_coefs(pr); names += [f'dept{k}:{dp}' for dp in DEPTS]
    return np.array(x, float), names


def unpack(spec, x, duration, dref=DUR_REF, marginal_logd=None, department=False):
    s = copy.deepcopy(spec); L = s['latent']; K1 = len(L['coefficients']); i = K1
    L['coefficients'] = [float(v) for v in x[:K1]]
    if duration:
        b = float(x[i]); i += 1
        L['location_covariates'] = [{'name': 'duration', 'transform': {'type': 'log_ratio', 'ref': dref}, 'coef': b,
                                     'note': 'charted anaesthesia duration, aneend - anestart, minutes'}]
        q = np.quantile(marginal_logd, (np.arange(40) + 0.5) / 40)
        L['marginal'] = {'covariate': 'duration', 'shifts': [float(b * v) for v in q],
                         'note': 'used when duration is missing: average of the latent over 40 quantiles of '
                                 'log(duration / ref) among the fitted cases'}
    else:
        L.pop('location_covariates', None); L.pop('marginal', None)
    for k, c in enumerate(s['allocation']):
        pr = c.get('prob')
        if not pr: continue
        pr['logit'] = float(x[i]); i += 1
        if pr['form'] == 'logistic_logx': pr['slope'] = float(x[i]); i += 1
        pr['covariates'] = [c for c in pr.get('covariates', []) if c['name'] != 'department']
        if department:
            pr['covariates'] += [{'name': 'department', 'transform': {'type': 'indicator', 'value': dp}, 'coef': float(x[i + j])}
                                 for j, dp in enumerate(DEPTS)]
            i += len(DEPTS)
        if not pr['covariates']: pr.pop('covariates')
    if department:
        s['required_covariates'] = sorted(set(s.get('required_covariates', [])) | {'department'})
    else:
        s.pop('required_covariates', None)
    return s


def make_nll(spec, data, logd, duration, dept=None):
    M = data['M']; L = spec['latent']; K = L['K']
    lo, hi = M.knots[0], M.knots[-1]
    fine = np.linspace(lo, hi, 60001)
    BF = jnp.asarray(np.column_stack([BSpline(M.knots, np.eye(K)[k], 3)(fine) for k in range(K)]))
    fine_j = jnp.asarray(fine); LX = jnp.asarray(data['LX'])
    ud, uinv = np.unique(np.round(logd, 12), return_inverse=True) if duration else (np.zeros(1), np.zeros(len(logd), int))
    UD = jnp.asarray(ud); uinv = jnp.asarray(uinv)
    g = jnp.asarray(data['g']); ok = jnp.asarray(data['ok']); hit = jnp.asarray(data['hit']); stat = jnp.asarray(data['stat'])
    vidx = jnp.asarray(data['vidx']); i0 = int(np.searchsorted(data['uv'], 0.0))
    alloc = spec['allocation']; K1 = K - 1
    department = dept is not None
    if department:
        Dn = np.column_stack([np.asarray(dept) == dp for dp in DEPTS]).astype(float)
        code = Dn @ (2 ** np.arange(len(DEPTS)))
        pairs, pinv = np.unique(np.column_stack([np.asarray(uinv), code]), axis=0, return_inverse=True)
        D = jnp.asarray(Dn); pinv = jnp.asarray(pinv.ravel()); pair_u = jnp.asarray(pairs[:, 0].astype(int))
        Dpair = jnp.asarray(np.array([Dn[np.where(code == c_)[0][0]] for c_ in pairs[:, 1]]))
    def spl(u, c): return jnp.interp(jnp.clip(u, lo, hi), fine_j, BF @ c)
    def nll(x):
        c = jnp.concatenate([jnp.zeros(1), x[:K1]]); i = K1
        b = 0.0
        if duration: b = x[i]; i += 1
        sh = b * UD                                                        # U shifts
        lwfull = jax.vmap(lambda s: spl(LX - s, c) - LX)(sh)               # U x nG
        lz = jax.scipy.special.logsumexp(lwfull, axis=1)                   # U
        def rec_mass(widx, s_idx, Dm=None):
            """mass recording value index widx under shift index s_idx (vector inputs); Dm: department indicators."""
            gi = g[widx]; s = sh[s_idx]
            lw = spl(LX[gi] - s[:, None], c) - LX[gi] - lz[s_idx][:, None]
            w = jnp.exp(lw) * ok[widx]
            rem = jnp.ones_like(w); tot = jnp.zeros(w.shape[0]); j = i
            for k, a in enumerate(alloc):
                pr = a.get('prob')
                if pr:
                    lg = x[j]; j += 1
                    sl = None
                    if pr['form'] == 'logistic_logx': sl = x[j]; j += 1
                    if department:
                        lg = lg + Dm @ x[j:j + len(DEPTS)]; j += len(DEPTS)
                        lg = lg[:, None]
                    if sl is not None:
                        p = jax.nn.sigmoid(lg + sl * (LX[gi] - math.log(pr['centre'])))
                    else:
                        p = jax.nn.sigmoid(lg) * jnp.ones_like(w)
                else:
                    p = jnp.ones_like(w)
                take = rem * p * stat[widx, k]
                tot = tot + (w * take * hit[widx, k]).sum(1)
                rem = rem - take
            return tot
        trunc = spec.get('truncation', {}).get('exclude_recorded_zero')
        if department:
            q = rec_mass(vidx, uinv, D)
            q0 = rec_mass(jnp.full(len(pair_u), i0), pair_u, Dpair)[pinv] if trunc else 0.0
            ll = jnp.log(jnp.clip(q, 1e-300)) - (jnp.log1p(-q0) if trunc else 0.0)
        else:
            q = rec_mass(vidx, uinv)
            q0 = rec_mass(jnp.full(len(ud), i0), jnp.arange(len(ud)))
            ll = jnp.log(jnp.clip(q, 1e-300)) - (jnp.log1p(-q0[uinv]) if trunc else 0.0)
        return -ll.sum()
    return jax.jit(nll), jax.jit(jax.grad(nll))


def engine_loglik(spec, vals, durs, depts=None):
    M = engine.Model(spec)
    tot = 0.0; cache = {}
    usecov = bool(spec['latent'].get('location_covariates')); usedept = 'department' in spec.get('required_covariates', [])
    for i, (v, dd) in enumerate(zip(vals, durs)):
        cov = {}
        if usecov: cov['duration'] = float(dd)
        if usedept: cov['department'] = depts[i]
        key = (round(float(dd), 9) if usecov else 0, cov.get('department'))
        if key not in cache: cache[key] = M.pmf(spec, cov or None)
        q = cache[key]
        tot += math.log(max(q[int(round(v / M.ru))], 1e-300))
    return tot


def fit(spec, vals, durs, duration, log=print, tag='', depts=None):
    t0 = time.time(); logd = np.log(durs / DUR_REF); department = depts is not None
    data = build(spec, vals); x0, names = pack(spec, duration, department)
    f, g = make_nll(spec, data, logd, duration, depts)
    fun = lambda x: float(f(jnp.asarray(x))); jac = lambda x: np.asarray(g(jnp.asarray(x)))
    hist = []
    def cb(intermediate_result):
        hist.append(float(intermediate_result.fun))
        if len(hist) % 25 == 0: log(f'{tag}:   iteration {len(hist)}, nll {hist[-1]:.4f}  ({time.time() - t0:.0f} s)')
        # stop once 25 iterations gain less than 0.001 in loglik (BIC differences here are in the hundreds)
        if len(hist) > 25 and hist[-26] - hist[-1] < 1e-3: raise StopIteration
    r = optimize.minimize(fun, x0, jac=jac, method='L-BFGS-B', callback=cb,
                          options={'maxiter': 4000, 'ftol': 1e-13, 'gtol': 1e-7})
    new = unpack(spec, r.x, duration, marginal_logd=logd, department=department)
    ll = engine_loglik(new, vals, durs, depts); k = len(r.x); n = len(vals)
    log(f'{tag:28s} n={n} k={k} loglik(engine)={ll:.3f} (jax {-r.fun:.3f}) BIC={-2 * ll + k * math.log(n):.2f} '
        f'[{time.time() - t0:.0f} s]')
    return new, ll, k, dict(zip(names, map(float, r.x)))


def finalize(new, ll, vals, d, var, method):
    """Fit block, zero model (urine output) and notes for an installed model."""
    k = len(pack(new, bool(new['latent'].get('location_covariates')), 'department' in new.get('required_covariates', []))[0])
    profiled = sum(1 for c in new['allocation'] if c.get('capture_window') is not None)   # capture window chosen by profile likelihood
    n = len(vals); kk = k + profiled
    prev = {x: new['fit'][x] for x in ('n', 'loglik', 'bic') if x in new.get('fit', {})}
    new['schema_version'] = '0.5'
    new['fit'] = {'n': n, 'loglik': ll, 'n_params': kk, 'aic': -2 * ll + 2 * kk, 'bic': -2 * ll + kk * math.log(n),
                  'data': f'VitalDB clinical_data.csv ({var}), positive values on the {new["resolution"]["effective"]:g} mL grid '
                          'with a usable anaesthesia duration', 'method': method, 'date': time.strftime('%Y-%m-%d'),
                  'previous': prev}
    if profiled: new['fit']['note'] = 'n_params counts the capture window, chosen by profile likelihood'
    if var == 'intraop_uo':
        x = d[var]; nz = int((x == 0).sum()); nr = int(x.notna().sum())
        new['zero_model'] = {
            'p0': nz / nr, 'n_zero': nz, 'n_recorded': nr,
            'note': 'A recorded 0 is a point mass at 0 with probability p0 (zeros / recorded values), mixed with the latent '
                    'mass that records as 0. The latent and rounding model are fitted on positive values only. Missing '
                    'values are not zeros: they mean not measured.',
            'rules': {
                'zero_anuric': {'test': {'any': [{'field': 'preop_cr', 'op': '>', 'value': 2}, {'field': 'asa', 'op': '==', 'value': 6}]},
                                'meaning': 'true zero: preoperative creatinine > 2 mg/dL (renal failure, dialysis) or ASA 6 (organ donor)'},
                'zero_unexplained': {'test': None,
                                     'meaning': 'recorded 0 with normal creatinine: possibly a blocked catheter or 0 meaning not measured; kept but uncertain'}}}
        new['special_values'] = [{'value': 0, 'handling': 'zero_model: point mass at 0, flagged zero_anuric or zero_unexplained'}]
    return new


if __name__ == '__main__':
    a = [x for x in sys.argv[1:] if not x.startswith('--')]
    d = pd.read_csv(a[0], encoding='utf-8-sig'); var = a[1]
    spec = json.load(open(os.path.join(MODELS, f'{var}.model.json')))
    if '--anchor' in sys.argv: spec = set_anchor(spec, sys.argv[sys.argv.index('--anchor') + 1])
    vals, durs, depts = records(d, var, spec, with_derived='--with-derived' in sys.argv)
    new, ll, k, p = fit(spec, vals, durs, '--duration' in sys.argv, tag=var,
                        depts=depts if '--department' in sys.argv else None)
    print(json.dumps(p, indent=1))
    if '--write' in sys.argv:
        new = finalize(new, ll, vals, d, var, 'tools/fit_volume.py' + (' --duration' if '--duration' in sys.argv else '') +
                       ': library likelihood, JAX gradient, L-BFGS-B then BFGS')
        json.dump(new, open(os.path.join(MODELS, f'{var}.model.json'), 'w'), indent=1)
