"""Refit the four timestamp models by maximising the likelihood the library itself evaluates.

The original fits (Oct 2026) used separate fitters whose discretisation differed slightly from
`engine.TimestampModel` (op start: the first-case clock prior was integrated on its own 0.1-min grid), so the
stored log-likelihoods did not match what the library recomputes. This fitter builds, for every included case,
the exact rounding masks that `TimestampModel.components` produces (they do not depend on the parameters), and
maximises the same discretised likelihood with JAX gradients. Knots, grids and the rounding structure are kept;
only the latent coefficients, the clock-prior coefficients and the mode logits / covariate effects are refitted.
The stored fit block is then the engine's own evaluation of the refitted parameters.

usage: python tools/fit_timestamps.py <clinical_data.csv> [anestart opstart opend aneend] [--write]
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
import vitaldb_resolution as vr
from vitaldb_resolution import engine
from vitaldb_resolution.registry import _ref_window

MODELS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'vitaldb_resolution', 'data', 'models')


def case_inputs(reg, d, var):
    """Included cases (status ok through the API) with their clock minute and covariates."""
    rows = []
    for r in d.to_dict('records'):
        dist = reg.case_distribution(r, var)
        if not dist.ok: continue
        ctx = reg.timestamp_context(var, r[var], r)
        c = {'ref': ctx['ref']}
        if var == 'anestart': c.update(ref_window=_ref_window(ctx['recording_start_clock']), emop=int(r['emop']))
        rows.append((int(r['caseid']), ctx['M'], c))
    return rows


def build(spec, rows):
    """Static arrays: per case a window of grid cells holding every cell that can record as M or that falls in a
    nominal window, the rounding/nominal masks on it, and the clock-time inputs of the prior."""
    TM = engine.TimestampModel(spec); G = TM.G; nG = len(G); dx = TM.dx
    alloc = spec['allocation']; L = spec['latent']; P = L.get('time_of_day_prior')
    n = len(rows); masks = [[] for _ in alloc]; wins = [[] for _ in alloc]; lo_idx = []
    applies = np.zeros((n, len(alloc))); covx = []; tod_on = np.zeros(n, bool); xc_win = []; xc_full = []
    spans = []
    for i, (_, M, c) in enumerate(rows):
        X, comps = TM.components(c)
        sel = np.zeros(nG, bool)
        for k, (lab, rec, take) in enumerate(comps):
            hit = np.round(rec) == M
            a = alloc[k]
            if a['kind'] == 'nominal':        # a nominal mode records M only inside its window
                w = a['window']; hit &= ((X % 1440) >= w[0]) & ((X % 1440) < w[1])
            sel |= hit
        spans.append(np.where(sel)[0])        # the cells themselves, not the span between them (memory)
    W = max(len(ix) for ix in spans)
    gidx = np.zeros((n, W), np.int64); valid = np.zeros((n, W), bool)
    for i, ((_, M, c), ix) in enumerate(zip(rows, spans)):
        X, comps = TM.components(c)
        g = np.zeros(W, np.int64); g[:len(ix)] = ix; ok = np.arange(W) < len(ix)
        gidx[i] = g; valid[i] = ok
        for k, (lab, rec, take) in enumerate(comps):
            m = (np.round(rec) == M)[g] & ok; masks[k].append(m)
            al = alloc[k]
            if al['kind'] == 'nominal':
                w = al['window']; xc = X % 1440; wins[k].append((((xc >= w[0]) & (xc < w[1]))[g]) & ok)
                ap = al.get('applies_if_ref_in'); applies[i, k] = 1.0 if (not ap or ap[0] <= c['ref'] % 1440 < ap[1]) else 0.0
            else:
                wins[k].append(np.zeros(W, bool)); applies[i, k] = 1.0
        row = []
        for al in alloc:
            pr = al.get('prob') or {}
            row.append([engine.transform(cv['transform'], c[cv['name']]) for cv in pr.get('covariates', [])])
        covx.append(row)
        if P:
            rc = c['ref'] % 1440; tod_on[i] = P['applies_if_ref_in'][0] <= rc < P['applies_if_ref_in'][1]
            day = np.floor(c['ref'] / 1440) * 1440
            xc_win.append((X - day)[g])
            if tod_on[i]: xc_full.append((X - day).astype(np.float64))
    # B-spline bases
    t = np.r_[[spec['grid']['lo']] * 3, L['inner_knots'], [spec['grid']['hi']] * 3]; K = len(t) - 4
    BG = np.column_stack([BSpline(t, np.eye(K)[k], 3)(G) for k in range(K)])
    data = dict(gidx=gidx, valid=valid, masks=np.stack([np.stack(m) for m in masks], 1),
                wins=np.stack([np.stack(w) for w in wins], 1), applies=applies, BG=BG, n=n, W=W)
    data['covx'] = [np.array([covx[i][k] for i in range(n)], float).reshape(n, -1) for k in range(len(alloc))]
    if P:
        kn = np.array(P['knots']); th = np.r_[[kn[0]] * 3, kn, [kn[-1]] * 3]; Kh = len(th) - 4
        fine = np.arange(kn[0], kn[-1] + 1e-9, 0.002)                      # prior evaluated by interpolation
        data['tod'] = dict(on=tod_on, fine=fine, BH=np.column_stack([BSpline(th, np.eye(Kh)[k], 3)(fine) for k in range(Kh)]),
                           lo=kn[0], hi=kn[-1], xc_win=np.array(xc_win), xc_full=np.array(xc_full))
    return data


def pack(spec):
    """Parameter vector <-> spec paths."""
    paths = [('latent', 'coefficients', j) for j in range(len(spec['latent']['coefficients']))]
    P = spec['latent'].get('time_of_day_prior')
    if P: paths += [('tod', j) for j in range(len(P['coefficients']))]
    for k, a in enumerate(spec['allocation']):
        pr = a.get('prob')
        if not pr: continue
        paths.append(('logit', k))
        paths += [('cov', k, j) for j in range(len(pr.get('covariates', [])))]
    def get(s):
        out = []
        for p in paths:
            if p[0] == 'latent': out.append(s['latent']['coefficients'][p[2]])
            elif p[0] == 'tod': out.append(s['latent']['time_of_day_prior']['coefficients'][p[1]])
            elif p[0] == 'logit': out.append(s['allocation'][p[1]]['prob']['logit'])
            else: out.append(s['allocation'][p[1]]['prob']['covariates'][p[2]]['coef'])
        return np.array(out, float)
    def put(s, x):
        s = copy.deepcopy(s)
        for p, v in zip(paths, map(float, x)):
            if p[0] == 'latent': s['latent']['coefficients'][p[2]] = v
            elif p[0] == 'tod': s['latent']['time_of_day_prior']['coefficients'][p[1]] = v
            elif p[0] == 'logit': s['allocation'][p[1]]['prob']['logit'] = v
            else: s['allocation'][p[1]]['prob']['covariates'][p[2]]['coef'] = v
        return s
    return paths, get, put


def make_nll(spec, data, paths, chunk=800):
    """Negative log-likelihood and its gradient. Cases are processed in chunks, each under jax.checkpoint, so the
    gradient's memory is one chunk's rather than all cases' (anaesthesia start needs > 6 GB otherwise)."""
    alloc = spec['allocation']; tod = data.get('tod'); n = data['n']
    BG = jnp.asarray(data['BG'])
    nlat = len(spec['latent']['coefficients']); ntod = len(spec['latent']['time_of_day_prior']['coefficients']) if tod else 0
    if tod:
        fine = jnp.asarray(tod['fine']); BH = jnp.asarray(tod['BH']); tlo, thi = tod['lo'], tod['hi']
        on_all = np.asarray(tod['on']); onrow = np.cumsum(on_all) - 1          # row of each on-case in xc_full

    def chunk_fn(a, b):
        sl = slice(a, b)
        C = {k: jnp.asarray(data[k][sl]) for k in ('gidx', 'valid', 'masks', 'wins', 'applies')}
        cov = [jnp.asarray(c[sl]) for c in data['covx']]; m = b - a
        if tod:
            on = on_all[sl]; loc = np.where(on)[0]
            xw = jnp.asarray(tod['xc_win'][sl]); xf = jnp.asarray(tod['xc_full'][onrow[a:b][on]]) if len(loc) else None
            onj = jnp.asarray(on)
        def f(x):
            c = jnp.concatenate([jnp.zeros(1), x[:nlat]]); i = nlat
            lw = BG @ c; wG = jnp.exp(lw - lw.max()); wG = wG / wG.sum()
            w = wG[C['gidx']] * C['valid']
            if tod:
                ch = jnp.concatenate([jnp.zeros(1), x[i:i + ntod]]); i += ntod
                lf = BH @ ch
                def lh(xc):
                    inside = (xc >= tlo) & (xc < thi)
                    return jnp.where(inside, jnp.interp(jnp.clip(xc, tlo, thi), fine, lf), -jnp.inf)
                if len(loc):
                    lfull = lh(xf); lwin = lh(xw[loc])
                    mx = jnp.max(jnp.where(jnp.isfinite(lfull), lfull, -1e300), axis=1, keepdims=True)
                    Z = (wG[None, :] * jnp.exp(lfull - mx)).sum(1)
                    wt = wG[C['gidx'][loc]] * jnp.exp(lwin - mx) / Z[:, None] * C['valid'][loc]
                    w = w.at[loc].set(wt)
            rem = jnp.ones_like(w); L = jnp.zeros(m)
            for k, al in enumerate(alloc):
                pr = al.get('prob')
                if pr:
                    lg = x[i] * jnp.ones(m); i += 1
                    nc = cov[k].shape[1]
                    if nc: lg = lg + cov[k] @ x[i:i + nc]; i += nc
                    p = jax.nn.sigmoid(lg)
                else:
                    p = jnp.ones(m)
                if al['kind'] == 'nominal':
                    take = rem * (p * C['applies'][:, k])[:, None] * C['wins'][:, k]
                else:
                    take = rem * p[:, None]
                L = L + (w * take * C['masks'][:, k]).sum(1)
                rem = rem - take
            return -jnp.log(jnp.clip(L, 1e-300)).sum()
        return jax.checkpoint(f)

    fns = [chunk_fn(a, min(a + chunk, n)) for a in range(0, n, chunk)]
    def nll(x):
        return sum(f(x) for f in fns)
    vg = jax.jit(jax.value_and_grad(nll))
    return jax.jit(nll), vg


def engine_loglik(spec, rows):
    TM = engine.TimestampModel(spec)
    return float(sum(math.log(max(TM.prob_recorded(M, c), 1e-300)) for _, M, c in rows))


def refit(var, d, reg, write=False, log=print):
    spec = json.load(open(os.path.join(MODELS, f'{var}.model.json')))
    rows = case_inputs(reg, d, var); t0 = time.time()
    data = build(spec, rows); paths, get, put = pack(spec)
    f, vg = make_nll(spec, data, paths); x0 = get(spec)
    ll_engine_old = engine_loglik(spec, rows)
    log(f'{var}: n={len(rows)} W={data["W"]} params={len(x0)}; stored {spec["fit"]["loglik"]:.3f}, engine {ll_engine_old:.3f}, '
        f'jax {-float(f(jnp.asarray(x0))):.3f}  ({time.time() - t0:.0f} s build)')
    fun = lambda x: float(f(jnp.asarray(x)))
    def fg(x):
        v, gr = vg(jnp.asarray(x)); return float(v), np.asarray(gr)
    hist = []
    def cb(intermediate_result):
        hist.append(float(intermediate_result.fun))
        np.save(f'/tmp/{var}_x_checkpoint.npy', intermediate_result.x)
        if len(hist) % 10 == 0: log(f'{var}:   iteration {len(hist)}, nll {hist[-1]:.4f}  ({time.time() - t0:.0f} s)')
        # stop once 25 iterations gain less than 0.01 in loglik (these surfaces are flat along a ridge; the
        # stored loglik is the engine's evaluation of the parameters reached, so it is exact either way)
        if len(hist) > 25 and hist[-26] - hist[-1] < 1e-2: raise StopIteration
    ck = f'/tmp/{var}_x_checkpoint.npy'
    xs = np.load(ck) if os.environ.get('RESUME') and os.path.exists(ck) else x0
    r = optimize.minimize(fg, xs, jac=True, method='L-BFGS-B', callback=cb,
                          options={'maxiter': 1500, 'ftol': 1e-10, 'gtol': 1e-5})
    log(f'{var}: L-BFGS-B {r.nit} iterations, {r.nfev} evaluations, message: {r.message}')
    if fun(x0) - r.fun < 0.01:          # no real gain: keep the stored parameters, re-evaluate only
        r.x = x0; log(f'{var}: gain < 0.01; parameters kept')
    new = put(spec, r.x)
    ll = engine_loglik(new, rows); k = len(r.x) + sum(1 for a in spec['allocation'] if a['kind'] == 'nominal' and False)
    n_params = spec['fit'].get('n_params', len(r.x))
    refit_ = not np.array_equal(r.x, x0)
    new['fit'] = {'n': len(rows), 'loglik': ll, 'n_params': n_params, 'bic': -2 * ll + n_params * math.log(len(rows)),
                  'source': 'clinical_data.csv',
                  'method': ('tools/fit_timestamps.py: refitted with the library likelihood (JAX gradient, L-BFGS-B)' if refit_
                             else 'parameters as fitted on ' + spec['fit'].get('date', '?') + ' (' + spec['fit'].get('method', '') +
                             '); fit statistics re-evaluated with the library likelihood by tools/fit_timestamps.py'),
                  'date': time.strftime('%Y-%m-%d'),
                  'previous': {k: spec['fit'][k] for k in ('n', 'loglik', 'bic', 'method', 'date') if k in spec['fit']}}
    log(f'{var}: refitted engine loglik {ll:.3f} (jax {-r.fun:.3f}); change vs engine on old params {ll - ll_engine_old:+.3f}; '
        f'max |dparam| {np.abs(r.x - x0).max():.3g}')
    if write:
        json.dump(new, open(os.path.join(MODELS, f'{var}.model.json'), 'w'), indent=1)
    return spec, new, rows


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    d = pd.read_csv(args[0], encoding='utf-8-sig'); reg = vr.load()
    for v in (args[1:] or ['aneend', 'opend', 'opstart', 'anestart']):
        refit(v, d, reg, write='--write' in sys.argv, log=lambda m: print(m, flush=True))
