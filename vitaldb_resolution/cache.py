"""Precomputed whole-dataset posterior cache for fast Monte Carlo draws.

Build once (minutes), load in seconds, then draw every recorded value in the dataset in one vectorised step:

    python -m vitaldb_resolution.cache build --clinical clinical_data.csv --lab lab_data.csv \
        --out vitaldb_cache --dtype int16 --knots 1024

    cache = vr.Cache.load('vitaldb_cache')
    s = cache.sampler(seed=1, variability=True)       # height/weight measurement variability added at draw time
    d = s.draw()                                       # dict: variable -> array over cases; 'lab' -> array over rows

How each value is stored
  - simple shapes (exact count, uniform interval, triangular) as 1-3 float64 parameters: exact.
  - posteriors from fitted models (heaped values, timestamps, mixtures) as quantile tables: K equally spaced
    probability levels from 0 to 1, stored as int16 (per-table offset and scale), float32 or float64. A draw is
    one uniform number and a linear interpolation, i.e. the CDF is treated as piecewise linear between knots.
  - height and weight keep two tables per case (measured modes, written-in modes) and the written-in
    probability, so measurement variability can differ between them at draw time.
  - derived variables (BMI, length of stay is stored directly; the four intervals) are computed from the
    draws of their inputs in the same iteration, so shared inputs stay coherent.
"""
import json
import os
import time

import numpy as np

from . import engine
from .registry import Registry, _missing

KIND = {'none': 0, 'point': 1, 'uniform': 2, 'triangular': 3, 'table': 4, 'split': 5}
INTERVALS = {'ind_to_incision': ('anestart', 'opstart'), 'op_duration': ('opstart', 'opend'),
             'ane_duration': ('anestart', 'aneend'), 'emergence': ('opend', 'aneend')}


# ---------------------------------------------------------------- exact piecewise-linear CDF of a mixture
def breakpoints(comps):
    """Exact (x, p) breakpoints of the CDF of a mixture of tabulated, uniform and point components.

    With them, x(p) = np.interp(p, P, X) is the exact quantile function. Returns None if a component has
    another shape.
    """
    if any(c.kind not in ('tabulated', 'uniform', 'point') for c in comps): return None
    tab = [c for c in comps if c.kind == 'tabulated']
    if tab and len(tab) == len(comps) and len({round(c.dx, 9) for c in tab}) == 1:
        dx = tab[0].dx; x0 = min(c.x[0] for c in tab)
        keys = np.concatenate([np.round((c.x - x0) / dx).astype(np.int64) for c in tab])
        mass = np.concatenate([c.m * c.weight for c in tab])
        k, inv = np.unique(keys, return_inverse=True)
        m = np.bincount(inv, weights=mass); m = m / m.sum()
        left = x0 + k * dx - dx / 2
        X = np.empty(2 * len(k)); P = np.empty(2 * len(k))
        X[0::2] = left; X[1::2] = left + dx
        c = np.cumsum(m); P[1::2] = c; P[0::2] = c - m
        return X, P
    pieces = []                                              # (a, b, w)
    for c in comps:
        if c.kind == 'point': pieces.append((c.p['value'], c.p['value'], c.weight))
        elif c.kind == 'uniform': pieces.append((c.p['lower'], c.p['upper'], c.weight))
        else:
            pieces += [(xi - c.dx / 2, xi + c.dx / 2, c.weight * mi) for xi, mi in zip(c.x, c.m)]
    tot = sum(w for *_, w in pieces)
    pts = sorted({a for a, _, _ in pieces} | {b for _, b, _ in pieces})
    X, P = [], []
    for e in pts:
        below = sum(w for a, b, w in pieces if b <= e and not (a == b == e))     # mass entirely at or below e
        at = sum(w for a, b, w in pieces if a == b == e)                         # point masses at e
        inside = sum(w * (e - a) / (b - a) for a, b, w in pieces if a < e < b)
        X += [e, e]; P += [(below + inside) / tot, (below + inside + at) / tot]
    return np.array(X), np.array(P)


def knot_levels(K):
    """Probability levels of the K knots: 0, K-2 interior mid-levels (k + 0.5)/(K - 2), and 1."""
    return np.r_[0.0, (np.arange(K - 2) + 0.5) / (K - 2), 1.0]


def table_from(X, P, K):
    """Quantile table at knot_levels(K).

    The two end knots are set so that the linear segment reproduces the exact conditional mean of each tail
    (probability 0.5/(K-2)), instead of the support ends: a thin, long tail otherwise gets 1/(2(K-2)) of the
    probability spread evenly across it, which inflates the SD.
    """
    pk = knot_levels(K); q = np.interp(pk, P, X); h = pk[1]
    t = (np.arange(256) + 0.5) / 256 * h
    q[0] = 2 * np.interp(t, P, X).mean() - q[1]
    q[-1] = 2 * np.interp(1 - h + t, P, X).mean() - q[-2]
    return q


def quantize(Q, dtype, chunk=4096):
    """Q: (n, K) float64 tables -> stored array + per-table offset/scale (int16 only). Chunked to bound memory."""
    if dtype == 'float64': return Q.astype(np.float64, copy=False), None, None
    if dtype == 'float32': return Q.astype(np.float32), None, None
    lo = Q[:, 0].copy(); rng = Q[:, -1] - lo
    scale = np.where(rng > 0, rng / 65535.0, 1.0)
    out = np.empty(Q.shape, np.int16)
    for s in range(0, len(Q), chunk):
        sl = slice(s, s + chunk)
        out[sl] = (np.round((Q[sl] - lo[sl, None]) / scale[sl, None]) - 32768).astype(np.int16)
    return out, lo, scale


def tables_matrix(tables, K):
    """Resample posterior tables onto ``K`` quantile knots each.

    Args:
        tables: ``(X, P)`` pairs: grid values and probabilities.
        K: Knots per table.

    Returns:
        An array of shape ``(len(tables), K)``.
    """
    Q = np.empty((len(tables), K))
    for i, (X, P) in enumerate(tables): Q[i] = table_from(X, P, K)
    return Q


def dequantize(q, lo, scale, rows=None):
    """Undo :func:`quantize` for some or all rows.

    Args:
        q: Stored knots (integers when quantised, floats otherwise).
        lo: Per-row offsets, or ``None`` when the tables are stored as floats.
        scale: Per-row scales.
        rows: Rows to return. Default: all.

    Returns:
        The knots as floats.
    """
    if lo is None: return q if rows is None else q[rows]
    if rows is None: return lo[:, None] + (q.astype(np.float64) + 32768) * scale[:, None]
    return lo[rows, None] + (q[rows].astype(np.float64) + 32768) * scale[rows, None]


def table_draw(q, lo, scale, rows, u):
    """Draw from tables `rows` at uniforms u (same length): linear interpolation between knots."""
    K = q.shape[1]; pk = knot_levels(K)
    i = np.clip(np.searchsorted(pk, u, side='right') - 1, 0, K - 2); f = (u - pk[i]) / (pk[i + 1] - pk[i])
    a = q[rows, i].astype(np.float64); b = q[rows, i + 1].astype(np.float64)
    v = a + f * (b - a)
    if lo is not None: v = lo[rows] + (v + 32768) * scale[rows]
    return v


# ---------------------------------------------------------------- builder
class _Collector:
    def __init__(self, n):
        self.kind = np.zeros(n, np.int8); self.a = np.full(n, np.nan); self.b = np.full(n, np.nan)
        self.c = np.full(n, np.nan); self.t1 = np.full(n, -1, np.int64); self.t2 = np.full(n, -1, np.int64)
        self.status = np.empty(n, object); self.flags = np.empty(n, object)


def _classify(dist, col, i, tables, split=None):
    col.status[i] = dist.status; col.flags[i] = ','.join(dist.flags)
    if not dist.ok: return
    cs = dist.components
    if len(cs) == 1 and cs[0].kind == 'point':
        col.kind[i] = 1; col.a[i] = cs[0].p['value']; return
    if len(cs) == 1 and cs[0].kind == 'uniform':
        col.kind[i] = 2; col.a[i] = cs[0].p['lower']; col.b[i] = cs[0].p['upper']; return
    if len(cs) == 1 and cs[0].kind == 'triangular':
        col.kind[i] = 3; col.a[i] = cs[0].p['lower']; col.b[i] = cs[0].p['mode']; col.c[i] = cs[0].p['upper']; return
    if split:
        wi = [c for c in cs if c.label in split]; me = [c for c in cs if c.label not in split]
        pw = sum(c.weight for c in wi) / sum(c.weight for c in cs)
        col.kind[i] = 5; col.a[i] = pw
        col.t1[i] = len(tables); tables.append(breakpoints(me) if me else breakpoints(wi))
        col.t2[i] = len(tables); tables.append(breakpoints(wi) if wi else breakpoints(me))
        return
    bp = breakpoints(cs)
    if bp is None: raise ValueError(f'cannot cache {dist.variable}={dist.recorded_value}: {[c.kind for c in cs]}')
    col.kind[i] = 4; col.t1[i] = len(tables); tables.append(bp)


def collect(reg, clinical, lab, log=print):
    """Exact posteriors for the whole dataset: per-variable collectors + list of breakpoint tables."""
    tables, cols = [], {}
    rows = clinical.to_dict('records'); n = len(rows)
    skip = {'bmi'} | set(INTERVALS)
    for v, e in reg.vars.items():
        if e['source'] == 'lab' or v in skip or e['method'] == 'not_applicable': continue
        if e['source'] == 'clinical' and v not in clinical.columns: continue
        t0 = time.time(); col = _Collector(n)
        split = engine.WRITTEN_IN_MODES.get(v) if v in ('height', 'weight') else None
        for i, r in enumerate(rows):
            _classify(reg.case_distribution(r, v), col, i, tables, split)
        cols[v] = col
        log(f'  {v:22s} {time.time() - t0:6.1f} s, tables so far {len(tables)}')
    # lab: vectorised for the simple analytes, API for the few mixtures
    t0 = time.time(); m = len(lab); col = _Collector(m)
    name = lab['name'].values; res = lab['result'].values.astype(float)
    col.status[:] = 'ok'; col.flags[:] = ''
    miss = ~np.isfinite(res); col.status[miss] = 'missing'
    ph = set(map(tuple, lab.loc[lab['name'] == 'ph', ['caseid', 'dt']].values.tolist()))
    n_at = lab.groupby(['caseid', 'dt']).size()
    cnt = n_at.reindex(list(map(tuple, lab[['caseid', 'dt']].values.tolist()))).values
    for nm in np.unique(name):
        e = reg.entry(nm); idx = np.where((name == nm) & ~miss)[0]; v = res[idx]
        if e['method'] == 'uniform':
            r = e['resolution']; col.kind[idx] = 2; col.a[idx] = v - r / 2; col.b[idx] = v + r / 2
            off = np.abs(v / r - np.round(v / r)) > 1e-6; col.flags[idx[off]] = 'off_grid'
        elif e['method'] == 'neighbours':
            for j, x in zip(idx, v):
                _classify(reg.distribution(nm, x), col, j, tables)
        elif nm == 'hct':
            keys = list(map(tuple, lab.iloc[idx][['caseid', 'dt']].values.tolist()))
            for j, x, k in zip(idx, v, keys):
                if cnt[j] == 1:
                    d = reg.distribution(nm, x); d.flags.append('stand_alone_probable_body_fluid')
                else:
                    d = reg.distribution(nm, x, source='blood_gas' if k in ph else 'cbc')
                _classify(d, col, j, tables)
        else:
            raise ValueError(f'lab analyte {nm}: method {e["method"]} not handled')
    cols['lab'] = col
    log(f'  lab                    {time.time() - t0:6.1f} s, tables {len(tables)}')
    return cols, tables


def write(cols, tables, clinical, lab, reg, out, dtype='int16', knots=1024):
    """Write a cache folder from collected columns and tables.

    Args:
        cols: Per-variable columns from :func:`collect`.
        tables: Posterior tables from :func:`collect`.
        clinical: The ``clinical_data.csv`` DataFrame.
        lab: The ``lab_data.csv`` DataFrame.
        reg: The registry.
        out: Output folder.
        dtype: Storage type for the knots: ``"int16"`` (default), ``"float32"`` or ``"float64"``.
        knots: Quantile knots per fitted posterior; sets the precision (see the README's table).

    Returns:
        A summary of what was written.
    """
    os.makedirs(out, exist_ok=True)
    Q = tables_matrix(tables, knots)
    q, lo, scale = quantize(Q, dtype); del Q
    np.save(os.path.join(out, 'tables.npy'), q)
    if lo is not None:
        np.save(os.path.join(out, 'tables_lo.npy'), lo); np.save(os.path.join(out, 'tables_scale.npy'), scale)
    meta = {'version': 2, 'dtype': dtype, 'knots': knots, 'knot_layout': 'mid-levels, mean-preserving ends', 'n_tables': len(tables), 'n_cases': len(clinical),
            'n_lab': len(lab), 'variables': sorted(cols), 'built': time.strftime('%Y-%m-%d %H:%M'),
            'registry_built': reg.reg.get('built')}
    statuses = sorted({str(x) for c in cols.values() for x in c.status})
    flagsets = sorted({str(x) for c in cols.values() for x in c.flags})
    meta['status_codes'] = statuses; meta['flag_codes'] = flagsets
    si = {k: i for i, k in enumerate(statuses)}; fi = {k: i for i, k in enumerate(flagsets)}
    for v, c in cols.items():
        np.savez(os.path.join(out, f'var_{v}.npz'), kind=c.kind, a=c.a, b=c.b, c=c.c,
                 t1=c.t1.astype(np.int32), t2=c.t2.astype(np.int32),
                 status=np.array([si[str(x)] for x in c.status], np.int8),
                 flags=np.array([fi[str(x)] for x in c.flags], np.int16))
    # case-level inputs the sampler needs: covariates for variability, and ids
    np.savez(os.path.join(out, 'case.npz'), caseid=clinical['caseid'].values,
             sex=np.asarray(clinical['sex'].astype(str).tolist(), dtype=str),
             age=np.array([float(str(a).lstrip('>')) + (1 if str(a).startswith('>') else 0)
                           if not _missing(a) else np.nan for a in clinical['age']]))
    np.savez(os.path.join(out, 'lab.npz'), caseid=lab['caseid'].values, dt=lab['dt'].values,
             name=np.asarray(lab['name'].astype(str).tolist(), dtype=str))
    json.dump(meta, open(os.path.join(out, 'meta.json'), 'w'), indent=1)
    return meta


def build(clinical_csv, lab_csv, out, dtype='int16', knots=1024, log=print):
    """Build the whole-dataset Monte Carlo cache from VitalDB's CSV files.

    Also available as ``python -m vitaldb_resolution.cache build``. About 2 minutes and 214 MB with the
    defaults. The cache holds distributions derived from the data; keep it with the data, not in a repository.

    Args:
        clinical_csv: Path to ``clinical_data.csv``.
        lab_csv: Path to ``lab_data.csv``.
        out: Output folder.
        dtype: Storage type for the knots.
        knots: Quantile knots per fitted posterior.
        log: Progress printer.

    Returns:
        A summary of what was written.
    """
    import pandas as pd
    reg = Registry.load()
    clinical = pd.read_csv(clinical_csv, encoding='utf-8-sig'); lab = pd.read_csv(lab_csv)
    cols, tables = collect(reg, clinical, lab, log)
    return write(cols, tables, clinical, lab, reg, out, dtype, knots)


# ---------------------------------------------------------------- loading and drawing
class Cache:
    """A whole-dataset cache of every record's distribution, for fast Monte Carlo draws.

    Load with :meth:`Cache.load`; draw with :meth:`sampler`.

    Args:
        path: The cache folder written by :func:`build`.
        mmap: Memory-map the tables instead of reading them into memory.
    """
    def __init__(self, path, mmap=False):
        self.path = path; self.meta = json.load(open(os.path.join(path, 'meta.json')))
        mm = 'r' if mmap else None
        self.q = np.load(os.path.join(path, 'tables.npy'), mmap_mode=mm)
        lp = os.path.join(path, 'tables_lo.npy')
        self.lo = np.load(lp) if os.path.exists(lp) else None
        self.scale = np.load(os.path.join(path, 'tables_scale.npy')) if self.lo is not None else None
        self.vars = {v: dict(np.load(os.path.join(path, f'var_{v}.npz'), allow_pickle=False))
                     for v in self.meta['variables']}
        self.case = dict(np.load(os.path.join(path, 'case.npz'), allow_pickle=False))
        self.lab = dict(np.load(os.path.join(path, 'lab.npz'), allow_pickle=False))
        self.reg = Registry.load()

    @classmethod
    def load(cls, path, mmap=False):
        """Load a cache folder.

        Args:
            path: The folder written by :func:`build`.
            mmap: Memory-map the tables.

        Returns:
            A :class:`Cache`.
        """
        return cls(path, mmap)

    def nbytes(self):
        """Memory held by the cache's arrays (bytes)."""
        n = self.q.nbytes + (0 if self.lo is None else self.lo.nbytes + self.scale.nbytes)
        n += sum(a.nbytes for d in self.vars.values() for a in d.values())
        n += sum(a.nbytes for a in self.case.values()) + sum(a.nbytes for a in self.lab.values())
        return n

    def status(self, variable):
        """Status string per record (ok, missing, imputed, ...); draws are NaN where it is not ok."""
        return np.array(self.meta['status_codes'])[self.vars[variable]['status']]

    def flags(self, variable):
        """Flags per record of a variable (comma-separated codes; empty when none).

        Args:
            variable: Variable name.
        """
        return np.array(self.meta['flag_codes'])[self.vars[variable]['flags']]

    def sampler(self, seed=None, variability=False):
        """A sampler that draws one value per record per call.

        Args:
            seed: Random seed.
            variability: Add the measurement variability of written-in heights and weights at draw time.

        Returns:
            A :class:`Sampler`.
        """
        return Sampler(self, np.random.default_rng(seed), variability)


class Sampler:
    """Draws true values for every record at once; create it with :meth:`Cache.sampler`.

    Args:
        cache: The cache.
        rng: A ``numpy.random.Generator``.
        variability: Add height and weight measurement variability.
    """
    def __init__(self, cache, rng, variability):
        self.c, self.rng, self.variability = cache, rng, variability
        self._prep_variability()

    def _prep_variability(self):
        reg = self.c.reg; sex = self.c.case['sex']; age = self.c.case['age']
        self.mv = {}
        for v in ('height', 'weight'):
            if v not in self.c.vars: continue
            mv = reg.spec(f'{v}.model.json')['measurement_variability']; wi = mv['written_in']
            b = wi.get('bias_self_minus_measured', 0)
            bias = np.zeros(len(sex))
            for i, (s, a) in enumerate(zip(sex, age)):
                x = b
                if isinstance(x, dict): x = x.get(s, 0.0)
                if isinstance(x, dict): x = x[engine._age_band(a if np.isfinite(a) else 50)]
                bias[i] = x
            self.mv[v] = {'measured': mv['measured'], 'wi_sd': wi['sd'], 'bias': bias}

    def _draw_var(self, d, u=None):
        k = d['kind']; n = len(k); rng = self.rng
        u = rng.random(n) if u is None else u
        out = np.full(n, np.nan)
        m = k == 1; out[m] = d['a'][m]
        m = k == 2; out[m] = d['a'][m] + u[m] * (d['b'][m] - d['a'][m])
        m = k == 3
        if m.any():
            a, c, b = d['a'][m], d['b'][m], d['c'][m]; uu = u[m]; fc = np.where(b > a, (c - a) / (b - a), 0.5)
            out[m] = np.where(uu < fc, a + np.sqrt(uu * (b - a) * (c - a)), b - np.sqrt((1 - uu) * (b - a) * (b - c)))
        m = k == 4
        if m.any(): out[m] = table_draw(self.c.q, self.c.lo, self.c.scale, d['t1'][m], u[m])
        m = k == 5
        written = np.zeros(n, bool)
        if m.any():
            w = rng.random(n) < np.nan_to_num(d['a']); written = m & w
            rows = np.where(written, d['t2'], d['t1'])
            out[m] = table_draw(self.c.q, self.c.lo, self.c.scale, rows[m], u[m])
        return out, written

    def draw(self, variables=None, include_lab=True, u=None):
        """One Monte Carlo iteration: a draw of the true value of every cached record.

        Records whose status is not ok are NaN. ``u`` optionally maps variable -> uniforms (common random
        numbers).
        """
        rng = self.rng; res = {}
        names = variables or [v for v in self.c.vars if v != 'lab']
        for v in names:
            x, written = self._draw_var(self.c.vars[v], None if u is None else u.get(v))
            if self.variability and v in self.mv:
                p = self.mv[v]; me = p['measured']; n = len(x)
                sd_me = me['sd'] if 'sd' in me else me['sd_relative'] * np.abs(x)
                noise = np.where(written, -p['bias'] + rng.normal(0, p['wi_sd'], n), rng.normal(0, 1, n) * sd_me)
                x = x + noise
            res[v] = x
        if variables is None:
            if 'height' in res and 'weight' in res: res['bmi'] = res['weight'] / (res['height'] / 100) ** 2
            for k, (a, b) in INTERVALS.items():
                if a in res and b in res: res[k] = (res[b] - res[a]) / 60
        if include_lab and 'lab' in self.c.vars:
            res['lab'], _ = self._draw_var(self.c.vars['lab'], None if u is None else u.get('lab'))
        return res


# ---------------------------------------------------------------- CLI
if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='Build the whole-dataset posterior cache')
    sp = ap.add_subparsers(dest='cmd', required=True)
    b = sp.add_parser('build')
    b.add_argument('--clinical', required=True); b.add_argument('--lab', required=True)
    b.add_argument('--out', required=True)
    b.add_argument('--dtype', default='int16', choices=['int16', 'float32', 'float64'])
    b.add_argument('--knots', type=int, default=1024)
    a = ap.parse_args()
    meta = build(a.clinical, a.lab, a.out, a.dtype, a.knots)
    print(json.dumps(meta, indent=1))
