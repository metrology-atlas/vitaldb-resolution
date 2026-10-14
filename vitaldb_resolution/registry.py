"""Registry + API: variable, recorded value and covariates -> Distribution of the true value."""
import json
import math
import os
from functools import cache

import numpy as np

from . import engine
from .distribution import Component, Distribution, uniform_interval

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
SENTINEL = -1e8          # 1900-01-01 placeholders sit far below this (seconds from casestart)


def _missing(v):
    if v is None: return True
    if isinstance(v, str): return v.strip() == ''
    try: return bool(np.isnan(v))
    except TypeError: return False


def _on_grid(v, r):
    return abs(v / r - round(v / r)) < 1e-6


def _ref_window(rs_clock_min):
    rc = rs_clock_min % 1440; h = math.floor(rc / 60)
    if 390 <= rc < 540: return 'first_case'
    return 'evening_night' if (h >= 19 or h < 6) else 'day'


def _hms(clock_min):
    s = int(round((clock_min % 1440) * 60)); return f'{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}'


class Registry:
    """The variable registry and the API: variable, recorded value and covariates to a distribution.

    Each of the 114 entries says how one VitalDB variable was recorded (method, resolution, anchor, model
    file, required covariates, rules and evidence). Create it with :func:`load` or :meth:`Registry.load`.

    Args:
        registry: The parsed ``registry.json``.
        models_dir: Folder holding the ``*.model.json`` files.
    """
    def __init__(self, registry, models_dir):
        self.reg = registry
        self.vars = registry['variables']
        self.models_dir = models_dir
        self.rules = json.load(open(os.path.join(DATA, 'exclusion_rules.json')))

    @classmethod
    def load(cls, path=None):
        """Load a registry and its models.

        Args:
            path: Path to a ``registry.json``. Default: the one shipped with the package; its models are read
                from the ``models`` folder beside it.

        Returns:
            A :class:`Registry`.
        """
        path = path or os.path.join(DATA, 'registry.json')
        return cls(json.load(open(path)), os.path.join(os.path.dirname(path), 'models'))

    # ------------------------------------------------------------------ lookup
    def entry(self, variable):
        """The registry entry of one variable.

        Args:
            variable: Variable name, e.g. ``"intraop_uo"`` or ``"lab_dt"``.

        Returns:
            The entry: method, resolution, anchor, model file, covariates, rules, evidence.

        Raises:
            KeyError: The variable is not in the registry.
        """
        if variable not in self.vars: raise KeyError(f'{variable!r} is not in the registry')
        return self.vars[variable]

    def variables(self, method=None):
        """Variable names, optionally only those of one method.

        Args:
            method: E.g. ``"model"``, ``"timestamp_model"``, ``"uniform"`` or ``"derived"``.

        Returns:
            The names, in registry order.
        """
        return [k for k, e in self.vars.items() if method is None or e['method'] == method]

    @cache
    def spec(self, filename):
        """The parsed JSON of one model file (cached).

        Args:
            filename: Model file name, e.g. ``"intraop_uo.model.json"``.
        """
        return json.load(open(os.path.join(self.models_dir, filename)))

    @cache
    def model(self, filename):
        """The evaluator for one model file (cached), from :func:`~vitaldb_resolution.engine.load_model`.

        Args:
            filename: Model file name.
        """
        return engine.load_model(self.spec(filename))

    # ------------------------------------------------------------------ main entry point
    def distribution(self, variable, value=None, variability=False, n_samples=20000, rng=None, **cov):
        """Distribution of the true value of a variable, given its recorded value.

        Args:
            variable: Variable name, as in the registry.
            value: The recorded value. Derived variables (BMI, intervals) take their inputs as keywords instead.
            variability: For height and weight, add the measurement variability of a written-in value.
            n_samples: Draws used for sampled posteriors (derived variables, variability).
            rng: Random generator for those draws. Default: seeded with 0, so results repeat.
            **cov: Covariates the entry lists in ``required_covariates`` (e.g. ``sex``, ``age``, ``emop``,
                ``department``, ``adm``, ``caseend``), or a derived variable's inputs.

        Returns:
            A :class:`~vitaldb_resolution.Distribution`. ``status`` is ``"ok"`` with components, or says why
            there is none: ``"missing"``, ``"invalid"``, ``"imputed"``, ``"not_applicable"``,
            ``"zero_or_not_recorded"``, ``"top_coded"`` or ``"error"`` (missing covariates, in ``note``).

        Raises:
            KeyError: The variable is not in the registry.

        Example:
            >>> reg = load()
            >>> d = reg.distribution("intraop_uo", 100)
            >>> d.status, d.kind
            ('ok', 'heaped')
        """
        e = self.entry(variable)
        m = e['method']
        if isinstance(value, np.generic): value = value.item()
        D = lambda **kw: Distribution(variable, value, e.get('unit', ''), **kw)
        if m == 'not_applicable':
            return D(status='not_applicable', kind='categorical')
        if m != 'derived' and _missing(value):
            return D(status='missing')
        cov_flags = []
        uses_age = 'age' in e.get('required_covariates', []) or variable == 'bmi'
        if uses_age and 'age' in cov and isinstance(cov['age'], str):
            a = cov['age'].strip()
            if a.startswith('>'): cov = {**cov, 'age': float(a[1:]) + 1}; cov_flags.append('age_covariate_top_coded')
            else:
                try: cov = {**cov, 'age': float(a)}
                except ValueError: cov = {**cov, 'age': None}
        missing_cov = [c for c in e.get('required_covariates', []) if c not in cov or _missing(cov[c])]
        if missing_cov and m not in ('derived',):
            return D(status='error', note=f'missing covariates: {missing_cov}')
        h = getattr(self, '_h_' + m)
        out = h(e, value, D, cov=cov, variability=variability, n=n_samples,
                rng=np.random.default_rng(0) if rng is None else rng)
        out.flags += [f for f in cov_flags if f not in out.flags]
        return out

    # ------------------------------------------------------------------ handlers
    def _h_not_applicable(self, e, v, D, **_):
        return D(status='not_applicable', kind='categorical')

    def _h_exact(self, e, v, D, cov, **_):
        v = float(v); flags = []
        if v == 0 and e.get('zero_status'): return D(status=e['zero_status'], kind='exact')
        for r in e.get('flag_rules', []):
            if r['test'] == 'value >= 50 and intraop_ebl < 1000' and v >= 50:
                ebl = cov.get('intraop_ebl')
                if _missing(ebl) or float(ebl) < 1000: flags.append(r['flag'])
        if e.get('unit_dose') and not _on_grid(v, e['unit_dose']): flags.append('off_unit_dose_grid')
        return D(kind='exact', components=[Component('point', label='exact', value=v)], flags=flags)

    def _uniform(self, e, v, D, res, anchor, kind='single_resolution', flags=(), label=None, extra=None):
        lo, hi, closed = uniform_interval(float(v), res, anchor)
        return D(kind=kind, flags=list(flags), extra=extra,
                 components=[Component('uniform', label=label or f'{res:g} {e.get("unit", "")}'.strip(),
                                       lower=lo, upper=hi, resolution=res, anchor=anchor, closed=closed)])

    def _h_uniform(self, e, v, D, **_):
        v = float(v); flags = []
        if e['type'] == 'date' and v < SENTINEL: return D(status='invalid', note='1900-01-01 placeholder')
        if v == 0 and e.get('zero_status'): return D(status=e['zero_status'])
        r = e['resolution']
        if e['type'] not in ('date', 'device_time') and not _on_grid(v, r): flags.append('off_grid')
        return self._uniform(e, v, D, r, e['anchor'], flags=flags)

    def _h_source_dependent(self, e, v, D, cov, **_):
        v = float(v); src = cov.get(e['covariate']); rs = e['resolution_by_source']
        if src in rs:
            return self._uniform(e, v, D, rs[src], e['anchor'], label=src, extra={'source': src})
        if not _on_grid(v, 1):          # a decimal digit only comes from the 0.1 % analyser
            return self._uniform(e, v, D, rs['cbc'], e['anchor'], label='cbc', extra={'source': 'cbc (inferred)'})
        pr = e['unknown_source_prior']['whole_number']
        comps = []
        for s, w in pr.items():
            lo, hi, closed = uniform_interval(v, rs[s], e['anchor'])
            comps.append(Component('uniform', weight=w, label=s, lower=lo, upper=hi, resolution=rs[s],
                                   anchor=e['anchor'], closed=closed))
        return D(kind='mixed_source', components=comps, flags=['source_unknown'],
                 note='source not given; mixture weighted by the share of whole-number results from each analyser')

    def _h_decimal_dependent(self, e, v, D, **_):
        v = float(v); whole = _on_grid(v, 1)
        r = e['resolution_if_whole'] if whole else e['resolution_if_decimal']
        return self._uniform(e, v, D, r, e['anchor'], extra={'decided_by': 'whole number' if whole else 'decimal'})

    def _h_neighbours(self, e, v, D, **_):
        v = float(v); rv = np.asarray(e['reachable_values'])
        i = int(np.argmin(np.abs(rv - v)))
        if abs(rv[i] - v) > 1e-6:
            return self._uniform(e, v, D, 1, 'nearest', flags=['off_grid', 'not_a_reachable_value'])
        left = rv[i] - rv[i - 1] if i > 0 else rv[i + 1] - rv[i]
        right = rv[i + 1] - rv[i] if i < len(rv) - 1 else left
        return D(kind='computed_value', components=[Component('uniform', label='neighbour midpoints',
                 lower=v - left / 2, upper=v + right / 2, anchor='nearest', closed='lower')])

    def _h_triangular_count(self, e, v, D, **_):
        n = float(v)
        if n == 0: return D(kind='exact', components=[Component('point', label='no ICU stay', value=0.0)])
        w = e.get('half_width', 1)
        return D(kind='count_uncertain', flags=['counting_rule_unknown'],
                 components=[Component('triangular', label=f'n ± {w}', lower=n - w, mode=n, upper=n + w)])

    def _h_age(self, e, v, D, **_):
        if isinstance(v, str) and v.strip() == e['top_code']['text']:
            return D(status='top_coded', kind='top_coded', extra={'lower': e['top_code']['lower'], 'upper': None},
                     note='Top-coded: true age is at least 90; no upper bound, so no distribution')
        v = float(v)
        r = e['infant_resolution'] if (v < 1 and not _on_grid(v, 1)) else e['resolution']
        return self._uniform(e, v, D, r, e['anchor'], label='completed years' if r == 1 else 'tenths of a year')

    def _h_colloid_rules(self, e, v, D, cov, **_):
        v = float(v); op = str(cov.get('opname', '')).lower()
        if v == 0: return D(status=e['zero_status'])
        if ('kidney' in op or 'nephrectomy' in op) and v <= 50:
            return self._uniform(e, v, D, 1, 'nearest', flags=['unit_uncertain'], label='native 1 mL')
        if 'liver transplant' in op and _on_grid(v, 100):
            return D(kind='exact', components=[Component('point', label='100 mL vials', value=v)])
        if _on_grid(v, 500):
            return D(kind='exact', components=[Component('point', label='500 mL bags', value=v)])
        if 100 <= v <= 400 and _on_grid(v, 50) and 'liver transplant' not in op:
            return self._uniform(e, v, D, 100, 'nearest', flags=['possibly_albumin'], label='estimated to 100 mL')
        return self._uniform(e, v, D, 1, 'nearest', label='native 1 mL')

    # value models -----------------------------------------------------------------------------
    def _h_model(self, e, v, D, cov, variability, n, rng, **_):
        spec = self.spec(e['model']); M = self.model(e['model'])
        v = float(v); flags = []
        if v == 0 and (e.get('zero_status') or spec.get('truncation', {}).get('exclude_recorded_zero')):
            return D(status=e.get('zero_status', 'special_value'), note='0 is not modelled (special value)')
        res = spec['resolution']['effective']
        pop = spec.get('population', {})
        if pop.get('children') and not _missing(cov.get('age')) and float(cov['age']) < 18:
            return self._uniform(e, v, D, spec.get('record_unit', res), 'nearest', flags=['child_native_resolution'],
                                 label='native resolution (model fitted to adults)')
        if not _on_grid(v, res):
            flags.append('off_grid'); v = round(v / res) * res
        c = {k: cov[k] for k in spec.get('required_covariates', []) if k in cov} if spec.get('required_covariates') else None
        w = M.latent(spec['latent'].get('coefficients'), c, spec)
        comps, written = [], 0.0
        for lab, rec, take in M.components(spec, c):
            wm = w * take * (np.round(rec / M.ru) == round(v / M.ru))
            if wm.sum() > 0:
                comps.append(Component('tabulated', weight=wm.sum(), label=lab, x=M.X, p=wm, dx=M.dx))
                if lab in engine.WRITTEN_IN_MODES.get(spec['variable'], set()): written += wm.sum()
        if not comps:
            d = self._uniform(e, v, D, res, 'nearest', flags=flags + ['outside_model_support'])
            d.note = 'value outside the fitted model support; uniform at the working resolution'; return d
        tot = sum(cc.weight for cc in comps)
        extra = {'possibly_written_in': round(float(written / tot), 4)} if spec.get('flags', {}).get('possibly_written_in') else None
        model = {'id': spec['variable'], 'version': spec.get('schema_version'), 'file': e['model'],
                 'fit_date': spec.get('fit', {}).get('date'), 'latent': spec['latent']['family']}
        kind = 'heaped' if len(spec['allocation']) > 1 else 'single_resolution'
        if variability and spec.get('measurement_variability'):
            s, _ = engine.sample_posterior(M, v, c, n, rng, variability=True)
            return D(kind=kind + '+measurement_variability', model=model, flags=flags, extra=extra,
                     components=[Component('samples', label='posterior with measurement variability', samples=s)])
        return D(kind=kind, components=comps, model=model, flags=flags, extra=extra)

    # timestamps -------------------------------------------------------------------------------
    def timestamp_context(self, variable, t, cov):
        """Reconstruct the clock time around a charted event.

        Args:
            variable: A timestamp variable, e.g. ``"opstart"``.
            t: The recorded time (s from the recording start).
            cov: The case's covariates, including ``adm`` (admission, whole days before the start).

        Returns:
            ``recording_start_clock`` (clock minutes, exact), ``ref`` (the reference the model uses: recording
            start or end, clock minutes), ``M`` (the event's clock minute) and ``offset_mismatch``.
        """
        adm = float(cov['adm']); t = float(t)
        clock = lambda x: (((-adm + math.ceil(x / 60 - 1e-9) * 60) % 86400) / 60)
        M = round(clock(t))
        # Recording start clock (exact): charted events are whole clock minutes, so t fixes the sub-minute offset.
        # Anaesthesia start is the anchor used in the fits; fall back to the event itself when it is absent.
        Rs = M - t / 60          # as in the fits: the event's own record fixes the sub-minute offset
        a = cov.get('anestart')
        t0 = float(a) if (a is not None and not _missing(a) and float(a) > SENTINEL) else t
        mismatch = abs(((t - t0) % 60 + 30) % 60 - 30) > 1e-3     # other charted times disagree on the offset
        ref = Rs if variable in ('anestart', 'opstart') else Rs + float(cov['caseend']) / 60
        while M - ref > 720: M -= 1440
        while ref - M > 720: M += 1440
        return {'recording_start_clock': Rs, 'ref': ref, 'M': int(M), 'offset_mismatch': mismatch}

    def _h_timestamp_model(self, e, v, D, cov, **_):
        var = e['variable']; t = float(v); flags = []
        if t < SENTINEL: return D(status='invalid', note='1900 placeholder')
        if float(cov['adm']) < SENTINEL: return D(status='invalid', note='admission date is a 1900 placeholder')
        if var == 'opend' and 598 <= float(cov['caseend']) - t <= 604:
            return D(status='imputed', note='op end set about 601 s before recording end')
        if var == 'aneend' and abs(t - round(t)) > 1e-9:
            return D(status='imputed', note='anaesthesia end derived from an imputed op end (not a whole second)')
        if float(cov['adm']) % 60 != 0: flags.append('clock_uncertain')
        if var in ('opend', 'aneend'):
            pair = {**{k: cov.get(k) for k in ('opend', 'aneend')}, var: t}
            if not _missing(pair['opend']) and not _missing(pair['aneend']) and float(pair['aneend']) > SENTINEL:
                if float(pair['opend']) > float(pair['aneend']): flags.append('order_violation')
                elif float(pair['opend']) == float(pair['aneend']): flags.append('same_time')
        spec0 = self.spec(e['model'])
        base = 0.0 if var in ('anestart', 'opstart') else float(cov['caseend'])
        G = (base - t) / 60 if spec0['latent']['direction'] == 'before' else (t - base) / 60
        lo, hi = spec0['grid']['lo'], spec0['grid']['hi']
        if not (lo <= G < hi):
            hint = next((m for h, m in ((12, '12-hour clock error?'), (24, 'date error, +24 h?')) if lo <= abs(G) - h * 60 < hi), None)
            return D(status='invalid', flags=flags, extra={'offset_min': round(G, 2), 'model_range_min': [lo, hi]},
                     note=f'{G:.0f} min from the {spec0["reference"]["name"]}, outside the model range'
                          + (f' ({hint})' if hint else ''))
        ctx = self.timestamp_context(var, t, cov)
        if ctx['offset_mismatch']: flags.append('clock_offset_mismatch')
        TM = self.model(e['model']); spec = self.spec(e['model'])
        c = {'ref': ctx['ref']}
        if var == 'anestart': c.update(ref_window=_ref_window(ctx['recording_start_clock']), emop=int(cov['emop']))
        X, w = TM._weights(c); X, comps = TM.components(c)
        out = []
        for lab, rec, take in comps:
            wm = w * take * (np.round(rec) == ctx['M'])
            if wm.sum() > 0:
                out.append(Component('tabulated', weight=wm.sum(), label=lab,
                                     x=(X - ctx['recording_start_clock']) * 60, p=wm, dx=TM.dx * 60))
        if not out:
            return D(status='invalid', note='recorded minute has zero probability under the model', flags=flags)
        d = D(kind='heaped', components=out, flags=flags,
              model={'id': var, 'version': spec.get('schema_version'), 'file': e['model'],
                     'fit_date': spec.get('fit', {}).get('date'), 'latent': spec['latent']['family']})
        mu = d.mean()
        d.extra = {'recorded_clock': _hms(ctx['M']), 'mean_clock': _hms(ctx['recording_start_clock'] + mu / 60),
                   'reference': spec['reference']['name'], 'ref_clock': _hms(ctx['ref']),
                   **({'ref_window': c['ref_window']} if 'ref_window' in c else {})}
        return d

    # derived ----------------------------------------------------------------------------------
    def _h_derived(self, e, v, D, cov, variability, n, rng, **_):
        var = e['variable']
        if var == 'bmi':
            need = ['height', 'weight', 'sex', 'age', 'emop', 'department']
            miss = [k for k in need if _missing(cov.get(k))]
            if miss: return D(status='error', note=f'missing inputs/covariates: {miss}')
            h = self.distribution('height', cov['height'], **cov)
            wt = self.distribution('weight', cov['weight'], **cov)
            for x in (h, wt):
                if not x.ok: return D(status=x.status, note=f'{x.variable}: {x.note}')
            sh = h.sample(n, rng) if not variability else self.distribution('height', cov['height'], variability=True, rng=rng, n_samples=n, **cov).sample(n, rng)
            sw = wt.sample(n, rng) if not variability else self.distribution('weight', cov['weight'], variability=True, rng=rng, n_samples=n, **cov).sample(n, rng)
            s = sw / (sh / 100) ** 2
            flags = []
            if not _missing(v):
                calc = round(float(cov['weight']) / (float(cov['height']) / 100) ** 2, 1)
                if abs(calc - float(v)) > 0.051: flags.append('recorded_bmi_mismatch')
            return D(kind='derived', flags=flags, extra={'inputs': ['height', 'weight']},
                     components=[Component('samples', label='weight / (height/100)^2', samples=s)])
        if var == 'los':
            if _missing(cov.get('adm')) or _missing(cov.get('dis')): return D(status='missing')
            adm, dis = float(cov['adm']), float(cov['dis'])
            if adm < SENTINEL or dis < SENTINEL: return D(status='invalid', note='1900 placeholder date')
            k = round((dis - adm) / 86400)
            return Distribution(var, k if _missing(v) else v, 'days', kind='derived',
                                components=[Component('triangular', label='n ± 1 day', lower=max(k - 1, 0),
                                                      mode=k, upper=k + 1)])
        a, b = e['inputs']
        if _missing(cov.get(a)) or _missing(cov.get(b)): return D(status='missing')
        da = self.distribution(a, cov[a], **{k: x for k, x in cov.items() if k != a})
        db = self.distribution(b, cov[b], **{k: x for k, x in cov.items() if k != b})
        for x in (da, db):
            if not x.ok: return D(status=x.status, note=f'{x.variable}: {x.note}', flags=x.flags)
        s = (db.sample(n, rng) - da.sample(n, rng)) / 60
        charted = (float(cov[b]) - float(cov[a])) / 60
        return Distribution(var, charted if _missing(v) else v, 'min', kind='derived',
                            flags=sorted(set(da.flags + db.flags)), extra={'charted': charted, 'inputs': [a, b]},
                            components=[Component('samples', label=e['formula'], samples=s)])

    # ------------------------------------------------------------------ convenience
    def case_distribution(self, row, variable, **kw):
        """Distribution of one variable for one ``clinical_data.csv`` row.

        Covariates are taken from the row; derived variables (``bmi``, ``op_duration``, ...) take their
        inputs from it too.

        Args:
            row: A row as a dict or a pandas Series.
            variable: Variable name.
            **kw: Passed to :meth:`distribution`, e.g. ``variability=True``.

        Returns:
            A :class:`~vitaldb_resolution.Distribution`. For timestamps, the clock time is in ``extra``.
        """
        row = dict(row)
        self.entry(variable)                 # raises KeyError for an unknown variable
        value = row.get(variable)
        return self.distribution(variable, value, **kw, **{k: x for k, x in row.items() if k != variable})

    def distributions(self, variable, rows, **kw):
        """One distribution per row of ``clinical_data.csv``.

        Args:
            variable: Variable name.
            rows: A DataFrame or a list of dicts.
            **kw: Passed to :meth:`distribution`.

        Returns:
            A list of :class:`~vitaldb_resolution.Distribution`, in row order.
        """
        it = rows.to_dict('records') if hasattr(rows, 'to_dict') else rows
        return [self.case_distribution(r, variable, **kw) for r in it]

    def lab_distributions(self, lab):
        """One distribution per row of ``lab_data.csv``.

        The haematocrit's analyser is inferred from the panel: blood gas if a pH result shares the timestamp,
        otherwise the CBC; a haematocrit alone is flagged ``source_unknown`` and gets a mixture of both.

        Args:
            lab: A DataFrame with columns ``caseid``, ``dt``, ``name`` and ``result``.

        Returns:
            A list of :class:`~vitaldb_resolution.Distribution` for the result values, in row order. These are
            distributions of the values, not of when they became known: see the ``lab_dt`` entry.
        """
        ph = set(map(tuple, lab.loc[lab['name'] == 'ph', ['caseid', 'dt']].values.tolist()))
        n_at = lab.groupby(['caseid', 'dt']).size().to_dict()
        out = []
        for c, t, nm, r in lab[['caseid', 'dt', 'name', 'result']].itertuples(index=False):
            if nm == 'hct' and n_at.get((c, t), 0) == 1:          # stand-alone: probably a body-fluid sample
                d = self.distribution(nm, r); d.flags.append('stand_alone_probable_body_fluid')
            else:
                d = self.distribution(nm, r, **({'source': 'blood_gas' if (c, t) in ph else 'cbc'} if nm == 'hct' else {}))
            out.append(d)
        return out

    @staticmethod
    def summary_frame(dists):
        """Summarise distributions as a table (needs pandas).

        Args:
            dists: Distributions, e.g. from :meth:`distributions`.

        Returns:
            A DataFrame with ``variable``, ``recorded``, ``status``, ``kind``, ``mean``, ``sd``, ``lo95``,
            ``hi95`` and ``flags``; NaN where there is no distribution.
        """
        import pandas as pd
        rows = []
        for d in dists:
            s = d.summary()
            rows.append({'variable': d.variable, 'recorded': d.recorded_value, 'status': d.status, 'kind': d.kind,
                         'mean': s['mean'] if s else np.nan, 'sd': s['sd'] if s else np.nan,
                         'lo95': s['ci95'][0] if s else np.nan, 'hi95': s['ci95'][1] if s else np.nan,
                         'flags': ','.join(d.flags)})
        return pd.DataFrame(rows)


def load(path=None):
    """Load the registry shipped with the package (or another one).

    Args:
        path: Path to a ``registry.json``. Default: the packaged one.

    Returns:
        A :class:`Registry`.
    """
    return Registry.load(path)
