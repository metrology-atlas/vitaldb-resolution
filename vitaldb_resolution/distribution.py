"""Distribution of a true value given a recorded one: a weighted mixture of simple components."""
import math

import numpy as np


class Component:
    """One way the recorded value could have arisen: a rounding mode, a source, a fitted posterior.

    Args:
        kind: The density: ``"uniform"`` (``lower``, ``upper``), ``"point"`` (``value``), ``"triangular"``
            (``lower``, ``mode``, ``upper``), ``"tabulated"`` (``x`` cell centres, ``p`` masses, ``dx`` cell
            width) or ``"samples"`` (``samples``, draws).
        weight: Mixture weight; :class:`Distribution` normalises the weights.
        label: Name of this component, e.g. the rounding mode (``"5 mL"``) or the analyser.
        **kw: The density's parameters (above), plus optional descriptive keys kept in :meth:`to_dict`:
            ``resolution``, ``anchor``, ``closed``, ``mode``, ``value``, ``interval`` (the rounding interval of a
            latent component: the true values this mode records as the recorded value).
    """

    def __init__(self, kind, weight=1.0, label=None, **kw):
        self.kind, self.weight, self.label, self.p = kind, float(weight), label, kw
        if kind == 'tabulated':
            x, m = np.asarray(kw['x'], float), np.asarray(kw['p'], float)
            keep = m > 0
            o = np.argsort(x[keep])
            self.x, self.m, self.dx = x[keep][o], m[keep][o] / m[keep].sum(), float(kw['dx'])
        if kind == 'samples':
            self.s = np.sort(np.asarray(kw['samples'], float))

    # bounds -------------------------------------------------------------------------------
    @property
    def lower(self):
        """Lowest true value with positive probability."""
        k = self.kind
        if k == 'point': return self.p['value']
        if k in ('uniform', 'triangular'): return self.p['lower']
        if k == 'tabulated': return float(self.x[0] - self.dx / 2)
        return float(self.s[0])

    @property
    def upper(self):
        """Highest true value with positive probability."""
        k = self.kind
        if k == 'point': return self.p['value']
        if k in ('uniform', 'triangular'): return self.p['upper']
        if k == 'tabulated': return float(self.x[-1] + self.dx / 2)
        return float(self.s[-1])

    # moments --------------------------------------------------------------------------------
    def moments(self):
        """Mean and variance of the component.

        Returns:
            ``(mean, variance)``. A tabulated density adds the within-cell variance ``dx**2 / 12``.
        """
        k = self.kind
        if k == 'point':
            return self.p['value'], 0.0
        if k == 'uniform':
            a, b = self.p['lower'], self.p['upper']
            return (a + b) / 2, (b - a) ** 2 / 12
        if k == 'triangular':
            a, c, b = self.p['lower'], self.p['mode'], self.p['upper']
            return (a + b + c) / 3, (a * a + b * b + c * c - a * b - a * c - b * c) / 18
        if k == 'tabulated':
            mu = float((self.m * self.x).sum())
            return mu, float((self.m * (self.x - mu) ** 2).sum() + self.dx ** 2 / 12)
        return float(self.s.mean()), float(self.s.var())

    # cdf / pdf ------------------------------------------------------------------------------
    def cdf(self, t):
        """Cumulative probability at ``t``.

        Args:
            t: Value(s) of the true quantity.

        Returns:
            ``P(true <= t)``, same shape as ``t``.
        """
        t = np.asarray(t, float); k = self.kind
        if k == 'point':
            return (t >= self.p['value']).astype(float)
        if k == 'uniform':
            a, b = self.p['lower'], self.p['upper']
            return np.clip((t - a) / (b - a), 0, 1) if b > a else (t >= a).astype(float)
        if k == 'triangular':
            a, c, b = self.p['lower'], self.p['mode'], self.p['upper']
            lo = np.where(t <= c, (np.clip(t, a, c) - a) ** 2 / ((b - a) * (c - a) if c > a else 1), 0)
            hi = np.where(t > c, 1 - (b - np.clip(t, c, b)) ** 2 / ((b - a) * (b - c) if b > c else 1), 0)
            return np.where(t < a, 0, np.where(t >= b, 1, lo + hi))
        if k == 'tabulated':
            frac = np.clip((t[..., None] - (self.x - self.dx / 2)) / self.dx, 0, 1)
            return (frac * self.m).sum(-1)
        return np.searchsorted(self.s, t, side='right') / len(self.s)

    def pdf(self, t):
        """Probability density at ``t``.

        Args:
            t: Value(s) of the true quantity.

        Returns:
            The density, same shape as ``t``. A point mass gives ``inf`` at its value and 0 elsewhere; samples
            are histogrammed.
        """
        t = np.asarray(t, float); k = self.kind
        if k == 'point':
            return np.where(t == self.p['value'], np.inf, 0.0)
        if k == 'uniform':
            a, b = self.p['lower'], self.p['upper']
            return np.where((t >= a) & (t < b), 1 / (b - a), 0.0)
        if k == 'triangular':
            a, c, b = self.p['lower'], self.p['mode'], self.p['upper']
            up = np.where((t >= a) & (t < c), 2 * (t - a) / ((b - a) * (c - a) if c > a else 1), 0)
            dn = np.where((t >= c) & (t <= b), 2 * (b - t) / ((b - a) * (b - c) if b > c else 1), 0)
            return up + dn
        if k == 'tabulated':
            i = np.floor((t - (self.x[0] - self.dx / 2)) / self.dx).astype(int)
            full = np.zeros(int(round((self.x[-1] - self.x[0]) / self.dx)) + 1)
            full[np.round((self.x - self.x[0]) / self.dx).astype(int)] = self.m / self.dx
            ok = (i >= 0) & (i < len(full))
            return np.where(ok, full[np.clip(i, 0, len(full) - 1)], 0.0)
        h, e = np.histogram(self.s, bins=min(200, max(10, int(len(self.s) ** 0.5))), density=True)
        i = np.searchsorted(e, t, side='right') - 1
        return np.where((i >= 0) & (i < len(h)), h[np.clip(i, 0, len(h) - 1)], 0.0)

    def sample(self, n, rng):
        """Draw from the component.

        Args:
            n: Number of draws.
            rng: A ``numpy.random.Generator``.

        Returns:
            ``n`` draws.
        """
        k = self.kind
        if k == 'point': return np.full(n, float(self.p['value']))
        if k == 'uniform': return rng.uniform(self.p['lower'], self.p['upper'], n)
        if k == 'triangular':
            a, c, b = self.p['lower'], self.p['mode'], self.p['upper']
            return rng.triangular(a, c, b, n) if b > a else np.full(n, c)
        if k == 'tabulated':
            return rng.choice(self.x, size=n, p=self.m) + rng.uniform(-self.dx / 2, self.dx / 2, n)
        return rng.choice(self.s, size=n, replace=True)

    def to_dict(self, include_grid=False):
        """JSON-ready description of the component.

        Args:
            include_grid: Include a tabulated density's grid and cell probabilities.

        Returns:
            ``label``, ``density`` (the kind), ``lower``, ``upper``, ``weight`` and any descriptive keys.
        """
        out = {'label': self.label, 'density': self.kind, 'lower': round(float(self.lower), 6),
               'upper': round(float(self.upper), 6), 'weight': round(float(self.weight), 6)}
        for k in ('resolution', 'anchor', 'closed', 'mode', 'value'):
            if k in self.p: out[k] = self.p[k]
        if 'interval' in self.p:
            lo, hi = self.p['interval']
            out['interval'] = [round(float(lo), 6), round(float(hi), 6)]
            out['mean'] = round(self.moments()[0], 6)
            out['uniform_mean'] = round((lo + hi) / 2, 6)
        if self.kind == 'tabulated' and include_grid:
            out.update(grid=self.x.tolist(), probabilities=self.m.tolist(), cell_width=self.dx)
        return out


def uniform_interval(v, resolution, anchor):
    """The interval of true values that record as ``v`` at a resolution and anchor.

    Args:
        v: The recorded value.
        resolution: The recording resolution, in the variable's unit.
        anchor: How values are rounded: ``"floor"`` (truncated down), ``"ceiling"`` (rounded up) or
            anything else for rounding to the nearest.

    Returns:
        ``(lower, upper, closed)``, where ``closed`` says which end is included: ``"lower"`` or ``"upper"``.
    """
    r = float(resolution)
    if anchor == 'floor':   return v, v + r, 'lower'
    if anchor == 'ceiling': return v - r, v, 'upper'
    return v - r / 2, v + r / 2, 'lower'


DENSITIES = ('latent', 'uniform', 'exact', 'triangular', 'derived', 'none')


def _infer_density(components):
    kinds = {c.kind for c in components}
    if not kinds: return 'none'
    if 'tabulated' in kinds: return 'latent'
    if kinds == {'uniform'}: return 'uniform'
    if kinds == {'point'}: return 'exact'
    if kinds == {'triangular'}: return 'triangular'
    return 'derived'


class Distribution:
    """The distribution of a true value given a recorded one: a weighted mixture of components.

    Returned by :meth:`~vitaldb_resolution.Registry.distribution` and its relatives. Only ``status == "ok"``
    responses carry components; the other statuses explain why there is no distribution, and the
    summary methods raise ``ValueError`` for them.

    Args:
        variable: Variable name, as in the registry.
        recorded_value: The value as recorded.
        unit: The variable's unit.
        status: ``"ok"``, ``"missing"``, ``"invalid"``, ``"imputed"``, ``"not_applicable"``,
            ``"zero_or_not_recorded"``, ``"top_coded"`` or ``"error"``.
        kind: How the distribution arises: ``"single_resolution"``, ``"heaped"``, ``"mixed_source"``,
            ``"computed_value"``, ``"exact"``, ``"count_uncertain"`` or ``"derived"``.
        components: The mixture components; their weights are normalised to sum to 1.
        model: The model file used, if any.
        flags: Warnings that do not prevent a distribution, e.g. ``"off_grid"`` or ``"source_unknown"``.
        note: Free-text explanation.
        extra: Further fields, e.g. the clock time of a timestamp.
        density: Where the shape within each interval comes from: ``"latent"`` (a fitted latent truncated to each
            component's interval, computed at call time), ``"uniform"`` (rectangular over the interval, GUM: midpoint,
            width/sqrt(12); used wherever no latent is fitted), ``"exact"``, ``"triangular"``, ``"derived"`` or
            ``"none"`` (no distribution). Inferred from the components when not given.
    """

    def __init__(self, variable, recorded_value, unit='', status='ok', kind=None, components=None, model=None,
                 flags=None, note=None, extra=None, density=None):
        self.variable, self.recorded_value, self.unit = variable, recorded_value, unit
        self.status, self.kind = status, kind
        self.components = components or []
        tot = sum(c.weight for c in self.components)
        for c in self.components: c.weight = c.weight / tot if tot > 0 else c.weight
        self.model, self.flags, self.note, self.extra = model, list(flags or []), note, dict(extra or {})
        self.density = density or _infer_density(self.components)

    def latent_uniform_gap(self):
        """Largest gap between a latent component's mean and its interval midpoint (the uniform mean), as a
        fraction of the interval width; ``None`` when no component carries an interval.
        """
        gaps = []
        for c in self.components:
            if 'interval' in c.p:
                lo, hi = c.p['interval']
                if hi > lo: gaps.append(abs(c.moments()[0] - (lo + hi) / 2) / (hi - lo))
        return max(gaps) if gaps else None

    @property
    def ok(self):
        """Whether a distribution exists (the response has components)."""
        return bool(self.components)

    def _need(self):
        if not self.components:
            raise ValueError(f'{self.variable}={self.recorded_value!r}: no distribution (status {self.status})')

    def mean(self):
        """Mean of the true value.

        Raises:
            ValueError: The response has no distribution.
        """
        self._need(); return float(sum(c.weight * c.moments()[0] for c in self.components))

    def var(self):
        """Variance of the true value (within-component plus between-component).

        Raises:
            ValueError: The response has no distribution.
        """
        self._need(); mu = self.mean()
        return float(sum(c.weight * (c.moments()[1] + (c.moments()[0] - mu) ** 2) for c in self.components))

    def std(self):
        """Standard deviation of the true value.

        Raises:
            ValueError: The response has no distribution.
        """
        return math.sqrt(max(self.var(), 0.0))

    def cdf(self, x):
        """Cumulative probability ``P(true <= x)``.

        Args:
            x: Value(s).

        Raises:
            ValueError: The response has no distribution.
        """
        self._need(); return sum(c.weight * c.cdf(x) for c in self.components)

    def pdf(self, x):
        """Probability density at ``x``.

        Args:
            x: Value(s).

        Raises:
            ValueError: The response has no distribution.
        """
        self._need(); return sum(c.weight * c.pdf(x) for c in self.components)

    def ppf(self, q):
        """Quantile function: the true value below which a share ``q`` of the probability lies.

        Found by bisection over the mixture's support.

        Args:
            q: Probability in ``[0, 1]``.

        Returns:
            The quantile.

        Raises:
            ValueError: The response has no distribution.
        """
        self._need()
        lo = min(c.lower for c in self.components); hi = max(c.upper for c in self.components)
        if hi == lo: return float(lo)
        a, b = lo, hi
        for _ in range(60):
            m = (a + b) / 2
            if float(self.cdf(m)) < q: a = m
            else: b = m
        return float(b)

    def interval(self, level=0.95):
        """Central interval holding ``level`` of the probability.

        Args:
            level: Coverage, e.g. 0.95.

        Returns:
            ``(lower, upper)``.
        """
        return self.ppf((1 - level) / 2), self.ppf(1 - (1 - level) / 2)

    def sample(self, n=10000, rng=None):
        """Draw true values from the mixture.

        Args:
            n: Number of draws.
            rng: A ``numpy.random.Generator``. Default: a new unseeded one.

        Returns:
            ``n`` draws.

        Raises:
            ValueError: The response has no distribution.
        """
        self._need()
        rng = np.random.default_rng() if rng is None else rng
        w = np.array([c.weight for c in self.components])
        k = rng.choice(len(self.components), size=n, p=w / w.sum())
        out = np.empty(n)
        for j, c in enumerate(self.components):
            m = k == j
            if m.any(): out[m] = c.sample(int(m.sum()), rng)
        return out

    def summary(self):
        """Mean, SD and 95% interval, or ``None`` when there is no distribution.

        Returns:
            ``{"mean", "sd", "ci95": [lower, upper]}`` rounded to 8 significant digits, or ``None``.
        """
        if not self.components: return None
        lo, hi = self.interval(0.95)
        r = lambda x: float(f'{x:.8g}')
        return {'mean': r(self.mean()), 'sd': r(self.std()), 'ci95': [r(lo), r(hi)]}

    def to_dict(self, include_grid=False):
        """The JSON response for this distribution.

        Args:
            include_grid: Include tabulated densities' grids, e.g. for plotting the posterior.

        Returns:
            ``variable``, ``recorded_value``, ``unit``, ``status``, ``kind``, ``components``, ``model``,
            ``summary``, ``flags``, ``note`` and, when present, ``extra``.
        """
        return {'variable': self.variable, 'recorded_value': self.recorded_value, 'unit': self.unit,
                'status': self.status, 'kind': self.kind, 'density': self.density,
                'components': [c.to_dict(include_grid) for c in self.components],
                'model': self.model, 'summary': self.summary(), 'flags': self.flags, 'note': self.note,
                **({'extra': self.extra} if self.extra else {})}

    def __repr__(self):
        s = self.summary()
        body = f"mean {s['mean']:.4g}, sd {s['sd']:.3g}, 95% [{s['ci95'][0]:.4g}, {s['ci95'][1]:.4g}]" if s else '-'
        return f"<Distribution {self.variable}={self.recorded_value!r} {self.status} {self.kind}: {body}" + \
               (f" flags={self.flags}" if self.flags else '') + '>'
