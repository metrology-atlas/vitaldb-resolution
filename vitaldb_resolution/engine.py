"""Generic rounding-model engine (v0.4).

Values: latent (log_spline, log_student_t_regression, normal_regression) + ordered (stick-breaking) allocation of rounding modes.
Timestamps (kind: timestamp): reference-offset spline latent + optional time-of-day prior + nominal modes (TimestampModel).
Derived variables (kind: derived): sampled from their inputs' posteriors (derived_samples).
A model is fully described by a JSON file; load_model(path) returns the right evaluator.
"""
import copy
import json

import numpy as np
from scipy.interpolate import BSpline

sig=lambda t:1/(1+np.exp(-t))

class Model:
    """Evaluator for a fitted value model (``kind: value``): a latent distribution plus rounding modes.

    Internal: use :meth:`~vitaldb_resolution.Registry.distribution`. The model file's schema describes the
    latent family (log-spline, log-Student-t or normal regression), the grid and the ordered allocation of
    rounding modes.

    Args:
        spec: The parsed model JSON.
    """
    def __init__(self, spec):
        self.spec=spec
        L=spec['latent']; g=spec['grid']
        self.dx=g['dx']; self.X=np.arange(g['lo']+self.dx/2, g['hi'], self.dx); self.LX=np.log(self.X)
        self.ru=spec.get('record_unit',1)   # unit in which recorded values are keyed (1 for mL/mg; 0.1 for height)
        if L['family']!='log_spline': self.base=spec['resolution']['effective']; return
        lo,hi=np.log(L['domain'][0]),np.log(L['domain'][1]); K=L['K']
        inner=np.linspace(lo,hi,K-2); t=np.r_[[lo]*3,inner,[hi]*3]
        self.knots=t
        self.B=np.column_stack([BSpline(t,np.eye(K)[k],3)(np.clip(self.LX,lo,hi)) for k in range(K)])
        self.base=spec['resolution']['effective']
    # ---- latent
    def latent(self, coef=None, cov=None, spec=None):
        """Probability of each true value on the grid before rounding (sums to 1)."""
        L=(self.spec if spec is None else spec)['latent']
        if L['family']=='log_student_t_regression':
            from scipy import stats
            mu=L['intercept']+sum(c['coef']*transform(c['transform'], cov[c['name']]) for c in L['covariates'])
            w=stats.t.pdf((self.LX-mu)/L['scale'], L['df'])/self.X; return w/w.sum()
        if L['family']=='normal_regression':
            mu=L['intercept']+sum(c['coef']*transform(c['transform'], cov[c['name']]) for c in L['covariates'])
            lsd=L['log_sd']+sum(c['coef']*transform(c['transform'], cov[c['name']]) for c in L.get('sd_covariates',[]))
            z=(self.X-mu)/np.exp(lsd); w=np.exp(-0.5*z*z); return w/w.sum()
        coef=np.asarray(L['coefficients'] if coef is None else coef)
        if L.get('location_covariates'):
            s=self.shift(L, cov)
            if s is None:                       # covariate not available: marginal over the fitted cases
                ws=[self._log_spline(coef, q) for q in L['marginal']['shifts']]
                w=np.mean(ws, axis=0); return w/w.sum()
            return self._log_spline(coef, s)
        return self._log_spline(coef, 0.0)
    @staticmethod
    def shift(L, cov):
        """Location shift of a log-spline latent on the log scale (sum of coef x transform), or None when a
        location covariate is missing from cov."""
        s=0.0
        for c in L['location_covariates']:
            v=None if cov is None else cov.get(c['name'])
            if v is None or (isinstance(v,float) and not np.isfinite(v)): return None
            s+=c['coef']*transform(c['transform'], v)
        return s
    def _log_spline(self, coef, s):
        """Grid probabilities of a log-spline latent shifted by s on the log scale: f(x) ~ g(log x - s) / x."""
        if s==0.0: B=self.B
        else:
            key=round(float(s),12); cache=self.__dict__.setdefault('_bcache',{})
            if key not in cache:
                if len(cache)>256: cache.clear()
                lo,hi=self.knots[0],self.knots[-1]; K=self.B.shape[1]
                u=np.clip(self.LX-s,lo,hi)
                cache[key]=np.column_stack([BSpline(self.knots,np.eye(K)[k],3)(u) for k in range(K)])
            B=cache[key]
        lw=B@np.r_[0,coef]-self.LX; w=np.exp(lw-lw.max()); return w/w.sum()
    # ---- rounding
    def _round(self, mode):
        if mode['type']=='set':
            vals=np.asarray(mode['values'],float); idx=np.abs(self.X[:,None]-vals[None,:]).argmin(1); return vals[idx]
        step=self._step(mode); a=mode.get('anchor','nearest')
        if a=='ceiling': return np.ceil(self.X/step-1e-9)*step
        if a=='floor': return np.floor(self.X/step+1e-9)*step
        return np.round(self.X/step)*step
    def _step(self, mode):
        if mode['type']=='absolute': return np.full_like(self.X, mode['resolution'])
        u=10.0**np.floor(np.log10(self.X)); return np.maximum(u*mode['fraction'], self.base)
    def _prob(self, p, cov=None):
        lg=p['logit']+sum(c['coef']*transform(c['transform'], cov[c['name']]) for c in p.get('covariates',[]))
        if p['form']=='constant': return np.full_like(self.X, sig(lg))
        if p['form']=='logistic_logx': return sig(lg+p['slope']*(self.LX-np.log(p['centre'])))
        raise ValueError(p['form'])
    def components(self, spec=None, cov=None):
        """Return list of (label, recorded_value_array, weight_array) over the true-value grid.

        A propensity covariate missing from ``cov`` (or an unknown level) is averaged over its fitted shares
        (``covariate_marginals`` in the model file): the weights become the share-weighted mixture."""
        s=self.spec if spec is None else spec
        for name,shares in s.get('covariate_marginals',{}).items():
            if cov is None or cov.get(name) not in shares:
                out=None
                for lev,sh in shares.items():
                    part=self.components(s,{**(cov or {}),name:lev})
                    out=[(l,r,sh*t) for l,r,t in part] if out is None else [(l,r,t0+sh*t) for (l,r,t0),(_,_,t) in zip(out,part)]
                return out
        remaining=np.ones_like(self.X); out=[]
        for comp in s['allocation']:
            if comp['kind']=='nominal':
                take=remaining*sig(comp['prob']['logit'])*(self.X<=comp['applies_below'])
                rec=np.full(self.X.shape, comp['recorded_value'], dtype=float)
            else:
                rec=self._round(comp['mode'])
                p=self._prob(comp['prob'],cov) if comp.get('prob') else np.ones_like(self.X)
                cw=comp.get('capture_window')
                if isinstance(cw,dict): dd=rec-self.X; p=p*((dd>=cw['lo']-1e-9)&(dd<cw['hi']-1e-9))   # one-sided window: lo <= recorded - true < hi
                elif cw is not None: p=p*(np.abs(self.X-rec)<cw)
                if comp.get('prob'): p=p*(rec>0)
                if comp.get('applies_below') is not None: p=p*(self.X<=comp['applies_below'])   # a non-base mode never records zero: mass passes to the next mode
                take=remaining*p
            out.append((comp['label'],rec,take)); remaining=remaining-take
        return out
    def pmf(self, spec=None, cov=None):
        """Probability of each recorded value (indexed in ``record_unit`` steps)."""
        s=self.spec if spec is None else spec
        w=self.latent(s['latent'].get('coefficients'), cov, s); top=int(self.X.max()/self.ru*1.2)+2000
        q=np.zeros(top)
        for _,rec,take in self.components(s,cov):
            q+=np.bincount(np.round(rec/self.ru).astype(int), weights=w*take, minlength=top)[:top]
        if s.get('truncation',{}).get('exclude_recorded_zero'): q[0]=0
        return q/q.sum()
    def loglik(self, vals, cnt, spec=None):
        """Log-likelihood of recorded values with counts ``cnt`` under the model."""
        q=self.pmf(spec); return float((cnt*np.log(np.clip(q[vals],1e-300,None))).sum())
    def posterior(self, v, spec=None, cov=None):
        """Posterior over the true value given recorded value ``v``.

        One entry per rounding mode that can produce it.
        """
        s=self.spec if spec is None else spec
        w=self.latent(s['latent'].get('coefficients'), cov, s); comps=[]; tot=0
        for lab,rec,take in self.components(s,cov):
            m=(np.round(rec/self.ru)==round(v/self.ru)); wm=w*take*m
            if wm.sum()>0: comps.append((lab,wm)); tot+=wm.sum()
        res=[]
        for lab,wm in comps:
            sup=self.X[wm>0]; res.append({'component':lab,'lower':float(sup.min()-self.dx/2),'upper':float(sup.max()+self.dx/2),'weight':float(wm.sum()/tot)})
        dens=sum(wm for _,wm in comps)/tot
        mean=float((dens*self.X).sum()); c=np.cumsum(dens)
        return {'recorded':v,'components':res,'mean':mean,'ci95':[float(self.X[np.searchsorted(c,.025)]),float(self.X[np.searchsorted(c,.975)])],'density':dens}

def transform(t, x):
    """Apply a covariate transform from a model file.

    Types: ``log_ratio``, ``linear``, ``indicator`` or ``quadratic``.

    Raises:
        ValueError: Unknown transform type.
    """
    if t['type']=='log_ratio': return np.log(float(x)/t['ref'])
    if t['type']=='linear': return (float(x)-t['centre'])/t['scale']
    if t['type']=='indicator': return 1.0 if x==t['value'] else 0.0
    if t['type']=='quadratic': return ((float(x)-t['centre'])/t['scale'])**2
    raise ValueError(t['type'])
def loglik_records(model, values, covs, spec=None):
    """Per-record log-likelihood for conditional-latent models."""
    tot=0.0
    for v,c in zip(values,covs):
        q=model.pmf(spec,c); tot+=np.log(max(q[int(round(v/model.ru))],1e-300))
    return tot
def free_params(spec):
    """List of (path, kind) for fitted parameters (latent coefs + logits/slopes)."""
    P=[('latent',i) for i in range(spec['latent']['K']-1)]
    for j,c in enumerate(spec['allocation']):
        p=c.get('prob')
        if p is None: continue
        if p.get('fixed'): continue
        P.append((j,'logit'))
        if p['form']=='logistic_logx': P.append((j,'slope'))
    return P
def get_vec(spec):
    """The free parameters of a model spec as one vector (for fitting)."""
    v=list(spec['latent']['coefficients'])
    for j,c in enumerate(spec['allocation']):
        p=c.get('prob')
        if p is None or p.get('fixed'): continue
        v.append(p['logit'])
        if p['form']=='logistic_logx': v.append(p['slope'])
    return np.array(v,float)
def set_vec(spec,vec):
    """A copy of a model spec with its free parameters set from a vector."""
    s=copy.deepcopy(spec); K=s['latent']['K']; s['latent']['coefficients']=list(map(float,vec[:K-1])); i=K-1
    for c in s['allocation']:
        p=c.get('prob')
        if p is None or p.get('fixed'): continue
        p['logit']=float(vec[i]); i+=1
        if p['form']=='logistic_logx': p['slope']=float(vec[i]); i+=1
    return s
def fit(model, spec, vals, cnt, thorough=True):
    """Fit a model spec to recorded values by maximum likelihood.

    Returns a copy with a ``fit`` block (n, log-likelihood, AIC, BIC).
    """
    from scipy import optimize
    f=lambda v: -model.loglik(vals,cnt,set_vec(spec,v))
    r=optimize.minimize(f,get_vec(spec),method='L-BFGS-B',options={'maxiter':5000,'maxfun':100000})
    if thorough:
        r=optimize.minimize(f,r.x,method='Powell',options={'maxiter':30000,'xtol':1e-6,'ftol':1e-9})
        r=optimize.minimize(f,r.x,method='L-BFGS-B',options={'maxiter':5000,'maxfun':100000})
    s=set_vec(spec,r.x); k=len(r.x)+sum(1 for c in spec['allocation'] if c['kind']=='nominal')  # +1 for profiled threshold
    s['fit']={'n':int(cnt.sum()),'loglik':-float(r.fun),'n_params':int(k),'aic':2*float(r.fun)+2*k,'bic':2*float(r.fun)+k*np.log(cnt.sum())}
    return s
def init_latent(model, median, sigma):
    """Starting spline coefficients for a log-normal-shaped latent with the given median and log-SD."""
    target=-(model.LX-np.log(median))**2/(2*sigma**2)
    b=np.linalg.lstsq(model.B,target,rcond=None)[0]; return list(map(float,(b-b[0])[1:]))

# ---------------- v0.3: sampling posteriors and propagating to derived variables ----------------
WRITTEN_IN_MODES={'height':{'tens','whole_cm','half_cm','round_up_from_x9'},'weight':{'whole_kg'}}
def _age_band(a):
    a=float(a)
    for lo,hi,lab in [(0,30,'19-29'),(30,40,'30-39'),(40,50,'40-49'),(50,60,'50-59'),(60,70,'60-69')]:
        if a<hi: return lab
    return '70+'
def sample_posterior(model, v, cov, n=20000, rng=None, variability=False):
    """Draw n samples of the true value given recorded v.

    Samples are drawn per rounding mode, so that measurement_variability (if requested) can differ between
    measured and written-in values.
    """
    rng=np.random.default_rng(0) if rng is None else rng
    s=model.spec; w=model.latent(s['latent'].get('coefficients'), cov, s)
    comps=[]
    for lab,rec,take in model.components(s,cov):
        wm=w*take*(np.round(rec/model.ru)==round(v/model.ru))
        if wm.sum()>0: comps.append((lab,wm))
    tot=sum(wm.sum() for _,wm in comps); probs=np.array([wm.sum()/tot for _,wm in comps])
    k=rng.choice(len(comps),size=n,p=probs); out=np.empty(n); labs=np.array([c[0] for c in comps])[k]
    for j,(lab,wm) in enumerate(comps):
        m=(k==j) 
        if not m.any(): continue
        x=rng.choice(model.X,size=m.sum(),p=wm/wm.sum())+rng.uniform(-model.dx/2,model.dx/2,m.sum())
        if variability and s.get('measurement_variability'):
            mv=s['measurement_variability']; wi=lab in WRITTEN_IN_MODES.get(s['variable'],set())
            if wi:
                b=mv['written_in'].get('bias_self_minus_measured',0)
                if isinstance(b,dict):
                    b=b[cov['sex']]; b=b[_age_band(cov['age'])] if isinstance(b,dict) else b
                x=x-b+rng.normal(0,mv['written_in']['sd'],m.sum())
            else:
                me=mv['measured']; sd=me['sd'] if 'sd' in me else me['sd_relative']*x
                x=x+rng.normal(0,1,m.sum())*sd
        out[m]=x
    return out,labs
def derived_samples(spec, models, recorded, cov, n=20000, rng=None, variability=False):
    """Samples of a derived variable from its inputs' posteriors.

    ``spec`` is a derived-variable spec with a supported ``formula``; ``models`` maps name -> Model; ``recorded``
    maps name -> value. Inputs are sampled independently given their own records and the covariates (the
    separate latent models ignore any correlation between inputs).
    """
    rng=np.random.default_rng(0) if rng is None else rng
    S={k:sample_posterior(models[k],recorded[k],cov,n,rng,variability)[0] for k in spec['inputs']}
    if spec['formula']=='weight / (height/100)^2': return S['weight']/(S['height']/100)**2
    raise ValueError(spec['formula'])

# ---------------- v0.4: charted timestamps anchored to a device time ----------------
class TimestampModel:
    """Evaluator for a charted-time model (``kind: timestamp``).

    True event clock time X (minutes since midnight) = ref - G (direction 'before') or ref + G ('after'), where
    ref is a device timestamp for the same case (recording start or end, exact to the second) and G has a
    log-spline density on a linear grid. The recorded value is the clock minute M produced by stick-breaking
    rounding modes on the clock grid.
    """
    def __init__(self, spec):
        from scipy.interpolate import BSpline
        self.spec=spec; L=spec['latent']; g=spec['grid']
        self.dx=g['dx']; self.G=np.arange(g['lo']+self.dx/2, g['hi'], self.dx)
        t=np.r_[[g['lo']]*3, L['inner_knots'], [g['hi']]*3]; K=len(t)-4
        B=np.column_stack([BSpline(t,np.eye(K)[k],3)(self.G) for k in range(K)])
        lw=B@np.r_[0,L['coefficients']]; w=np.exp(lw-lw.max()); self.wG=w/w.sum()
        self.tod=None
        if L.get('time_of_day_prior'):
            P=L['time_of_day_prior']; th=np.r_[[P['knots'][0]]*3,P['knots'],[P['knots'][-1]]*3]; Kh=len(th)-4
            self.tod=(P, BSpline(th, np.eye(Kh), 3), np.r_[0,P['coefficients']])
    def _weights(self, cov):
        X=self._true(cov['ref'])
        if self.tod is None: return X,self.wG
        P,bsp,c=self.tod; rc=cov['ref']%1440
        if not (P['applies_if_ref_in'][0]<=rc<P['applies_if_ref_in'][1]): return X,self.wG
        day=np.floor(cov['ref']/1440)*1440; xc=X-day; inside=(xc>=P['knots'][0])&(xc<P['knots'][-1])
        lh=np.where(inside, bsp(np.clip(xc,P['knots'][0],P['knots'][-1]))@c, -np.inf)
        w=self.wG*np.exp(lh-lh[inside].max()); return X,w/w.sum()
    def _true(self, ref):
        return ref-self.G if self.spec['latent']['direction']=='before' else ref+self.G
    def components(self, cov):
        """Recorded clock minute and weight of each nominal mode over the grid of true times."""
        X,_=self._weights(cov); remaining=np.ones_like(X); out=[]
        for comp in self.spec['allocation']:
            if comp['kind']=='nominal':
                rec=np.full_like(X, comp['recorded_value']); w=comp['window']; xc=X%1440
                p=sig(comp['prob']['logit'])*((xc>=w[0])&(xc<w[1]))
                if comp.get('applies_if_ref_in') and not (comp['applies_if_ref_in'][0]<=cov['ref']%1440<comp['applies_if_ref_in'][1]): p=p*0
                take=remaining*p; out.append((comp['label'],rec,take)); remaining=remaining-take; continue
            step=comp['mode']['resolution']; a=comp['mode'].get('anchor','nearest')
            if a=='floor': rec=np.floor(X/step+1e-9)*step
            elif a=='ceiling': rec=np.ceil(X/step-1e-9)*step
            else: rec=np.floor(X/step+0.5)*step
            if comp.get('prob') is None: p=np.ones_like(X)
            else:
                p=comp['prob']; lg=p['logit']+sum(c['coef']*transform(c['transform'], cov[c['name']]) for c in p.get('covariates',[]))
                p=np.full_like(X, sig(lg))
            take=remaining*p; out.append((comp['label'],rec,take)); remaining=remaining-take
        return X,out
    def prob_recorded(self, M, cov):
        """Probability of recording clock minute ``M``."""
        X,w=self._weights(cov); X,comps=self.components(cov); return float(sum((w*take*(np.round(rec)==M)).sum() for _,rec,take in comps))
    def posterior(self, M, cov):
        """Posterior over the true event time given the recorded clock minute ``M``."""
        X,w=self._weights(cov); X,comps=self.components(cov); res=[]; dens=np.zeros_like(X)
        for lab,rec,take in comps:
            wm=w*take*(np.round(rec)==M)
            if wm.sum()>0: res.append((lab,wm)); dens+=wm
        tot=dens.sum(); dens=dens/tot; o=np.argsort(X); c=np.cumsum(dens[o])
        out=[{'component':lab,'weight':float(wm.sum()/tot),'lower':float(X[wm>0].min()-self.dx/2),'upper':float(X[wm>0].max()+self.dx/2)} for lab,wm in res]
        return {'recorded':M,'components':out,'mean':float((dens*X).sum()),'ci95':[float(X[o][np.searchsorted(c,.025)]),float(X[o][np.searchsorted(c,.975)])],'X':X,'density':dens}


def load_model(path_or_spec):
    """Load a *.model.json (or spec dict) and return Model, TimestampModel, or the spec itself for kind 'derived'."""
    spec=json.load(open(path_or_spec)) if isinstance(path_or_spec,str) else path_or_spec
    kind=spec.get('kind','value')
    if kind=='timestamp': return TimestampModel(spec)
    if kind=='derived': return spec
    return Model(spec)
