"""Re-evaluate a model's fit block with the library's own likelihood, keeping its parameters.

Used where refitting with the library likelihood changes the log-likelihood by less than 0.1 (propofol: +0.04;
weight: the original Nelder-Mead fit stopped 0.04 short), so the parameters stay as fitted and only the stored
statistics are brought in line with what the library recomputes.

usage: python tools/reevaluate_fit.py <clinical_data.csv> <variable> [...] [--write]
"""
import json, math, os, sys, time
import pandas as pd
from vitaldb_resolution import Registry, fitcheck

MODELS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'vitaldb_resolution', 'data', 'models')

if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    d = pd.read_csv(args[0], encoding='utf-8-sig'); reg = Registry.load()
    for var in args[1:]:
        p = os.path.join(MODELS, f'{var}.model.json'); spec = json.load(open(p))
        r = fitcheck.recompute(var, d, reg); f = spec['fit']; k = f['n_params']
        print(f"{var}: n {r['n']} (stored {f['n']}), loglik {r['loglik']:.4f} (stored {f['loglik']:.4f})")
        prev = {x: f[x] for x in ('n', 'loglik', 'bic') if x in f}
        f.update(n=r['n'], loglik=r['loglik'], bic=-2 * r['loglik'] + k * math.log(r['n']))
        if 'aic' in f: f['aic'] = -2 * r['loglik'] + 2 * k
        f['reevaluated'] = {'date': time.strftime('%Y-%m-%d'), 'by': 'tools/reevaluate_fit.py (library likelihood; parameters unchanged)',
                            'previous': prev}
        if '--write' in sys.argv: json.dump(spec, open(p, 'w'), indent=1)
