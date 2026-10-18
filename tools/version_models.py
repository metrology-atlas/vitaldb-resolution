"""Assign model ids and versions, archive superseded versions and write data/models.lock.

v1 of every model is its file at the baseline commit (tag v0.5.0). A model whose fit_hash differs from v1 becomes v2:
v1 is written to data/models/archive/<id>.v1.model.json (with its id, version and fingerprint added; fitted numbers
untouched) and the current file records `supersedes`.

usage: python tools/version_models.py <baseline commit> <release>
"""
import hashlib, json, os, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from vitaldb_resolution import versions as V

BASE, RELEASE = sys.argv[1:3]
DATA_RELEASE = 'VitalDB 1.0.0 (clinical_data.csv, lab_data.csv)'
REASONS = {
    'aneend': 'excluded 2 records outside the model range (not corrected); fit block re-evaluated with the library likelihood',
    'opstart': 'refitted with the library likelihood: the old fit integrated the time-of-day prior on its own grid (loglik gap 10.6)',
    'anestart': 'fit block re-evaluated with the library likelihood (refit gained < 0.01); parameters unchanged',
    'opend': 'fit block re-evaluated with the library likelihood; parameters unchanged',
    'intraop_uo': 'anaesthesia duration as a latent location covariate; department on the heaping propensities; zero rule (point mass, zero_anuric / zero_unexplained)',
    'intraop_crystalloid': 'anaesthesia duration as a latent location covariate',
    'intraop_ebl': 'anaesthesia duration as a latent location covariate',
    'intraop_ppf': 'fit block re-evaluated with the library (grid) likelihood; parameters unchanged',
    'weight': 'fit block re-evaluated with the library likelihood; parameters unchanged',
}


def git(*a):
    return subprocess.run(['git', '-C', ROOT, *a], capture_output=True, check=True).stdout


def sha(b): return hashlib.sha256(b).hexdigest()


def fitted_on(commit, code):
    rules = (git('show', f'{commit}:vitaldb_resolution/data/exclusion_rules.json') if commit else
             open(os.path.join(V.DATA, 'exclusion_rules.json'), 'rb').read())
    return {'data': DATA_RELEASE, 'exclusion_rules_sha256': sha(rules),
            'code': code, 'seed': None,
            'seed_note': 'deterministic optimisers (no random draws in fitting)'}


def ordered(spec, meta):
    """Put id, version and the fingerprint first, keep the rest in order."""
    return {**meta, **{k: v for k, v in spec.items() if k not in meta}}


lock = {'release': RELEASE, 'baseline': {'tag': 'v0.5.0', 'commit': BASE}, 'models': {}, 'published': {}}
os.makedirs(V.ARCHIVE, exist_ok=True)
for fn in sorted(os.listdir(V.MODELS)):
    if not fn.endswith('.model.json'): continue
    path = os.path.join(V.MODELS, fn)
    cur = json.load(open(path, encoding='utf-8'))
    mid = cur['variable']
    old = json.loads(git('show', f'{BASE}:vitaldb_resolution/data/models/{fn}'))
    h1, h2 = V.fit_hash(old), V.fit_hash(cur)
    v1meta = {'id': mid, 'version': 1, 'fit_hash': h1,
              'fitted_on': fitted_on(BASE, f'baseline {BASE[:7]} (fitting scripts of the original analysis; see fit.method)'),
              'supersedes': None}
    lock['published'][f'{mid}@1'] = {'fit_hash': h1, 'release': '0.5.0'}
    if h1 == h2:
        meta = dict(v1meta)
        lock['models'][mid] = {'version': 1, 'fit_hash': h1, 'file': fn}
    else:
        arch = V.archive_file(mid, 1)
        with open(os.path.join(V.MODELS, arch), 'w', encoding='utf-8') as fh:
            json.dump(ordered(old, {**v1meta, 'superseded_by': 2}), fh, indent=1, ensure_ascii=False); fh.write('\n')
        meta = {'id': mid, 'version': 2, 'fit_hash': h2,
                'fitted_on': fitted_on(None, f'vitaldb-resolution {RELEASE} (tag v{RELEASE}): tools/fit_timestamps.py, '
                                             'tools/fit_volume.py, tools/fit_logt.py, tools/reevaluate_fit.py'),
                'supersedes': {'version': 1, 'file': arch, 'reason': REASONS.get(mid, 'refit')}}
        lock['models'][mid] = {'version': 2, 'fit_hash': h2, 'file': fn}
        lock['published'][f'{mid}@2'] = {'fit_hash': h2, 'release': RELEASE}
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(ordered(cur, meta), fh, indent=1, ensure_ascii=False); fh.write('\n')
    print(f'{mid}: v{meta["version"]} {meta["fit_hash"][:12]}')
with open(V.LOCK, 'w', encoding='utf-8') as fh:
    json.dump(lock, fh, indent=1); fh.write('\n')
