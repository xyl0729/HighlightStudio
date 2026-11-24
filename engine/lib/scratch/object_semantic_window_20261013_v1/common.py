import os
os.environ.update(CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
from pathlib import Path
from scratch.open_vocabulary_spatial_fit_20261012_v1 import common as S
from . import core
import numpy as np
D=Path(__file__).resolve().parent;R=S.R;P=S.P;E=S.E;read=S.read;sha=S.sha;save=S.save;atomic=S.atomic
CONFIG=dict(features=152,ridge=.01,intercept=False,candidate_grid=17,window_samples=17,min_seconds=.5,source_clip_window_mass=True,folds=2,threads=2,bootstrap_draws=20000,bootstrap_seed=20261008,ci_quantiles=[.0125,.9875],joint_gate=.002,temporal_and_joint_positive=True,old_calibration_frozen_historical_exposure=True)

def prepare():
    if (D/'manifest.json').exists():raise FileExistsError('Frozen experiment exists')
    S.check();inp=read(P.D/'inputs.json');cases=E.checked_population(inp)
    complete=read(S.EXTRACT/'detection_complete.json');audit=read(S.EXTRACT/'independent_extraction_recheck_20261012.json')
    if not audit['passed'] or audit['complete_sha256']!=sha(S.EXTRACT/'detection_complete.json') or complete['records']!=621:raise ValueError('Full subject detector prerequisite')
    files=dict(read(S.D/'manifest.json')['files'])
    paths=list(D.glob('*.py'))+[P.D/'original_posteriors.json',R/'scratch/fusai/calibration.npz',S.D/'manifest.json',S.EXTRACT/'detection_complete.json',S.EXTRACT/'independent_extraction_recheck_20261012.json',R/'scratch/structured_window_20261005_v1/core.py',R/'docs/superpowers/specs/2026-10-13-object-semantic-window-design.md',R/'docs/superpowers/plans/2026-10-13-object-semantic-window.md']
    files.update({p.relative_to(R).as_posix():sha(p) for p in paths})
    for f in (0,1):
        tr={c['group'] for c in cases if c['fold']!=f};va={c['group'] for c in cases if c['fold']==f}
        proof=read(R/f'scratch/spatial_object_ranker/oof_v1/fold{f}_training.json')
        if set(proof['train_source_ids'])!=tr or tr&va:raise ValueError('SOR train provenance differs')
    save(D/'manifest.json',dict(config=CONFIG,files=files,versions=P.versions(),population=[c['video_id'] for c in cases],source_hashes=read(S.D/'manifest.json')['sources'],unquantized_parameter_upper_bound=5361977800,historical_weak_labels_not_blind=True,test_labels_used=False,parent='S15 new detector signal, D06 label-free interval pool; new semantic temporal use'))
    return check()

def check():
    m=read(D/'manifest.json')
    if m['config']!=CONFIG or m['versions']!=P.versions():raise ValueError('Config/runtime drift')
    for rel,h in m['files'].items():
        if sha(R/rel)!=h:raise ValueError('Frozen file drift:'+rel)
    return sha(D/'manifest.json')

def context():
    inp=read(P.D/'inputs.json');cases=E.checked_population(inp);post=read(P.D/'original_posteriors.json');native=P.arithmetic()
    with np.load(R/'scratch/fusai/calibration.npz',allow_pickle=False) as z:cal={k:z[k] for k in ('w','mu','sd')}
    return inp,cases,post,native,cal

def prepare_case(c,arm,post,native,cal):
    r=post[c['video_id']]
    if P.baseline_window(c,r,cal,native)!=c['current_window']:raise ValueError('Scored D temporal identity drift')
    q=np.maximum(native['logit_pred'](cal['w'],(native['features'](r,c['n'],c['fps'],use_pos=False)-cal['mu'])/cal['sd'])-.05,0)
    ids,rows=S.objects(c,arm);sem=np.stack([core.semantic(o,c['W'],c['H']) for o in rows]);windows=core.pool(c['n'],c['fps'],c['current_window'])
    return dict(windows=windows,X=core.window_features(windows,q,sem,ids,c['n']),ids=ids,semantic=sem,q=q)

def summary(rows):
    if len(rows)!=621 or len({r['video_id'] for r in rows})!=621 or len({r['group'] for r in rows})!=564:raise ValueError('Complete fixed population')
    means={a:{k:float(np.mean([r[a][k] for r in rows])) for k in ('temporal','joint')} for a in ('baseline','control','candidate')}
    contrasts={a:{k:dict(delta=float(np.mean([r['candidate'][k]-r[a][k] for r in rows])),ci975=E.source_ci([r['candidate'][k]-r[a][k] for r in rows],[r['group'] for r in rows])) for k in ('temporal','joint')} for a in ('baseline','control')}
    folds={str(f):{k:float(np.mean([r['candidate'][k]-r['baseline'][k] for r in rows if r['fold']==f])) for k in ('temporal','joint')} for f in (0,1)}
    gate=contrasts['baseline']['joint']['delta']>=.002 and contrasts['baseline']['temporal']['delta']>0 and all(contrasts[a]['joint']['ci975'][0]>0 for a in contrasts) and all(v>0 for fold in folds.values() for v in fold.values())
    return dict(records=621,source_groups=564,means=means,contrasts=contrasts,fold_deltas=folds,screen_pass=bool(gate),changed_windows={a:sum(r[a+'_window']!=r['baseline_window'] for r in rows) for a in ('control','candidate')},historical_reused_weak_train_labels_not_blind=True,actual_platform_score=None,package_generated=False)
