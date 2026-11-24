import os
os.environ.update(CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
from pathlib import Path
import numpy as np
from scratch.open_vocabulary_spatial_fit_20261012_v1 import common as S
from . import core
D=Path(__file__).resolve().parent;R=S.R;P=S.P;E=S.E;read=S.read;sha=S.sha;save=S.save;atomic=S.atomic
SPEECH=R/'scratch/speech_text_source_oof_20261012_v2';VISUAL=R/'scratch/clip_temporal_supervised'
CONFIG=dict(features=20,ridge=.01,points=16,speech_shift=8,candidate_grid=17,window_samples=17,min_seconds=.5,folds=2,source_clip_window_mass=True,bootstrap_draws=20000,bootstrap_seed=20261008,ci_quantiles=[.0125,.9875],joint_gate=.002,full_chain_parameter_upper_bound=5581375053,parameter_limit=9000000000,missing_cache_fallback=True,no_download=True)

def context():
    inp=read(P.D/'inputs.json');cases=E.checked_population(inp);post=read(P.D/'original_posteriors.json');native=P.arithmetic()
    with np.load(R/'scratch/fusai/calibration.npz',allow_pickle=False) as z:cal={k:z[k] for k in ('w','mu','sd')}
    speech={c['video_id']:c for c in read(SPEECH/'inputs.json')['cases']}
    return inp,cases,post,native,cal,speech

def prepare():
    if (D/'manifest.json').exists():raise FileExistsError('No duplicate experiment')
    S.check();inp,cases,post,native,cal,speech=context();speech_complete=read(SPEECH/'cache_complete.json');viscomplete=read(VISUAL/'extraction_complete.json');files={}
    if speech_complete['records']!=719 or viscomplete['records']!=621:raise ValueError('Complete paired parent assets')
    parent=read(S.D/'manifest.json');files.update(parent['files'])
    for rel,h in parent['sources'].items():
        if sha(R/rel)!=h:raise ValueError('Source video byte drift:'+rel)
        files[rel]=h
    used=[];missing=[];hashes=speech_complete['hashes'];identity=('n','fps','start','source_sha256')
    for c in cases:
        name=c['video_id'];p=VISUAL/'features'/(name+'.npz')
        if sha(p)!=viscomplete['cache_hashes'][p.name]:raise ValueError('Visual cache drift')
        with np.load(p) as z:
            if str(z['fingerprint'])!=viscomplete['fingerprint'] or z['features'].shape!=(len(z['ids']),512):raise ValueError('CLIP512 projection lineage')
        files[p.relative_to(R).as_posix()]=sha(p)
        if name not in speech:missing.append(name);continue
        if any(c[k]!=speech[name][k] for k in identity):raise ValueError('ASR native source mismatch')
        for ext in ('.npz','.json'):
            p=SPEECH/'features'/(name+ext)
            if sha(p)!=hashes[p.name]:raise ValueError('ASR frozen cache drift')
            files[p.relative_to(R).as_posix()]=hashes[p.name]
        used.append(name)
    if len(used)!=449 or len(missing)!=172:raise ValueError('Fixed paired population drift')
    for f in (0,1):
        tr=[c for c in cases if c['fold']!=f];va=[c for c in cases if c['fold']==f];proof=P.read(R/f'scratch/spatial_object_ranker/oof_v1/fold{f}_training.json')
        if set(proof['train_source_ids'])!={c['group'] for c in tr} or {c['group'] for c in tr}&{c['group'] for c in va} or {c['source_sha256'] for c in tr}&{c['source_sha256'] for c in va}:raise ValueError('Source/byte/SOR train proof')
    paths=list(D.glob('*.py'))+[P.D/'inputs.json',P.D/'original_posteriors.json',R/'scratch/fusai/calibration.npz',P.LABEL,SPEECH/'cache_complete.json',SPEECH/'inputs.json',VISUAL/'extraction_complete.json',VISUAL/'extraction_manifest.json',R/'scratch/speech_text_source_oof_fit_20261012_v1/RESULT_SUMMARY.json',R/'scratch/object_semantic_window_20261013_v1/core.py',R/'docs/superpowers/specs/2026-10-13-audiovisual-alignment-design.md',R/'docs/superpowers/plans/2026-10-13-audiovisual-alignment.md']
    files.update({p.relative_to(R).as_posix():sha(p) for p in paths})
    save(D/'manifest.json',dict(config=CONFIG,files=files,sources=parent['sources'],population=[c['video_id'] for c in cases],matched_cached=used,missing_speech_cached=missing,historical_weak_train_labels_not_blind=True,test_labels_used=False,versions=P.versions(),parent='M12 failed text residual, T34 failed subject temporal gain; new local crossmodal alignment/halfclip mismatch signal'))
    return check()

def check():
    m=read(D/'manifest.json')
    if m['config']!=CONFIG or m['versions']!=P.versions():raise ValueError('Fixed runtime/config drift')
    for rel,h in m['files'].items():
        if sha(R/rel)!=h:raise ValueError('Frozen input/code drift:'+rel)
    return sha(D/'manifest.json')

def features(c,post,native,cal,speech):
    v=c['video_id'];base=c['current_window']
    if P.baseline_window(c,post[v],cal,native)!=base:raise ValueError('Actual D time identity')
    windows=core.pool(c['n'],c['fps'],base)
    if v not in speech:return dict(windows=windows,eligible=False,reason='missing_existing_ASR_cache')
    with np.load(VISUAL/'features'/(v+'.npz')) as z:vi=z['ids'];vx=z['features'].astype(float)
    with np.load(SPEECH/'features'/(v+'.npz')) as z:ids=z['ids'].copy();local=z['local'].copy();topic=z['topic'].copy();mask=z['mask'].copy()
    if len(ids)!=16 or ids[0]!=0 or ids[-1]!=c['n']-1:raise ValueError('16-point native endpoints')
    visual=np.column_stack([np.interp(ids,vi,vx[:,k]) for k in range(512)])
    a,b,valid=core.alignment(visual,local,topic,mask)
    if valid.sum()<2:return dict(windows=windows,eligible=False,reason='insufficient_common_speech_points')
    q=np.maximum(native['logit_pred'](cal['w'],(native['features'](post[v],c['n'],c['fps'],use_pos=False)-cal['mu'])/cal['sd'])-.05,0)
    return dict(windows=windows,eligible=True,reason=None,ids=ids,semantic={'candidate':a,'control':b},X={arm:core.windows_features(windows,q,x,ids,c['n']) for arm,x in (('candidate',a),('control',b))},q=q,common_points=int(valid.sum()))

def summary(rows):
    from scratch.object_semantic_window_20261013_v1.common import summary as fixed_summary
    return fixed_summary(rows)
