import os,sys
os.environ['CUDA_VISIBLE_DEVICES']='-1'
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='2'
from pathlib import Path
import numpy as np
D=Path(__file__).resolve().parent;R=D.parents[1];sys.path.insert(0,str(R))
from scratch.open_vocabulary_spatial_oof_20261012_v1 import common as Q,core
P=Q.P;E=Q.E;read=P.read;sha=P.sha;save=P.save;atomic=P.atomic
EXTRACT=R/'scratch/open_vocabulary_spatial_recovery_20261012_v2'
CONFIG=dict(Q.CONFIG);CONFIG.update(actual_head_elements=5633,normalization_elements=108,head_activation='relu_relu_tanh',standardization='training_row_mean_std',weighting='source_clip_frame_candidate',fit_device='cpu',fit_threads=2)
BEST=R/'outputs/fusai/four_explore_20261011_v2/submission_fusai_D_spatial_center010_20261011_v2.zip'
BEST_SHA='50a9d9ed4ca7b50c81b7ae3bbf71ce103a51fa7d527904d14827cacb9d1b2a32'

def check():
    m=read(D/'manifest.json')
    if m['config']!=CONFIG or m['versions']!=P.versions():raise ValueError('Quality config/runtime drift')
    for rel,h in m['files'].items():
        if sha(R/rel)!=h:raise ValueError('Frozen quality file drift:'+rel)
    return sha(D/'manifest.json')

def prepare():
    Q.check(full_sources=True)
    if (D/'manifest.json').exists():raise FileExistsError('Frozen fit outputs exist')
    completed=read(EXTRACT/'detection_complete.json');recheck=read(EXTRACT/'independent_extraction_recheck_20261012.json')
    if int((EXTRACT/'recovery_exit.txt').read_text(encoding='utf-8-sig'))!=0 or completed['records']!=621 or completed['images']!=7452 or not recheck['passed'] or recheck['complete_sha256']!=sha(EXTRACT/'detection_complete.json'):raise ValueError('Full detector prerequisite missing')
    inp=read(P.D/'inputs.json');cases=E.checked_population(inp)
    if sha(BEST)!=BEST_SHA:raise ValueError('ScoredD package drift')
    parent=read(Q.D/'manifest.json');files=dict(parent['files'])
    files.update({(EXTRACT/rel).relative_to(R).as_posix():h for rel,h in completed['files_sha256'].items()})
    paths=list(D.glob('*.py'))+[D/'PROTOCOL.md',EXTRACT/'manifest.json',EXTRACT/'recovery_manifest.json',EXTRACT/'RESULT_SUMMARY.json',EXTRACT/'detection_complete.json',EXTRACT/'independent_extraction_recheck_20261012.json',Q.D/'manifest.json',BEST]
    files.update({p.relative_to(R).as_posix():sha(p) for p in paths})
    for f in (0,1):
        tr=[c for c in cases if c['fold']!=f];va=[c for c in cases if c['fold']==f]
        if {c['group'] for c in tr}&{c['group'] for c in va} or {c['source_sha256'] for c in tr}&{c['source_sha256'] for c in va}:raise ValueError('Source/byte fold leak')
        proof=read(R/f'scratch/spatial_object_ranker/oof_v1/fold{f}_training.json')
        if set(proof['train_source_ids'])&{c['group'] for c in va}:raise ValueError('Baseline SOR validation-source leak')
    save(D/'manifest.json',dict(config=CONFIG,versions=P.versions(),files=files,sources=parent['source_hashes'],population=[c['video_id'] for c in cases],
        original_quality_manifest_sha256=sha(Q.D/'manifest.json'),detector_manifest_sha256=sha(EXTRACT/'manifest.json'),
        total_parameter_envelope=parent['unquantized_parameter_envelope'],label_sha256=sha(P.LABEL),historical_weak_development_not_blind=True,no_test_inputs=True))
    return check()

def objects(c,arm):
    from .learning import map_yolo,nearest
    ids=core.sampled_frames(c['n'])
    if arm=='candidate':
        row=read(EXTRACT/'detections'/(c['video_id']+'.json'))
        if [x['local_frame'] for x in row['frames']]!=ids or row['source_sha256']!=c['source_sha256']:raise ValueError('Detector native lineage')
        return ids,[core.checked_objects(x['objects'],c['W'],c['H']) for x in row['frames']]
    if arm!='control':raise ValueError('Unknown arm')
    records=read(R/c['detector_path'])['sample']['recs'];clock=[x['fid'] for x in records]
    if any(type(f) is not int or not 0<=f<c['n'] for f in clock):raise ValueError('YOLO original local clock')
    return ids,[map_yolo(records[nearest(clock,f)].get('dets',[]),c['W'],c['H']) for f in ids]

def baseline(c,inputs,reference_fold=None):return Q.baseline(c,c['fold'] if reference_fold is None else reference_fold,inputs)

def dataset(cases,labels,arm,inputs,fold):
    from .learning import nearest,gain,source_weights,training_frames
    X=[];y=[];metadata=[];items=[]
    for c in cases:
        if c['fold']==fold:raise ValueError('Validation clip supplied to training')
        b=baseline(c,inputs,fold);truth=E.checked_truth(c,labels[c['video_id']]);ids,rows=objects(c,arm);frames=[]
        for frame in training_frames(truth):
            obj=rows[nearest(ids,frame)];pool=core.proposals(b[frame],obj,c['W'],c['H']);frames.append((frame,pool))
            for x in pool:X.append(core.features(x,b[frame],obj,c['W'],c['H']));y.append(gain(x,b[frame],truth[frame]))
            metadata.append(dict(video_id=c['video_id'],group=c['group'],frame=frame,pool=pool))
        items.append((c['group'],c['video_id'],frames))
    return np.array(X,np.float32),np.array(y,np.float32),source_weights(items),metadata

def infer(c,arm,model,inputs):
    from .model import predict
    b=baseline(c,inputs);ids,rows=objects(c,arm);chosen=[];decisions=[]
    for frame,obj in zip(ids,rows):
        pool=core.proposals(b[frame],obj,c['W'],c['H']);X=np.array([core.features(x,b[frame],obj,c['W'],c['H']) for x in pool]);scores=predict(model,X)
        x=core.choose_position(pool,scores,int(b[frame,0]));chosen.append(x);decisions.append(dict(frame=frame,pool=pool,scores=scores.tolist(),chosen=x))
    return core.apply(b,ids,chosen,c['W'],c['fps']),decisions

def summary(rows):
    if len(rows)!=621 or len({x['group'] for x in rows})!=564:raise ValueError('Partial population')
    contrasts={}
    for other in ('baseline','control'):
        delta=[r['candidate']['joint']-r[other]['joint'] for r in rows]
        contrasts[other]=dict(delta=float(np.mean(delta)),ci975=E.source_ci(delta,[r['group'] for r in rows]))
    folds={str(f):float(np.mean([r['candidate']['joint']-r['baseline']['joint'] for r in rows if r['fold']==f])) for f in (0,1)}
    if any(r[a]['temporal']!=r['baseline']['temporal'] for r in rows for a in ('candidate','control')):raise ValueError('Time changed')
    gate=contrasts['baseline']['delta']>=.002 and contrasts['baseline']['ci975'][0]>0 and contrasts['control']['ci975'][0]>0 and min(folds.values())>0
    return dict(records=621,source_groups=564,means={a:float(np.mean([r[a]['joint'] for r in rows])) for a in ('baseline','control','candidate')},contrasts=contrasts,fold_deltas=folds,screen_pass=bool(gate),temporal_unchanged=True,actual_platform_score=None,package_generated=False,confidence=.975,evidence='Historical reused model-generated weak TRAIN labels, not blind')
