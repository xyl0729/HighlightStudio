"""Predeclared training-only UNISAL comparison; no submission writer."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
from .core import sample_ids,choose_box,source_half

ROOT=Path(__file__).resolve().parents[2]
CACHE=ROOT/'scratch/joint_audit/run_v2'
LABELS=ROOT/'data/train/高光剪辑训练集/qvhighlights-videos/train.jsonl'
UPSTREAM=ROOT/'models/UNISAL-upstream'
CONFIG={'stride':5,'chunk':16,'input_hw':[224,384],'source':'DHF1K','static':False,
        'smooth_sec':.4,'mass_tie_tolerance':1e-12,'primary_scale':'1.7','dtype':'float32',
        'parameters':4552439173,'bootstrap_seed':20261002,'bootstrap_n':10000}
CODE=['scratch/unisal_crop/core.py','scratch/unisal_crop/model.py','scratch/unisal_crop/run.py',
      'tests/test_unisal_crop.py','docs/superpowers/specs/2026-10-02-unisal-crop-design.md']
BASELINE='outputs/fusai/submission_fusai_ora4b_ef_v1.zip'
BASEHASH='f435fb79abae2d4a4672459779fa2326b8a24f0629f80e25a6edb60c2d8a23e1'

def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def read_json(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def canonical_hash(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,allow_nan=False).encode()).hexdigest()

def save_json(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    text=json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)
    if p.exists():
        if read_json(p)!=obj:raise ValueError('Refusing output overwrite: '+str(p))
        return
    partial=p.with_suffix(p.suffix+'.partial')
    with partial.open('x',encoding='utf-8') as f:f.write(text)
    partial.replace(p)

def check_frozen():
    if digest(ROOT/BASELINE)!=BASEHASH:raise ValueError('Baseline changed')
    frozen=read_json(ROOT/'scratch/fusai/run_manifest.json')['files'];stable={}
    for p,h in frozen.items():
        if p.endswith('.py') or p=='scratch/fusai/calibration.npz':
            if digest(ROOT/p)!=h:raise ValueError('Frozen dependency changed '+p)
            stable[p]=h
    return stable

def prepare(out):
    from scratch.fusai.baseline import full_track
    from hlcut.io_utils import video_meta
    from hlcut.spatial import SUBJECT_CLASS_WEIGHT
    cm=read_json(CACHE/'manifest.json');stable=check_frozen()
    if digest(LABELS)!=cm['inputs']['labels']:raise ValueError('Labels changed')
    labels={r['video_id']:r for r in map(json.loads,LABELS.read_text(encoding='utf-8-sig').splitlines())}
    clips=[];sources={};caches={}
    for vid in cm['ids']:
        p=CACHE/'clips'/(vid+'.json');caches[p.relative_to(ROOT).as_posix()]=digest(p)
        c=read_json(p);s=c['sample'];lab=labels[vid]
        if s['axis']!='x' or lab['targetRatioWH']!=[9,16]:raise ValueError('Unexpected target axis')
        source=lab['provenance']['source_group'];group=source.rsplit('_',2)[0]
        candidates=[ROOT/base/(source+'.mp4') for base in ['data/train/src','data/train/qvh_src']]
        video=next((p for p in candidates if p.exists()),None)
        if video is None:raise ValueError('Missing source '+source)
        rel=video.relative_to(ROOT).as_posix()
        if rel not in sources:sources[rel]=digest(video)
        if sources[rel]!=c['source_sha256']:raise ValueError('Source hash drift')
        ns,fps,W,H=video_meta(str(video));start=round(lab['clip']['start_sec']*fps)
        end=min(ns,round(lab['clip']['end_sec']*fps))
        if end-start!=s['n'] or abs(fps-s['fps'])>1e-6 or (W,H)!=(s['W'],s['H']):
            raise ValueError('Clip metadata drift')
        counts=[sum(SUBJECT_CLASS_WEIGHT.get(int(d[5]),0)>0 for d in r.get('dets',[])) for r in s['recs']]
        stratum='low' if np.mean(np.asarray(counts)>0)<.5 else ('multi' if np.mean(np.asarray(counts)>=2)>=.5 else 'single')
        clips.append({'video_id':vid,'group':group,'half':source_half(group),'stratum':stratum,
                      'source':rel,'start':start,'n':s['n'],'fps':fps,'W':W,'H':H,
                      'window':c['row']['1.7']['baseline']['window'],
                      'baseline_boxes':full_track(s,9,16,12.,2.,.03)})
    clips.sort(key=lambda c:hashlib.sha256(('unisal-run:'+c['video_id']).encode()).hexdigest())
    if len(clips)!=270 or len({c['video_id'] for c in clips})!=270:raise ValueError('Dataset drift')
    # No teacher boxes/scores are persisted in inference inputs.
    inputs={'clips':clips}
    m={'config':CONFIG,'inputs_sha256':canonical_hash(inputs),'labels_sha256':digest(LABELS),
       'cache_manifest_sha256':digest(CACHE/'manifest.json'),'cache_files':caches,'sources':sources,
       'stable':stable,'upstream_acquisition_sha256':digest(UPSTREAM/'ACQUISITION.json'),
       'code':{p:digest(ROOT/p) for p in CODE},'baseline_sha256':BASEHASH,
       'limitations':['Reused training replay; not a new blind test','9:16 only','No test inference or submission in this runner']}
    save_json(out/'inputs.json',inputs);save_json(out/'manifest.json',m)
    print(json.dumps({'prepared':len(clips),'groups':len({c['group'] for c in clips}),
                      'half_counts':dict(Counter(c['half'] for c in clips)),
                      'sampled_frames':sum(len(sample_ids(c['n'])) for c in clips)}),flush=True)

def check_inputs(out,check_sources=True):
    m=read_json(out/'manifest.json');inputs=read_json(out/'inputs.json')
    if CONFIG!=m['config'] or canonical_hash(inputs)!=m['inputs_sha256']:raise ValueError('Inputs/config drift')
    if {p:digest(ROOT/p) for p in CODE}!=m['code']:raise ValueError('Experiment code drift')
    if check_frozen()!=m['stable']:raise ValueError('Frozen set drift')
    if digest(UPSTREAM/'ACQUISITION.json')!=m['upstream_acquisition_sha256']:raise ValueError('Upstream acquisition drift')
    if digest(LABELS)!=m['labels_sha256'] or digest(CACHE/'manifest.json')!=m['cache_manifest_sha256']:
        raise ValueError('Evaluation inputs drift')
    for p,h in m['cache_files'].items():
        if digest(ROOT/p)!=h:raise ValueError('Cache drift')
    if check_sources:
        for p,h in m['sources'].items():
            if digest(ROOT/p)!=h:raise ValueError('Source drift')
    return m,inputs

def frame_batches(clip):
    from hlcut.io_utils import iter_frames
    from .model import preprocess
    ids=sample_ids(clip['n']);seen=[];batch=[]
    for fid,frame in iter_frames(str(ROOT/clip['source']),[clip['start']+f for f in ids]):
        local=fid-clip['start']
        if frame.shape[:2]!=(clip['H'],clip['W']):raise ValueError('Frame dimensions mismatch')
        seen.append(local);batch.append((local,preprocess(frame)))
        if len(batch)==CONFIG['chunk']:yield batch;batch=[]
    if batch:yield batch
    if seen!=ids:raise ValueError('Incomplete sequential frame decode')

def validate_boxes(boxes,clip):
    from hlcut.metric import check_prediction
    if len(boxes)!=clip['n']:raise ValueError('Incomplete prediction track')
    for f,b in enumerate(boxes):
        if len(b)!=3 or any(type(v)!=int for v in b):raise ValueError('Noninteger box')
        if b[2]!=clip['baseline_boxes'][f][2]:raise ValueError('Crop size drift')
        fid,why=check_prediction({'frame':f,'bboxes':b},clip['n'],clip['W'],clip['H'],9,16)
        if fid!=f:raise ValueError('Invalid prediction: '+str(why))

def infer(out,limit):
    import torch,cv2
    from .model import Unisal
    from hlcut.pipeline import _anchor_track,Config
    from hlcut.geometry import place_box
    torch.set_num_threads(1);cv2.setNumThreads(1)
    m,inputs=check_inputs(out)
    todo=[]
    for clip in inputs['clips']:
        p=out/'clips'/(clip['video_id']+'.json')
        if p.exists():
            old=read_json(p)
            if old['input_sha256']!=canonical_hash(clip):raise ValueError('Cached prediction input drift')
            validate_boxes(old['boxes'],clip)
        else:todo.append(clip)
    if limit:todo=todo[:limit]
    if not todo:print(json.dumps({'nothing_to_infer':True}),flush=True);return
    begin=time.perf_counter();model=Unisal(UPSTREAM)
    if model.device=='cuda':torch.cuda.reset_peak_memory_stats()
    for i,clip in enumerate(todo,1):
        start=time.perf_counter();anchors=[];invalid=0
        for fid,logmap in model.predict_batches(frame_batches(clip)):
            base=clip['baseline_boxes'][fid]
            if not np.isfinite(logmap).all():box=base;invalid+=1
            else:box=choose_box(logmap,clip['W'],clip['H'],base)
            anchors.append({'fid':fid,'s_anchor':box[0]+box[2]/2.,'s_weight':1.})
        if [a['fid'] for a in anchors]!=sample_ids(clip['n']):raise ValueError('Inference missing frames')
        track=_anchor_track(anchors,clip['n'],clip['fps'],'x',clip['W'],clip['H'],Config(spatial='detect',smooth_sec=.4))
        boxes=[list(place_box(float(x),clip['H']/2.,clip['W'],clip['H'],clip['baseline_boxes'][f][2],9,16)) for f,x in enumerate(track)]
        validate_boxes(boxes,clip)
        if model.device=='cuda':torch.cuda.synchronize()
        payload={'video_id':clip['video_id'],'input_sha256':canonical_hash(clip),'boxes':boxes,
                 'sample_ids':[a['fid'] for a in anchors],'sample_x':[a['s_anchor'] for a in anchors],
                 'invalid_maps':invalid,'seconds':time.perf_counter()-start}
        save_json(out/'clips'/(clip['video_id']+'.json'),payload)
        print(json.dumps({'done_this_run':i,'todo_this_run':len(todo),'video_id':clip['video_id'],
                          'seconds':payload['seconds'],'invalid_maps':invalid,'elapsed':time.perf_counter()-begin,
                          'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20 if model.device=='cuda' else None}),flush=True)

def metrics(*args, **kwargs):
    raise NotImplementedError('Offline evaluation helper is not included in the public inference runtime.')

def cluster_ci(values,groups,n_boot=10000):
    values=np.asarray(values,float);unique=sorted(set(groups))
    if len(values)!=len(groups) or not unique or not np.isfinite(values).all():raise ValueError('Bad paired values')
    sums=np.array([values[np.array(groups)==g].sum() for g in unique])
    counts=np.array([sum(g==x for x in groups) for g in unique])
    idx=np.random.default_rng(20261002).integers(0,len(unique),size=(n_boot,len(unique)))
    return np.quantile(sums[idx].sum(1)/counts[idx].sum(1),[.025,.975]).tolist()

def score(out):
    from hlcut.geometry import place_box
    m,inputs=check_inputs(out);labels={r['video_id']:r for r in map(json.loads,LABELS.read_text(encoding='utf-8-sig').splitlines())}
    rows=[];invalid=total_maps=0;max_drift=0;output_hashes={}
    for clip in inputs['clips']:
        p=out/'clips'/(clip['video_id']+'.json');result=read_json(p)
        if result['input_sha256']!=canonical_hash(clip):raise ValueError('Prediction mismatch')
        validate_boxes(result['boxes'],clip)
        if result['sample_ids']!=sample_ids(clip['n']):raise ValueError('Sampling mismatch')
        invalid+=result['invalid_maps'];total_maps+=len(result['sample_ids']);output_hashes[p.name]=digest(p)
        cbox=list(place_box(clip['W']/2.,clip['H']/2.,clip['W'],clip['H'],clip['baseline_boxes'][0][2],9,16))
        tracks={'baseline':clip['baseline_boxes'],'unisal':result['boxes'],'center':[cbox]*clip['n']}
        row={k:clip[k] for k in ['video_id','group','half','stratum']}
        for name,boxes in tracks.items():row[name]=metrics(boxes,labels[clip['video_id']]['cropRois'],clip['window'],clip['W'],clip['H'])
        old=read_json(CACHE/'clips'/(clip['video_id']+'.json'))['row']['1.7']['baseline']['joint']
        max_drift=max(max_drift,abs(old-row['baseline']['joint']));rows.append(row)
    if max_drift>1e-12:raise ValueError('Baseline score reproduction failed')
    if max(r[n]['reference_error'] for r in rows for n in tracks)>1e-6:raise ValueError('Reference score mismatch')
    def summarize(rs):
        return {'n':len(rs),'groups':len({r['group'] for r in rs}),
                **{name:float(np.mean([r[name]['joint'] for r in rs])) for name in tracks},
                'delta':float(np.mean([r['unisal']['joint']-r['baseline']['joint'] for r in rs]))}
    summary=summarize(rows);summary['halves']={str(h):summarize([r for r in rows if r['half']==h]) for h in [0,1]}
    summary['strata']={s:summarize([r for r in rows if r['stratum']==s]) for s in sorted({r['stratum'] for r in rows})}
    groups=[r['group'] for r in rows]
    for reference in ['baseline','center']:
        values=[r['unisal']['joint']-r[reference]['joint'] for r in rows]
        summary['ci95_vs_'+reference]=cluster_ci(values,groups)
    summary['gt_iou']={name:float(np.mean([r[name]['gt_iou'] for r in rows])) for name in tracks}
    ds=np.array([r['unisal']['joint']-r['baseline']['joint'] for r in rows])
    summary['improved_worse_same']=[int((ds>1e-10).sum()),int((ds < -1e-10).sum()),int((np.abs(ds)<=1e-10).sum())]
    passed=invalid/max(total_maps,1)<=.01 and all(summary['ci95_vs_'+r][0]>0 for r in ['baseline','center']) and all(summary['halves'][str(h)]['delta']>0 for h in [0,1])
    summary.update({'complete':True,'invalid_maps':invalid,'sampled_maps':total_maps,'max_baseline_drift':max_drift,
                    'decision':'needs_other_source_validation' if passed else 'reject_configuration',
                    'new_submission_recommended':False,'limitations':m['limitations']})
    save_json(out/'rows.json',rows);save_json(out/'score.json',summary)
    save_json(out/'verification.json',{'output_hashes':output_hashes,'baseline_unchanged':digest(ROOT/BASELINE)==BASEHASH,
                                    'reference_error_max':max(r[n]['reference_error'] for r in rows for n in tracks),
                                    'baseline_score_max_difference':max_drift,'code_sha256':m['code']})
    print(json.dumps(summary,indent=2),flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['prepare','infer','score'])
    ap.add_argument('--out',default='scratch/unisal_crop/run_v1');ap.add_argument('--limit',type=int,default=0)
    args=ap.parse_args();out=(ROOT/args.out).resolve()
    if ROOT not in out.parents:raise ValueError('Output outside workspace')
    if args.phase=='prepare':prepare(out)
    elif args.phase=='infer':infer(out,args.limit)
    else:score(out)

if __name__=='__main__':main()
