"""Joint temporal-spatial utility helpers."""
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import time
import types
import numpy as np

from hlcut.io_utils import iter_frames, video_meta
from hlcut.pipeline import Config, _analyse
from hlcut.geometry import safe_crop_width, derived_height, free_axis, iou_xywh

ROOT=Path(__file__).resolve().parents[2]


def source_ids(local_ids,start):
    if not isinstance(start,(int,np.integer)) or start<0:raise ValueError('Invalid source offset')
    ids=list(local_ids)
    if any(not isinstance(i,(int,np.integer)) or i<0 for i in ids):raise ValueError('Invalid local index')
    return [int(i+start) for i in ids]


def analyse_clip(path,start,n,fps,W,H,axis,crop_len,cfg,detector):
    source_ids([],start)
    def clip_frames(input_path,local_ids):
        ids=list(local_ids); mapped=source_ids(ids,start); got=[]
        for fid,frame in iter_frames(input_path,mapped):
            if frame.shape[:2]!=(H,W):raise ValueError('Decoded size differs from source metadata')
            got.append(fid)
            yield fid-start,frame
        if got!=mapped:raise ValueError('Source decoding did not provide every requested frame')
    scope=dict(_analyse.__globals__);scope['iter_frames']=clip_frames
    function=types.FunctionType(_analyse.__code__,scope,_analyse.__name__,_analyse.__defaults__,_analyse.__closure__)
    return function(path,n,fps,W,H,axis,crop_len,cfg,detector)


def score_window(window,gt_mask,iou_vector):
    g=np.asarray(gt_mask,bool);u=np.asarray(iou_vector,float);a,b=window
    if g.ndim!=1 or u.shape!=g.shape or not 0<=a<=b<len(g):raise ValueError('Invalid window or mask shape')
    if not np.isfinite(u).all() or np.any((u<0)|(u>1)) or np.any(u[~g]!=0):raise ValueError('Invalid IoU vector')
    denom=b-a+1+int(g.sum())
    return {'temporal':float(2*g[a:b+1].sum()/denom),'joint':float(2*u[a:b+1].sum()/denom)}


def roi_mask(rois,n,W,H):
    g=np.zeros(n,bool);boxes={}
    for fid,box in rois:
        if not isinstance(fid,int) or not 0<=fid<n or fid in boxes:raise ValueError('Invalid/duplicate GT frame')
        if len(box)!=4 or not np.isfinite(box).all():raise ValueError('Malformed GT box')
        x,y,w,h=box
        if x<0 or y<0 or w<=0 or h<=0 or x+w>W or y+h>H:raise ValueError('GT box outside source')
        g[fid]=True;boxes[fid]=box
    if not boxes:raise ValueError('No spatial GT; do not invent labels')
    return g,boxes


def screen_roi_label(rois,n,W,H):
    try:return roi_mask(rois,n,W,H),None
    except ValueError as exc:return None,str(exc)


def write_json(path,value):
    def scalar(v):
        if isinstance(v,np.generic):return v.item()
        raise TypeError('Unsupported cache value: '+type(v).__name__)
    tmp=path.with_suffix(path.suffix+'.partial')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False,default=scalar)+'\n',encoding='utf-8')
    tmp.replace(path)


def load_inputs():
    from scratch.fusai.prepare import digest
    paths={'labels':ROOT/'data/train/高光剪辑训练集/qvhighlights-videos/train.jsonl',
           'posteriors':ROOT/'scratch/post/train_all986.jsonl',
           'split':ROOT/'scratch/selection_audit/snapshot490/manifest.json',
           'windows':ROOT/'scratch/selection_audit/target55_results/rows.json',
           'calibration':ROOT/'scratch/fusai/calibration.npz'}
    labels={r['video_id']:r for r in map(json.loads,paths['labels'].read_text(encoding='utf-8-sig').splitlines())}
    posts={r['video_id']:r for r in map(json.loads,paths['posteriors'].read_text(encoding='utf-8-sig').splitlines())}
    split=json.loads(paths['split'].read_text());wins={r['video_id']:r for r in json.loads(paths['windows'].read_text())}
    ids=sorted(v for v in split['replay'] if labels[v].get('cropRois'))
    if len(ids)!=271:raise ValueError('The fixed replay population has changed')
    train_groups={labels[v]['provenance']['source_group'].rsplit('_',2)[0] for v in split['train']}
    items=[];source_hashes={};exclusions=[]
    for vid in ids:
        label=labels[vid];p=posts[vid];group=label['provenance']['source_group']
        if group.rsplit('_',2)[0] in train_groups:raise ValueError('Source-group leakage')
        candidates=[ROOT/base/(group+'.mp4') for base in ('data/train/src','data/train/qvh_src')]
        candidates=[path for path in candidates if path.exists()]
        if not candidates:raise ValueError('Training source missing')
        path=candidates[0];rel=str(path.relative_to(ROOT)).replace('\\','/')
        if rel not in source_hashes:source_hashes[rel]=digest(path)
        ns,fps,W,H=video_meta(str(path))
        start=max(0,round(label['clip']['start_sec']*fps));end=min(ns,round(label['clip']['end_sec']*fps))
        if end-start!=p['n'] or abs(fps-p['fps'])>1e-6:raise ValueError('Clip length/fps mismatch')
        if label['targetRatioWH']!=[9,16]:raise ValueError('Unexpected validation target ratio')
        valid,reason=screen_roi_label(label['cropRois'],p['n'],W,H)
        if reason:
            exclusions.append({'video_id':vid,'reason':reason,'action':'quarantine entire training clip; do not clip or repair GT'})
            continue
        g,boxes=valid
        cw=safe_crop_width(W,H,9,16);axis=free_axis(W,H,cw,9,16)
        items.append({'video_id':vid,'group':group.rsplit('_',2)[0],'path':str(path),'source_sha256':source_hashes[rel],
                      'start':start,'n':p['n'],'fps':fps,'W':W,'H':H,'axis':axis,
                      'crop_len':cw if axis=='x' else derived_height(cw,9,16),
                      'g':g,'gt_boxes':boxes,'label':label,'post':p,'windows':wins[vid]})
    return items,{'label_quarantine':exclusions,'original_spatial_candidates':len(ids),'inputs':{name:digest(path) for name,path in paths.items()},'sources':source_hashes}


def score_clip(item,sample,cal):
    from scratch.fusai.baseline import full_track,feats,logit_pred,choose_window
    from hlcut.geometry import place_box
    n=item['n'];g=item['g'];fps=item['fps'];W=item['W'];H=item['H']
    if sample['axis'] in ('x','y'):track=full_track(sample,9,16,12.,2.,.03)
    else:track=[list(place_box(W/2,H/2,W,H,safe_crop_width(W,H,9,16),9,16))]*n
    u=np.zeros(n)
    for fid,box in item['gt_boxes'].items():
        x,y,w=track[fid];u[fid]=iou_xywh([x,y,w,derived_height(w,9,16)],box)
    w,mu,sd=cal;q=logit_pred(w,(feats(item['post'],n,fps)-mu)/sd)
    segments=np.zeros(n,bool)
    for seg in item['label']['segments']:segments[max(0,seg['start_frame']):min(n,seg['end_frame']+1)]=True
    row={'video_id':item['video_id'],'group':item['group'],'duration':n/fps,'gt_frames':int(g.sum()),
         'segments_vs_roi_frame_difference':int(np.count_nonzero(g!=segments)),
         'mean_crop_iou_on_gt':float(u[g].mean())}
    for sc,ratio in [('1.0',[16,9]),('1.7',[9,16])]:
        # Both scales are historical diagnostics on the SAME 9:16 spatial target.
        # This is not a validation set for target16:9.
        wins=item['windows'][sc]['windows'];baseline=choose_window(q,n,fps,ratio)
        if tuple(baseline)!=tuple(wins['baseline']):raise ValueError('Frozen baseline window drift')
        results={}
        for name,window in [('baseline',baseline),('failed_qrank',wins['selected'])]:
            results[name]=score_window(window,g,u)
            results[name]['segment_temporal']=score_window(window,segments,np.zeros(n))['temporal']
            results[name]['window']=list(window)
        row[sc]=results
    return row,u


def summarize(rows,full):
    from scratch.selection_audit.core import cluster_ci
    groups=[r['group'] for r in rows];summary={'records':len(rows),'groups':len(set(groups)),'complete':full,
        'target_ratio':'9:16 only','primary_scale':'1.7','mean_crop_iou_on_gt':float(np.mean([r['mean_crop_iou_on_gt'] for r in rows])),
        'clips_with_segment_roi_difference':sum(r['segments_vs_roi_frame_difference']>0 for r in rows),
        'limitations':['Repeatedly-used training replay, not fresh blind validation.',
                       'No target16:9 GT; cannot justify landscape-target changes.',
                       'Temporal and perfect-spatial decompositions are diagnostics, not achievable score promises.']}
    for sc in ('1.0','1.7'):
        result={}
        for metric in ('segment_temporal','temporal','joint'):
            base=np.array([r[sc]['baseline'][metric] for r in rows]);rank=np.array([r[sc]['failed_qrank'][metric] for r in rows]);delta=rank-base
            result[metric]={'baseline':float(base.mean()),'failed_qrank':float(rank.mean()),'delta':float(delta.mean()),
                            'ci95_by_source':cluster_ci(delta,groups)}
        j=result['joint'];result['validation_still_favors_failed_qrank']=j['delta']>0
        result['fails_known_failure_direction_check']=j['delta']>0
        summary[sc]=result
    summary['new_package_recommended']=False
    return summary


def main():
    from scratch.fusai.prepare import digest
    ap=argparse.ArgumentParser();ap.add_argument('--limit',type=int,default=0);ap.add_argument('--device',default='0')
    args=ap.parse_args();out=ROOT/'scratch/joint_audit/run_v2';out.mkdir(exist_ok=True);(out/'clips').mkdir(exist_ok=True)
    cfg=Config(sample_fps=4.,analysis_width=256,spatial='detect',det_batch=16,y_head=True,headroom=.03,keep_dets=True,temporal='none')
    items,input_hashes=load_inputs()
    code_paths=['scratch/joint_audit/audit.py','scratch/fusai/baseline.py','scripts/build_final.py','scripts/build_anchor_rule.py',
                'hlcut/pipeline.py','hlcut/io_utils.py','hlcut/geometry.py','hlcut/spatial.py','scratch/post/build_ef.py','scratch/post/fit_decode.py']
    manifest={**input_hashes,'config':asdict(cfg),'detector':'yolo11s.pt','detector_sha256':digest(ROOT/'yolo11s.pt'),
              'code':{p:digest(ROOT/p) for p in code_paths},'ids':[r['video_id'] for r in items]}
    mp=out/'manifest.json'
    if mp.exists() and json.loads(mp.read_text())!=manifest:raise ValueError('Audit manifest changed; do not reuse caches')
    if not mp.exists():write_json(mp,manifest)
    with np.load(ROOT/'scratch/fusai/calibration.npz',allow_pickle=False) as c:cal=tuple(c[k] for k in ('w','mu','sd'))
    selected=items[:args.limit] if args.limit else items;detector=None;rows=[];t0=time.monotonic();inferred=0
    for i,item in enumerate(selected,1):
        path=out/'clips'/(item['video_id']+'.json')
        if path.exists():
            stored=json.loads(path.read_text());row=stored['row']
            if stored['source_sha256']!=item['source_sha256']:raise ValueError('Source cache fingerprint changed')
        else:
            if detector is None:
                from hlcut.spatial import YoloDetector
                detector=YoloDetector(str(ROOT/'yolo11s.pt'),device=args.device,conf=.25,imgsz=640)
            sample={k:item[k] for k in ('video_id','n','fps','W','H','axis','crop_len')}
            sample['recs']=analyse_clip(item['path'],item['start'],item['n'],item['fps'],item['W'],item['H'],item['axis'],item['crop_len'],cfg,detector)
            row,u=score_clip(item,sample,cal);inferred+=1
            write_json(path,{'source_sha256':item['source_sha256'],'sample':sample,'iou_vector':u.tolist(),'row':row})
        rows.append(row)
        if i%10==0 or i==len(selected):
            state={'status':'running','completed':i,'requested':len(selected),'total':len(items),'newly_inferred':inferred,'elapsed_sec':round(time.monotonic()-t0,1)}
            write_json(out/'state.json',state);print(json.dumps(state),flush=True)
    summary=summarize(rows,len(rows)==len(items));summary['label_quarantine']=input_hashes['label_quarantine'];write_json(out/('summary.json' if summary['complete'] else 'smoke_summary.json'),summary)
    write_json(out/'state.json',{'status':'complete' if summary['complete'] else 'smoke_complete','completed':len(rows),'total':len(items),'elapsed_sec':round(time.monotonic()-t0,1)})
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
