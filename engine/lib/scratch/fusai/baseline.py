"""Frozen first fusai baseline. No test-specific windows, ranking, or manual edits."""
import argparse
import hashlib
import json
import math
import sys
import zipfile
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
sys.path.insert(0,str(ROOT/'scratch/post'))
from build_final import full_track
from build_ef import fit, feats
from fit_decode import logit_pred, ef_decode
from hlcut.geometry import place_box, safe_crop_width
from hlcut.io_utils import load_index, read_jsonl
from hlcut.metric import check_prediction
from hlcut.pipeline import label_end
from scratch.fusai.prepare import digest


def temporal_scale(ratio):
    if list(ratio)==[16,9]:return 1.0
    if list(ratio)==[9,16]:return 1.7
    raise ValueError('Unsupported target ratio')


def require_complete(index, rows):
    expected=[str(it['video_id']) for it in index]
    ids=[str(r['video_id']) for r in rows]
    if len(set(expected))!=len(expected) or len(set(ids))!=len(ids) or set(ids)!=set(expected):
        raise ValueError('Missing, extra, or duplicate records')
    return dict(zip(ids,rows))


def choose_window(q,n,fps,ratio):
    if n<=0 or fps<=0 or len(q)!=n or not np.isfinite(q).all():
        raise ValueError('Invalid temporal input')
    return ef_decode(q,label_end(n,fps),max(1,int(fps/4)),temporal_scale(ratio))


def build_records(index,detections,posterior,calibration,model_params_b):
    samples=require_complete(index,detections)
    posts=require_complete(index,posterior)
    w,mu,sd=calibration
    out=[];fallbacks=0
    for it in index:
        vid=str(it['video_id']);s=samples[vid];p=posts[vid]
        n,fps,W,H=s['n'],s['fps'],s['W'],s['H']
        if n<=0 or fps<=0 or W<=0 or H<=0 or abs(p['dur']-n/fps)>1e-6:
            raise ValueError('Posterior/detection metadata mismatch')
        sizes=[len(p[k]) for k in ('S','E','logp')]
        if len(set(sizes))!=1 or not np.isfinite(p['logp']).all():
            raise ValueError('Malformed posterior')
        tw,th=map(float,it['targetRatioWH'])
        if s['axis'] in ('x','y'):
            boxes=full_track(s,tw,th,12.,2.,.03)
        else:
            box=list(place_box(W/2,H/2,W,H,safe_crop_width(W,H,tw,th),tw,th))
            boxes=[box]*n
        if sizes[0]:
            q=logit_pred(w,(feats(p,n,fps)-mu)/sd)
            a,b=choose_window(q,n,fps,it['targetRatioWH'])
        else:
            if not isinstance(p.get('greedy'),str) or not p['greedy'].strip():
                raise ValueError('Empty model output; not substituting predictions')
            a,b=0,label_end(n,fps)-1
            fallbacks+=1
        if not 0<=a<=b<n:raise ValueError('Decoded frame interval invalid')
        out.append({'video_id':vid,'targetRatioWH':it['targetRatioWH'],
                    'model_params_b':model_params_b,
                    'predictions':[{'frame':f,'bboxes':boxes[f]} for f in range(a,b+1)]})
    return out,{'model_answer_without_posterior_fallbacks':fallbacks}


def validate_records(records,index,metadata):
    rows=require_complete(index,records)
    total=empty=0
    for it in index:
        vid=str(it['video_id']);r=rows[vid];m=metadata[vid]
        if r['targetRatioWH']!=it['targetRatioWH']:raise ValueError('Target ratio mismatch')
        params=r.get('model_params_b',float('nan'))
        if not isinstance(params,(int,float)) or not math.isfinite(params) or not 0<params<=9:
            raise ValueError('Parameter declaration missing or over limit')
        preds=r['predictions']
        if not isinstance(preds,list):raise ValueError('Predictions must be a list')
        previous=-1
        for p in preds:
            box=p.get('bboxes',[]) if isinstance(p,dict) else []
            if not all(isinstance(v,(int,float)) and math.isfinite(v) for v in box):
                raise ValueError('Nonfinite/malformed box')
            f,reason=check_prediction(p,m['n'],m['W'],m['H'],*it['targetRatioWH'])
            if f is None or f<=previous:raise ValueError('Invalid/duplicate/unsorted prediction')
            previous=f
        total+=len(preds);empty+=not preds
    return {'records':len(records),'frames':total,'empty':empty,'invalid':0}


def write_package(records,path):
    path=Path(path);zpath=path.with_suffix('.zip')
    if path.exists() or zpath.exists():raise FileExistsError('Refusing to overwrite prediction artifacts')
    payload=(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in records)).encode('utf-8')
    with path.open('xb') as f:f.write(payload)
    with zipfile.ZipFile(zpath,'x',zipfile.ZIP_DEFLATED) as z:
        z.writestr(path.name,payload)
    with zipfile.ZipFile(zpath) as z:
        if z.testzip() is not None or z.read(path.name)!=payload:raise ValueError('ZIP validation failed')
    return {'jsonl':str(path),'zip':str(zpath),'sha256':digest(zpath),'jsonl_sha256':digest(path)}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--index',required=True);ap.add_argument('--dump',required=True)
    ap.add_argument('--post',required=True);ap.add_argument('--calibration',required=True)
    ap.add_argument('--parameter-audit',required=True);ap.add_argument('--out',required=True)
    args=ap.parse_args()
    idx=load_index(args.index);dets=read_jsonl(args.dump);posts=read_jsonl(args.post)
    audit=json.loads(Path(args.parameter_audit).read_text(encoding='utf-8'))
    with np.load(args.calibration,allow_pickle=False) as c:cal=tuple(c[k] for k in ('w','mu','sd'))
    rows,stats=build_records(idx,dets,posts,cal,audit['total_parameters']/1e9)
    stats.update(validate_records(rows,idx,{str(r['video_id']):r for r in dets}))
    stats.update(write_package(rows,args.out))
    Path(args.out).with_suffix('.report.json').write_text(json.dumps(stats,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(stats,indent=2))


if __name__=='__main__':main()
