"""Build predictions from delivered inference signals; no previous result or decision lookup."""
import os
os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
from pathlib import Path
import sys,json,zipfile,numpy as np,time,hashlib
PACKAGE=Path(__file__).resolve().parent
SNAP=PACKAGE/'lib'
R=PACKAGE/'signals'
sys.path[:0]=[str(SNAP),str(SNAP/'scripts'),str(SNAP/'scratch/post')]
from replay_contract import cutoff,remove,validate
from algorithm_core import gap,proposal_features
import argparse,joblib
from scratch.baseline_continuity_path_20261010_v1.common import arithmetic,current_boxes,baseline_window
from scratch.spatial_track_reference.track_parity import own_track
from scratch.selected_y_smooth_20261005_v1.core import smooth_y_row
from scratch.four_explore_20261011_v1.core import center_x
from scratch.audiovisual_alignment_20261013_v1 import core
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',required=True);args=ap.parse_args()
    D=Path(args.out);D.mkdir(parents=True,exist_ok=False);tick=time.monotonic();native=arithmetic()
    index=read(R/'data/fusai/index.json');meta={str(c['video_id']):c for c in read(R/'scratch/clip_temporal_candidate/inputs.json')['cases']}
    det={str(x['video_id']):x for x in map(json.loads,(R/'scratch/fusai/detections.jsonl').read_text(encoding='utf-8-sig').splitlines())}
    posts={str(x['video_id']):x for x in map(json.loads,(R/'scratch/temporal_tighten/post_test_merged.jsonl').read_text(encoding='utf-8-sig').splitlines())}
    with np.load(R/'scratch/fusai/calibration.npz') as z:cal={k:z[k] for k in ('w','mu','sd')}
    t=R/'scratch/audiovisual_exploratory_delivery_20261013_v1'
    with np.load(t/'final_model.npz') as z:beta=z['beta'];sd=z['sd']
    rows=[]
    for it in index:
        v=str(it['video_id']);c=meta[v];ratio=it['targetRatioWH'];n=c['n'];fps=c['fps'];s=det[v]
        raw=native['features'](posts[v],n,fps,use_pos=False);q=native['logit_pred'](cal['w'],(raw-cal['mu'])/cal['sd'])
        base=native['ef_decode'](q,native['label_end'](n,fps),max(1,int(fps/4)),1.7 if ratio==[9,16] else 1.)
        track=np.array(own_track(s,ratio),int)
        if ratio==[9,16]:
            qf=np.maximum(q-.05,0)
            base=native['ef_decode'](qf,native['label_end'](n,fps),max(1,int(fps/4)),1.7)
            p=R/'scratch/spatial_object_ranker/final_v1/test_clips'/f'{v}.json'
            if p.exists():
                base=baseline_window(c,posts[v],cal,native)
                track=current_boxes(c,np.array(read(p)['boxes'],int),base)
                track[:,0]=center_x(track[:,0].tolist(),track[:,2].tolist(),c['W'])
                speech=read(t/'features'/f'{v}.json')
                with np.load(t/'features'/f'{v}.npz') as z:ids=z['ids'];local=z['local'].astype(float);topic=z['topic'].astype(float);mask=z['mask']
                with np.load(R/'scratch/clip_temporal_candidate/features'/f'{v}.npz') as z:vi=z['ids'];vx=z['features'].astype(float)
                visual=np.column_stack([np.interp(ids,vi,vx[:,k]) for k in range(512)])
                aligned,_,valid=core.alignment(visual,local,topic,mask)
                if valid.sum()>=2 and speech['raw_transcript']['parse_failed_chunks']==0:
                    ws=core.pool(n,fps,base);X=core.windows_features(ws,np.maximum(q-.05,0),aligned,ids,n)
                    base=ws[core.select((X/sd)@beta)]
        elif c['W']>c['H'] and np.all(track==np.array([0,0,c['W']])):
            qf=np.maximum(q-.05,0);div=8
            if qf.sum()<=0:qf=q;div=4
            base=native['ef_decode'](qf,native['label_end'](n,fps),max(1,int(fps/div)),2.)
        a,b=map(int,base)
        row=dict(video_id=it['video_id'],targetRatioWH=ratio,predictions=[dict(frame=f,bboxes=track[f].tolist()) for f in range(a,b+1)],model_params_b=5.59)
        if ratio==[16,9]:row=smooth_y_row(row,c)
        rows.append(row)

    sources=read(PACKAGE/'replay_inputs/source_identity.json')
    models=[joblib.load(PACKAGE/f'learned/model{i}.joblib') for i in (0,1)]
    cuts=[read(PACKAGE/f'learned/policy{i}.json')['cutoff'] for i in (0,1)]
    decisions=[]
    for i,row in enumerate(rows):
        v=str(row['video_id']);c=meta[v];k=int(sources[v]['source_SHA256'],16)%2
        fs=[p['frame'] for p in row['predictions']]
        raw=native['features'](posts[v],c['n'],c['fps'],use_pos=False)
        q0=native['logit_pred'](cal['w'],(raw-cal['mu'])/cal['sd'])
        if row['targetRatioWH']==[9,16]:anchor,floor,scale='T35',.05,1.7
        elif c['W']>c['H'] and all(p['bboxes']==[0,0,c['W']] for p in row['predictions']):anchor,floor,scale='op20',.05,2.
        else:anchor,floor,scale='op10',0.,1.
        q=np.maximum(q0-floor,0) if floor else q0
        if q.sum()<=0:q=q0
        g=gap(q,fs[0],fs[-1]+1,c['fps'],scale)
        base=np.zeros(c['n'],bool);base[fs]=True;keep=base.copy()
        if g:keep[g[0]:g[1]]=False
        X=proposal_features(raw,q,base,keep,c['fps'],scale)
        prob=float(models[k].predict_proba(X[None])[0,1]);cut=cutoff(k,anchor,*cuts)
        take=bool(g and prob>cut)
        if take:rows[i]=remove(row,g)
        decisions.append(dict(video_id=v,member=k,anchor=anchor,probability=prob,selected=take,gap=g))
    validate(rows,meta)
    output=D/'predictions.jsonl'
    output.write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in rows),encoding='utf8')
    multis=sum(any(b['frame']!=a['frame']+1 for a,b in zip(r['predictions'],r['predictions'][1:])) for r in rows)
    result=dict(actual_exit=0,records=len(rows),multisegment_records=multis,raw_encoder_replayed=False,
                old_submission_not_in_prediction_inputs=True,old_decisions_not_in_prediction_inputs=True,
                seconds=time.monotonic()-tick)
    (D/'RESULT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    (D/'decisions.json').write_text(json.dumps(decisions,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
