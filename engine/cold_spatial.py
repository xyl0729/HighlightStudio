import os
os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',CUDA_VISIBLE_DEVICES='-1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
from cold_runtime import *
import cv2,numpy as np,torch
from cold_contract import eligible_sor
from hlcut.io_utils import iter_frames
from hlcut.geometry import place_box
from hlcut.pipeline import Config,_anchor_track
from scratch.spatial_track_reference.track_parity import own_track
from scratch.unisal_crop.model import Unisal,preprocess
from scratch.unisal_crop.core import sample_ids,choose_box
from scratch.spatial_object_ranker.model import Ranker
from scratch.spatial_object_ranker.run import infer_case

def run():
    install_inference_IO_guard();assert read(WORK/'detector_terminal.json')['actual_exit']==0;torch.set_num_threads(2);cv2.setNumThreads(1)
    detector={str(x['video_id']):x for x in map(json.loads,(SIGNALS/'scratch/fusai/detections.jsonl').read_text(encoding='utf8').splitlines())}
    model=Unisal(asset('UNISAL-upstream'));assert model.device=='cpu';state=torch.load(SIGNALS/'scratch/spatial_object_ranker/final_v1/model.pt',map_location='cpu',weights_only=True);head=Ranker();head.load_state_dict(state['state']);head.eval();mu=np.array(state['mu'],np.float32);sd=np.array(state['sd'],np.float32);rows=[]
    for c in cases():
        v=c['video_id'];s=detector[v]
        if not eligible_sor(dict(c,axis=s['axis'])):continue
        validate_source(c);base=np.array(own_track(s,c['targetRatioWH']),int);ids=sample_ids(c['n']);anchors=[];bad=0
        def batches():
            chunk=[]
            for fid,bgr in iter_frames(c['source'],ids):
                chunk.append((fid,preprocess(bgr)))
                if len(chunk)==16:yield chunk;chunk=[]
            if chunk:yield chunk
        for fid,logmap in model.predict_batches(batches()):
            if np.isfinite(logmap).all():box=choose_box(logmap,c['W'],c['H'],base[fid])
            else:box=base[fid];bad+=1
            anchors.append(dict(fid=fid,s_anchor=float(box[0]+box[2]/2),s_weight=1.))
        assert [x['fid'] for x in anchors]==ids
        curve=_anchor_track(anchors,c['n'],c['fps'],'x',c['W'],c['H'],Config(spatial='detect',smooth_sec=.4))
        saliency=np.array([place_box(float(x),c['H']/2,c['W'],c['H'],int(base[f,2]),9,16) for f,x in enumerate(curve)],int)
        recs=s['recs'];row=infer_case(dict(c,fold='cold',baseline_boxes=saliency.tolist()),(saliency,base,recs,[r['fid'] for r in recs]),head,mu,sd)
        p=SIGNALS/'scratch/spatial_object_ranker/final_v1/test_clips'/f'{v}.json';write(p,row);rows.append(dict(video_id=v,SHA256=sha(p),nonfinite_map_original_fallback_count=bad));progress('spatial',completed_cases=len(rows),total_eligible=121,device='cpu',CPU_threads=2)
    assert len(rows)==sum(eligible_sor(dict(c,axis=detector[c['video_id']]['axis'])) for c in cases());write(WORK/'spatial_seal.json',dict(rows=rows,UNISAL_parameters=model.parameters,SOR_parameters=sum(p.numel() for p in head.parameters())))
    return dict(records=len(rows),UNISAL_parameters=model.parameters,SOR_parameters=sum(p.numel() for p in head.parameters()))

if __name__=='__main__':raise SystemExit(finish('spatial',run))
