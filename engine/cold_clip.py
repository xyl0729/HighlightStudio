import os
os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUDA_LAUNCH_BLOCKING='1')
from cold_runtime import *
import cv2,numpy as np,torch
from transformers import CLIPVisionConfig,CLIPVisionModelWithProjection,CLIPImageProcessor
from hlcut.io_utils import iter_frames

def run():
    install_inference_IO_guard();torch.set_num_threads(2);cv2.setNumThreads(1);torch.backends.cudnn.allow_tf32=True
    root=asset('CLIP-ViT-B-32');cfg=CLIPVisionConfig.from_pretrained(root,local_files_only=True);model=CLIPVisionModelWithProjection.from_pretrained(root,config=cfg,local_files_only=True,weights_only=True,dtype=torch.float32).cuda().eval();processor=CLIPImageProcessor.from_pretrained(root,local_files_only=True)
    cs=[c for c in cases() if (SIGNALS/'scratch/spatial_object_ranker/final_v1/test_clips'/f"{c['video_id']}.json").exists()];assert cs or not any(c['targetRatioWH']==[9,16] and c['W']>c['H']*9/16 for c in cases())
    files={}
    for i,c in enumerate(cs):
        validate_source(c);ids=np.arange(0,c['n'],15,dtype=np.int64);values=[];batch=[];seen=[]
        def encode():
            x=processor(images=batch,return_tensors='pt')['pixel_values'].cuda()
            with torch.inference_mode():y=model(pixel_values=x).image_embeds;y=y/y.norm(dim=-1,keepdim=True)
            a=y.cpu().numpy();assert np.isfinite(a).all();return a
        def flush():
            a=encode()
            if i<2:
                b=encode();assert np.max(abs(a-b))<=1e-5,'CLIP repeat failed'
            values.extend(a);batch.clear()
        for fid,bgr in iter_frames(c['source'],ids):
            batch.append(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB));seen.append(fid)
            if len(batch)==16:flush()
        if batch:flush()
        assert seen==ids.tolist();a=np.asarray(values,np.float32);p=SIGNALS/'scratch/clip_temporal_candidate/features'/f"{c['video_id']}.npz";p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('xb') as f:np.savez_compressed(f,ids=ids,features=a)
        files[str(p)]=sha(p);progress('clip',completed_cases=i+1,total_cases=len(cs))
    write(WORK/'clip_seal.json',dict(files=files,parameters=sum(p.numel() for p in model.parameters())))
    return dict(records=121,parameters=sum(p.numel() for p in model.parameters()))

if __name__=='__main__':raise SystemExit(finish('clip',run))
