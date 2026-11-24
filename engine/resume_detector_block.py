import os
os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',CUDA_LAUNCH_BLOCKING='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
from cold_runtime import *
import argparse,numpy as np,cv2,torch
from hlcut.geometry import derived_height,free_axis,safe_crop_width
from hlcut.pipeline import Config,_analyse
from hlcut.spatial import YoloDetector
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--start',type=int,required=True);ap.add_argument('--end',type=int,required=True);a=ap.parse_args();stage=f'detector_block{a.start:03d}'
    def run():
        install_inference_IO_guard();torch.set_num_threads(2);cv2.setNumThreads(1);detector=YoloDetector(str(asset('yolo11s.pt')),device=load_config()['detector_device'],conf=.25,imgsz=640)
        detector([np.zeros((144,256,3),np.uint8)]);torch.set_num_threads(2)
        cfg=Config(sample_fps=4.,analysis_width=256,spatial='detect',det_batch=16,y_head=True,headroom=.03,keep_dets=True,temporal='none');rows=[]
        cs=cases();assert 0<=a.start<a.end<=len(cs) and a.end-a.start<=32
        for i in range(a.start,a.end):
            c=cs[i];validate_source(c);tw,th=c['targetRatioWH'];cw=safe_crop_width(c['W'],c['H'],tw,th);axis=free_axis(c['W'],c['H'],cw,tw,th);length=cw if axis=='x' else derived_height(cw,tw,th)
            recs=_analyse(c['source'],c['n'],c['fps'],c['W'],c['H'],axis,length,cfg,detector);torch.cuda.synchronize()
            row=dict(video_id=c['video_id'],n=c['n'],fps=c['fps'],W=c['W'],H=c['H'],axis=axis,crop_len=length,recs=recs)
            assert [r['fid'] for r in recs]==list(range(0,c['n'],max(1,int(round(c['fps']/4.)))))
            for rec in recs:assert np.isfinite(np.asarray(rec['dets'],float)).all()
            path=WORK/'fresh_detections'/f"{c['video_id']}.json";write(path,row);rows.append(dict(video_id=c['video_id'],path=str(path),SHA256=sha(path)))
            progress(stage,completed_new_cases=len(rows),first_case=a.start,last_completed_case=i,remaining_total_cases=len(cs)-i-1,CPU_threads=2,device='0')
        return dict(start=a.start,end=a.end,rows=rows,parameters=sum(p.numel() for p in detector.model.model.parameters()),synchronous_CUDA=True)
    return finish(stage,run)
if __name__=='__main__':raise SystemExit(main())
