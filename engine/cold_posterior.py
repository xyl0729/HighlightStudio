import os
os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUDA_LAUNCH_BLOCKING='1')
from cold_runtime import *
import runpy,gc

def run_original(index,out):
    import torch
    from hlcut.vlm import VideoLLM
    counts=[];original=VideoLLM.__init__
    def counted(self,*args,**kwargs):
        original(self,*args,**kwargs);quant={}
        for module in self.model.modules():
            if module.__class__.__name__=='Linear4bit':quant[id(module.weight)]=int(module.in_features*module.out_features)
        counts.append(sum(quant.get(id(p),p.numel()) for p in self.model.parameters()))
    VideoLLM.__init__=counted
    sys.argv=[str(LIB/'scratch/post/ora_posterior.py'),'--index',str(index),'--video-dir',load_config()['video_dir'],'--out',str(out),'--fps','2.5','--model',str(asset('Video-ORA-4B')),'--max-s','400','--top-s','12','--batch','16','--max-px',str(256*448)]
    try:runpy.run_path(str(LIB/'scratch/post/ora_posterior.py'),run_name='__main__')
    finally:VideoLLM.__init__=original
    assert len(counts)==1;gc.collect();torch.cuda.empty_cache();return counts[0]

def run():
    install_inference_IO_guard();import torch,numpy as np
    torch.set_num_threads(2);index=read(SIGNALS/'data/fusai/index.json');pref=WORK/'posterior_preflight_index.json';write(pref,index[:2]);dest=WORK/'raw_posteriors';dest.mkdir()
    counts=[]
    for key in ('A','B'):counts.append(run_original(pref,dest/(key+'.jsonl')))
    a=[json.loads(s) for s in (dest/'A.jsonl').read_text(encoding='utf8').splitlines()];b=[json.loads(s) for s in (dest/'B.jsonl').read_text(encoding='utf8').splitlines()];error=0.
    assert len(a)==len(b)==2
    for x,y in zip(a,b):
        assert x['video_id']==y['video_id'] and x['S']==y['S'] and x['E']==y['E'] and x['greedy']==y['greedy']
        if x['logp']:error=max(error,float(np.max(abs(np.array(x['logp'])-np.array(y['logp'])))))
    assert error<=1e-4+1e-12,('posterior repeated quantized inference',error)
    write(WORK/'posterior_preflight.json',dict(passed=True,real_sources=2,independent_model_loads=2,raw_logp_repeat_error=error,prequantization_parameters=counts))
    raw=dest/'all426.jsonl';counts.append(run_original(SIGNALS/'data/fusai/index.json',raw));rows=[json.loads(s) for s in raw.read_text(encoding='utf8').splitlines()];metadata={c['video_id']:c for c in cases()};assert len(rows)==len(index) and {str(r['video_id']) for r in rows}==set(metadata)
    p=SIGNALS/'scratch/temporal_tighten/post_test_merged.jsonl';p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf8') as f:
        for r in rows:
            c=metadata[str(r['video_id'])];assert abs(r['dur']-c['n']/c['fps'])<1e-6;r['n']=c['n'];r['fps']=c['fps'];f.write(json.dumps(r,allow_nan=False)+'\n')
    assert len(set(counts))==1;write(WORK/'posterior_seal.json',dict(records=len(rows),SHA256=sha(p),prequantization_parameters=counts[-1],preflight_error=error,old_posterior_input=False))
    return dict(records=len(rows),prequantization_parameters=counts[-1],preflight_error=error)

if __name__=='__main__':raise SystemExit(finish('posterior',run))
