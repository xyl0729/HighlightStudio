"""Prepare fresh video inference and run the delivered stages in sequence."""
from pathlib import Path
import argparse, json, shutil, hashlib, subprocess, sys, os, time
from datetime import datetime

HERE=Path(__file__).resolve().parent

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf8') as f:json.dump(x,f,ensure_ascii=False,indent=2)

def prepare(index_path,video_dir,models,work,device):
    if work.exists():raise FileExistsError('Choose a new work directory; existing evidence is never replaced')
    import cv2
    sys.path.insert(0,str(HERE))
    from cold_contract import validate_index,resolve_source
    content=index_path.read_text(encoding='utf-8-sig')
    try:index=json.loads(content)
    except json.JSONDecodeError:index=[json.loads(x) for x in content.splitlines() if x.strip()]
    if isinstance(index,dict):index=[index]
    for x in index:x['video_id']=str(x['video_id'])
    validate_index(index)
    for name in ('Video-ORA-4B','CLIP-ViT-B-32','Whisper-small','UNISAL-upstream','yolo11s.pt'):
        if not (models/name).exists():raise FileNotFoundError('Required model asset: '+name)
    work.mkdir(parents=True)
    code=work/'code'
    shutil.copytree(HERE,code,ignore=shutil.ignore_patterns('signals','training_evidence','replay_inputs',
                     '__pycache__','*.pyc','reference_video_manifest.json'))
    signals=code/'signals'
    for rel in ('scratch/fusai/calibration.npz',
                'scratch/spatial_object_ranker/final_v1/model.pt',
                'scratch/audiovisual_exploratory_delivery_20261013_v1/final_model.npz'):
        q=signals/rel;q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(HERE/'signals'/rel,q)
    cases=[];sources={}
    for x in index:
        path=resolve_source(video_dir,x['video_id']);cap=cv2.VideoCapture(str(path))
        try:
            if not cap.isOpened():raise ValueError('Cannot read source '+x['video_id'])
            n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=float(cap.get(cv2.CAP_PROP_FPS))
            W=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));H=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            ok,first=cap.read()
            if min(n,W,H)<=0 or fps<=0 or not ok:raise ValueError('Invalid media metadata '+x['video_id'])
        finally:cap.release()
        digest=sha(path)
        cases.append(dict(video_id=x['video_id'],source=str(path),source_sha256=digest,start=0,
                          n=n,fps=fps,W=W,H=H,targetRatioWH=x['targetRatioWH']))
        sources[x['video_id']]=dict(source_SHA256=digest,member=int(digest,16)%2)
    write(code/'raw_manifest.json',dict(cases=cases))
    write(signals/'data/fusai/index.json',index)
    write(signals/'scratch/clip_temporal_candidate/inputs.json',dict(cases=cases))
    write(code/'replay_inputs/source_identity.json',sources)
    package=HERE.parent
    write(code/'config.json',dict(models_root=str(models),video_dir=str(video_dir),
          original_project=str(package),detector_device=device,
          model_overrides={'UNISAL-upstream':str(models/'UNISAL-upstream')}))
    static={str(p.relative_to(code)):sha(p) for p in code.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    write(work/'INPUT_MANIFEST.json',dict(records=len(cases),source_index_SHA256=sha(index_path),
          files=static,
          model_files={str(p.relative_to(models)):sha(p) for p in (models/'yolo11s.pt',models/'UNISAL-upstream/ACQUISITION.json') if p.is_file()}))
    return code,cases

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--index',type=Path,required=True);ap.add_argument('--video-dir',type=Path,required=True)
    ap.add_argument('--models',type=Path,required=True);ap.add_argument('--work',type=Path,required=True)
    ap.add_argument('--detector-device',default='0');ap.add_argument('--prepare-only',action='store_true')
    args=ap.parse_args();work=args.work.resolve();started=time.monotonic()
    code,cases=prepare(args.index.resolve(),args.video_dir.resolve(),args.models.resolve(),work,args.detector_device)
    write(work/'PREPARATION.json',dict(actual_exit=0,records=len(cases),inference_executed=False,
          seconds=time.monotonic()-started))
    if args.prepare_only:
        print(json.dumps(dict(actual_exit=0,prepared_records=len(cases),inference_executed=False)));return
    stages=[('detector',[sys.executable,'-u',str(code/'fresh_detector.py')]),
            ('spatial',[sys.executable,'-u',str(code/'cold_spatial.py')]),
            ('clip',[sys.executable,'-u',str(code/'cold_clip.py')]),
            ('speech',[sys.executable,'-u',str(code/'cold_speech.py')]),
            ('posterior',[sys.executable,'-u',str(code/'cold_posterior.py')]),
            ('build',[sys.executable,'-u',str(code/'build_from_signals.py'),'--out',str(work/'built')]),
            ('semifinal_postprocess',[sys.executable,'-u',str(code/'postprocess_semifinal.py')]),
            ('validate',[sys.executable,'-u',str(code/'validate_output.py')])]
    env=dict(os.environ,PYTHONIOENCODING='utf8',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
             OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
             TORCHINDUCTOR_CACHE_DIR=str(work/'runtime_cache'))
    results={};rc=0
    for stage,command in stages:
        write(work/(stage+'_COMMAND.json'),dict(command=command,time=datetime.now().astimezone().isoformat()))
        with (work/(stage+'.stdout.log')).open('xb') as out,(work/(stage+'.stderr.log')).open('xb') as err:
            rc=subprocess.call(command,cwd=code,env=env,stdout=out,stderr=err)
        results[stage]=rc;write(work/(stage+'_EXIT.json'),dict(actual_exit=rc))
        if rc:break
    result=dict(actual_exit=rc,stages=results,records=len(cases),seconds=time.monotonic()-started)
    write(work/'PROCESS_EXIT.json',result)
    print(json.dumps(result,ensure_ascii=False))
    raise SystemExit(rc)

if __name__=='__main__':main()
