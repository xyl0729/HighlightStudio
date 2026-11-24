import os,sys,json,hashlib,time
from pathlib import Path
from datetime import datetime
CODE=Path(__file__).resolve().parent
LIB=CODE/'lib';WORK=CODE.parent;SIGNALS=CODE/'signals'
sys.path.insert(0,str(LIB));sys.path.insert(0,str(LIB/'scripts'));sys.path.insert(0,str(LIB/'scratch/post'))

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,value):
    p=Path(p);p.parent.mkdir(exist_ok=True,parents=True)
    with p.open('x',encoding='utf8') as f:json.dump(value,f,ensure_ascii=False,allow_nan=False,indent=2,default=lambda a:a.tolist() if hasattr(a,'tolist') else float(a))
def load_config():return read(CODE/'config.json')
def progress(stage,**kw):
    p=WORK/(stage+'.progress.next');p.write_text(json.dumps(dict(time=datetime.now().astimezone().isoformat(),pid=os.getpid(),stage=stage,**kw),ensure_ascii=False),encoding='utf8');os.replace(p,WORK/(stage+'.progress.json'))
    print(json.dumps(dict(stage=stage,**kw),ensure_ascii=True),flush=True)
def cases():return read(CODE/'raw_manifest.json')['cases']
def asset(name):
    c=load_config();return Path(c.get('model_overrides',{}).get(name,str(Path(c['models_root'])/name)))
def validate_source(c):assert sha(c['source'])==c['source_sha256'],c['video_id']
def install_inference_IO_guard():
    project=Path(load_config()['original_project']).resolve();allowed=[WORK.resolve(),Path(load_config()['models_root']).resolve(),Path(load_config()['video_dir']).resolve(),project/'.venv']
    def hook(event,args):
        if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
            try:p=Path(os.fsdecode(args[0])).resolve()
            except (OSError,ValueError):return
            if p.is_relative_to(project) and not any(p.is_relative_to(a) for a in allowed):raise PermissionError('cold inference blocked original project read: '+str(p))
    sys.addaudithook(hook)

def finish(stage,fn):
    start=time.monotonic();rc=1;error=None;proof=None
    try:proof=fn();rc=0
    except Exception as e:
        import traceback
        error=repr(e);traceback.print_exc()
    result=dict(stage=stage,actual_exit=rc,error=error,seconds=time.monotonic()-start,proof=proof,raw_input_recomputed=rc==0 and stage in ('detector','spatial','clip','speech','posterior','pipeline'),cached_signal_inputs=False,retained_parent_raw_detection_cases=0,cached_scored_TEST_signal_inputs=False,TEST_labels=False,actual_platform_score=None)
    write(WORK/(stage+'_terminal.json'),result);return rc
