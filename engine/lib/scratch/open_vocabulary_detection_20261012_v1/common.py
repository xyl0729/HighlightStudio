import hashlib,json,importlib.metadata
from pathlib import Path
import numpy as np

D=Path(__file__).resolve().parent;R=D.parents[1];MODEL=R/'models/GroundingDINO-tiny'
REV='a2bb814dd30d776dcf7e30523b00659f4f141c71';REPO='IDEA-Research/grounding-dino-tiny'
PROMPT='person. face. hand. dog. cat. bird. horse. fish. food. plate. cup. bottle. tool. toy. musical instrument. ball. vehicle. bicycle. plant. book. phone. computer. clothing. furniture.'
CONFIG=dict(revision=REV,repo=REPO,prompt=PROMPT,threshold=.30,text_threshold=.25,dtype='float32',batch_size=1,disable_custom_kernels=True,
            repeat_tolerance=1e-5,cpu_gpu_tolerance=1e-3,reserved_head_elements=4096)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,v):
    with Path(p).open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)
def atomic(p,v):
    p=Path(p);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8');q.replace(p)
def versions():return {x:importlib.metadata.version(x) for x in ('torch','transformers','opencv-python','numpy','safetensors','huggingface-hub')}
def check():
    m=read(D/'manifest.json')
    if m['config']!=CONFIG or m['versions']!=versions():raise ValueError('Frozen config/runtime drift')
    for rel,h in m['files'].items():
        if sha(R/rel)!=h:raise ValueError('Frozen asset/code drift:'+rel)
    for c in m['cases']:
        if sha(R/c['source'])!=c['source_sha256']:raise ValueError('Source video drift')
    return m
def decode(c,absolute):
    import cv2
    cap=cv2.VideoCapture(str(R/c['source']))
    if not cap.isOpened():raise ValueError('Source cannot open')
    try:
        if abs(cap.get(cv2.CAP_PROP_FPS)-c['fps'])>1e-4:raise ValueError('FPS clock drift')
        cap.set(cv2.CAP_PROP_POS_FRAMES,absolute);ok,bgr=cap.read()
        if not ok or bgr.shape[:2]!=(c['H'],c['W']) or abs(cap.get(cv2.CAP_PROP_POS_FRAMES)-(absolute+1))>.1:raise ValueError('Native frame decode drift')
        return cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
    finally:cap.release()
