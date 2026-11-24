from pathlib import Path
import re

def validate_index(rows):
    ids=[]
    for r in rows:
        v=str(r['video_id'])
        if not re.fullmatch(r'[A-Za-z0-9_-]+',v) or list(r['targetRatioWH']) not in ([9,16],[16,9]):raise ValueError('invalid source identity or target ratio')
        ids.append(v)
    if not ids or len(ids)!=len(set(ids)):raise ValueError('nonempty unique index required')
    return ids

def eligible_sor(c):return c['targetRatioWH']==[9,16] and c['axis']=='x'

def ensure_fresh(path):
    if Path(path).exists():raise FileExistsError('fresh output required: '+str(path))

def resolve_source(root,video_id):
    if not re.fullmatch(r'[A-Za-z0-9_-]+',str(video_id)):raise ValueError('unsafe ID')
    root=Path(root).resolve();p=root/(str(video_id)+'.mp4')
    if not p.is_relative_to(root) or not p.is_file():raise ValueError('raw source missing/outside input root')
    return p
