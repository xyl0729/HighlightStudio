import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scratch.open_vocabulary_spatial_oof_20261012_v1 import core

YOLO_MAP={0:0,16:3,15:4,14:5,17:6,39:11,41:10,45:9,2:16,3:16,5:16,7:16,1:17,58:18,73:19,67:20,63:21,62:21,56:23,57:23,59:23,60:23,24:22,26:22,27:22,32:15,**{i:8 for i in range(46,56)}}

def map_yolo(rows,W,H):
    a=np.asarray(rows,float).reshape(-1,6)
    if not np.isfinite(a).all() or np.any((a[:,4]<0)|(a[:,4]>1)) or np.any(a[:,5]!=np.floor(a[:,5])):raise ValueError('Malformed cached YOLO')
    out=[]
    for x1,y1,x2,y2,confidence,cls in a:
        if int(cls) not in YOLO_MAP:continue
        x1,x2=np.clip([x1,x2],0,W);y1,y2=np.clip([y1,y2],0,H)
        if x2<=x1 or y2<=y1:continue
        out.append([x1,y1,x2,y2,confidence,YOLO_MAP[int(cls)]])
    return core.suppress_objects(np.array(out).reshape(-1,6),W,H)

def nearest(ids,frame):
    if not len(ids) or any(b<=a for a,b in zip(ids,ids[1:])):raise ValueError('Bad sample clocks')
    return int(np.argmin(np.abs(np.asarray(ids)-frame)))

def training_frames(truth):
    ids=sorted(truth)
    if not ids:raise ValueError('Missing truth cannot become negatives')
    return [ids[i] for i in sorted(set(np.rint(np.linspace(0,len(ids)-1,min(48,len(ids)))).astype(int).tolist()))]

def source_weights(items):
    from collections import Counter
    if not items or len({v for g,v,frames in items})!=len(items):raise ValueError('Duplicate/missing training clips')
    counts=Counter(g for g,v,frames in items);out=[]
    for group,vid,frames in items:
        if not frames or len({f for f,p in frames})!=len(frames) or any(not len(pool) for f,pool in frames):raise ValueError('Missing training frame candidates')
        for frame,pool in frames:out.extend([1/(len(counts)*counts[group]*len(frames)*len(pool))]*len(pool))
    a=np.asarray(out,np.float64)
    if abs(a.sum()-1)>1e-12:raise ValueError('Source unitmass drift')
    return a

def gain(x,base,truth):
    bx,by,w=base;h=w*16/9
    def iou(at):
        tx,ty,tw,th=truth;inter=max(0,min(at+w,tx+tw)-max(at,tx))*max(0,min(by+h,ty+th)-max(by,ty))
        return inter/(w*h+tw*th-inter)
    return float(iou(x)-iou(bx))
