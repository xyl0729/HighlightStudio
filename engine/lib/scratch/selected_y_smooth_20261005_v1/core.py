"""Fixed nonlearned y-only transform; never changes temporal selection."""
import copy, math
import numpy as np
from hlcut.spatial import smooth_track

def smooth_y_row(row, meta):
    W,H,n=meta['W'],meta['H'],meta['n'];fps=meta['fps'];ratio=row['targetRatioWH']
    if any(type(v)!=int or v<=0 for v in [W,H,n]) or not math.isfinite(fps) or fps<=0:raise ValueError('invalid metadata')
    if len(ratio)!=2 or any(type(v)!=int or v<=0 for v in ratio):raise ValueError('invalid ratio')
    params=row['model_params_b']
    if not math.isfinite(params) or not 0<params<=9:raise ValueError('invalid parameter declaration')
    tw,th=ratio;preds=row['predictions'];widths=set();prev=None
    for p in preds:
        f=p['frame'];b=p['bboxes']
        if type(f)!=int or not 0<=f<n or (prev is not None and f!=prev+1):raise ValueError('nonconsecutive or invalid frame')
        if len(b)!=3 or any(type(v)!=int for v in b):raise ValueError('noninteger box')
        x,y,w=b
        if min(x,y)<0 or w<=0 or x+w>W or y*tw+w*th>H*tw:raise ValueError('invalid continuous geometry')
        widths.add(w);prev=f
    if len(widths)>1:raise ValueError('variable-width crop unsupported')
    out=copy.deepcopy(row)
    if len(preds)<4:return out
    w=next(iter(widths));upper_num=H*tw-w*th
    if W-w>2 or upper_num<=2*tw:return out
    ys=np.array([p['bboxes'][1] for p in preds],np.float64)
    smoothed=smooth_track(ys,max(1.,.25*fps))
    values=np.clip(np.rint(smoothed),0,upper_num//tw).astype(np.int64)
    for p,y in zip(out['predictions'],values):p['bboxes'][1]=int(y)
    return out
