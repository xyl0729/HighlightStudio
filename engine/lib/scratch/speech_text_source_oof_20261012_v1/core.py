import numpy as np
from scratch.internvideo2_temporal_oof_20261012_v1.core import uniform16,source_folds,duration_weights,fit,predict
from scratch.speech_transcript_feasibility_20261012_v1.core import pool_segments,masked_candidate


def native_segments(segments,n,fps):
    if not isinstance(n,int) or n<1 or not np.isfinite(fps) or fps<=0:raise ValueError('native frame clock')
    out=[];clipped=0;end=n/fps
    for s in segments:
        a,b=float(s['start']),float(s['end'])
        if not np.isfinite([a,b]).all() or b<a:raise ValueError('invalid speech interval')
        lo=max(0.,a);hi=min(end,b);clipped+=int(lo!=a or hi!=b)
        if hi>lo:out.append(dict(text=s['text'],start=lo,end=hi))
    return out,clipped


def source_weights(cases,grids,masks):
    if len(cases)!=len(grids) or len(cases)!=len(masks):raise ValueError('source case coverage')
    parent={c['group']:c['group']for c in cases};owner={}
    def root(g):
        while parent[g]!=g:parent[g]=parent[parent[g]];g=parent[g]
        return g
    for c in cases:
        g,h=c['group'],c['source_sha256']
        if h in owner:
            a,b=root(g),root(owner[h]);parent[max(a,b)]=min(a,b)
        else:owner[h]=g
    roots=[root(c['group'])for c in cases];mass={g:0. for g in roots};raw=[]
    for c,ids,mask,g in zip(cases,grids,masks,roots):
        mask=np.asarray(mask)
        if mask.shape!=np.asarray(ids).shape or mask.dtype.kind!='b':raise ValueError('explicit paired valid mask')
        w=duration_weights(ids,c['n'])*mask;mass[g]+=w.sum();raw.append(w)
    return [w/mass[g] if mass[g]>0 else np.zeros_like(w)for w,g in zip(raw,roots)],roots
