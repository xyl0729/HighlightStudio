import numpy as np
from collections import Counter
from scratch.structured_window_20261005_v1.core import pool

def semantic(rows,W,H):
    a=np.asarray(rows,float).reshape(-1,6);out=np.zeros(48)
    if W<=0 or H<=0 or not np.isfinite(a).all():raise ValueError('Nonfinite detector data')
    for x1,y1,x2,y2,conf,cls in a:
        if not 0<=cls<24 or cls!=int(cls) or not 0<=conf<=1 or not 0<=x1<x2<=W or not 0<=y1<y2<=H:raise ValueError('Invalid semantic geometry')
        k=int(cls);out[k]=max(out[k],conf);out[24+k]=max(out[24+k],conf*(x2-x1)*(y2-y1)/(W*H))
    return out

def window_features(windows,q,x,ids,n):
    q=np.asarray(q,float);x=np.asarray(x,float);ids=np.asarray(ids);ws=np.asarray(windows)
    if q.shape!=(n,) or x.shape!=(len(ids),48) or ids.ndim!=1 or len(ids)<2 or ids.dtype.kind not in 'iu' or ids[0]!=0 or ids[-1]!=n-1 or np.any(np.diff(ids)<=0) or not all(np.isfinite(v).all() for v in (q,x)):raise ValueError('Native feature clock')
    if ws.ndim!=2 or ws.shape[1]!=2 or ws.dtype.kind not in 'iu' or np.any(ws[:,0]<0) or np.any(ws[:,1]>=n) or np.any(ws[:,1]<ws[:,0]):raise ValueError('Candidate window')
    a,b=ws.T;pos=a[:,None]+(b-a)[:,None]*np.linspace(0,1,17)
    j=np.clip(np.searchsorted(ids,pos,side='right')-1,0,len(ids)-2);frac=(pos-ids[j])/(ids[j+1]-ids[j]);values=x[j]*(1-frac[:,:,None])+x[j+1]*frac[:,:,None]
    visual=np.column_stack([values.mean(1),values.max(1),values.std(1)])
    length=b-a+1;prefix=np.r_[0,np.cumsum(q)];mass=prefix[b+1]-prefix[a]
    scalar=np.column_stack([length/n,mass/length,mass/n,2*mass/(length+q.sum()),q[a],q[b],a/n,b/n])
    f=np.column_stack([visual,scalar]);f-=f[0]
    if not np.isfinite(f).all():raise ValueError('Nonfinite window features')
    return f

def weights(items):
    groups=Counter(g for g,n in items)
    if not groups or any(type(n) is not int or n<1 for g,n in items):raise ValueError('Training candidate mass')
    return np.concatenate([np.full(n,1/(len(groups)*groups[g]*n)) for g,n in items])

def fit(x,y,w):
    x=np.asarray(x,float);y=np.asarray(y,float);w=np.asarray(w,float)
    if x.ndim!=2 or x.shape[1]!=152 or y.shape!=(len(x),) or w.shape!=y.shape or not len(x) or any(not np.isfinite(a).all() for a in (x,y,w)) or np.any(w<=0) or not np.isclose(w.sum(),1):raise ValueError('Ridge data')
    sd=np.maximum(np.sqrt(w@(x*x)),1e-5);z=x/sd
    normal=z.T@(w[:,None]*z)+.01*np.eye(152);rhs=z.T@(w*y);beta=np.linalg.solve(normal,rhs)
    err=float(np.max(np.abs(normal@beta-rhs)))
    if err>1e-9:raise ValueError('Ridge failed normal equation')
    return dict(beta=beta,sd=sd,error=err)

def select(scores):
    s=np.asarray(scores,float)
    if s.ndim!=1 or not len(s) or not np.isfinite(s).all():raise ValueError('Window scores')
    s=s.copy();s[0]=0.;return int(np.argmax(s))

def metric(case,boxes,truth,window):
    a,b=window;t=np.asarray(boxes,float)
    if not truth or t.shape!=(case['n'],3) or not 0<=a<=b<case['n'] or not np.isfinite(t).all() or np.any(t[:,:2]<0) or np.any(t[:,2]<=0) or np.any(t[:,0]+t[:,2]>case['W']) or np.any(t[:,1]+t[:,2]*16/9>case['H']+1e-10):raise ValueError('Strict native track/truth/window')
    numerator=0.;overlap=0
    for f,roi in truth.items():
        if a<=f<=b:
            x,y,w=t[f];h=w*16/9;tx,ty,tw,th=roi
            inter=max(0,min(x+w,tx+tw)-max(x,tx))*max(0,min(y+h,ty+th)-max(y,ty))
            numerator+=inter/(w*h+tw*th-inter);overlap+=1
    den=b-a+1+len(truth);return dict(temporal=2*overlap/den,joint=2*numerator/den)

def window_targets(case,boxes,truth,windows):
    metric(case,boxes,truth,windows[0]);u=np.zeros(case['n']);b=np.asarray(boxes,float)
    for f,(tx,ty,tw,th) in truth.items():
        x,y,w=b[f];h=w*16/9;inter=max(0,min(x+w,tx+tw)-max(x,tx))*max(0,min(y+h,ty+th)-max(y,ty));u[f]=inter/(w*h+tw*th-inter)
    prefix=np.r_[0,np.cumsum(u)];ws=np.asarray(windows,int);a,z=ws.T
    score=2*(prefix[z+1]-prefix[a])/(z-a+1+len(truth));return score-score[0]
