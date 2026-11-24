import numpy as np
from hlcut.pipeline import label_end

def pool(n,fps,baseline):
    if n<=0 or fps<=0 or not 0<=baseline[0]<=baseline[1]<n:raise ValueError('invalid metadata/window')
    end=label_end(n,fps);points=np.unique(np.rint(np.linspace(0,end-1,17)).astype(int))
    minimum=max(1,int(fps*.5));windows=[list(map(int,baseline))]
    windows += [[int(a),int(b)] for a in points for b in points if b>=a and b-a+1>=minimum]
    return [list(v) for v in dict.fromkeys(map(tuple,windows))]

def window_features(windows,q,x,grid,n):
    q=np.asarray(q,float);x=np.asarray(x,float);grid=np.asarray(grid,float);ws=np.asarray(windows,int)
    if q.shape!=(n,) or x.shape!=(64,64) or grid.shape!=(64,) or not np.isfinite(q).all() or not np.isfinite(x).all():raise ValueError('features invalid')
    if (ws[:,0]<0).any() or (ws[:,1]>=n).any() or (ws[:,1]<ws[:,0]).any():raise ValueError('window invalid')
    a,b=ws[:,0],ws[:,1];length=b-a+1;prefix=np.r_[0,np.cumsum(q)];mass=prefix[b+1]-prefix[a]
    positions=a[:,None]+(b-a)[:,None]*np.linspace(0,1,17)[None]
    # The prepared feature sequence is on an exact uniform 64-token grid.
    if not np.allclose(grid,np.linspace(0,n-1,64),atol=1e-12):raise ValueError('unexpected feature grid')
    index=positions/(n-1)*63;lo=np.floor(index).astype(int);hi=np.minimum(lo+1,63);w=index-lo
    visual=(x[lo]*(1-w[:,:,None])+x[hi]*w[:,:,None]).mean(axis=1)
    scalar=np.column_stack([length/n,mass/length,mass/n,2*mass/(length+q.sum()),q[a],q[b],a/n,b/n])
    feats=np.column_stack([visual,scalar]);feats-=feats[0]
    if not np.isfinite(feats).all():raise ValueError('nonfinite window features')
    return feats

def fit(x,y,w):
    x=np.asarray(x,float);y=np.asarray(y,float);w=np.asarray(w,float)
    if x.ndim!=2 or x.shape[1]!=72 or y.shape!=(len(x),) or w.shape!=y.shape or (w<=0).any():raise ValueError('invalid fit shape')
    if not all(np.isfinite(v).all() for v in (x,y,w)):raise ValueError('invalid fit values')
    sd=np.maximum(np.sqrt((w[:,None]*x*x).sum(axis=0)/w.sum()),1e-5);z=x/sd
    beta=np.linalg.solve(z.T@(w[:,None]*z)+np.eye(72),z.T@(w*y))
    return beta,sd

def select(scores):
    s=np.asarray(scores,float)
    if s.ndim!=1 or not len(s) or not np.isfinite(s).all():raise ValueError('invalid scores')
    s=s.copy();s[0]=0
    return int(np.argmax(s))
