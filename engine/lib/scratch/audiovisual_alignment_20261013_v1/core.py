import numpy as np
from scratch.object_semantic_window_20261013_v1.core import pool,weights,metric,window_targets,select

def unit(x):
    x=np.asarray(x,float)
    if x.ndim!=2 or not np.isfinite(x).all():raise ValueError('Finite embedding matrix')
    return x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)

def alignment(visual,local,topic,mask):
    x,t,g=map(unit,(visual,local,topic));m=np.asarray(mask,bool)
    if x.shape!=t.shape or x.shape!=g.shape or len(x)<2 or len(x)%2 or m.shape!=(len(x),):raise ValueError('Paired native embeddings')
    shifted=np.roll(t,len(x)//2,axis=0);both=m & np.roll(m,len(x)//2)
    a=np.einsum('ij,ij->i',x,t);b=np.einsum('ij,ij->i',x,shifted);theme=np.einsum('ij,ij->i',x,g)
    aa=np.column_stack([a,a-theme,a-b,both]);bb=np.column_stack([b,b-theme,b-a,both])
    aa[~both]=0;bb[~both]=0
    return aa,bb,both

def windows_features(windows,q,x,ids,n):
    q=np.asarray(q,float);x=np.asarray(x,float);ids=np.asarray(ids);ws=np.asarray(windows)
    if q.shape!=(n,) or x.shape!=(len(ids),4) or ids.ndim!=1 or len(ids)<2 or ids[0]!=0 or ids[-1]!=n-1 or np.any(np.diff(ids)<=0) or not np.isfinite(q).all() or not np.isfinite(x).all():raise ValueError('Native feature clock')
    if ws.ndim!=2 or ws.shape[1]!=2 or ws.dtype.kind not in 'iu' or np.any(ws[:,0]<0) or np.any(ws[:,1]>=n) or np.any(ws[:,1]<ws[:,0]):raise ValueError('Legal windows')
    a,b=ws.T;pos=a[:,None]+(b-a)[:,None]*np.linspace(0,1,17)
    j=np.clip(np.searchsorted(ids,pos,side='right')-1,0,len(ids)-2);f=(pos-ids[j])/(ids[j+1]-ids[j]);v=x[j]*(1-f[:,:,None])+x[j+1]*f[:,:,None]
    visual=np.column_stack([v.mean(1),v.max(1),v.std(1)])
    length=b-a+1;p=np.r_[0,np.cumsum(q)];mass=p[b+1]-p[a];den=length+q.sum()
    scalar=np.column_stack([length/n,mass/length,mass/n,2*mass/den,q[a],q[b],a/n,b/n])
    z=np.column_stack([visual,scalar]);z-=z[0]
    if not np.isfinite(z).all():raise ValueError('Nonfinite features')
    return z

def fit(x,y,w):
    x=np.asarray(x,float);y=np.asarray(y,float);w=np.asarray(w,float)
    if x.ndim!=2 or x.shape[1]!=20 or y.shape!=(len(x),) or w.shape!=y.shape or any(not np.isfinite(a).all() for a in (x,y,w)) or np.any(w<=0) or not np.isclose(w.sum(),1):raise ValueError('Source weighted training arrays')
    sd=np.maximum(np.sqrt(w@(x*x)),1e-5);z=x/sd;normal=z.T@(w[:,None]*z)+.01*np.eye(20);rhs=z.T@(w*y);beta=np.linalg.solve(normal,rhs)
    err=float(abs(normal@beta-rhs).max())
    if err>1e-9:raise ValueError('Normal equation failure')
    return dict(beta=beta,sd=sd,error=err)
