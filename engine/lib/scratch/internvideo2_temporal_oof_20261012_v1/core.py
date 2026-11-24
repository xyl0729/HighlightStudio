import hashlib
import numpy as np

def uniform16(n):
 if type(n)!=int or n<16:raise ValueError('uniform16 requires at least16 native frames')
 return np.linspace(0,n-1,16).astype(np.int64)

def source_folds(cases):
 groups=sorted({c['group'] for c in cases});parent={g:g for g in groups};owner={}
 def root(g):
  while parent[g]!=g:parent[g]=parent[parent[g]];g=parent[g]
  return g
 for c in cases:
  g=c['group'];h=c['source_sha256']
  if h in owner:
   a,b=root(g),root(owner[h]);parent[max(a,b)]=min(a,b)
  else:owner[h]=g
 return {g:int(hashlib.sha256(root(g).encode()).hexdigest(),16)%5 for g in groups}

def duration_weights(ids,n):
 ids=np.asarray(ids)
 if type(n)!=int or n<1 or ids.ndim!=1 or not len(ids) or ids.dtype.kind not in 'iu' or ids[0]!=0 or ids[-1]!=n-1 or np.any(np.diff(ids)<=0):raise ValueError('full ordered native frame grid')
 edges=np.r_[-.5,(ids[:-1].astype(float)+ids[1:])/2,n-.5]
 return np.diff(edges)/n

def paired_features(projected,ids,fps):
 x=np.asarray(projected,float);ids=np.asarray(ids)
 if x.ndim!=2 or not min(x.shape) or x.shape[1]%3 or not np.isfinite(x).all() or ids.shape!=(len(x),) or ids.dtype.kind not in 'iu' or np.any(np.diff(ids)<=0) or not np.isfinite(fps) or fps<=0:raise ValueError('finite features and native clock')
 t=ids/fps;k=x.shape[1]//3
 before=np.column_stack([np.interp(t-1,t,x[:,j]) for j in range(k,2*k)])-x[:,k:2*k]
 after=np.column_stack([np.interp(t+1,t,x[:,j]) for j in range(2*k,3*k)])-x[:,2*k:]
 return x.copy(),np.column_stack([x[:,:k],before,after])

def fit(x,y,weights,ridge=.01):
 x=np.asarray(x,float);y=np.asarray(y,float);weights=np.asarray(weights,float)
 if x.ndim!=2 or not min(x.shape) or y.shape!=(len(x),) or weights.shape!=(len(x),) or not all(np.isfinite(z).all() for z in (x,y,weights)) or np.any(weights<=0) or not np.isfinite(ridge) or ridge<=0:raise ValueError('finite weighted training set')
 weights=weights/weights.sum();mu=weights@x;sd=np.sqrt(np.maximum(weights@((x-mu)**2),1e-12));z=(x-mu)/sd;intercept=float(weights@y)
 normal=z.T@(weights[:,None]*z)+ridge*np.eye(x.shape[1]);rhs=z.T@(weights*(y-intercept));w=np.linalg.solve(normal,rhs);error=float(np.max(np.abs(normal@w-rhs)))
 if error>1e-10:raise ValueError('ridge normal equation did not converge')
 return dict(w=w,mu=mu,sd=sd,intercept=np.array(intercept),normal_equation_max_error=np.array(error))

def predict(model,x):return (np.asarray(x,float)-model['mu'])/model['sd']@model['w']+model['intercept']
