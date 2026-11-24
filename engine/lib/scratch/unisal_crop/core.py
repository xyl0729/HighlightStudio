"""Fixed saliency-to-crop geometry, independent of labels/model state."""
import hashlib
import numpy as np

def sample_ids(n, stride=5):
    if not isinstance(n,int) or n<=0 or not isinstance(stride,int) or stride<=0:
        raise ValueError('Invalid frame count/stride')
    return sorted(set(range(0,n,stride))|{n-1})

def source_half(group):
    return int(hashlib.sha256(('unisal-v1:'+group).encode()).hexdigest()[-1],16)%2

def probability_map(logmap):
    a=np.asarray(logmap,dtype=np.float64)
    if a.ndim!=2 or not a.size or not np.isfinite(a).all():
        raise ValueError('Invalid log probability map')
    p=np.exp(a-a.max());return p/p.sum()

def rectangle_mass(prob,boxes,source_size):
    p=np.asarray(prob,np.float64);b=np.asarray(boxes,np.float64)
    W,H=source_size
    if p.ndim!=2 or not p.size or not np.isfinite(p).all() or (p<0).any() or W<=0 or H<=0:
        raise ValueError('Invalid density/source dimensions')
    if b.ndim!=2 or b.shape[1]!=4 or not np.isfinite(b).all() or (b[:,2:]<=0).any():
        raise ValueError('Invalid rectangles')
    x,y,w,h=b.T
    if (x<0).any() or (y<0).any() or (x+w>W+1e-8).any() or (y+h>H+1e-8).any():
        raise ValueError('Out-of-bounds rectangles')
    ph,pw=p.shape;integ=np.pad(p.cumsum(0).cumsum(1),((1,0),(1,0)))
    def query(xx,yy):
        xx=np.clip(xx*pw/W,0,pw);yy=np.clip(yy*ph/H,0,ph)
        x0=np.floor(xx).astype(int);y0=np.floor(yy).astype(int)
        x1=np.minimum(x0+1,pw);y1=np.minimum(y0+1,ph)
        dx=xx-x0;dy=yy-y0
        return ((1-dx)*(1-dy)*integ[y0,x0]+dx*(1-dy)*integ[y0,x1]
                +(1-dx)*dy*integ[y1,x0]+dx*dy*integ[y1,x1])
    return query(x+w,y+h)-query(x,y+h)-query(x+w,y)+query(x,y)

def choose_box(logmap,W,H,baseline):
    if len(baseline)!=3 or any(not isinstance(v,(int,np.integer)) for v in baseline):
        raise ValueError('Baseline must be integer x,y,w')
    x,y,w=map(int,baseline);h=w*16./9.
    if W<=0 or H<=0 or w<=0 or x<0 or y<0 or x+w>W or y+h>H+1e-8:
        raise ValueError('Illegal source/baseline')
    xx=np.arange(W-w+1,dtype=int)
    boxes=np.column_stack([xx,np.full_like(xx,y),np.full_like(xx,w),np.full(len(xx),h)])
    masses=rectangle_mass(probability_map(logmap),boxes,[W,H])
    tied=np.flatnonzero(masses>=masses.max()-1e-12)
    winner=min(tied,key=lambda i:(abs(int(xx[i])-x),int(xx[i])))
    return [int(xx[winner]),y,w]
