import numpy as np

def sampled_frames(n):
    if type(n) is not int or n<1:raise ValueError('Invalid clip length')
    return sorted(set(np.rint(np.linspace(0,n-1,min(n,12))).astype(int).tolist()))

def checked_objects(rows,W,H):
    a=np.asarray(rows,float).reshape(-1,6)
    if W<=0 or H<=0 or not np.isfinite(a).all() or np.any(a[:,:2]<0) or np.any(a[:,2]<=a[:,0]) or np.any(a[:,3]<=a[:,1]) or np.any(a[:,2]>W) or np.any(a[:,3]>H) or np.any((a[:,4]<0)|(a[:,4]>1)) or np.any((a[:,5]<0)|(a[:,5]>=24)|(a[:,5]!=np.floor(a[:,5]))):raise ValueError('Invalid detected objects')
    return a

def iou(a,b):
    area=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    return area/((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-area)

def suppress_objects(rows,W,H):
    a=checked_objects(rows,W,H);order=sorted(range(len(a)),key=lambda i:(-a[i,4]*np.sqrt((a[i,2]-a[i,0])*(a[i,3]-a[i,1])/(W*H)),i));out=[]
    for i in order:
        if all(iou(a[i],a[j])<=.5 for j in out):out.append(i)
        if len(out)==6:break
    return a[out]

def checked_baseline(b,W,H):
    a=np.asarray(b,float)
    if a.shape!=(3,) or not np.isfinite(a).all() or a[2]<=0 or min(a[:2])<0 or a[0]+a[2]>W or a[1]+a[2]*16/9>H:raise ValueError('Illegal actual baseline geometry')
    return a

def proposals(base,rows,W,H):
    x,y,w=checked_baseline(base,W,H);a=checked_objects(rows,W,H);positions=[int(x)]
    for r in a:
        positions.extend(int(np.clip(round(v),0,W-w)) for v in ((r[0]+r[2]-w)/2,r[0],r[2]-w))
    return list(dict.fromkeys(positions))

def features(x,base,rows,W,H):
    bx,by,w=checked_baseline(base,W,H);a=checked_objects(rows,W,H)
    if not isinstance(x,(int,np.integer)) or not 0<=x<=W-w:raise ValueError('Illegal candidate')
    free=max(1,W-w);offset=x-bx;values=[offset/w,abs(offset)/w,offset/free,abs(offset)/free,w/W,w*16/9/H]
    def content(at):
        cover=np.zeros(24);centres=np.zeros(24)
        for x1,y1,x2,y2,confidence,cls in a:
            k=int(cls);vy=max(0,min(by+w*16/9,y2)-max(by,y1))/(y2-y1)
            vx=max(0,min(at+w,x2)-max(at,x1))/(x2-x1)
            cover[k]=max(cover[k],confidence*vx*vy)
            close=confidence*(1-min(abs((x1+x2)/2-(at+w/2))/w,1))*vy
            centres[k]=max(centres[k],close)
        return np.r_[cover,centres]
    return np.r_[values,content(x)-content(bx)].astype(np.float32)

def choose_position(pool,scores,baseline):
    if baseline not in pool or len(pool)!=len(set(pool)) or len(pool)!=len(scores) or not np.isfinite(scores).all():raise ValueError('Candidate decision identity')
    usable=[(float(s),int(x)) for x,s in zip(pool,scores) if x!=baseline and s>0]
    if not usable:return int(baseline)
    return min(usable,key=lambda p:(-p[0],abs(p[1]-baseline),p[1]))[1]

def apply(base,ids,chosen,W,fps):
    from hlcut.spatial import smooth_track
    b=np.asarray(base,int);ids=np.asarray(ids,int);chosen=np.asarray(chosen,float)
    if b.ndim!=2 or b.shape[1]!=3 or len(ids)!=len(chosen) or not len(ids) or ids[0]<0 or ids[-1]>=len(b) or np.any(np.diff(ids)<=0) or not np.isfinite(chosen).all() or fps<=0:raise ValueError('Invalid track clocks')
    offset=chosen-b[ids,0]
    if not np.any(offset):return b.copy()
    full=smooth_track(np.interp(np.arange(len(b)),ids,offset),.4*fps);out=b.copy()
    out[:,0]=np.clip(np.rint(b[:,0]+full),0,W-b[:,2]).astype(int)
    return out
