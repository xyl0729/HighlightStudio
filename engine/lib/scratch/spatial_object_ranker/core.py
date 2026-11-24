import numpy as np
from hlcut.spatial import SUBJECT_CLASS_WEIGHT,smooth_track

def objects(dets,W,H):
 out=[]
 for row in dets:
  if len(row)!=6 or not np.isfinite(row).all():raise ValueError('invalid detection')
  x1,y1,x2,y2,conf,cls=map(float,row);weight=SUBJECT_CLASS_WEIGHT.get(int(cls),0)
  if weight>0 and x2>x1 and y2>y1:out.append((weight*conf*(x2-x1)*(y2-y1)/(W*H),row))
 return [r for _,r in sorted(out,key=lambda a:(-a[0],tuple(a[1])))[:3]]

def proposals(base,yolo,w,W,H,dets):
 if min(w,W,H)<=0 or not 0<=base<=W-w or not 0<=yolo<=W-w:raise ValueError('invalid geometry')
 xs=[int(base),int(yolo)]+[int(np.clip(round((r[0]+r[2]-w)/2),0,W-w)) for r in objects(dets,W,H)]
 return list(dict.fromkeys(xs))

def features(x,base,yolo,w,W,H,dets):
 out=[(x-base)/w,(x-yolo)/w,abs(x-base)/w,abs(x-yolo)/w]
 for row in objects(dets,W,H):
  x1,y1,x2,y2,conf,cls=row;dw=x2-x1;dh=y2-y1;cx=(x1+x2)/2
  cover=lambda a:max(0,min(a+w,x2)-max(a,x1))/dw
  out.extend([(cx-(x+w/2))/w,dw/w,dh/H,conf,float(int(cls)==0),SUBJECT_CLASS_WEIGHT.get(int(cls),0),cover(x),float(x<=cx<=x+w),cover(x)-cover(base),dw*dh/(W*H)])
 out.extend([0.]*(34-len(out)))
 if len(out)!=34 or not np.isfinite(out).all():raise ValueError('invalid features')
 return np.asarray(out,np.float32)

def offset_track(base,ids,selected,W,fps):
 b=np.asarray(base);ids=np.asarray(ids,int);selected=np.asarray(selected,float)
 if b.ndim!=2 or b.shape[1]!=3 or not len(b) or not len(ids) or len(ids)!=len(selected):raise ValueError('invalid track')
 if not np.isfinite(b).all() or not np.isfinite(selected).all() or fps<=0 or ids.min()<0 or ids.max()>=len(b) or np.any(np.diff(ids)<=0):raise ValueError('invalid sample positions')
 delta=selected-b[ids,0]
 if not np.any(delta):return b.copy()
 full=np.interp(np.arange(len(b)),ids,delta);full=smooth_track(full,.4*fps)
 out=b.copy();out[:,0]=np.clip(np.rint(b[:,0]+full),0,W-b[:,2]).astype(b.dtype)
 return out
