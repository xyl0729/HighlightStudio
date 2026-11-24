import numpy as np
from scipy.ndimage import median_filter
from hlcut.spatial import smooth_track

ARMS=('gaussian','median','jump_gaussian','median_gaussian','bounded_gaussian')

def filter_boxes(boxes,W,fps,arm):
    b=np.asarray(boxes)
    if arm not in ARMS or b.ndim!=2 or b.shape[1]!=3 or not len(b) or fps<=0 or not np.isfinite(b).all():raise ValueError('invalid filter input')
    if np.any(b[:,2]<=0) or np.any(b[:,0]<0) or np.any(b[:,0]+b[:,2]>W):raise ValueError('illegal crop')
    x=b[:,0].astype(float);sigma=max(1.,.25*fps);k=2*round(.125*fps)+1
    if arm=='median':y=median_filter(x,size=k,mode='nearest')
    elif arm=='median_gaussian':y=smooth_track(median_filter(x,size=k,mode='nearest'),sigma)
    elif arm=='jump_gaussian':
        bounds=np.r_[0,np.flatnonzero(np.abs(np.diff(x))>.30*np.minimum(b[1:,2],b[:-1,2]))+1,len(x)];y=x.copy()
        for a,z in zip(bounds[:-1],bounds[1:]):y[a:z]=smooth_track(x[a:z],sigma)
    else:
        y=smooth_track(x,sigma)
        if arm=='bounded_gaussian':y=x+np.clip(y-x,-.10*b[:,2],.10*b[:,2])
    result=b.copy();result[:,0]=np.clip(np.rint(y),0,W-b[:,2]).astype(b.dtype)
    return result
