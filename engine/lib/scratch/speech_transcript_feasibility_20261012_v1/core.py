import numpy as np


def place_samples(pcm,covered,samples,timestamp,start,sr):
    samples=np.asarray(samples,np.float32)
    if pcm.shape!=covered.shape or samples.ndim!=1 or not np.isfinite(samples).all() or not np.isfinite([timestamp,start,sr]).all() or sr<=0:
        raise ValueError('finite PTS and samples required')
    offset=round((timestamp-start)*sr);lo=max(0,offset);hi=min(len(pcm),offset+len(samples))
    if hi>lo:
        pcm[lo:hi]=samples[lo-offset:hi-offset];covered[lo:hi]=True


def clip_segments(offsets,chunk_start,chunk_duration):
    if not np.isfinite([chunk_start,chunk_duration]).all() or chunk_start<0 or chunk_duration<=0:
        raise ValueError('legal chunk clock')
    out=[];report={'clipped':0,'empty_or_outside':0}
    for x in offsets:
        a,b=map(float,x['timestamp']);text=x['text'].strip()
        if not np.isfinite([a,b]).all() or b<a:raise ValueError('invalid ASR timestamp')
        lo=max(0.,a);hi=min(chunk_duration,b)
        if a!=lo or b!=hi:report['clipped']+=1
        if hi<=lo or not text:report['empty_or_outside']+=1;continue
        out.append({'text':text,'start':chunk_start+lo,'end':chunk_start+hi})
    return out,report


def pool_segments(segments,embeddings,ids,fps,n,radius=3):
    x=np.asarray(embeddings,float);ids=np.asarray(ids)
    if x.ndim!=2 or x.shape[0]!=len(segments) or ids.ndim!=1 or ids.dtype.kind not in 'iu' or not np.isfinite(x).all() or not np.isfinite([fps,radius]).all() or fps<=0 or radius<=0 or n<1 or np.any(ids<0) or np.any(ids>=n):
        raise ValueError('legal speech embeddings/native frame clock')
    a=np.array([s['start']for s in segments]);b=np.array([s['end']for s in segments])
    if len(a) and (not np.isfinite([a,b]).all() or np.any(a<0) or np.any(b<=a) or np.any(b>n/fps+1e-6)):
        raise ValueError('segment outside native clip')
    local=np.zeros((len(ids),x.shape[1]));global_=local.copy();mask=np.zeros(len(ids),bool)
    if not len(segments):return local,global_,mask
    topic=(b-a)@x/(b-a).sum()
    for j,t in enumerate(ids/fps):
        w=np.maximum(0,np.minimum(b,min(n/fps,t+radius))-np.maximum(a,max(0,t-radius)))
        if w.sum()>0:local[j]=w@x/w.sum();global_[j]=topic;mask[j]=True
    return local,global_,mask


def masked_candidate(base,residual,ids,mask):
    base=np.asarray(base,float);residual=np.asarray(residual,float);ids=np.asarray(ids);mask=np.asarray(mask,bool)
    if base.ndim!=1 or mask.shape!=base.shape or residual.shape!=ids.shape or ids.ndim!=1 or ids.dtype.kind not in 'iu' or len(ids)<2 or ids[0]!=0 or ids[-1]!=len(base)-1 or np.any(np.diff(ids)<=0) or not np.isfinite(base).all() or not np.isfinite(residual).all():
        raise ValueError('finite full native interpolation')
    delta=np.interp(np.arange(len(base)),ids,residual);return np.where(mask,np.clip(base+delta,0,1),base)
