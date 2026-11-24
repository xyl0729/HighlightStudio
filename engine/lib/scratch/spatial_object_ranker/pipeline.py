import numpy as np

def nearest_index(ids,frame):
 ids=np.asarray(ids,int);pos=int(np.searchsorted(ids,frame));options=[i for i in [pos-1,pos] if 0<=i<len(ids)]
 if not options:raise ValueError('empty samples')
 return min(options,key=lambda i:(abs(int(ids[i])-frame),int(ids[i])))

def choose_x(xs,scores):
 if not xs or len(xs)!=len(scores) or not np.isfinite(scores).all():raise ValueError('bad candidate scores')
 s=list(scores);s[0]=0.
 i=min(range(len(xs)),key=lambda i:(-s[i],i!=0,abs(xs[i]-xs[0]),xs[i]))
 return xs[i]
