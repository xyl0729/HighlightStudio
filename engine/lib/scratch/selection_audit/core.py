"""CPU-only numerical helpers; no file or video access."""
import numpy as np
from scratch.post.fit_decode import features, ef_decode, post_probs
from hlcut.pipeline import label_end, _widen


def grouped_folds(groups, k=3, seed=20260929):
    unique=sorted(set(groups))
    if len(unique)<k: raise ValueError('Not enough independent groups')
    rng=np.random.RandomState(seed); rng.shuffle(unique)
    mapping={g:i%k for i,g in enumerate(unique)}
    return [[i for i,g in enumerate(groups) if mapping[g]==j] for j in range(k)]


def equal_weights(lengths):
    if not lengths or min(lengths)<=0: raise ValueError('Empty training samples')
    w=np.concatenate([np.full(n,1/n) for n in lengths])
    return w/w.mean()


def temporal_f(window,g):
    a,b=window
    if b<a: return 0.0
    return float(2*g[a:b+1].sum()/((b-a+1)+g.sum()))


def fit_calibration(items,equal=False):
    X=np.vstack([x for x,y in items]); y=np.concatenate([y for x,y in items])
    wt=equal_weights([len(y) for x,y in items]) if equal else np.ones(len(y))
    mu=np.average(X,axis=0,weights=wt)
    sd=np.sqrt(np.average((X-mu)**2,axis=0,weights=wt))+1e-9
    Z=np.column_stack([(X-mu)/sd,np.ones(len(y))]); w=np.zeros(Z.shape[1])
    reg=np.diag(np.r_[np.ones(len(w)-1),0.])
    for _ in range(30):
        pr=1/(1+np.exp(-np.clip(Z@w,-500,500)))
        grad=Z.T@(wt*(pr-y))+reg@w
        hess=(Z*(wt*pr*(1-pr))[:,None]).T@Z+reg
        w-=np.linalg.solve(hess,grad)
    return mu,sd,w


def predict_calibration(model,X):
    mu,sd,w=model
    return 1/(1+np.exp(-np.clip(np.column_stack([(X-mu)/sd,np.ones(len(X))])@w,-500,500)))


def fit_ridge(X,y,weights,lam):
    mu=np.average(X,axis=0,weights=weights)
    sd=np.sqrt(np.average((X-mu)**2,axis=0,weights=weights))+1e-9
    Z=np.column_stack([(X-mu)/sd,np.ones(len(y))])
    penalty=np.diag(np.r_[np.ones(X.shape[1]),0.])*lam
    w=np.linalg.solve((Z*weights[:,None]).T@Z+penalty,Z.T@(weights*y))
    return mu,sd,w


def predict_ridge(model,X):
    mu,sd,w=model
    return np.column_stack([(X-mu)/sd,np.ones(len(X))])@w


def candidates(r,q,base):
    n=r['n']; fps=r['fps']; use=label_end(n,fps); step=max(1,int(fps/4))
    wins=[tuple(base)]
    wins += [ef_decode(q,use,step,s) for s in (0.6,0.8,1.,1.2,1.4,1.7)]
    for i in np.argsort(-np.asarray(r['logp']),kind='stable')[:6]:
        a,b=r['S'][i]*fps,r['E'][i]*fps
        for frac in (0,0.3,0.6): wins.append(_widen(a,b,use,max(fps,frac*n)))
    return list(dict.fromkeys((max(0,min(use-1,int(a))),max(0,min(use-1,int(b)))) for a,b in wins))


def candidate_features(r,q,wins,base):
    n=r['n']; fps=r['fps']; use=label_end(n,fps); q=q[:use]
    p=post_probs(r); ss=np.asarray(r['S'])*fps; ee=np.asarray(r['E'])*fps
    center=float((p*(ss+ee)/2).sum()); entropy=float(-(p*np.log(np.maximum(p,1e-300))).sum())
    cs=np.r_[0.,np.cumsum(q)]; tot=cs[-1]; anslen=np.maximum(1,ee-ss+1)
    out=[]
    for a,b in wins:
        length=b-a+1; mass=cs[b+1]-cs[a]
        overlap=np.maximum(0,np.minimum(b,ee)-np.maximum(a,ss)+1)
        bi=max(0,min(b,base[1])-max(a,base[0])+1)
        out.append([length/n,a/n,b/n,(a+b)/2/n,abs((a+b)/2-center)/n,
                    mass/length,mass/max(tot,1e-9),(tot-mass)/max(1,use-length),
                    2*mass/(length+tot),2*mass/(length+1.7*tot),
                    float((p*2*overlap/(length+anslen)).sum()),
                    bi/max(1,length+base[1]-base[0]+1-bi),entropy,np.log(n/fps),
                    float(q[a]),float(q[b])])
    return np.asarray(out)


def cluster_ci(delta,groups,seed=20260929,reps=2000):
    delta=np.asarray(delta); unique=sorted(set(groups))
    sums=np.array([delta[np.array(groups)==g].sum() for g in unique])
    counts=np.array([sum(x==g for x in groups) for g in unique])
    rng=np.random.RandomState(seed); vals=[]
    for _ in range(reps):
        idx=rng.randint(len(unique),size=len(unique)); vals.append(sums[idx].sum()/counts[idx].sum())
    return np.quantile(vals,[0.025,0.975]).tolist()
