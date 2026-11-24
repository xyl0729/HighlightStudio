import numpy as np
import torch
from torch import nn

def network():return nn.Sequential(nn.Linear(54,64),nn.ReLU(),nn.Linear(64,32),nn.ReLU(),nn.Linear(32,1),nn.Tanh())

def fit(X,y,weights):
    X=np.asarray(X,np.float32);y=np.asarray(y,np.float32);w=np.asarray(weights,np.float32)
    if X.shape!=(len(y),54) or w.shape!=y.shape or not len(y) or not all(np.isfinite(a).all() for a in (X,y,w)) or np.any(w<=0):raise ValueError('Invalid training dataset')
    torch.set_num_threads(2);torch.manual_seed(20261012);torch.use_deterministic_algorithms(True)
    mu=X.mean(0);sd=np.maximum(X.std(0),1e-5);x=torch.from_numpy((X-mu)/sd);target=torch.from_numpy(y);mass=torch.from_numpy(w/w.mean())
    model=network();assert sum(p.numel() for p in model.parameters())==5633
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001);losses=[]
    for epoch in range(30):
        model.train();order=torch.randperm(len(X));total=0.
        for pos in range(0,len(X),4096):
            idx=order[pos:pos+4096];opt.zero_grad(set_to_none=True);pred=model(x[idx]).squeeze(-1);loss=((pred-target[idx]).square()*mass[idx]).mean()
            if not torch.isfinite(loss):raise ValueError('Nonfinite train loss')
            loss.backward();opt.step();total+=float(loss.detach())*len(idx)
        losses.append(total/len(X));print('EPOCH',epoch+1,'LOSS',losses[-1],flush=True)
    model.eval();return dict(state=model.state_dict(),mu=mu,sd=sd,losses=losses,head_elements=5633,normalization_elements=108)

def predict(saved,X):
    model=network();model.load_state_dict(saved['state']);model.eval()
    a=np.asarray(X,np.float32)
    with torch.inference_mode():v=model(torch.from_numpy((a-saved['mu'])/saved['sd'])).squeeze(-1).numpy()
    if not np.isfinite(v).all():raise ValueError('Nonfinite inference')
    return v
