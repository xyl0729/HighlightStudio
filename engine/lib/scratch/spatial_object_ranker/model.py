import os,tempfile;os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR',os.path.join(tempfile.gettempdir(),'torchinductor_cache'))
import numpy as np,torch
from torch import nn
class Ranker(nn.Module):
 def __init__(self):
  super().__init__();self.net=nn.Sequential(nn.Linear(34,64),nn.ReLU(),nn.Linear(64,32),nn.ReLU(),nn.Linear(32,1),nn.Tanh())
 def forward(self,x):return self.net(x).squeeze(-1)

def fit(X,Y,weight,epochs=30):
 X=np.asarray(X,np.float32);Y=np.asarray(Y,np.float32);weight=np.asarray(weight,np.float32)
 if X.ndim!=2 or X.shape[1]!=34 or not len(X) or Y.shape!=(len(X),) or weight.shape!=(len(X),):raise ValueError('training shapes')
 if not all(np.isfinite(v).all() for v in [X,Y,weight]) or (weight<=0).any():raise ValueError('nonfinite/nonpositive training input')
 torch.set_num_threads(2);torch.manual_seed(20261003)
 mu=X.mean(0);sd=X.std(0);sd=np.maximum(sd,1e-5)
 x=torch.from_numpy((X-mu)/sd);y=torch.from_numpy(Y);w=torch.from_numpy(weight/weight.mean());model=Ranker()
 opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001);losses=[]
 for epoch in range(epochs):
  model.train();order=torch.randperm(len(X));total=0.
  for pos in range(0,len(X),4096):
   ids=order[pos:pos+4096];opt.zero_grad(set_to_none=True);out=model(x[ids]);loss=((out-y[ids]).square()*w[ids]).mean()
   if not torch.isfinite(loss):raise ValueError('nonfinite training loss')
   loss.backward();opt.step();total+=float(loss.detach())*len(ids)
  losses.append(total/len(X))
 model.eval();return model,mu,sd,losses

def predict(model,mu,sd,X):
 X=np.asarray(X,np.float32)
 if X.ndim!=2 or X.shape[1]!=34 or not np.isfinite(X).all():raise ValueError('invalid prediction features')
 with torch.inference_mode():return model(torch.from_numpy((X-mu)/sd)).numpy()
