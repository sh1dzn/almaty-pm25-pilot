"""Isolated PyTorch worker: never loads XGBoost or scikit-learn OpenMP runtimes."""
import os,json,time,copy
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset,DataLoader
import sys
job=Path(sys.argv[1]);data=np.load(job);c=json.loads(str(data['config']));torch.set_num_threads(c['hardware_threads'])
tr,va,te=data['tr'],data['va'],data['te'];a=data['seq'];raw=data['raw'];y=data['y'];hidden=int(data['hidden']);seed=int(data['seed'])
torch.manual_seed(seed);np.random.seed(seed)
# Fit all normalization on eligible training issue hours only.
mean=raw[tr].astype(np.float64).mean(axis=0);std=raw[tr].astype(np.float64).std(axis=0);std[std==0]=1
ym=float(y[tr].mean());ys=float(y[tr].std())
def scale(ids):return ((a[ids]-mean)/std).astype(np.float32)
class LSTM(nn.Module):
 def __init__(self):
  super().__init__();self.lstm=nn.LSTM(raw.shape[1],hidden,batch_first=True);self.out=nn.Linear(hidden,1)
 def forward(self,x):return self.out(self.lstm(x)[0][:,-1]).squeeze(-1)
model=LSTM();opt=torch.optim.Adam(model.parameters(),lr=c['lstm_learning_rate'])
dl=DataLoader(TensorDataset(torch.tensor(scale(tr)),torch.tensor((y[tr]-ym)/ys,dtype=torch.float32)),batch_size=c['lstm_batch_size'],shuffle=True,generator=torch.Generator().manual_seed(seed))
vx=torch.tensor(scale(va));vy=torch.tensor((y[va]-ym)/ys,dtype=torch.float32)
best=float('inf');state=None;wait=0;bestepoch=0;start=time.time()
for epoch in range(c['lstm_max_epochs']):
 model.train()
 for xb,yb in dl:
  opt.zero_grad();loss=((model(xb)-yb)**2).mean();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
 model.eval()
 with torch.no_grad():vl=float(((model(vx)-vy)**2).mean())
 if vl<best-1e-5:best=vl;state=copy.deepcopy(model.state_dict());wait=0;bestepoch=epoch+1
 else:wait+=1
 if (epoch+1)%10==0:print(' LSTM epoch',epoch+1,'validation scaled MSE',round(vl,5),'elapsed s',round(time.time()-start,1),flush=True)
 if wait>=c['lstm_patience']:break
model.load_state_dict(state);model.eval()
def predict(ids):
 x=scale(ids);p=[]
 with torch.no_grad():
  for st in range(0,len(x),512):p.append(model(torch.tensor(x[st:st+512])).numpy()*ys+ym)
 return np.concatenate(p)
np.savez(job.with_suffix('.output.npz'),validation=predict(va),test=predict(te),best_epoch=bestepoch,train_seconds=time.time()-start)
print(' LSTM finished hidden',hidden,'seed',seed,'best epoch',bestepoch,'seconds',round(time.time()-start,1),flush=True)
