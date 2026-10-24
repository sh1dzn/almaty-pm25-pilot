"""Reproducible preliminary forecasting experiment. Run fetch_data.py first.
All hyperparameter choices use 2024 only. 2025 is held out until scoring.
"""
from pathlib import Path
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).parent/'.mpl-cache'))
import json, time, hashlib, platform, subprocess, importlib.metadata, copy
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, precision_recall_fscore_support, confusion_matrix
from xgboost import XGBRegressor
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R=Path(__file__).resolve().parent; OUT=R/'results'; OUT.mkdir(exist_ok=True)
C=json.loads((R/'config.json').read_text()); START=time.time()
torch.set_num_threads(C['hardware_threads'])

def metrics(y,p,ref=None):
    rmse=np.sqrt(mean_squared_error(y,p)); prec,rec,f1,_=precision_recall_fscore_support(y>35,p>35,average='binary',zero_division=0)
    return dict(n=len(y),MAE=mean_absolute_error(y,p),RMSE=rmse,R2=r2_score(y,p),NMB=100*np.sum(p-y)/np.sum(y),SS=0. if ref is None else 1-rmse/np.sqrt(mean_squared_error(y,ref)),Precision=prec,Recall=rec,F1=f1)

raw=pd.read_csv(R/'data/pm25.csv.gz'); raw['time']=pd.to_datetime(raw.datetime_utc,utc=True)
raw['year']=raw.time.dt.tz_convert(C['timezone']).dt.year
raw=raw[(raw.station_id==C['station_id']) & raw.year.between(2022,2025)].copy()
assert not raw.time.duplicated().any()
raw=raw.set_index('time').sort_index()
pm10=pd.read_csv(R/'data/pm10.csv.gz'); pm10=pm10[pm10.station_id.eq(C['station_id'])].copy();pm10['time']=pd.to_datetime(pm10.datetime_utc,utc=True)
pm10=pm10.set_index('time').value_ugm3
# Hourly UTC grid preserves elapsed time through the March 2024 zone change.
lo=pd.Timestamp('2022-01-01',tz=C['timezone']).tz_convert('UTC'); hi=pd.Timestamp('2026-01-01',tz=C['timezone']).tz_convert('UTC')
grid=pd.date_range(lo,hi,freq='h',inclusive='left'); d=pd.DataFrame(index=grid)
d['pm_raw']=raw.value_ugm3.reindex(grid);d['pm10']=pm10.reindex(grid)
neg=d.pm_raw<0; inconsistent=(d.pm_raw>d.pm10)&d.pm10.notna()
pm=d.pm_raw.mask(neg|inconsistent)
# The protocol's 24 h rolling rule is causal; comparison excludes current hour.
past=pm.shift(1); median=past.rolling(24,min_periods=12).median();sd=past.rolling(24,min_periods=12).std()
spike=(pm-median).abs()>4*sd
spike=spike & sd.gt(0)
d['pm']=pm.mask(spike)
d['local_year']=grid.tz_convert(C['timezone']).year
w=json.loads((R/'data/weather_era5.json').read_text()); w=pd.DataFrame(w['hourly']);w.index=pd.to_datetime(w.pop('time'),utc=True)
assert not w.index.duplicated().any()
for col in w.columns: d[col]=w[col].reindex(grid)
angle=np.deg2rad(d.pop('wind_direction_10m'));speed=d.pop('wind_speed_10m');d['wind_u']=-speed*np.sin(angle);d['wind_v']=-speed*np.cos(angle)
WEATHER=list(w.columns);WEATHER.remove('wind_speed_10m');WEATHER.remove('wind_direction_10m');WEATHER+=['wind_u','wind_v']
WEATHER=[col for col in WEATHER if col not in C.get('excluded_weather_variables',[])]
assert d[WEATHER].notna().all().all()
local=grid.tz_convert(C['timezone']);md=local.strftime('%m-%d')
d['heating']=((md>='10-15')|(md<'04-15')).astype(float)
CAL=['heating']
for name,v,period in [('hour',local.hour,24),('weekday',local.dayofweek,7),('month',local.month-1,12)]:
 for trig,fn in [('sin',np.sin),('cos',np.cos)]:
  col=f'{name}_{trig}';d[col]=fn(2*np.pi*v/period);CAL.append(col)
# Reset forward filling at every split; target is never filled.
d['filled']=d.groupby('local_year',sort=False).pm.ffill(limit=C['causal_fill_limit_hours'])
# Before training record every QC and availability decision.
quality=[]
for yr in range(2022,2026):
 k=d.local_year.eq(yr);obs=d.pm[k].dropna()
 quality.append(dict(year=yr,grid_hours=int(k.sum()),published=int(d.pm_raw[k].notna().sum()),absent=int(d.pm_raw[k].isna().sum()),negative_removed=int((neg&k).sum()),pm25_over_pm10_removed=int((inconsistent&k).sum()),rolling_rule_removed=int((spike&k).sum()),retained=len(obs),coverage=float(len(obs)/k.sum()),mean=float(obs.mean()),std=float(obs.std()),median=float(obs.median()),p95=float(obs.quantile(.95)),exceedance_percent=float((obs>35).mean()*100),short_gap_filled=int((d.pm.isna()&d.filled.notna()&k).sum())))
pd.DataFrame(quality).to_csv(OUT/'dataset_audit.csv',index=False)
d.to_csv(OUT/'hourly_prepared.csv.gz',compression='gzip')

def features(series):
 x=pd.DataFrame(index=grid)
 for lag in [0,1,2,3,6,12,24]:x[f'pm_lag_{lag}']=series.shift(lag)
 for window in [6,12,24]:
  x[f'pm_mean_{window}']=series.rolling(window,min_periods=window).mean()
  x[f'pm_std_{window}']=series.rolling(window,min_periods=window).std()
 return pd.concat([x,d[WEATHER+CAL]],axis=1)
X=features(d.filled); Xcomplete=features(d.pm)
SEQCOL=['filled']+WEATHER+CAL
seqraw=d[SEQCOL].to_numpy(dtype=np.float32)
sequences=np.lib.stride_tricks.sliding_window_view(seqraw,(25,seqraw.shape[1]))[:,0,:,:]
sequences=np.concatenate([np.full((24,25,seqraw.shape[1]),np.nan,dtype=np.float32),sequences],axis=0)
seqvalid=np.isfinite(sequences).all(axis=(1,2))
complete25=d.pm.notna().rolling(25,min_periods=25).sum().eq(25)

class LSTM(nn.Module):
 def __init__(self,n,hidden):
  super().__init__();self.lstm=nn.LSTM(n,hidden,batch_first=True);self.out=nn.Linear(hidden,1)
 def forward(self,x):return self.out(self.lstm(x)[0][:,-1]).squeeze(-1)

def train_lstm(tr,va,y,hidden,seed):
 torch.manual_seed(seed);np.random.seed(seed)
 # Scale each physical input once from eligible training hours only.
 sc=StandardScaler().fit(seqraw[np.isin(np.arange(len(d)),tr)])
 def transform(ids):return sc.transform(sequences[ids].reshape(-1,len(SEQCOL))).reshape(len(ids),25,len(SEQCOL)).astype(np.float32)
 xt,xv=transform(tr),transform(va); ym=float(y[tr].mean());ys=float(y[tr].std())
 model=LSTM(len(SEQCOL),hidden); opt=torch.optim.Adam(model.parameters(),lr=C['lstm_learning_rate'])
 dl=DataLoader(TensorDataset(torch.tensor(xt),torch.tensor((y[tr]-ym)/ys,dtype=torch.float32)),batch_size=C['lstm_batch_size'],shuffle=True,generator=torch.Generator().manual_seed(seed))
 vx=torch.tensor(xv);vy=torch.tensor((y[va]-ym)/ys,dtype=torch.float32)
 best=float('inf');state=None;wait=0;bestepoch=0
 for epoch in range(C['lstm_max_epochs']):
  model.train()
  for xb,yb in dl:
   opt.zero_grad();loss=((model(xb)-yb)**2).mean();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
  model.eval()
  with torch.no_grad():vl=float(((model(vx)-vy)**2).mean())
  if vl<best-1e-5:best=vl;state=copy.deepcopy(model.state_dict());wait=0;bestepoch=epoch+1
  else:wait+=1
  if wait>=C['lstm_patience']:break
 model.load_state_dict(state);model.eval()
 def predict(ids):
  batches=[];a=transform(ids)
  with torch.no_grad():
   for st in range(0,len(a),512):batches.append(model(torch.tensor(a[st:st+512])).numpy()*ys+ym)
  return np.concatenate(batches)
 return predict,bestepoch

def make_model(name,param,seed):
 if name=='Ridge':return make_pipeline(StandardScaler(),Ridge(alpha=param))
 if name=='Random forest':return RandomForestRegressor(n_estimators=C['rf_n_estimators'],max_depth=param,min_samples_leaf=C['rf_min_samples_leaf'],random_state=seed,n_jobs=4)
 if name=='XGBoost':return XGBRegressor(n_estimators=C['xgb_n_estimators'],max_depth=param,learning_rate=C['xgb_learning_rate'],objective='reg:squarederror',random_state=seed,n_jobs=4,tree_method='hist',subsample=.8,colsample_bytree=.8)
 raise ValueError(name)

runmetrics=[];predframes=[];tuning=[];abmetrics=[];dmrows=[];errorrows=[]
# Covariances retain actual hour separations; missing hours do not become adjacent.
def dm_test(y,pa,pb,times,h):
 delta=(pa-y)**2-(pb-y)**2;mean=delta.mean();full=pd.Series(delta-mean,index=times).reindex(pd.date_range(times.min(),times.max(),freq='h'))
 a=full.to_numpy();n=np.isfinite(a).sum();lag=max(24,h-1);v=np.nansum(a*a)
 for k in range(1,lag+1):v+=2*(1-k/(lag+1))*np.nansum(a[k:]*a[:-k])
 var=v/n**2
 z=mean/np.sqrt(var) if var>0 else np.nan
 return dict(mean_loss_difference=mean,DM=z,p=2*norm.sf(abs(z)) if np.isfinite(z) else np.nan,hac_lags=lag)

for h in C['horizons_hours']:
 target=d.pm.shift(-h).to_numpy(); target_time=grid+pd.Timedelta(hours=h);ty=target_time.tz_convert(C['timezone']).year
 valid=X.notna().all(axis=1).to_numpy()&seqvalid&np.isfinite(target)&(ty==d.local_year.to_numpy())
 tr=np.where(valid&d.local_year.isin(C['train_years']).to_numpy())[0]
 va=np.where(valid&d.local_year.isin(C['validation_years']).to_numpy())[0]
 te=np.where(valid&d.local_year.isin(C['test_years']).to_numpy())[0]
 assert target_time[tr].max()<grid[va].min() and target_time[va].max()<grid[te].min()
 assert len(set(tr)&set(te))==0 and len(va)>500 and len(te)>500
 print('HORIZON',h,'train/val/test',len(tr),len(va),len(te),flush=True)
 selected={};valrmse={}
 valrmse['Persistence']=np.sqrt(mean_squared_error(target[va],d.filled.to_numpy()[va]))
 for name,params in [('Ridge',C['ridge_alpha']),('Random forest',C['rf_max_depth']),('XGBoost',C['xgb_max_depth']),('LSTM',C['lstm_hidden_sizes'])]:
  candidates=[]
  for param in params:
   t=time.time()
   if name=='LSTM':predict,ep=train_lstm(tr,va,target,param,C['seeds'][0]);pv=predict(va)
   else:
    model=make_model(name,param,C['seeds'][0]);model.fit(X.iloc[tr],target[tr]);pv=model.predict(X.iloc[va]);ep=None
   err=float(np.sqrt(mean_squared_error(target[va],pv)));candidates.append((err,param))
   tuning.append(dict(horizon=h,model=name,parameter=param,validation_RMSE=err,epochs=ep,seconds=time.time()-t));print(' tune',name,param,round(err,3),flush=True)
  valrmse[name],selected[name]=min(candidates)
 # Selection is finalized using validation loss before held-out scoring.
 winner=min(valrmse,key=valrmse.get)
 (OUT/f'selected_h{h}.json').write_text(json.dumps(dict(horizon=h,parameters=selected,validation_RMSE=valrmse,validation_winner=winner,train_n=len(tr),validation_n=len(va),test_n=len(te)),indent=2))
 bymodel={}
 for name in ['Persistence','Ridge','Random forest','XGBoost','LSTM']:
  plist=[]
  seeds=[C['seeds'][0]] if name in ['Persistence','Ridge'] else C['seeds']
  for seed in seeds:
   st=time.time()
   if name=='Persistence':p=d.filled.to_numpy()[te];ep=None
   elif name=='LSTM':predict,ep=train_lstm(tr,va,target,selected[name],seed);p=predict(te)
   else:model=make_model(name,selected[name],seed);model.fit(X.iloc[tr],target[tr]);p=model.predict(X.iloc[te]);ep=None
   ref=d.filled.to_numpy()[te];times=target_time[te]
   pframe=pd.DataFrame(dict(issue_time=grid[te],target_time=times,horizon=h,model=name,seed=seed,observed=target[te],predicted=p,persistence=ref,heating=d.heating.shift(-h).iloc[te].to_numpy()))
   predframes.append(pframe);plist.append(p)
   for group,mask in [('All',np.ones(len(te),bool)),('Heating proxy',pframe.heating.eq(1).to_numpy()),('Non-heating proxy',pframe.heating.eq(0).to_numpy())]:
    runmetrics.append(dict(horizon=h,model=name,seed=seed,group=group,train_n=len(tr),validation_n=len(va),seconds=time.time()-st,epochs=ep,**metrics(target[te][mask],p[mask],ref[mask])))
   print(' test',h,name,seed,'RMSE',round(runmetrics[-3]['RMSE'],3),flush=True)
  bymodel[name]=np.stack(plist)
 # Inference uses first preregistered seed, rather than an unplanned ensemble.
 for name in bymodel:
  if name==winner:continue
  a=dm_test(target[te],bymodel[winner][0],bymodel[name][0],target_time[te],h)
  dmrows.append(dict(horizon=h,validation_winner=winner,competitor=name,seed=C['seeds'][0],**a))
 # Controlled XGBoost ablations and training-set attrition comparison.
 for variant in ['Full','Without weather','Without calendar','Complete training cases']:
  cols=[c for c in X if not(variant=='Without weather' and c in WEATHER) and not(variant=='Without calendar' and c in CAL)]
  train_ids=tr if variant!='Complete training cases' else tr[complete25.iloc[tr].to_numpy()]
  test_ids=te[complete25.iloc[te].to_numpy()] if variant=='Complete training cases' else te
  # Separate Full on common complete-case test set is produced below.
  for seed in C['seeds']:
   model=make_model('XGBoost',selected['XGBoost'],seed);model.fit(X.iloc[train_ids][cols],target[train_ids]);p=model.predict(X.iloc[test_ids][cols]);ref=d.filled.to_numpy()[test_ids]
   abmetrics.append(dict(horizon=h,variant=variant,seed=seed,train_n=len(train_ids),**metrics(target[test_ids],p,ref)))
   if variant=='Full':
    ids=te[complete25.iloc[te].to_numpy()];pc=model.predict(X.iloc[ids][cols]);abmetrics.append(dict(horizon=h,variant='Full on complete test cases',seed=seed,train_n=len(train_ids),**metrics(target[ids],pc,d.filled.to_numpy()[ids])))
 # Errors are reported for validation-selected winner and first seed.
 p=bymodel[winner][0];y=target[te];ix=np.argsort(np.abs(p-y))[-10:][::-1]
 for j in ix:errorrows.append(dict(horizon=h,model=winner,issue_time=str(grid[te[j]]),target_time=str(target_time[te[j]]),observed=y[j],predicted=p[j],absolute_error=abs(p[j]-y[j])))
 cm=confusion_matrix(y>35,p>35,labels=[False,True]);np.savetxt(OUT/f'confusion_h{h}.csv',cm,delimiter=',',fmt='%d',header='predicted_le35,predicted_gt35',comments='')
 pd.DataFrame(runmetrics).to_csv(OUT/'metrics_by_seed.csv',index=False)
 pd.DataFrame(tuning).to_csv(OUT/'tuning.csv',index=False)
 pd.DataFrame(abmetrics).to_csv(OUT/'ablation_by_seed.csv',index=False)
 pd.DataFrame(dmrows).to_csv(OUT/'dm_tests.csv',index=False)
 pd.DataFrame(errorrows).to_csv(OUT/'worst_errors.csv',index=False)
 pd.concat(predframes).to_csv(OUT/'predictions.csv.gz',index=False,compression='gzip')

# Holm family correction across the 16 planned winner/competitor comparisons.
dm=pd.DataFrame(dmrows);pvals=dm.p.to_numpy();order=np.argsort(pvals);adj=np.empty(len(dm));running=0
for rank,i in enumerate(order):running=max(running,(len(dm)-rank)*pvals[i]);adj[i]=min(1,running)
dm['p_holm']=adj;dm.to_csv(OUT/'dm_tests.csv',index=False)
m=pd.DataFrame(runmetrics)
cols=['MAE','RMSE','R2','NMB','SS','Precision','Recall','F1']
summary=m.groupby(['horizon','model','group'])[cols].agg(['mean','std']).fillna(0);summary.columns=['_'.join(k) for k in summary.columns];summary.to_csv(OUT/'metrics_summary.csv')
ab=pd.DataFrame(abmetrics);a=ab.groupby(['horizon','variant'])[cols].agg(['mean','std']);a.columns=['_'.join(k) for k in a.columns];a.to_csv(OUT/'ablation_summary.csv')
# Whole-week block resampling preserves episodes and actual spacing of missing hours.
pred=pd.concat(predframes);boot=[];rng=np.random.default_rng(C['bootstrap_seed'])
for h in C['horizons_hours']:
 for name in pred.model.unique():
  a=pred[(pred.horizon==h)&(pred.model==name)&(pred.seed==C['seeds'][0])].copy();a['block']=((pd.to_datetime(a.target_time,utc=True)-pd.Timestamp('2025-01-01',tz='UTC')).dt.total_seconds()//(168*3600)).astype(int)
  a['sse']=(a.predicted-a.observed)**2;a['ae']=abs(a.predicted-a.observed)
  b=a.groupby('block').agg(n=('sse','size'),sse=('sse','sum'),ae=('ae','sum')).to_numpy();draw=rng.integers(0,len(b),size=(C['bootstrap_replicates'],len(b)));tot=b[draw].sum(axis=1)
  r=np.sqrt(tot[:,1]/tot[:,0]);ma=tot[:,2]/tot[:,0]
  boot.append(dict(horizon=h,model=name,seed=C['seeds'][0],RMSE_low=np.quantile(r,.025),RMSE_high=np.quantile(r,.975),MAE_low=np.quantile(ma,.025),MAE_high=np.quantile(ma,.975),blocks=len(b),block_hours=168,replicates=C['bootstrap_replicates']))
pd.DataFrame(boot).to_csv(OUT/'bootstrap_ci.csv',index=False)
# Export scientific vector figures.
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'savefig.bbox':'tight'})
fig,axs=plt.subplots(1,2,figsize=(9,3.3));allm=m[m.group.eq('All')]
for name,g in allm.groupby('model'):
 s=g.groupby('horizon').agg(r=('RMSE','mean'),s=('RMSE','std'),skill=('SS','mean')).fillna(0)
 axs[0].errorbar(s.index,s.r,yerr=s.s,marker='o',capsize=3,label=name);axs[1].plot(s.index,s.skill,marker='o',label=name)
axs[0].set(xlabel='Forecast horizon (h)',ylabel='RMSE (µg/m³)',xticks=C['horizons_hours']);axs[1].set(xlabel='Forecast horizon (h)',ylabel='Skill relative to persistence',xticks=C['horizons_hours']);axs[1].axhline(0,color='.4',lw=.7)
axs[0].legend(fontsize=7);fig.tight_layout();fig.savefig(OUT/'horizon_performance.pdf');fig.savefig(OUT/'horizon_performance.png',dpi=160);plt.close(fig)
sel=json.loads((OUT/'selected_h24.json').read_text());winner=sel['validation_winner'];z=pred[(pred.horizon==24)&(pred.model==winner)&(pred.seed==11)].copy();z['time']=pd.to_datetime(z.target_time,utc=True);z['residual']=z.predicted-z.observed
fig,axs=plt.subplots(1,2,figsize=(9,3.4));vmax=max(z.observed.max(),z.predicted.max());axs[0].scatter(z.observed,z.predicted,s=5,alpha=.18);axs[0].plot([0,vmax],[0,vmax],color='.3',lw=.8);axs[0].set(xlabel='Observed PM₂.₅ (µg/m³)',ylabel='24 h forecast PM₂.₅ (µg/m³)')
cm=confusion_matrix(z.observed>35,z.predicted>35,labels=[False,True]);axs[1].imshow(cm,cmap='Blues')
for (i,j),v in np.ndenumerate(cm):axs[1].text(j,i,str(v),ha='center',va='center',color='white' if v>cm.max()/2 else 'black')
axs[1].set(xticks=[0,1],yticks=[0,1],xticklabels=['≤35','>35'],yticklabels=['≤35','>35'],xlabel='Predicted PM₂.₅ (µg/m³)',ylabel='Observed PM₂.₅ (µg/m³)');fig.suptitle(f'Preliminary 2025 errors: {winner}, seed 11, 24 h horizon');fig.tight_layout();fig.savefig(OUT/'error_analysis.pdf');fig.savefig(OUT/'error_analysis.png',dpi=160);plt.close(fig)
versions={p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','scikit-learn','xgboost','torch','matplotlib']}
meta=dict(config_sha256=hashlib.sha256((R/'config.json').read_bytes()).hexdigest(),software=versions,python=platform.python_version(),platform=platform.platform(),machine=platform.machine(),cpu=subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'],text=True).strip(),memory_bytes=int(subprocess.check_output(['sysctl','-n','hw.memsize'],text=True)),torch_threads=torch.get_num_threads(),training_device='CPU',total_seconds=time.time()-START,complete=True)
(OUT/'run_metadata.json').write_text(json.dumps(meta,indent=2));print('COMPLETE',json.dumps(meta),flush=True)
