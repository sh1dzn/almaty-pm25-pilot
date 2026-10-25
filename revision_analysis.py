"""Audit and extend the saved pilot, without refitting or replacing predictions.

Run with the environment pinned in requirements-lock.txt. Outputs go to revision_v2.
"""
from pathlib import Path
import hashlib, json, time, platform, importlib.metadata
import numpy as np
import pandas as pd
from scipy.stats import norm

R=Path(__file__).resolve().parent; O=R/'revision_v2'; O.mkdir(exist_ok=True)
MODELS=['Persistence','Ridge','Random forest','XGBoost','LSTM']
H=[1,6,12,24]; SEEDS=[11,22,33,44,55]; START=time.time()
C=json.loads((R/'config.json').read_text())
P=pd.read_csv(R/'results/predictions.csv.gz')
D=pd.read_csv(R/'results/hourly_prepared.csv.gz',index_col=0)
D.index=pd.to_datetime(D.index,utc=True)
for k in ['issue_time','target_time']: P[k]=pd.to_datetime(P[k],utc=True)
audit={'input_rows':len(P),'input_columns':P.columns.tolist(),'station':C['station_id'],
       'station_source':'config.json, single-station run_pilot.py',
       'split':'test, local issue and target year 2025', 'timezone':'UTC storage; Asia/Almaty calendars'}
keys=['horizon','model','seed','issue_time','target_time']
audit['duplicate_keys']=int(P.duplicated(keys).sum())
audit['missing_fields']=int(P.isna().sum().sum())
audit['nonfinite_numeric']=int((~np.isfinite(P.select_dtypes('number'))).sum().sum())
assert audit['duplicate_keys']==audit['missing_fields']==audit['nonfinite_numeric']==0
assert set(P.model)==set(MODELS) and set(P.horizon)==set(H)
assert ((P.target_time-P.issue_time).dt.total_seconds()==P.horizon*3600).all()
assert (P.target_time.dt.tz_convert('Asia/Almaty').dt.year==2025).all()
assert (P.issue_time.dt.tz_convert('Asia/Almaty').dt.year==2025).all()
assert P.groupby(['horizon','target_time'])['observed'].nunique().eq(1).all()
assert np.allclose(P.observed,D.pm.reindex(P.target_time).to_numpy(),atol=1e-12)
md=P.target_time.dt.tz_convert('Asia/Almaty').dt.strftime('%m-%d')
assert np.array_equal(P.heating.to_numpy(),((md>='10-15')|(md<'04-15')).to_numpy())
assert np.allclose(P.persistence,D.filled.reindex(P.issue_time).to_numpy(),atol=1e-12)
cohorts=[]; arrays={}
for h in H:
 a=P[P.horizon.eq(h)]
 ref=a[a.model.eq('Persistence')].sort_values('target_time')
 times=pd.DatetimeIndex(ref.target_time); y=ref.observed.to_numpy(); by={}
 for m in MODELS:
  b=a[a.model.eq(m)]; expected=[11] if m in MODELS[:2] else SEEDS
  assert sorted(b.seed.unique())==expected
  z=b.pivot(index='target_time',columns='seed',values='predicted').reindex(times)
  assert z.notna().all().all() and len(b)==len(times)*len(expected)
  by[m]=z.to_numpy()
  cohorts.append(dict(horizon=h,model=m,input_rows=len(b),output_rows=len(b),unique_targets=len(times),seeds=','.join(map(str,expected)),removed_rows=0))
 assert np.allclose(by['Persistence'][:,0],ref.persistence)
 arrays[h]=(times,y,by)
audit['output_rows']=len(P);audit['removed_rows']=0
pers=P[P.model.eq('Persistence')]
rawissue=D.pm.reindex(pers.issue_time).to_numpy()
audit['persistence_issue_observed_rows']=int(np.isfinite(rawissue).sum())
audit['persistence_issue_locf_rows']=int(np.isnan(rawissue).sum())
audit['persistence_mismatch_observed_issue']=int(np.sum(np.abs(pers.predicted.to_numpy()[np.isfinite(rawissue)]-rawissue[np.isfinite(rawissue)])>1e-12))
audit['alignment_rule']='No rows dropped; identical complete cohorts were already present for every model and seed within horizon.'
pd.DataFrame(cohorts).to_csv(O/'cohort_audit.csv',index=False)

def score(y,p,ref):
 e=p-y; rmse=float(np.sqrt(np.mean(e*e))); pr=float(np.sqrt(np.mean((ref-y)**2)))
 yp=y>35; pp=p>35; tp=int(np.sum(yp&pp));fp=int(np.sum(~yp&pp));fn=int(np.sum(yp&~pp))
 precision=tp/(tp+fp) if tp+fp else 0.;recall=tp/(tp+fn) if tp+fn else 0.
 return dict(n=len(y),positive_n=int(yp.sum()),RMSE=rmse,MAE=float(np.mean(abs(e))),
 R2=float(1-np.sum(e*e)/np.sum((y-y.mean())**2)),skill=1-rmse/pr,
 NMB=float(100*e.sum()/y.sum()),precision=precision,recall=recall,F1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)

rows=[]; season=[]
for h,(times,y,by) in arrays.items():
 ref=by['Persistence'][:,0];md=times.tz_convert('Asia/Almaty').strftime('%m-%d');heat=(md>='10-15')|(md<'04-15')
 for m,z in by.items():
  for j,seed in enumerate([11] if m in MODELS[:2] else SEEDS):
   rows.append(dict(horizon=h,model=m,seed=seed,**score(y,z[:,j],ref)))
   for name,mask in [('Heating',heat),('Non-heating',~heat)]:
    season.append(dict(horizon=h,season=name,model=m,seed=seed,**score(y[mask],z[mask,j],ref[mask])))
 for name,mask in [('Heating',heat),('Non-heating',~heat)]:
  n=int(mask.sum());pos=int((y[mask]>35).sum());pre=pos/n
  season.append(dict(horizon=h,season=name,model='Always exceed',seed=11,n=n,positive_n=pos,
    RMSE=np.nan,MAE=np.nan,R2=np.nan,skill=np.nan,NMB=np.nan,precision=pre,recall=1. if pos else 0.,F1=2*pos/(n+pos)))
M=pd.DataFrame(rows);S=pd.DataFrame(season)
M.to_csv(O/'metrics_by_seed.csv',index=False);S.to_csv(O/'seasonal_by_seed.csv',index=False)
MET=['RMSE','MAE','R2','skill','NMB','precision','recall','F1']
def summary(df,keys):
 a=df.groupby(keys,sort=False)[MET].agg(['mean','std']);a.columns=['_'.join(c) for c in a.columns]
 # Deterministic variability is defined as zero, except regression metrics undefined for classifier.
 for c in a:
  if c.endswith('_std'):
   mean_column=c[:-4]+'_mean'
   a[c]=a[c].fillna(0.).where(a[mean_column].notna())
 return a.reset_index()
ms=summary(M,['horizon','model']);ss=summary(S,['horizon','season','model'])
ms.to_csv(O/'main_performance.csv',index=False)
counts=S.groupby(['horizon','season','model'],sort=False)[['n','positive_n']].first().reset_index()
ss=ss.merge(counts,on=['horizon','season','model']);ss.to_csv(O/'seasonal_performance.csv',index=False)
old=pd.read_csv(R/'results/metrics_by_seed.csv');mx=0.
for _,z in M.iterrows():
 q=old[(old.horizon==z.horizon)&(old.model==z.model)&(old.seed==z.seed)&old.group.eq('All')].iloc[0]
 for new,orig in [('RMSE','RMSE'),('MAE','MAE'),('R2','R2'),('skill','SS'),('NMB','NMB'),('precision','Precision'),('recall','Recall'),('F1','F1')]:mx=max(mx,abs(z[new]-q[orig]))
audit['max_metric_reproduction_abs_difference']=mx;assert mx<1e-9

def holm(p):
 p=np.asarray(p);order=np.argsort(p);adj=np.empty(len(p));run=0.
 for rank,i in enumerate(order):run=max(run,(len(p)-rank)*p[i]);adj[i]=min(1.,run)
 return adj
def dm(y,a,b,times):
 diff=(a-y)**2-(b-y)**2;mu=float(diff.mean())
 v=pd.Series(diff-mu,index=times).reindex(pd.date_range(times.min(),times.max(),freq='h')).to_numpy()
 n=np.isfinite(v).sum();var=np.nansum(v*v)
 for lag in range(1,25):var+=2*(1-lag/25)*np.nansum(v[lag:]*v[:-lag])
 se=np.sqrt(var/n**2);z=mu/se
 return dict(mean_squared_loss_difference=mu,DM=float(z),raw_p=float(2*norm.sf(abs(z))),n=int(n),hac_lags=24)

selected={h:json.loads((R/f'results/selected_h{h}.json').read_text())['validation_winner'] for h in H}
testbest={h:ms[ms.horizon.eq(h)].sort_values('RMSE_mean').iloc[0].model for h in H}
family=[];seen=set()
for h in H:
 for origin,ref,comps in [('validation',selected[h],[m for m in MODELS if m!=selected[h]]),('exploratory',testbest[h],['Ridge','Persistence'])]:
  for m in comps:
   key=(h,*sorted([ref,m]))
   if ref==m or key in seen:continue
   seen.add(key);family.append((h,ref,m,origin))
assert len(family)==22
dmall=[]
for estimand in ['seed11','ensemble']:
 rr=[]
 for h,a,b,origin in family:
  times,y,z=arrays[h];pa=z[a][:,0] if estimand=='seed11' else z[a].mean(axis=1);pb=z[b][:,0] if estimand=='seed11' else z[b].mean(axis=1)
  rr.append(dict(estimand=estimand,horizon=h,reference_model=a,competitor=b,selection=origin,**dm(y,pa,pb,times)))
 df=pd.DataFrame(rr);df['p_holm_22']=holm(df.raw_p);df['reject_22']=df.p_holm_22<.05
 df['p_holm_16']=np.nan
 base=df.selection.eq('validation');df.loc[base,'p_holm_16']=holm(df.loc[base,'raw_p'])
 dmall.append(df)
DM=pd.concat(dmall,ignore_index=True);DM.to_csv(O/'dm_comparisons.csv',index=False)
old=pd.read_csv(R/'results/dm_tests.csv')
for _,z in DM[(DM.estimand=='seed11')&(DM.selection=='validation')].iterrows():
 b=old[(old.horizon==z.horizon)&(old.validation_winner==z.reference_model)&(old.competitor==z.competitor)].iloc[0]
 assert np.allclose([z.mean_squared_loss_difference,z.DM,z.raw_p,z.p_holm_16],[b.mean_loss_difference,b.DM,b.p,b.p_holm],atol=1e-10)
audit['original_DM_16_reproduced']=True
sense=DM[DM.estimand.eq('seed11')][['horizon','reference_model','competitor','selection','raw_p','p_holm_22','reject_22']].merge(
 DM[DM.estimand.eq('ensemble')][['horizon','reference_model','competitor','raw_p','p_holm_22','reject_22']],on=['horizon','reference_model','competitor'],suffixes=('_seed11','_ensemble'))
sense['decision_changed']=sense.reject_22_seed11!=sense.reject_22_ensemble
sense.to_csv(O/'dm_seed_sensitivity.csv',index=False)

# One set of calendar-block draws per horizon, shared across every model/seed/estimand.
boot=[];blockrows=[];rng=np.random.default_rng(20261025);anchor=pd.Timestamp('2025-01-01T00:00:00Z')
for h,(times,y,by) in arrays.items():
 block=np.asarray((times-anchor)//pd.Timedelta(hours=168),dtype=int);ids=np.unique(block)
 for b in range(int(block.min()),int(block.max())+1):
  blockrows.append(dict(horizon=h,block_id=b,start_utc=str(anchor+pd.Timedelta(hours=168*int(b))),end_exclusive_utc=str(anchor+pd.Timedelta(hours=168*(int(b)+1))),scored_hours=int((block==b).sum()),sampled=bool(b in ids)))
 n=np.array([(block==b).sum() for b in ids]);draw=rng.integers(0,len(ids),size=(1000,len(ids)));den=n[draw].sum(axis=1)
 np.savez_compressed(O/f'bootstrap_draws_h{h}.npz',block_ids=ids,draw_indices=draw,block_counts=n)
 ref=by['Persistence'][:,0];se_ref=np.array([np.sum((ref[block==b]-y[block==b])**2) for b in ids]);rb=np.sqrt(se_ref[draw].sum(axis=1)/den)
 for m,p in by.items():
  se=np.stack([np.sum((p[block==b]-y[block==b,None])**2,axis=0) for b in ids])
  rep=np.sqrt(se[draw].sum(axis=1)/den[:,None])
  ens=p.mean(axis=1);see=np.array([np.sum((ens[block==b]-y[block==b])**2) for b in ids]);re=np.sqrt(see[draw].sum(axis=1)/den)
  vals={'mean_metric':(rep.mean(axis=1),float(np.sqrt(np.mean((p-y[:,None])**2,axis=0)).mean())),
        'seed11':(rep[:,0],float(np.sqrt(np.mean((p[:,0]-y)**2)))),
        'ensemble':(re,float(np.sqrt(np.mean((ens-y)**2))))}
  pr=float(np.sqrt(np.mean((ref-y)**2)))
  for est,(rep_r,point) in vals.items():
   for metric,v,pv in [('RMSE',rep_r,point),('delta_RMSE',rep_r-rb,point-pr),('skill',1-rep_r/rb,1-point/pr)]:
    lo,hi=np.quantile(v,[.025,.975]);boot.append(dict(horizon=h,model=m,estimand=est,metric=metric,estimate=pv,ci_low=lo,ci_high=hi,n=len(y),nonempty_blocks=len(ids),replicates=1000))
B=pd.DataFrame(boot);B.to_csv(O/'bootstrap_intervals.csv',index=False);pd.DataFrame(blockrows).to_csv(O/'bootstrap_blocks.csv',index=False)
assert B.query('model=="Persistence" and metric=="delta_RMSE"')[['estimate','ci_low','ci_high']].eq(0).all().all()
assert np.allclose(B.query('estimand=="mean_metric" and metric=="RMSE"').estimate,ms.RMSE_mean)

# Common target hours across all horizons, for the persistence shape check.
common=arrays[H[0]][0]
for h in H[1:]:common=common.intersection(arrays[h][0])
commonrows=[]
for h,(times,y,by) in arrays.items():
 k=times.isin(common);ref=by['Persistence'][:,0]
 commonrows.append(dict(horizon=h,original_n=len(y),common_n=int(k.sum()),original_RMSE=np.sqrt(np.mean((ref-y)**2)),common_RMSE=np.sqrt(np.mean((ref[k]-y[k])**2))))
pd.DataFrame(commonrows).to_csv(O/'persistence_common_targets.csv',index=False)

# Exploratory event grouping at the global 95th percentile of selected-model |error|.
times,y,by=arrays[24];pred=by[selected[24]][:,0];err=pred-y;threshold=float(np.quantile(abs(err),.95))
bad=np.where(abs(err)>threshold)[0];event=[]
groups=np.split(bad,np.where(np.diff(times[bad])!=pd.Timedelta(hours=1).to_timedelta64())[0]+1)
for ids in groups:
 if len(ids)==0:continue
 event.append(dict(start=times[ids[0]].tz_convert('Asia/Almaty').isoformat(),end=times[ids[-1]].tz_convert('Asia/Almaty').isoformat(),duration_hours=int((times[ids[-1]]-times[ids[0]])/pd.Timedelta(hours=1))+1,scored_hours=len(ids),max_absolute_error=float(max(abs(err[ids]))),mean_absolute_error=float(np.mean(abs(err[ids]))),signed_bias=float(np.mean(err[ids])),threshold=threshold))
events=pd.DataFrame(event).sort_values('max_absolute_error',ascending=False);events.to_csv(O/'error_events.csv',index=False)
audit['event_threshold_95_percentile']=threshold
audit['event_rule']='Exploratory |error| strictly above global seed-11 24-h selected-model 95th percentile; group only adjacent exact hourly timestamps.'
pd.DataFrame(dict(target_time=times,observed=y,predicted=pred,persistence=by['Persistence'][:,0],error=err)).to_csv(O/'error_diagnostics_data.csv',index=False)

# Validation provenance and actual parameter values, without reconstructing validation forecasts.
val=[];params=[];tune=pd.read_csv(R/'results/tuning.csv');runs=pd.read_csv(R/'results/metrics_by_seed.csv')
for h in H:
 s=json.loads((R/f'results/selected_h{h}.json').read_text())
 for m in MODELS:
  v=s['validation_RMSE'][m]
  if m!='Persistence':assert np.isclose(v,tune[(tune.horizon==h)&tune.model.eq(m)].validation_RMSE.min())
  val.append(dict(horizon=h,model=m,validation_RMSE=v,selected=m==s['validation_winner'],provenance=f'results/selected_h{h}.json and tuning.csv; logged, not independently rescored'))
  grid={'Persistence':'none','Ridge':str(C['ridge_alpha']),'Random forest':str(C['rf_max_depth']),'XGBoost':str(C['xgb_max_depth']),'LSTM':str(C['lstm_hidden_sizes'])}[m]
  chosen={'Persistence':{'rule':'latest causal issue-time filled PM2.5'},
    'Ridge':{'alpha':s['parameters'].get('Ridge'),'scaling':'StandardScaler on train X'},
    'Random forest':{'max_depth':s['parameters'].get('Random forest'),'n_estimators':150,'min_samples_leaf':3,'n_jobs':4},
    'XGBoost':{'max_depth':s['parameters'].get('XGBoost'),'n_estimators':200,'learning_rate':.05,'subsample':.8,'colsample_bytree':.8,'objective':'reg:squarederror','tree_method':'hist','n_jobs':4},
    'LSTM':{'hidden_size':s['parameters'].get('LSTM'),'lookback_elapsed_hours':24,'sequence_length':25,'input_features':14,'layers':1,'dropout':0.,'optimizer':'Adam','learning_rate':.001,'batch_size':256,'max_epochs':30,'patience':10,'gradient_clip_norm':5,'scaling':'training-only input and target mean/population SD; see lstm_configuration.json','best_epoch':'metrics_by_seed.csv; lstm_epochs.csv','stopping_epoch':'explicit field missing; reconstruction labelled in lstm_epochs.csv'}}[m]
  params.append(dict(horizon=h,model=m,search_parameter={'Persistence':'none','Ridge':'alpha','Random forest':'max_depth','XGBoost':'max_depth','LSTM':'hidden_size'}[m],search_grid=grid,selected_value=s['parameters'].get(m,'none'),selected_parameters=json.dumps(chosen),selection_metric='2024 RMSE at seed 11',evidence=f'selected_h{h}.json; tuning.csv; run_pilot.py; lstm_worker.py'))
pd.DataFrame(val).to_csv(O/'validation_rmse.csv',index=False);pd.DataFrame(params).to_csv(O/'hyperparameters.csv',index=False)
lstm=runs[(runs.model=='LSTM')&(runs.group=='All')][['horizon','seed','epochs']].rename(columns={'epochs':'best_checkpoint_epoch'})
lstm['stopping_epoch_reconstructed']=np.minimum(lstm.best_checkpoint_epoch+10,30).astype(int)
lstm['stopping_epoch_provenance']='Not stored explicitly; reconstructed from worker break rule min(best checkpoint + patience 10, max epochs 30).'
lstm.to_csv(O/'lstm_epochs.csv',index=False)
lc={'lookback_elapsed_hours':24,'sequence_values_including_current':25,'input_features':['filled','temperature_2m','relative_humidity_2m','pressure_msl','precipitation','wind_u','wind_v','heating','hour_sin','hour_cos','weekday_sin','weekday_cos','month_sin','month_cos'],
 'layers':1,'dropout':0.,'layers_and_dropout_provenance':'torch.nn.LSTM defaults, not tuned; one recurrent layer plus final linear output',
 'optimizer':'Adam','learning_rate':.001,'batch_size':256,'maximum_epochs':30,'patience':10,'gradient_norm_clip':5,
 'scaling':'Input mean and population SD fitted on eligible training issue-hour feature vectors, applied to every sequence step; target mean/SD fitted on training targets; zero input SD replaced by 1.',
 'stop_epoch':'See lstm_epochs.csv; exact stop not explicitly saved; deterministic reconstruction is labelled.',
 'architecture_evidence':'lstm_worker.py; selected_h*.json; metrics_by_seed.csv'}
(O/'lstm_configuration.json').write_text(json.dumps(lc,indent=2))

inputs=['results/predictions.csv.gz','results/hourly_prepared.csv.gz','results/dm_tests.csv','results/metrics_by_seed.csv','results/tuning.csv','config.json','run_pilot.py','lstm_worker.py']
audit['sha256']={n:hashlib.sha256((R/n).read_bytes()).hexdigest() for n in inputs}
audit['analysis_seconds']=time.time()-START;audit['python']=platform.python_version()
audit['software']={p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','matplotlib']}
(O/'audit.json').write_text(json.dumps(audit,indent=2))
# Explicit CSV inputs for the five manuscript tables; Table 7 has yearly and eligibility parts.
q=pd.read_csv(R/'results/dataset_audit.csv');q['source']='AirData.kz';q['station']=C['station_id'];q['timezone']=C['timezone'];q['split']=['Train','Train','Validation','Test'];q.to_csv(O/'table_7_dataset.csv',index=False)
elig=[]
for h in H:
 s=json.loads((R/f'results/selected_h{h}.json').read_text());elig.append({k:s[k] for k in ['horizon','train_n','validation_n','test_n']})
pd.DataFrame(elig).to_csv(O/'table_7_eligibility.csv',index=False)
ms.to_csv(O/'table_8_main_performance.csv',index=False)
DM[DM.estimand.eq('seed11')].to_csv(O/'table_9_statistical_comparisons.csv',index=False)
ab=pd.read_csv(R/'results/ablation_summary.csv');ab[ab.variant.isin(['Full','Without weather','Without calendar'])].to_csv(O/'table_10_feature_ablation.csv',index=False)
ss.to_csv(O/'table_11_seasonal_performance.csv',index=False)
print(json.dumps(audit,indent=2))
print('Validation winners',selected,'Test-lowest mean RMSE',testbest)
print(DM.query('horizon==6 and reference_model=="LSTM" and competitor=="Persistence"').to_string(index=False))
print('Changed seed11/ensemble decisions:',int(sense.decision_changed.sum()))
print(pd.DataFrame(commonrows).to_string(index=False))
print(events.head(6).to_string(index=False))
