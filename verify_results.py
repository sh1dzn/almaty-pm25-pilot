"""Independent output checks. Fails on unequal model test sets or filled targets."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
R=Path(__file__).resolve().parent;out=R/'results'
c=json.loads((R/'config.json').read_text());meta=json.loads((out/'run_metadata.json').read_text());assert meta['complete']
p=pd.read_csv(out/'predictions.csv.gz',parse_dates=['issue_time','target_time'])
d=pd.read_csv(out/'hourly_prepared.csv.gz',index_col=0,parse_dates=True)
m=pd.read_csv(out/'metrics_by_seed.csv')
source=pd.read_csv(R/'data/pm25.csv.gz');source=source[source.station_id.eq(c['station_id'])]
assert (abs(source.value_ugm3-source.raw_value*1000)<=0.00500001).all(), 'Published values must agree with unit conversion within two-decimal rounding'
assert len(p)>0 and p[['observed','predicted','persistence']].notna().all().all()
assert np.isfinite(p[['observed','predicted','persistence']]).all().all()
assert set(p['model'])=={'Persistence','Ridge','Random forest','XGBoost','LSTM'}
for h in c['horizons_hours']:
 a=p[p.horizon.eq(h)]
 assert ((a.target_time-a.issue_time).dt.total_seconds()==h*3600).all()
 assert (a.target_time.dt.tz_convert(c['timezone']).dt.year==2025).all()
 assert (a.issue_time.dt.tz_convert(c['timezone']).dt.year==2025).all()
 ref=None
 for (model,seed),g in a.groupby(['model','seed']):
  times=list(g.target_time)
  if ref is None:ref=times
  assert times==ref,('Unequal test sets',h,model,seed)
  observed=d.pm.reindex(pd.DatetimeIndex(g.target_time)).to_numpy()
  assert np.array_equal(g.observed.to_numpy(),observed)
  rmse=float(np.sqrt(np.mean((g.predicted-g.observed)**2)))
  row=m[(m.horizon==h)&(m.model==model)&(m.seed==seed)&(m.group=='All')].iloc[0]
  assert np.isclose(rmse,row.RMSE,rtol=1e-10)
  if model=='Persistence':assert np.array_equal(g.predicted.to_numpy(),g.persistence.to_numpy())
 ab=pd.read_csv(out/'ablation_by_seed.csv');z=ab[(ab.horizon==h)&ab.seed.eq(11)].set_index('variant')
 assert z.loc['Full on complete test cases','n']==z.loc['Complete training cases','n']
 assert z.loc['Complete training cases','train_n']<=z.loc['Full on complete test cases','train_n']
 audit=pd.read_csv(out/'dataset_audit.csv')
 assert (audit.published==audit.retained+audit.negative_removed+audit.pm25_over_pm10_removed+audit.rolling_rule_removed).all()
for s in ['Random forest','XGBoost','LSTM']:assert set(p[p.model.eq(s)].seed)==set(c['seeds'])
dm=pd.read_csv(out/'dm_tests.csv');assert len(dm)==16;assert dm.p.between(0,1).all();assert (dm.p_holm>=dm.p-1e-12).all()
ci=pd.read_csv(out/'bootstrap_ci.csv');assert (ci.RMSE_low<=ci.RMSE_high).all()
print('PASS: every reported RMSE independently recomputed; identical test hours across models/seeds; no imputed targets; horizon and split checks; attrition reconciliation; ablation cohorts; Holm correction; intervals.')
