"""Independent checks of the extended bootstrap and event outputs."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
R=Path(__file__).resolve().parent;O=R/'revision_v2'
audit=json.loads((O/'audit.json').read_text())
for n,sha in audit['sha256'].items():assert hashlib.sha256((R/n).read_bytes()).hexdigest()==sha
allp=pd.read_csv(R/'results/predictions.csv.gz');h=6
p=allp[allp.horizon.eq(h)].copy();p.target_time=pd.to_datetime(p.target_time,utc=True)
ref=p[p.model.eq('Persistence')].sort_values('target_time');times=pd.DatetimeIndex(ref.target_time);y=ref.observed.to_numpy()
z=p[p.model.eq('LSTM')].pivot(index='target_time',columns='seed',values='predicted').reindex(times).to_numpy()
blocks=np.asarray((times-pd.Timestamp('2025-01-01T00:00Z'))//pd.Timedelta(hours=168))
draw=np.load(O/'bootstrap_draws_h6.npz');unique=draw['block_ids'];indices=draw['draw_indices']
# Frequency weights independently aggregate resampled calendar blocks.
w=np.stack([np.bincount(row,minlength=len(unique)) for row in indices])
sse=np.stack([((z[blocks==b]-y[blocks==b,None])**2).sum(axis=0) for b in unique]);n=np.array([(blocks==b).sum() for b in unique])
est=np.sqrt((w@sse)/(w@n)[:,None]).mean(axis=1)
direct_rows=np.concatenate([np.where(blocks==unique[j])[0] for j in indices[0]])
direct_rmse=np.sqrt(((z[direct_rows]-y[direct_rows,None])**2).mean(axis=0)).mean()
assert np.isclose(direct_rmse,est[0],rtol=1e-12)
ci=np.quantile(est,[.025,.975]);saved=pd.read_csv(O/'bootstrap_intervals.csv')
a=saved[(saved.horizon==h)&saved.model.eq('LSTM')&saved.estimand.eq('mean_metric')&saved.metric.eq('RMSE')].iloc[0]
assert np.allclose(ci,[a.ci_low,a.ci_high],rtol=1e-12)
assert np.isclose(a.estimate,np.sqrt(((z-y[:,None])**2).mean(axis=0)).mean())
events=pd.read_csv(O/'error_events.csv');a=events.iloc[0]
assert a.start=='2025-02-09T18:00:00+05:00' and a.end=='2025-02-10T08:00:00+05:00' and a.scored_hours==a.duration_hours==15
diagnostic=pd.read_csv(O/'error_diagnostics_data.csv');diagnostic.target_time=pd.to_datetime(diagnostic.target_time,utc=True)
for _,e in events.iterrows():
 b=diagnostic[diagnostic.target_time.between(pd.Timestamp(e.start),pd.Timestamp(e.end))]
 assert len(b)==e.scored_hours and (abs(b.error)>e.threshold).all()
 assert (b.target_time.diff().dropna()==pd.Timedelta(hours=1)).all()
 assert np.isclose(b.error.mean(),e.signed_bias)
classifier=pd.read_csv(O/'seasonal_performance.csv').query('model=="Always exceed"')
for col in ['RMSE','R2','skill']:
 assert classifier[col+'_mean'].isna().all() and classifier[col+'_std'].isna().all()
assert (diagnostic.predicted<0).sum()==32
summary={'input_hashes_verified':True,'six_hour_LSTM_mean_metric_CI_recomputed_with_frequency_weights':ci.tolist(),'direct_first_draw_RMSE':direct_rmse,'all_error_groups_consecutive_and_thresholded':True,'negative_24h_LSTM_forecasts':32,'undefined_classifier_regression_metrics_remain_missing':True}
(O/'verification_summary.json').write_text(json.dumps(summary,indent=2))
print('PASS',json.dumps(summary))
