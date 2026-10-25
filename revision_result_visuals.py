"""Results-focused matrices and hourly error diagnostics, from saved forecasts only."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm,TwoSlopeNorm
R=Path(__file__).resolve().parent;S=R/'revision_v2';O=R/'revision_v3';O.mkdir(exist_ok=True)
MODELS=['Persistence','Ridge','Random forest','XGBoost','LSTM'];SHORT=['Persist.','Ridge','RF','XGB','LSTM']
COLOR={'Persistence':'#707070','Ridge':'#0072B2','Random forest':'#009E73','XGBoost':'#D55E00','LSTM':'#AA4499'}
MARK={'Persistence':'x','Ridge':'o','Random forest':'s','XGBoost':'^','LSTM':'D'}
STYLE={'Persistence':'--','Ridge':'-','Random forest':':','XGBoost':'-.','LSTM':'-'}
plt.rcParams.update({'font.size':8.5,'axes.labelsize':8.5,'xtick.labelsize':8.5,'ytick.labelsize':8.5,'legend.fontsize':8.5,'font.family':'DejaVu Sans','svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'lines.linewidth':1.})
def save(fig,name):
 for ext in ['pdf','svg','png']:fig.savefig(O/f'{name}.{ext}',dpi=220,bbox_inches='tight',pad_inches=.025)
 plt.close(fig)
B=pd.read_csv(S/'bootstrap_intervals.csv').query('estimand=="mean_metric"')
season=pd.read_csv(S/'seasonal_performance.csv')
fig=plt.figure(figsize=(5.2,4.1));ax=[fig.add_axes(p) for p in [[.12,.59,.35,.28],[.63,.59,.35,.28],[.12,.18,.35,.25],[.63,.18,.35,.25]]]
for i,(metric,label) in enumerate([('RMSE','RMSE (µg/m³)'),('delta_RMSE','ΔRMSE (µg/m³)')]):
 for j,m in enumerate(MODELS):
  a=B[B.model.eq(m)&B.metric.eq(metric)].sort_values('horizon');x=np.arange(4)+(j-2)*.045
  ax[i].errorbar(x,a.estimate,yerr=np.vstack([a.estimate-a.ci_low,a.ci_high-a.estimate]),color=COLOR[m],marker=MARK[m],ls=STYLE[m],markersize=3.5,capsize=2,elinewidth=.7,label=m)
 ax[i].set_xticks(range(4),['1','6','12','24']);ax[i].set_xlabel('Forecast horizon (h)');ax[i].set_ylabel(label)
 ax[i].text(-.20,1.04,f'({chr(97+i)})',transform=ax[i].transAxes)
ax[1].axhline(0,color='#444',lw=.7,zorder=0)
fig.legend(*ax[0].get_legend_handles_labels(),loc='upper center',ncol=3,frameon=False,bbox_to_anchor=(.54,1.),columnspacing=1.1,handlelength=1.8)
norm=TwoSlopeNorm(vmin=-10,vcenter=0,vmax=40);matrices=[]
for i,g in enumerate(['Heating','Non-heating']):
 a=season[season.season.eq(g)&season.model.isin(MODELS)].pivot(index='model',columns='horizon',values='skill_mean').reindex(MODELS)[[1,6,12,24]]*100
 im=ax[i+2].imshow(a,cmap='RdBu',norm=norm,aspect='auto')
 for row in range(5):
  for col in range(4):
   v=float(a.iloc[row,col]);ax[i+2].text(col,row,f'{v:+.1f}',ha='center',va='center',fontsize=8.5,color='white' if norm(v)>.8 or norm(v)<.2 else 'black')
 ax[i+2].set_xticks(range(4),['1','6','12','24']);ax[i+2].set_yticks(range(5),SHORT);ax[i+2].set_xlabel('Forecast horizon (h)')
 ax[i+2].set_title(g+' skill (%)',fontsize=8.5,pad=7);ax[i+2].text(-.20,1.08,f'({chr(99+i)})',transform=ax[i+2].transAxes)
 for m in MODELS:
  for h in [1,6,12,24]:matrices.append(dict(season=g,model=m,horizon=h,skill_percent=float(a.loc[m,h])))
cb=fig.colorbar(im,cax=fig.add_axes([.29,.066,.53,.018]),orientation='horizontal',ticks=[-10,0,10,20,30,40]);cb.set_label('Skill relative to persistence (%)',fontsize=8.5)
pd.DataFrame(matrices).to_csv(O/'seasonal_skill_matrices.csv',index=False)
save(fig,'horizon_performance')

d=pd.read_csv(S/'error_diagnostics_data.csv');d.target_time=pd.to_datetime(d.target_time,utc=True)
local=d.target_time.dt.tz_convert('Asia/Almaty');md=local.dt.strftime('%m-%d');d['season']=np.where((md>='10-15')|(md<'04-15'),'Heating','Non-heating');d['target_hour']=local.dt.hour;d['absolute_error']=abs(d.predicted-d.observed)
hour=d.groupby(['season','target_hour']).agg(n=('absolute_error','size'),MAE=('absolute_error','mean'),median_absolute_error=('absolute_error','median'),signed_bias=('error','mean')).reset_index()
hour.to_csv(O/'hourly_error_diagnostics.csv',index=False)
original=pd.read_csv(R/'results/metrics_by_seed.csv')
for g,old in [('Heating','Heating proxy'),('Non-heating','Non-heating proxy')]:
 z=hour[hour.season.eq(g)];expected=original[(original.horizon==24)&original.model.eq('LSTM')&(original.seed==11)&original.group.eq(old)].iloc[0]
 assert z.n.sum()==expected.n and np.isclose(np.average(z.MAE,weights=z.n),expected.MAE)
wide=hour.pivot(index='target_hour',columns='season',values='MAE')
assert (wide.Heating>wide['Non-heating']).all()
y=d.observed.to_numpy();p=d.predicted.to_numpy();cm=np.array([[np.sum((y<=35)&(p<=35)),np.sum((y<=35)&(p>35))],[np.sum((y>35)&(p<=35)),np.sum((y>35)&(p>35))]]);percent=100*cm/cm.sum(axis=1,keepdims=True)
fig=plt.figure(figsize=(5.2,3.3));a=fig.add_axes([.10,.55,.30,.37]);b=fig.add_axes([.62,.55,.27,.37]);c=fig.add_axes([.11,.13,.83,.25])
hb=a.hexbin(y,p,gridsize=30,extent=(-10,340,-10,340),mincnt=1,norm=LogNorm(),cmap='cividis',linewidths=0);assert int(hb.get_array().sum())==len(d)
a.plot([-10,340],[-10,340],color='#555',ls='--',lw=.8)
for fun in [a.axhline,a.axvline]:fun(35,color='#444',ls=':',lw=.75)
a.set(xlim=(-10,340),ylim=(-10,340),xlabel='Observed PM₂.₅ (µg/m³)',ylabel='Predicted PM₂.₅ (µg/m³)');a.set_aspect('equal')
cb=fig.colorbar(hb,cax=fig.add_axes([.43,.55,.017,.37]));cb.ax.set_title('Count',fontsize=8.5)
im=b.imshow(percent,vmin=0,vmax=100,cmap='Blues');b.set_xticks([0,1],['≤35','>35']);b.set_yticks([0,1],['≤35','>35']);b.set(xlabel='Predicted class',ylabel='Measured class')
for i in range(2):
 for j in range(2):b.text(j,i,f'{cm[i,j]:,}\n{percent[i,j]:.1f}%',ha='center',va='center',fontsize=8.5,color='white' if percent[i,j]>55 else 'black')
cb=fig.colorbar(im,cax=fig.add_axes([.93,.55,.016,.37]));cb.ax.set_title('%',fontsize=8.5)
for g,col,mark,ls in [('Heating','#0072B2','o','-'),('Non-heating','#D55E00','s','--')]:
 z=hour[hour.season.eq(g)].sort_values('target_hour');c.plot(z.target_hour,z.MAE,color=col,marker=mark,markersize=3,ls=ls,label=g)
c.set(xlim=(0,23),ylim=(0,hour.MAE.max()*1.45),xlabel='Target hour (Asia/Almaty)',ylabel='MAE (µg/m³)');c.set_xticks(range(0,24,3));c.legend(loc='upper center',ncol=2,frameon=False,handlelength=1.7,columnspacing=1.5)
for axes,l in [(a,'(a)'),(b,'(b)'),(c,'(c)')]:axes.text(-.10 if axes is c else -.19,1.07,l,transform=axes.transAxes)
save(fig,'error_analysis')
for g,z in hour.groupby('season'):
 print(g,'hourly MAE min/max:',z.loc[z.MAE.idxmin()].to_dict(),z.loc[z.MAE.idxmax()].to_dict(),'min cell n',z.n.min())
print('New matrices and hourly diagnostics use original saved metrics/forecasts; no training.')
