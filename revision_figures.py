"""Vector figures from revision_analysis.py outputs; no model fitting."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.dates import DateFormatter,DayLocator
from zoneinfo import ZoneInfo

R=Path(__file__).resolve().parent;O=R/'revision_v2'
MODELS=['Persistence','Ridge','Random forest','XGBoost','LSTM']
COLOR={'Persistence':'#707070','Ridge':'#0072B2','Random forest':'#009E73','XGBoost':'#D55E00','LSTM':'#AA4499'}
MARK={'Persistence':'x','Ridge':'o','Random forest':'s','XGBoost':'^','LSTM':'D'}
STYLE={'Persistence':'--','Ridge':'-','Random forest':':','XGBoost':'-.','LSTM':'-'}
plt.rcParams.update({'font.size':8.5,'axes.labelsize':8.5,'xtick.labelsize':8.5,'ytick.labelsize':8.5,
 'legend.fontsize':8.5,'font.family':'DejaVu Sans','svg.fonttype':'none','pdf.fonttype':42,
 'axes.spines.top':False,'axes.spines.right':False,'lines.linewidth':1.0})
def save(fig,name):
 for ext in ['pdf','svg','png']:fig.savefig(O/f'{name}.{ext}',dpi=220,bbox_inches='tight',pad_inches=.025)
 plt.close(fig)
B=pd.read_csv(O/'bootstrap_intervals.csv');B=B[B.estimand.eq('mean_metric')]
fig,ax=plt.subplots(1,2,figsize=(5.2,2.4));fig.subplots_adjust(left=.11,right=.98,bottom=.19,top=.76,wspace=.40)
for i,(metric,label) in enumerate([('RMSE','RMSE (µg/m³)'),('delta_RMSE','ΔRMSE vs persistence (µg/m³)')]):
 for j,m in enumerate(MODELS):
  a=B[B.model.eq(m)&B.metric.eq(metric)].sort_values('horizon')
  x=np.arange(4)+(j-2)*.045
  ax[i].errorbar(x,a.estimate,yerr=np.vstack([a.estimate-a.ci_low,a.ci_high-a.estimate]),color=COLOR[m],marker=MARK[m],linestyle=STYLE[m],markersize=3.5,capsize=2,elinewidth=.7,label=m)
 ax[i].set_xticks(np.arange(4),['1','6','12','24']);ax[i].set_xlabel('Forecast horizon (h)');ax[i].set_ylabel(label)
 ax[i].text(-.17,1.04,'(a)' if i==0 else '(b)',transform=ax[i].transAxes)
ax[1].axhline(0,color='#444',lw=.7,zorder=0)
fig.legend(*ax[0].get_legend_handles_labels(),loc='upper center',ncol=3,frameon=False,bbox_to_anchor=(.53,1.0),columnspacing=1.2,handlelength=2.)
save(fig,'horizon_performance')

d=pd.read_csv(O/'error_diagnostics_data.csv');d.target_time=pd.to_datetime(d.target_time,utc=True)
y=d.observed.to_numpy();p=d.predicted.to_numpy();cm=np.array([[np.sum((y<=35)&(p<=35)),np.sum((y<=35)&(p>35))],[np.sum((y>35)&(p<=35)),np.sum((y>35)&(p>35))]])
percent=cm/cm.sum(axis=1,keepdims=True)*100
pd.DataFrame({'observed_class':np.repeat(['≤35','>35'],2),'predicted_class':np.tile(['≤35','>35'],2),'count':cm.ravel(),'row_percent':percent.ravel()}).to_csv(O/'confusion_counts.csv',index=False)
fig,ax=plt.subplots(1,2,figsize=(5.2,2.35));fig.subplots_adjust(left=.10,right=.94,bottom=.24,top=.89,wspace=.70)
lower=-10;lim=340
hb=ax[0].hexbin(y,p,gridsize=30,extent=(lower,lim,lower,lim),mincnt=1,norm=LogNorm(),cmap='cividis',linewidths=0)
assert int(hb.get_array().sum())==len(y), 'Every diagnostic prediction must appear, including negative forecasts.'
ax[0].plot([lower,lim],[lower,lim],color='#555',ls='--',lw=.8)
for fn in [ax[0].axhline,ax[0].axvline]:fn(35,color='#444',ls=':',lw=.75)
ax[0].set(xlim=(lower,lim),ylim=(lower,lim),xlabel='Observed PM₂.₅ (µg/m³)',ylabel='Predicted PM₂.₅ (µg/m³)');ax[0].set_aspect('equal')
cb=fig.colorbar(hb,ax=ax[0],fraction=.055,pad=.045);cb.ax.set_title('Count',fontsize=8.5);cb.ax.tick_params(labelsize=8.5)
im=ax[1].imshow(percent,vmin=0,vmax=100,cmap='Blues')
ax[1].set_xticks([0,1],['≤35','>35']);ax[1].set_yticks([0,1],['≤35','>35']);ax[1].set(xlabel='Predicted class (µg/m³)',ylabel='Measured class (µg/m³)')
for i in range(2):
 for j in range(2):ax[1].text(j,i,f'{cm[i,j]:,}\n{percent[i,j]:.1f}%',ha='center',va='center',fontsize=8.5,color='white' if percent[i,j]>55 else 'black')
cb=fig.colorbar(im,ax=ax[1],fraction=.055,pad=.045);cb.ax.set_title('%',fontsize=8.5);cb.ax.tick_params(labelsize=8.5)
for i in range(2):ax[i].text(-.22,1.11,f'({chr(97+i)})',transform=ax[i].transAxes,fontsize=8.5)
save(fig,'error_analysis')

tz=ZoneInfo('Asia/Almaty');lo=pd.Timestamp('2025-02-03',tz=tz);hi=pd.Timestamp('2025-02-17',tz=tz)
grid=pd.date_range(lo,hi,freq='h',inclusive='left')
raw=pd.read_csv(R/'results/hourly_prepared.csv.gz',index_col=0);raw.index=pd.to_datetime(raw.index,utc=True)
series=pd.DataFrame(index=grid);series['observed']=raw.pm.reindex(grid)
scored=d.set_index('target_time');series['predicted']=scored.predicted.reindex(grid);series['persistence']=scored.persistence.reindex(grid)
assert len(grid)==336 and series.predicted.isna().any()
series.to_csv(O/'winter_window.csv',index_label='target_time_local')
fig,ax=plt.subplots(figsize=(5.2,1.95));fig.subplots_adjust(left=.11,right=.98,bottom=.23,top=.77)
ax.plot(grid,series.observed,label='Measured',color='#151515',lw=1.1)
ax.plot(grid,series.predicted,label='LSTM, seed 11, 24 h',color=COLOR['LSTM'],lw=1.0,marker='D',markersize=2,markevery=24)
ax.plot(grid,series.persistence,label='Persistence, 24 h',color=COLOR['Persistence'],ls='--',lw=.9)
ax.axhline(35,color='#555',ls=':',lw=.8,label='Threshold 35 µg/m³')
events=pd.read_csv(O/'error_events.csv');event=events.iloc[0];es=pd.Timestamp(event.start);ee=pd.Timestamp(event.end)+pd.Timedelta(hours=1)
ax.axvspan(es,ee,color='#ddd',alpha=.6,zorder=0)
ax.annotate('15 consecutive high-error hours',xy=(es+(ee-es)/2,225),xytext=(lo+pd.Timedelta(days=.3),285),arrowprops={'arrowstyle':'->','lw':.7},fontsize=8.5)
ax.set(xlim=(lo,hi-pd.Timedelta(hours=1)),ylim=(0,320),ylabel='PM₂.₅ (µg/m³)',xlabel='Target time, February 2025 (Asia/Almaty)')
ax.xaxis.set_major_locator(DayLocator(interval=2,tz=tz));ax.xaxis.set_major_formatter(DateFormatter('%d %b',tz=tz))
fig.legend(*ax.get_legend_handles_labels(),loc='upper center',ncol=2,frameon=False,bbox_to_anchor=(.53,1.0),columnspacing=1.)
save(fig,'winter_timeseries')
print('Saved three figures as PDF, SVG and PNG; minimum text size 8 pt at 5.2 inches.')
