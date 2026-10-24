"""Render figures from saved predictions; does not refit models."""
from pathlib import Path
import os,json
R=Path(__file__).resolve().parent;OUT=R/'results'
os.environ.setdefault('MPLCONFIGDIR',str(R/'.mpl-cache'))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
C=json.loads((R/'config.json').read_text())
m=pd.read_csv(OUT/'metrics_by_seed.csv')
pred=pd.read_csv(OUT/'predictions.csv.gz')
# Export scientific vector figures.
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.bbox':'tight'})
fig,axs=plt.subplots(1,2,figsize=(6.5,3.2));allm=m[m.group.eq('All')]
for name,g in allm.groupby('model'):
 s=g.groupby('horizon').agg(r=('RMSE','mean'),s=('RMSE','std'),skill=('SS','mean')).fillna(0)
 axs[0].errorbar(s.index,s.r,yerr=s.s,marker='o',capsize=3,label=name);axs[1].plot(s.index,s.skill,marker='o',label=name)
axs[0].set(xlabel='Forecast horizon (h)',ylabel='RMSE (µg/m³)',xticks=C['horizons_hours']);axs[1].set(xlabel='Forecast horizon (h)',ylabel='Skill relative to persistence',xticks=C['horizons_hours']);axs[1].axhline(0,color='.4',lw=.7)
axs[0].legend(fontsize=8);fig.tight_layout();fig.savefig(OUT/'horizon_performance.pdf');fig.savefig(OUT/'horizon_performance.png',dpi=160);plt.close(fig)
sel=json.loads((OUT/'selected_h24.json').read_text());winner=sel['validation_winner'];z=pred[(pred.horizon==24)&(pred.model==winner)&(pred.seed==11)].copy();z['time']=pd.to_datetime(z.target_time,utc=True);z['residual']=z.predicted-z.observed
fig,axs=plt.subplots(1,2,figsize=(6.5,3.2));vmax=max(z.observed.max(),z.predicted.max());axs[0].scatter(z.observed,z.predicted,s=5,alpha=.18);axs[0].plot([0,vmax],[0,vmax],color='.3',lw=.8);axs[0].set(xlabel='Observed PM₂.₅ (µg/m³)',ylabel='24 h forecast PM₂.₅ (µg/m³)')
cm=confusion_matrix(z.observed>35,z.predicted>35,labels=[False,True]);axs[1].imshow(cm,cmap='Blues')
for (i,j),v in np.ndenumerate(cm):axs[1].text(j,i,str(v),ha='center',va='center',color='white' if v>cm.max()/2 else 'black')
axs[1].set(xticks=[0,1],yticks=[0,1],xticklabels=['≤35','>35'],yticklabels=['≤35','>35'],xlabel='Predicted PM₂.₅ (µg/m³)',ylabel='Observed PM₂.₅ (µg/m³)');fig.suptitle(f'Preliminary 2025 errors: {winner}, seed 11, 24 h horizon',fontsize=10);fig.tight_layout();fig.savefig(OUT/'error_analysis.pdf');fig.savefig(OUT/'error_analysis.png',dpi=160);plt.close(fig)
