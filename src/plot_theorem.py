import pandas as pd, numpy as np, matplotlib.pyplot as plt
from pathlib import Path
root=Path(str(Path(__file__).resolve().parents[1]/'rerun'))
df=pd.read_csv(root/'exact_dgd_paired_contrasts_full.csv')
fig,ax=plt.subplots(figsize=(6.6,4.3))
for p,g in df.groupby('placement'):
 ax.scatter(g.predicted_delta,g.observed_delta,s=12,alpha=.5,label=p.replace('_',' '))
q=np.quantile(np.r_[df.predicted_delta.values,df.observed_delta.values],[.01,.99]);ax.plot([q[0],q[1]],[q[0],q[1]],linestyle='--',linewidth=1,color='black')
ax.set(xlabel='Exact expected difference in DGD MSE',ylabel='Observed difference in DGD MSE',title='Model-predicted versus realized placement effects')
ax.legend(fontsize=8,frameon=False);fig.tight_layout();fig.savefig(root/'fig_exact_risk.png',dpi=190);fig.savefig(root/'fig_exact_risk.pdf');plt.close(fig)
rank=pd.read_csv(root/'rank_observability_bounds.csv')
m=rank.groupby(['rounds','placement']).bayes_lower_bound.mean().reset_index()
fig,ax=plt.subplots(figsize=(6.6,4))
for p,g in m.groupby('placement'):
 ax.plot(g.rounds,g.bayes_lower_bound,marker='o',label=p.replace('_',' '))
ax.set(xlabel='Number of DGD updates',ylabel='Local information lower bound on MSE',title='Finite-time sensing directions remain inaccessible')
ax.legend(fontsize=8,frameon=False);fig.tight_layout();fig.savefig(root/'fig_rank_bound.png',dpi=190);fig.savefig(root/'fig_rank_bound.pdf');plt.close(fig)
