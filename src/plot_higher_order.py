import pandas as pd,numpy as np,matplotlib.pyplot as plt
from pathlib import Path
p=Path(__file__).resolve().parents[1]/"rerun"
s=pd.read_csv(p/'higher_order_summary.csv')
fig,ax=plt.subplots(figsize=(3.55,2.2))
ys=np.array([1,0]);names=['Fixed complementary roles','Permuted roles']
for y,(_,row) in zip(ys,s.sort_values('condition').iterrows()):
    # lexicographic sorting fixed then permuted
    ax.plot([row.CI_low,row.CI_high],[y,y],lw=3,solid_capstyle='round')
    ax.plot(row.MSE_B_minus_A,y,'o',ms=7)
ax.axvline(0,color='.4',ls='--',lw=.8)
ax.set_yticks(ys,names,fontsize=8)
ax.set_xlabel('Paired MSE difference: B minus A',fontsize=8)
ax.set_xlim(-.023,.092)
ax.grid(axis='x',alpha=.15)
ax.spines[['top','right']].set_visible(False)
ax.tick_params(axis='x',labelsize=8)
fig.tight_layout(pad=.7)
fig.savefig(p/'fig_hypergroup.pdf',bbox_inches='tight')
fig.savefig(p/'fig_hypergroup.png',dpi=170,bbox_inches='tight')
