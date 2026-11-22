import pandas as pd,numpy as np
from pathlib import Path
p=Path(str(Path(__file__).resolve().parents[1]/'rerun'))
d=pd.read_csv(p/'new_design_controls.csv');h=pd.read_csv(p/'higher_order_dyadic_control.csv')
rng=np.random.default_rng(772)
rows=[]
for mode in sorted(d['mode'].unique()):
 for method in sorted(d.method.unique()):
  x=d[(d['mode']==mode)&(d.method==method)]
  # graph-instance resampling, preserves two noise conditions per instance
  v=x.groupby(['family','rep']).reduction_pct.mean().values
  boot=v[rng.integers(0,len(v),size=(4000,len(v)))].mean(axis=1)
  rows.append(dict(mode=mode,method=method,mean_percent=v.mean(),ci_low=np.quantile(boot,.025),ci_high=np.quantile(boot,.975),wins=int(sum(v>0)),clusters=len(v),proposals=x.proposals.mean(),mean_seconds=x.seconds.mean()))
for mode in sorted(d['mode'].unique()):
 q=d[d['mode']==mode].pivot_table(index=['family','rep','noise'],columns='method',values='reduction_pct')
 for other in ['heterogeneity','cross','misspecified']:
  v=(q['info']-q[other]).groupby(['family','rep']).mean().values
  b=v[rng.integers(0,len(v),size=(4000,len(v)))].mean(axis=1)
  rows.append(dict(mode=mode,method='info minus '+other,mean_percent=v.mean(),ci_low=np.quantile(b,.025),ci_high=np.quantile(b,.975),wins=int(sum(v>0)),clusters=len(v),proposals=np.nan,mean_seconds=np.nan))
pd.DataFrame(rows).to_csv(p/'new_design_control_summary.csv',index=False)
print(pd.DataFrame(rows).round(3).to_string(index=False))
rows=[]
for (connected,role),x in h.groupby(['connected','role_assignment']):
 v=x['difference'].to_numpy();b=v[rng.integers(0,len(v),size=(4000,len(v)))].mean(axis=1)
 rows.append(dict(connected=connected,assignment=role,mean_diff=v.mean(),low=np.quantile(b,.025),high=np.quantile(b,.975),max_dyadic_gap=x.max_protocol_gap.max(),trials=len(v)))
pd.DataFrame(rows).to_csv(p/'higher_order_dyadic_summary.csv',index=False)
print(pd.DataFrame(rows).round(5).to_string(index=False))
