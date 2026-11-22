from pathlib import Path
import pandas as pd, numpy as np
root=Path(__file__).resolve().parent
pairs=[('placement_causal_runs.csv','placement_causal_runs.csv'),('exact_dgd_risks_full.csv','exact_dgd_risks_full.csv'),('exact_dgd_paired_contrasts_full.csv','exact_dgd_paired_contrasts_full.csv'),('rank_observability_bounds.csv','rank_observability_bounds.csv'),('exact_risk_design.csv','exact_risk_design.csv'),('higher_order_group_inference.csv','higher_order_group_inference.csv'),('higher_order_summary.csv','higher_order_summary.csv'),('higher_order_dyadic_control.csv','higher_order_dyadic_control.csv')]
for generated,archived in pairs:
 a=pd.read_csv(root/'rerun'/generated);b=pd.read_csv(root/'data'/archived)
 assert a.equals(b),(generated,a.shape,b.shape)
 print('EXACT PASS',generated,len(a))
x=pd.read_csv(root/'rerun'/'new_design_controls.csv');y=pd.read_csv(root/'data'/'new_design_controls.csv')
a=x[x['mode']=='fixed_proposals'].reset_index(drop=True);b=y[y['mode']=='fixed_proposals'].reset_index(drop=True)
assert a.drop(columns='seconds').equals(b.drop(columns='seconds'))
print('FIXED-PROPOSAL PASS (excluding runtime seconds)')
print('Wall-clock-budget contrasts are expected to vary with hardware and scheduling.')
print('Fresh-seed replication is archived-only; generator not included.')
