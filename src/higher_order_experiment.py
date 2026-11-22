"""Matched higher-order group-inference counterexample. Requires numpy/pandas."""
import itertools
import numpy as np
import pandas as pd
from pathlib import Path
OUT=Path(__file__).resolve().parents[1]/"rerun"
A=[(0,1,2),(0,3,4),(1,3,5),(2,4,5)]
B=[(0,1,3),(0,2,4),(1,2,5),(3,4,5)]
ROLES=np.array([0,0,0,1,1,2])
def proj(H):
    x=np.zeros((6,6),int)
    for h in H:
        for u,v in itertools.combinations(h,2):x[u,v]+=1;x[v,u]+=1
    return x
assert np.array_equal(proj(A),proj(B))
def run(reps=500,blocks=12,sigma=.2,seed=24680):
    rng=np.random.default_rng(seed); records=[]
    for condition in ('fixed','permuted'):
        for trial in range(reps):
            errA=errB=0.0;completeA=completeB=0
            for block in range(blocks):
                roles=ROLES.copy() if condition=='fixed' else rng.permutation(ROLES)
                # Matched task state, node-and-task noise for the two hypergraphs.
                # Each of the four group IDs has the same latent 3-dimensional target.
                theta=rng.standard_normal((4,3))
                noise=rng.normal(0,sigma,(4,6))
                for H,which in ((A,'A'),(B,'B')):
                    for task,edge in enumerate(H):
                        estimate=np.zeros(3)
                        for k in range(3):
                            specialists=[v for v in edge if roles[v]==k]
                            if specialists:
                                estimate[k]=np.mean([theta[task,k]+noise[task,v] for v in specialists])
                        sq=np.mean((estimate-theta[task])**2)
                        if which=='A':errA+=sq;completeA+=int(len(set(roles[list(edge)]))==3)
                        else:errB+=sq;completeB+=int(len(set(roles[list(edge)]))==3)
            records.append(dict(condition=condition,trial=trial,mean_mse_A=errA/(blocks*4),mean_mse_B=errB/(blocks*4),difference_B_minus_A=(errB-errA)/(blocks*4),complete_A=completeA,complete_B=completeB))
    df=pd.DataFrame(records);df.to_csv(OUT/'higher_order_group_inference.csv',index=False)
    summ=[]
    for name,g in df.groupby('condition'):
        rngb=np.random.default_rng(321)
        v=g.difference_B_minus_A.to_numpy()
        idx=rngb.integers(0,len(v),(4000,len(v)))
        ci=np.quantile(v[idx].mean(axis=1),[.025,.975])
        summ.append(dict(condition=name,repetitions=len(g),MSE_A=g.mean_mse_A.mean(),MSE_B=g.mean_mse_B.mean(),MSE_B_minus_A=v.mean(),CI_low=ci[0],CI_high=ci[1],complete_groups_A=g.complete_A.mean(),complete_groups_B=g.complete_B.mean()))
    s=pd.DataFrame(summ);s.to_csv(OUT/'higher_order_summary.csv',index=False)
    print(s.to_string(index=False))
if __name__=='__main__':run()
