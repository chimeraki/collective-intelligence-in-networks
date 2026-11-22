from pathlib import Path
import sys,time
import numpy as np,pandas as pd,networkx as nx
sys.path.insert(0,str(Path(__file__).resolve().parent))
from decentralized_collaboration_benchmark import graph_of,D
from placement_causal_validation import FAMILIES,assignment
from test_exact_dgd import bundles_sigma
rows=[];tic=time.time()
for n in [48,96]:
 for fi,f in enumerate(FAMILIES):
  for rep in range(6):
   gs=505201+n*131+fi*37+rep; g=graph_of(f,n,gs)
   reach=np.asarray(nx.floyd_warshall_numpy(g,nodelist=list(range(n))))
   for noise in ['homogeneous']:
    seed=777001+gs*3+(noise=='heterogeneous')*1543
    A,y,th,typ,sigma=bundles_sigma(n,seed,noise)
    for placement in ['random','clustered','interleaved','heterophily_swap']:
     perm=assignment(typ,g,placement,gs+1903);a=A[perm]
     for t in [1,2,3,4,6]:
      ranks=np.array([np.linalg.matrix_rank(a[reach[i]<=t-1],tol=1e-9) for i in range(n)])
      rows.append(dict(n=n,family=f,rep=rep,placement=placement,rounds=t,mean_rank=ranks.mean(),bayes_lower_bound=(D-ranks.mean())/D,locally_identifiable=(ranks==D).mean()))
 print('n',n,'elapsed',time.time()-tic,flush=True)
df=pd.DataFrame(rows);df.to_csv(str(Path(__file__).resolve().parents[1]/'rerun'/'rank_observability_bounds.csv'),index=False)
print(df.groupby(['rounds','placement']).bayes_lower_bound.mean().unstack().round(3).to_string())
print('ring at round3',df[(df.family=='ring')&(df.rounds==3)].groupby('placement').bayes_lower_bound.mean())
print('elapsed',time.time()-tic)
