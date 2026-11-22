"""Targeted design comparisons. Same graph and measurement bundles; no hidden state/response in design objectives.
Scores: complementary edges; finite-time logdet; neighborhood-heterogeneity-inspired covariance dispersion;
noisy misspecified logdet. Fixed proposal and fixed wall-time settings."""
import sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np,pandas as pd
from placement_causal_validation import make_bundles,assignment
from decentralized_collaboration_benchmark import graph_of,mixing,grad,D
from pathlib import Path
OUT=Path(str(Path(__file__).resolve().parents[1]/'rerun'))
F=['ring','small_world','random_regular','erdos_renyi','modular','geometric','bipartite','torus']

def sim(w,a,y,theta,T=120):
 z=np.zeros((len(y),D))
 for _ in range(T):z=w@z-.22*grad(z,a,y)
 return float(np.mean((z-theta)**2))

def optimize(g,w,a,typ,initial,method,seed,budget=80,time_budget=None,perturb=0):
 start=time.perf_counter();rng=np.random.default_rng(seed);order=initial.copy(); n=len(order);edge=np.asarray(list(g.edges()),int)
 p=np.linalg.matrix_power(w.toarray(),4)
 aa=a.copy()
 if perturb:
  aa=aa+rng.normal(size=aa.shape)*perturb
  aa=aa/np.linalg.norm(aa,axis=1,keepdims=True)
 q=np.einsum('ni,nj->nij',aa,aa)
 qbar=q.mean(axis=0)
 def score(o):
  if method=='cross':return float(np.count_nonzero(typ[o][edge[:,0]]!=typ[o][edge[:,1]]))
  if method in ('info','misspecified'):
   fisher=np.einsum('ij,jab->iab',p,q[o],optimize=True)
   return float(np.linalg.slogdet(np.eye(D)[None,:,:]+fisher)[1].mean())
  if method=='heterogeneity':
   # Le Bars-inspired local neighborhood mismatch, using sensing covariance instead of gradients.
   local=np.einsum('ij,jab->iab',w.toarray(),q[o],optimize=True)
   return -float(np.mean((local-qbar)**2))
  return 0.
 calls=0;accepted=0
 prev=score(order) if method!='random' else 0
 while (calls<budget if time_budget is None else (time.perf_counter()-start)<time_budget):
  i,j=rng.choice(n,2,replace=False);calls+=1
  if typ[order[i]]==typ[order[j]]:continue
  order[i],order[j]=order[j],order[i]
  if method=='random':accepted+=1;continue
  s=score(order)
  if s>=prev+1e-12:prev=s;accepted+=1
  else:order[i],order[j]=order[j],order[i]
 return order,calls,accepted,time.perf_counter()-start

def main():
 rows=[]
 for fi,f in enumerate(F):
  for rep in range(2):
   n=48;gs=607001+fi*41+rep*7;g=graph_of(f,n,gs);w=mixing(g)
   for noise in ['homogeneous','heterogeneous']:
    a,y,theta,typ=make_bundles(n,770003+gs*3+int(noise=='heterogeneous')*98,noise)
    initial=assignment(typ,g,'random',gs+1500)
    base=sim(w,a[initial],y[initial],theta)
    for mode in ('fixed_proposals','fixed_time'):
     for method in ('cross','heterogeneity','info','misspecified','random'):
      order,calls,accepted,elapsed=optimize(g,w,a,typ,initial,method,gs*100+sum(map(ord,method))+int(noise=='heterogeneous')*31,
                      budget=80,time_budget=(.16 if mode=='fixed_time' else None),perturb=(.7 if method=='misspecified' else 0))
      mse=sim(w,a[order],y[order],theta)
      rows.append(dict(family=f,rep=rep,noise=noise,mode=mode,method=method,baseline_mse=base,mse=mse,
            reduction_pct=100*(base-mse)/base,proposals=calls,accepted=accepted,seconds=elapsed))
   print('done',f,rep,flush=True)
 df=pd.DataFrame(rows);df.to_csv(OUT/'new_design_controls.csv',index=False)
 print(df.groupby(['mode','method']).agg(mean_pct=('reduction_pct','mean'),median_pct=('reduction_pct','median'),wins=('reduction_pct',lambda x:int((x>0).sum())),proposals=('proposals','mean'),seconds=('seconds','mean')).round(3).to_string())
if __name__=='__main__':main()
