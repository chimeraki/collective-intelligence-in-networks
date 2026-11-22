"""Equal-candidate-budget direct placement heuristic comparison (existing measurements unchanged)."""
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from placement_causal_validation import make_bundles,assignment
from decentralized_collaboration_benchmark import graph_of,mixing,grad,D
import numpy as np,pandas as pd
from pathlib import Path
O=Path(str(Path(__file__).resolve().parents[1]/'rerun'));F=['ring','small_world','random_regular','erdos_renyi','modular','geometric','bipartite','torus']

def score_info(p,a):
 q=np.einsum('ni,nj->nij',a,a);f=np.einsum('ij,jab->iab',p,q)
 return np.linalg.slogdet(np.eye(D)[None]+f)[1].mean()

def optimize(g,w,a,typ,initial,criterion,seed,budget=80):
 rng=np.random.default_rng(seed);order=initial.copy(); ed=np.array(list(g.edges()),int);p=np.linalg.matrix_power(w.toarray(),4)
 def score(o):
  if criterion=='cross':
   tt=typ[o];return float(np.sum(tt[ed[:,0]]!=tt[ed[:,1]]))
  return score_info(p,a[o])
 prev=score(order) if criterion!='random' else 0
 for _ in range(budget):
  i,j=rng.choice(len(a),2,replace=False)
  order[i],order[j]=order[j],order[i]
  if criterion!='random':
   nxt=score(order)
   if nxt>=prev:prev=nxt
   else:order[i],order[j]=order[j],order[i]
 return order

def run(w,a,y,theta):
 z=np.zeros((len(y),D))
 for t in range(120):z=w@z-.22*grad(z,a,y)
 return np.mean((z-theta)**2)

def main():
 rows=[]
 for fi,f in enumerate(F):
  for rep in range(3):
   n=48;gs=505201+n*131+fi*37+rep;g=graph_of(f,n,gs);w=mixing(g)
   for noise in ('homogeneous','heterogeneous'):
    seed=777001+gs*3+(noise=='heterogeneous')*1543
    a,y,th,typ=make_bundles(n,seed,noise)
    initial=assignment(typ,g,'random',gs+1903)
    for method in ['baseline','random_swaps','cross_swaps','info_swaps']:
     order=initial if method=='baseline' else optimize(g,w,a,typ,initial,{'random_swaps':'random','cross_swaps':'cross','info_swaps':'info'}[method],gs*100+len(method),80)
     mse=run(w,a[order],y[order],th)
     rows.append(dict(family=f,rep=rep,noise=noise,method=method,mse=mse))
  print('complete',f,flush=True)
 d=pd.DataFrame(rows);d.to_csv(O/'reviewer_placement_budget.csv',index=False)
 q=d.pivot(index=['family','rep','noise'],columns='method',values='mse');out=[]
 for method in ['random_swaps','cross_swaps','info_swaps']:
  x=(q[method]-q.baseline).groupby(['family','rep']).mean().values
  r=np.random.default_rng(234);b=x[r.integers(len(x),size=(5000,len(x)))].mean(axis=1)
  out.append(dict(method=method,mean_delta=x.mean(),win=int(sum(x<0)),clusters=len(x),ci_low=np.quantile(b,.025),ci_high=np.quantile(b,.975)))
 print(pd.DataFrame(out).to_string(index=False))
 pd.DataFrame(out).to_csv(O/'reviewer_placement_budget_summary.csv',index=False)
if __name__=='__main__':main()
