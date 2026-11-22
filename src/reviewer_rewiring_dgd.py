"""Same-data, same DGD protocol edge-rewiring pilot with fixed candidate budget."""
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from placement_causal_validation import make_bundles,assignment
from decentralized_collaboration_benchmark import graph_of,mixing,grad,D
import numpy as np, pandas as pd, networkx as nx
from pathlib import Path
O=Path(str(Path(__file__).resolve().parents[1]/'rerun'))
F=['ring','small_world','random_regular','erdos_renyi','modular','geometric','bipartite','torus']; r=np.random.default_rng(211126)

def dgd(g,a,y,theta):
 w=mixing(g);z=np.zeros((len(y),D));out={}
 for t in range(1,241):
  z=w@z-.22*grad(z,a,y)
  if t in (120,240):out[t]=float(np.mean((z-theta)**2))
 return out

def score(g,a,objective):
 w=mixing(g).toarray()
 if objective=='spectral':
  vals=np.linalg.eigvalsh(w);return 1-max(abs(vals[0]),abs(vals[-2]))
 p=np.linalg.matrix_power(w,4)
 q=np.einsum('ni,nj->nij',a,a)
 fisher=np.einsum('ij,jab->iab',p,q)
 return np.linalg.slogdet(np.eye(D)[None]+fisher)[1].mean()

def propose(g,r):
 ed=list(g.edges());
 for _ in range(30):
  e1,e2=r.choice(len(ed),2,replace=False);a,b=ed[e1];c,d=ed[e2]
  if len({a,b,c,d})<4:continue
  if r.random()<.5:c,d=d,c
  if a==c or b==d or g.has_edge(a,c) or g.has_edge(b,d):continue
  q=g.copy();q.remove_edge(a,b);q.remove_edge(c,d);q.add_edge(a,c);q.add_edge(b,d)
  if nx.is_connected(q):return q
 return None

def optimize(g,a,obj,r, budget=30):
 q=g.copy();old=score(q,a,obj) if obj!='random' else 0
 for k in range(budget):
  cand=propose(q,r)
  if cand is None:continue
  if obj=='random':q=cand
  else:
   sc=score(cand,a,obj)
   if sc>old+1e-12:q,old=cand,sc
 return q

def main():
 results=[];designs=[]
 for fi,f in enumerate(F):
  for rep in range(3):
   n=48; gs=905201+n*131+fi*37+rep
   g=graph_of(f,n,gs);rng=np.random.default_rng(gs)
   for noise in ('homogeneous','heterogeneous'):
    seed=777001+gs*3+(noise=='heterogeneous')*1543
    a,y,theta,typ=make_bundles(n,seed,noise)
    perm=assignment(typ,g,'random',gs+1903);aa=a[perm];yy=y[perm]
    designspec={'baseline':g}
    for obj in ('random','spectral','task'):
     gg=optimize(g,aa,'info' if obj=='task' else obj,np.random.default_rng(gs*11+len(obj)*7),budget=30)
     designspec[obj+'_edges']=gg
    # same 300 candidate-swap heuristic as primary experiment, fixed graph
    p2=assignment(typ,g,'heterophily_swap',gs+1903)
    for label,gg in designspec.items():
     inputs=(a[p2],y[p2]) if label=='baseline_placement' else (aa,yy)
     res=dgd(gg,*inputs,theta)
     for t,v in res.items():results.append(dict(family=f,rep=rep,noise=noise,method=label,rounds=t,mse=v))
     designs.append(dict(family=f,rep=rep,noise=noise,method=label,edges=gg.number_of_edges(),same_degrees=sorted(dict(gg.degree()).values())==sorted(dict(g.degree()).values())))
    res=dgd(g,a[p2],y[p2],theta)
    for t,v in res.items():results.append(dict(family=f,rep=rep,noise=noise,method='heterophily_placement',rounds=t,mse=v))
   print('finished',f,rep,flush=True)
 df=pd.DataFrame(results);df.to_csv(O/'reviewer_rewiring_dgd.csv',index=False)
 p=df.pivot(index=['family','rep','noise','rounds'],columns='method',values='mse');out=[]
 for t in (120,240):
  q=p.xs(t,level='rounds')
  for method in [x for x in q.columns if x!='baseline']:
   delta=(q[method]-q.baseline).groupby(['family','rep']).mean();v=delta.values
   rr=np.random.default_rng(390);bs=v[rr.integers(len(v),size=(4000,len(v)))].mean(axis=1)
   out.append(dict(rounds=t,method=method,mean_delta=v.mean(),improved=int(sum(v<0)),clusters=len(v),ci_low=np.quantile(bs,.025),ci_high=np.quantile(bs,.975)))
 pd.DataFrame(out).to_csv(O/'reviewer_rewiring_dgd_summary.csv',index=False)
 print(pd.DataFrame(out).to_string(index=False),flush=True)
if __name__=='__main__':main()
