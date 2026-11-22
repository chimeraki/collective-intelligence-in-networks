"""Matched-data causal placement validation: same sensor/response bundles permuted across fixed nodes.
Note: this is a fresh correction to earlier placement comparisons, not an independent task family.
"""
import numpy as np,pandas as pd, networkx as nx
from pathlib import Path
from scipy import sparse
from decentralized_collaboration_benchmark import graph_of,mixing,grad,REG,D
OUT=Path(__file__).resolve().parents[1]/"rerun"; FAMILIES=['ring','small_world','random_regular','erdos_renyi','modular','geometric','bipartite','torus']
NREPS=6; SIZES=[48,96]; NOISES=['homogeneous','heterogeneous']; TMAX=240; HORIZONS=[40,120,240]; ETA=.22

def make_bundles(n,seed,noise):
 r=np.random.default_rng(seed); th=r.normal(size=D);th=th/np.linalg.norm(th)*np.sqrt(D)
 typ=np.repeat(np.arange(4),n//4); A=np.zeros((n,D))
 for j,k in enumerate(typ):
  A[j,2*k:2*k+2]=r.normal(size=2)
  A[j,2*((k+1)%4):2*((k+1)%4)+2]+=.12*r.normal(size=2)
 A/=np.linalg.norm(A,axis=1,keepdims=True)
 sig=np.full(n,.10) if noise=='homogeneous' else np.where(r.random(n)<.25,.35,.07)
 y=A@th+r.normal(size=n)*sig
 return A,y,th,typ

def assignment(typ,g,mode,seed):
 n=len(typ);r=np.random.default_rng(seed)
 if mode=='clustered': return np.arange(n)
 if mode=='random': return r.permutation(n)
 if mode=='interleaved':
  inds=[list(np.flatnonzero(typ==k)) for k in range(4)]
  return np.array([inds[i%4][i//4] for i in range(n)])
 if mode=='heterophily_swap':
  order=r.permutation(n);edge=np.asarray(g.edges(),dtype=int)
  t=typ[order].copy(); old=np.count_nonzero(t[edge[:,0]]!=t[edge[:,1]])
  for _ in range(300):
   i,j=r.choice(n,2,replace=False)
   if t[i]==t[j]:continue
   order[i],order[j]=order[j],order[i];t[i],t[j]=t[j],t[i]
   new=np.count_nonzero(t[edge[:,0]]!=t[edge[:,1]])
   if new>=old:old=new
   else:order[i],order[j]=order[j],order[i];t[i],t[j]=t[j],t[i]
  return order
 raise ValueError(mode)

def simulate(w,A,y,th):
 n=len(y); z=np.zeros((n,D)); rec={};opt=np.linalg.solve(A.T@A/n+REG*np.eye(D),A.T@y/n)
 for t in range(1,TMAX+1):
  z=w@z-ETA*grad(z,A,y)
  if t in HORIZONS:
   rec[t]=(float(np.mean((z-th)**2)),float(np.mean((z-opt)**2)),float(np.mean((z-z.mean(axis=0))**2)))
 return rec,float(np.mean((opt-th)**2))

def main():
 out=[]
 for n in SIZES:
  for fi,f in enumerate(FAMILIES):
   for rep in range(NREPS):
    gs=505201+n*131+fi*37+rep;g=graph_of(f,n,gs);w=mixing(g)
    for noise in NOISES:
     seed=777001+gs*3+(noise=='heterogeneous')*1543
     A,y,theta,typ=make_bundles(n,seed,noise)
     for method in ['random','clustered','interleaved','heterophily_swap']:
      perm=assignment(typ,g,method,gs+1903)
      a=A[perm]; yy=y[perm];rec,oracle=simulate(w,a,yy,theta)
      for t,(mse,excess,consensus) in rec.items():
       out.append(dict(n=n,family=f,rep=rep,noise=noise,placement=method,rounds=t,
             mse=mse,excess_mse=excess,consensus_error=consensus,oracle_mse=oracle,edges=g.number_of_edges(),
             n_messages=2*g.number_of_edges()*t,scalars=2*g.number_of_edges()*t*D))
   print('complete',n,f,flush=True)
 df=pd.DataFrame(out);assert len(df)==len(SIZES)*len(FAMILIES)*NREPS*len(NOISES)*4*len(HORIZONS)
 df.to_csv(OUT/'placement_causal_runs.csv',index=False)
 # inferential unit: graph seed; noise settings retained as repeated measures, not independent graph realizations
 p=df.pivot(index=['n','family','rep','noise','rounds'],columns='placement',values='mse')
 results=[];rng=np.random.default_rng(2602)
 for t in HORIZONS:
  q=p.xs(t,level='rounds');base=q['random'];
  for alt in ['clustered','interleaved','heterophily_swap']:
   diff=(q[alt]-base).groupby(level=['n','family','rep']).mean().values
   boots=np.mean(diff[rng.integers(len(diff),size=(4000,len(diff)))],axis=1)
   denom=base.mean()
   results.append(dict(rounds=t,placement=alt,graph_clusters=len(diff),baseline_mse=denom,
     mean_delta_mse=diff.mean(),relative_change_pct=100*diff.mean()/denom,
     ci_low=float(np.quantile(boots,.025)),ci_high=float(np.quantile(boots,.975)),
     graph_clusters_improved=int(np.sum(diff<0))))
 pd.DataFrame(results).to_csv(OUT/'placement_causal_clustered_ci.csv',index=False)
 print(pd.DataFrame(results).to_string(index=False));print('oracle difference',df.groupby(['n','family','rep','noise','rounds']).oracle_mse.agg(lambda v:v.max()-v.min()).max())
if __name__=='__main__':main()
