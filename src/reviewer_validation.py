"""Targeted reviewer checks, seeded, fixed-bundle comparisons."""
import sys, time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd, networkx as nx
from scipy import stats
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error
from decentralized_collaboration_benchmark import graph_of,mixing,grad,REG,D
from placement_causal_validation import make_bundles,assignment
OUT=Path(str(Path(__file__).resolve().parents[1]/'rerun')); rng=np.random.default_rng(20261121)
FAMILIES=['ring','small_world','random_regular','erdos_renyi','modular','geometric','bipartite','torus']
BASE=pd.read_csv(str(Path(__file__).resolve().parents[1]/'data'/'placement_causal_runs.csv'))

def setup(n,fi,rep,noise):
 gs=505201+n*131+fi*37+rep; g=graph_of(FAMILIES[fi],n,gs);w=mixing(g)
 seed=777001+gs*3+(noise=='heterogeneous')*1543
 a,y,th,typ=make_bundles(n,seed,noise)
 return gs,g,w,a,y,th,typ

def features(w,g,a,typ,perm):
 n=len(perm); a=a[perm];typ=typ[perm]; ed=np.asarray(g.edges(),int)
 mixed=np.mean(typ[ed[:,0]]!=typ[ed[:,1]])
 # Finite-time distribution of task sensing directions through the same W used by DGD
 p=(w@w@w@w).toarray(); p=np.clip(p,0,1)
 quad=np.einsum('ni,nj->nij',a,a)
 fisher=np.einsum('ij,jab->iab',p,quad)
 vals=np.linalg.eigvalsh(fisher)
 logdet=np.mean(np.log1p(np.maximum(vals,0)).sum(axis=1))
 loweig=np.mean(vals[:,0]);
 # Role-specific mass and quadratic task info derived from a (no responses, no true hidden state)
 mass=np.stack([p@(typ==k).astype(float) for k in range(4)],axis=1)
 coverage=np.mean(mass.min(axis=1))
 return dict(mixed=mixed,info_logdet=logdet,info_min=loweig,role_coverage=coverage)

def train_predict():
 rows=[]
 for n in (48,96):
  for fi,f in enumerate(FAMILIES):
   for rep in range(6):
    gs,g,w,a,y,theta,typ=setup(n,fi,rep,'homogeneous')
    for place in ('random','clustered','interleaved','heterophily_swap'):
     perm=assignment(typ,g,place,gs+1903); feat=features(w,g,a,typ,perm)
     rec=BASE.query('n==@n and family==@f and rep==@rep and noise=="homogeneous" and placement==@place and rounds==120').iloc[0]
     rows.append(dict(n=n,family=f,rep=rep,placement=place,mse=float(rec.mse),**feat))
 met=pd.DataFrame(rows)
 met['cluster']=met.n.astype(str)+'_'+met.family+'_'+met.rep.astype(str)
 for c in ['mse','mixed','info_logdet','info_min','role_coverage']:
  met[c+'_diff']=met[c]-met.groupby('cluster')[c].transform('mean')
 met.to_csv(OUT/'reviewer_predictor_data.csv',index=False)
 # graph-only descriptors are constant under placement, a necessary negative control
 # all diagnostics evaluated via leave-one-GRAPH-FAMILY-out. Deviations make prediction target within-instance effects
 mods={'cross_specialty':['mixed_diff'],'info':['info_logdet_diff','info_min_diff','role_coverage_diff'],'combined':['mixed_diff','info_logdet_diff','info_min_diff','role_coverage_diff']}
 res=[];logo=LeaveOneGroupOut()
 for name,cols in mods.items():
  y=met.mse_diff.to_numpy();pred=np.zeros(len(y))
  for tr,te in logo.split(met[cols],y,groups=met.family):
   model=make_pipeline(StandardScaler(),Ridge(alpha=10));model.fit(met[cols].iloc[tr],y[tr]);pred[te]=model.predict(met[cols].iloc[te])
  res.append(dict(model=name,heldout_family_r2=r2_score(y,pred),mae=mean_absolute_error(y,pred),spearman=stats.spearmanr(y,pred).statistic))
 # graph-only metric has zero within-graph signal, predicts deviation 0
 y=met.mse_diff.values
 res.append(dict(model='graph-only (within-instance)',heldout_family_r2=r2_score(y,np.zeros(len(y))),mae=np.mean(abs(y)),spearman=np.nan))
 pd.DataFrame(res).to_csv(OUT/'reviewer_prediction_cv.csv',index=False)
 print('CV',pd.DataFrame(res).to_string(index=False),flush=True)
 # stratified effect with sign and cluster-aware CI
 p=BASE.query('rounds==120').pivot(index=['n','family','rep','noise'],columns='placement',values='mse')
 rows=[]
 for fam in FAMILIES:
  q=p.xs(fam,level='family')
  for place in ('clustered','interleaved','heterophily_swap'):
   d=(q[place]-q.random).groupby(level=['n','rep']).mean().values
   rr=np.random.default_rng(300+FAMILIES.index(fam));b=d[rr.integers(0,len(d),size=(5000,len(d)))].mean(axis=1)
   rows.append(dict(family=fam,placement=place,n_clusters=len(d),mean_delta=d.mean(),median_delta=np.median(d),improved=int(sum(d<0)),ci_low=np.quantile(b,.025),ci_high=np.quantile(b,.975)))
 pd.DataFrame(rows).to_csv(OUT/'reviewer_family_effects.csv',index=False)
 print('Family',pd.DataFrame(rows).query('placement=="clustered"')[['family','mean_delta','improved']].to_string(index=False),flush=True)
 return met

def bfs_order(g):
 visited=set();arr=[]
 for v in sorted(g):
  if v not in visited:
   for u in nx.bfs_tree(g,v):
    visited.add(u);arr.append(u)
 return np.array(arr)

def alternative_perm(typ,g,mode,seed):
 n=len(typ);r=np.random.default_rng(seed); by=[list(np.flatnonzero(typ==k)) for k in range(4)]
 if mode=='bfs_cluster':
  nodes=bfs_order(g);order=np.empty(n,dtype=int);order[nodes]=np.arange(n);return order
 if mode=='relabel_cluster':return r.permutation(n)
 if mode=='random_swap':return r.permutation(n)
 raise ValueError(mode)

def evalrun(w,a,y,theta,eta=.22,T=240):
 z=np.zeros((len(y),D)); res={}
 for t in range(1,T+1):
  z=w@z-eta*grad(z,a,y)
  if t in (40,120,240):res[t]=float(np.mean((z-theta)**2))
 return res

def ordering_and_step_control():
 rows=[]
 for n in (48,96):
  for fi,f in enumerate(FAMILIES):
   for rep in range(3):
    for noise in ('homogeneous','heterogeneous'):
     gs,g,w,a,y,theta,typ=setup(n,fi,rep,noise)
     for place in ('random','clustered','interleaved','heterophily_swap','bfs_cluster','relabel_cluster','random_swap'):
      perm=assignment(typ,g,place,gs+1903) if place in ['random','clustered','interleaved','heterophily_swap'] else alternative_perm(typ,g,place,gs+347)
      aa=a[perm];yy=y[perm]
      for eta in (.12,.22):
       res=evalrun(w,aa,yy,theta,eta)
       for t,mse in res.items():rows.append(dict(n=n,family=f,rep=rep,noise=noise,placement=place,eta=eta,rounds=t,mse=mse))
   print('ordering',n,f,flush=True)
 df=pd.DataFrame(rows);df.to_csv(OUT/'reviewer_order_step_controls.csv',index=False)
 print(df.query('rounds==120').groupby(['eta','placement']).mse.mean().to_string(),flush=True)

if __name__=='__main__':
 train_predict();ordering_and_step_control()
