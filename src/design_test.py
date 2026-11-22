from pathlib import Path
import sys,time
import numpy as np,pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parent))
from decentralized_collaboration_benchmark import graph_of,mixing
from placement_causal_validation import FAMILIES,assignment,simulate
from test_exact_dgd import bundles_sigma, exact_risk

def design(g,w,A,y,th,typ,sig,seed,mode,proposals=80):
 rng=np.random.default_rng(seed);n=len(A);order=assignment(typ,g,'random',seed); edges=np.asarray(g.edges(),dtype=int)
 def cost(q):
  if mode=='edge':return -int(np.sum(typ[q][edges[:,0]]!=typ[q][edges[:,1]]))
  if mode=='risk':return exact_risk(w,A[q],sig[q],120)[0]
  return 0.
 old=cost(order)
 for k in range(proposals):
  i,j=rng.choice(n,2,replace=False)
  if typ[order[i]]==typ[order[j]]:continue
  order[i],order[j]=order[j],order[i]
  new=cost(order)
  if new<=old:old=new
  else:order[i],order[j]=order[j],order[i]
 return order

def main():
 rows=[];tic=time.time()
 for fi,f in enumerate(FAMILIES):
  for rep in range(3):
   n=48;gs=505201+n*131+fi*37+rep;g=graph_of(f,n,gs);w=mixing(g)
   seed=777001+gs*3;A,y,th,typ,sig=bundles_sigma(n,seed,'homogeneous')
   initseed=gs+1903
   init=assignment(typ,g,'random',initseed)
   for mode in ['random','edge','risk']:
    if mode=='random':order=init
    else:order=design(g,w,A,y,th,typ,sig,initseed,mode,80)
    r,b,v,_=exact_risk(w,A[order],sig[order],120)
    mse=simulate(w,A[order],y[order],th)[0][120][0]
    rows.append(dict(family=f,rep=rep,method=mode,expected_mse=r,realized_mse=mse))
  print(f,round(time.time()-tic,1),flush=True)
 df=pd.DataFrame(rows);df.to_csv(str(Path(__file__).resolve().parents[1]/'rerun'/'exact_risk_design.csv'),index=False)
 for field in ['expected_mse','realized_mse']:
  p=df.pivot(index=['family','rep'],columns='method',values=field)
  print(field, 'mean differences from random:\n', (p.subtract(p['random'],axis=0)).mean().to_string())
  print('instances improved risk',sum(p.risk<p.random),'edge',sum(p.edge<p.random))
 print('sec',time.time()-tic)
if __name__=='__main__':main()
