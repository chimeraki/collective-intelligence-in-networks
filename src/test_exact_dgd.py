from pathlib import Path
import sys,time
import numpy as np, pandas as pd
from scipy import sparse
sys.path.insert(0,str(Path(__file__).resolve().parent))
from decentralized_collaboration_benchmark import graph_of,mixing,REG,D
from placement_causal_validation import FAMILIES,make_bundles,assignment,simulate,ETA

def bundles_sigma(n,seed,noise):
 r=np.random.default_rng(seed);th=r.normal(size=D);th=th/np.linalg.norm(th)*np.sqrt(D)
 typ=np.repeat(np.arange(4),n//4);A=np.zeros((n,D))
 for j,k in enumerate(typ):
  A[j,2*k:2*k+2]=r.normal(size=2)
  A[j,2*((k+1)%4):2*((k+1)%4)+2]+=.12*r.normal(size=2)
 A/=np.linalg.norm(A,axis=1,keepdims=True)
 sig=np.full(n,.10) if noise=='homogeneous' else np.where(r.random(n)<.25,.35,.07)
 y=A@th+r.normal(size=n)*sig
 return A,y,th,typ,sig

def exact_risk(w,a,sigma,T=120,eta=ETA):
 n=len(a);d=a.shape[1]; m=n+d
 # Transfer operator maps [theta; e] into local estimates.
 B=np.zeros((n,d,m)); B[:,:,:d]=np.einsum('ni,nj->nij',a,a)
 B[np.arange(n),:,d+np.arange(n)]=a*sigma[:,None]
 U=np.zeros_like(B)
 A2=a.reshape(n,d,1)
 ww=w.tocsr()
 for _ in range(T):
  mix=(ww @ U.reshape(n,-1)).reshape(n,d,m)
  scalar=np.einsum('nd,ndk->nk',a,U)
  U=mix-eta*(a[:,:,None]*scalar[:,None,:]+REG*U)+eta*B
 bias=U[:,:,:d]-np.eye(d)[None,:,:]
 return (np.sum(bias*bias)+np.sum(U[:,:,d:]**2))/(n*d), np.sum(bias*bias)/(n*d),np.sum(U[:,:,d:]**2)/(n*d),U

def main():
 rows=[]; tic=time.time()
 for fi,f in enumerate(FAMILIES):
  for rep in range(3):
   n=48; gs=505201+n*131+fi*37+rep;g=graph_of(f,n,gs);w=mixing(g)
   for noise in ['homogeneous','heterogeneous']:
    seed=777001+gs*3+(noise=='heterogeneous')*1543
    A,y,theta,typ,sigma=bundles_sigma(n,seed,noise)
    for method in ['random','clustered','interleaved','heterophily_swap']:
     perm=assignment(typ,g,method,gs+1903)
     a=A[perm];yy=y[perm];sig=sigma[perm]
     r,b,v,U=exact_risk(w,a,sig,120)
     z=U[:,:,:D]@theta + np.einsum('ndk,k->nd',U[:,:,D:],(yy-a@theta)/sig)
     mse_direct=float(np.mean((z-theta[None,:])**2))
     observed=simulate(w,a,yy,theta)[0][120][0]
     assert abs(mse_direct-observed)<2e-12,(mse_direct,observed)
     rows.append(dict(n=n,family=f,rep=rep,noise=noise,placement=method,analytic_bayes_risk=r,bias=b,variance=v,observed_mse=observed,graph_seed=gs))
  print('family',f,'sec',round(time.time()-tic,1),flush=True)
 df=pd.DataFrame(rows);df.to_csv(str(Path(__file__).resolve().parents[1]/'rerun'/'exact_dgd_risks.csv'),index=False)
 p=df.pivot(index=['family','rep','noise'],columns='placement',values=['analytic_bayes_risk','observed_mse'])
 contrasts=[]
 for alt in ['clustered','interleaved','heterophily_swap']:
  for (fam,rep,noise),row in p.iterrows():
   contrasts.append(dict(family=fam,rep=rep,noise=noise,placement=alt,predicted_delta=row['analytic_bayes_risk',alt]-row['analytic_bayes_risk','random'],observed_delta=row['observed_mse',alt]-row['observed_mse','random']))
 cd=pd.DataFrame(contrasts);cd.to_csv(str(Path(__file__).resolve().parents[1]/'rerun'/'exact_dgd_paired_contrasts.csv'),index=False)
 from scipy.stats import pearsonr,spearmanr
 print('config correlation',pearsonr(df.analytic_bayes_risk,df.observed_mse))
 print('contrast correlation',pearsonr(cd.predicted_delta,cd.observed_delta),spearmanr(cd.predicted_delta,cd.observed_delta))
 print(cd.groupby('placement')[['predicted_delta','observed_delta']].mean().to_string())
 print('sign agreement',np.mean(np.sign(cd.predicted_delta)==np.sign(cd.observed_delta)))
 print('total seconds',time.time()-tic)
if __name__=='__main__':main()
