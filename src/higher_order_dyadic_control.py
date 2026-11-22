"""Budget-accounted dyadic emulation and connected-block extension of matched group inference."""
import numpy as np, pandas as pd
from pathlib import Path
from higher_order_experiment import A,B,ROLES,proj
OUT=Path(str(Path(__file__).resolve().parents[1]/'rerun'))
assert np.array_equal(proj(A),proj(B))

def run(seed=20261122,reps=500,blocks=12,sigma=.2):
 rng=np.random.default_rng(seed); rows=[]
 # Bridge triads are identical in both designs, preserving equal projected adjacency.
 def edges(H,connected):
  groups=[tuple(6*b+v for v in h) for b in range(blocks) for h in H]
  if connected: groups += [(6*b+5,6*(b+1),6*(b+1)+1) for b in range(blocks-1)]
  return groups
 for connected in (False,True):
  GA=edges(A,connected);GB=edges(B,connected)
  # confirm identical weighted pairwise projection, plus common bridges connect blocks
  def projection(groups):
   mat=np.zeros((6*blocks,6*blocks),int)
   for x,y,z in groups:
    for i,j in [(x,y),(x,z),(y,z)]:mat[i,j]+=1;mat[j,i]+=1
   return mat
  assert np.array_equal(projection(GA),projection(GB))
  for condition in ('fixed','randomized'):
   for t in range(reps):
    role=np.concatenate([ROLES if condition=='fixed' else rng.permutation(ROLES) for _ in range(blocks)])
    # Same task-specific state and per-node potential measurements across designs.
    latent=rng.normal(size=(len(GA),3));noise=rng.normal(scale=sigma,size=(len(GA),6*blocks))
    mse=[]; dyadic=[]
    for groups in (GA,GB):
     loss=[];same=[]
     for q,h in enumerate(groups):
      est=np.zeros(3)
      # Group meeting receives three observations. Dyadic star: two members transmit one
      # scalar each to a coordinator, who combines with their local scalar.
      messages=[(int(role[v]),float(latent[q,role[v]]+noise[q,v])) for v in h]
      for c in range(3):
       vals=[val for label,val in messages if label==c]
       if vals:est[c]=np.mean(vals)
      loss.append(np.mean((est-latent[q])**2))
      # Identical computation with two dyadic scalar deliveries, assuming the
      # coordinator can receive both members' tagged observations.
      paired=[messages[0],messages[1],messages[2]]
      alt=np.array([np.mean([v for r,v in paired if r==c]) if any(r==c for r,v in paired) else 0 for c in range(3)])
      same.append(np.mean((alt-latent[q])**2))
     mse.append(np.mean(loss));dyadic.append(np.mean(same))
    rows.append(dict(connected=connected,role_assignment=condition,trial=t,
      MSE_A=mse[0],MSE_B=mse[1],difference=mse[1]-mse[0],
      dyadic_MSE_A=dyadic[0],dyadic_MSE_B=dyadic[1],
      max_protocol_gap=max(abs(np.array(mse)-np.array(dyadic))),
      group_input_scalars=3*len(GA),dyadic_transmitted_scalars=2*len(GA)))
 df=pd.DataFrame(rows);df.to_csv(OUT/'higher_order_dyadic_control.csv',index=False)
 print(df.groupby(['connected','role_assignment']).agg(mean_diff=('difference','mean'),mean_A=('MSE_A','mean'),mean_B=('MSE_B','mean'),protocol_gap=('max_protocol_gap','max')).round(5))
if __name__=='__main__':run()
