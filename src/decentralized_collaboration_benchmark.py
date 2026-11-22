"""Decentralized collaborative reconstruction with private sensor equations.
No agent sees other agents' observations; only d-dimensional state and (for gradient tracking)
gradient-tracker messages cross graph edges. Reproducible seeded, paired design.
"""
import numpy as np, pandas as pd, networkx as nx, scipy.sparse as sp
from scipy.stats import spearmanr
from pathlib import Path
OUT=Path(str(Path(__file__).resolve().parents[1]/'rerun')); D=8; REG=.025; HORIZONS=(20,60,120)

def graph_of(f,n,seed):
    r=np.random.default_rng(seed)
    if f=='ring': g=nx.watts_strogatz_graph(n,4,0,seed=seed)
    elif f=='small_world': g=nx.watts_strogatz_graph(n,4,.15,seed=seed)
    elif f=='random_regular': g=nx.random_regular_graph(4,n,seed=seed)
    elif f=='erdos_renyi': g=nx.gnm_random_graph(n,2*n,seed=seed)
    elif f=='preferential': g=nx.barabasi_albert_graph(n,2,seed=seed)
    elif f=='modular':
        g=nx.stochastic_block_model([n//4]*4,[[.22 if i==j else .006 for j in range(4)] for i in range(4)],seed=seed)
    elif f=='geometric': g=nx.random_geometric_graph(n,.22 if n==48 else .16,seed=seed)
    elif f=='bipartite':
        g=nx.bipartite.random_graph(n//2,n//2,4/(n//2),seed=seed)
    elif f=='core_periphery':
        core=max(6,n//5); g=nx.Graph();g.add_nodes_from(range(n))
        for i in range(core):
            for j in range(i):
                if r.random()<.36:g.add_edge(i,j)
        for j in range(core,n):
            for i in r.choice(core,size=2,replace=False):g.add_edge(j,int(i))
    elif f=='torus':
        g=nx.grid_2d_graph(4,n//4,periodic=True);g=nx.convert_node_labels_to_integers(g)
    if not nx.is_connected(g):
        c=list(nx.connected_components(g));
        for a,b in zip(c,c[1:]):g.add_edge(min(a),min(b))
    return nx.convert_node_labels_to_integers(g)

def mixing(g):
    n=len(g); a=nx.to_scipy_sparse_array(g,nodelist=list(range(n)),dtype=float,format='csr')
    deg=np.asarray(a.sum(axis=1)).reshape(-1)
    # symmetric doubly stochastic Metropolis weights
    i,j=a.nonzero(); vals=1/(1+np.maximum(deg[i],deg[j])); w=sp.csr_matrix((vals,(i,j)),shape=(n,n))
    w=w+sp.diags(1-np.asarray(w.sum(axis=1)).reshape(-1))
    return w

def roles_of(g,design,seed):
    n=len(g);rng=np.random.default_rng(seed);deg=np.array([g.degree(i) for i in range(n)])
    if design in ('imbalance','rare_hubs'):
        counts=np.array([n-int(3*round(.1*n)),round(.1*n),round(.1*n),round(.1*n)])
    else:counts=np.array([n//4]*4)
    base=np.repeat(np.arange(4),counts)
    if design in ('balanced_random','imbalance'):rng.shuffle(base);return base
    if design=='clustered':return base
    if design=='interleaved':return np.arange(n)%4
    if design=='degree_mixed':
        ranks=np.argsort(-deg,kind='stable');r=np.zeros(n,dtype=int);r[ranks]=np.arange(n)%4;return r
    if design=='rare_hubs':
        ranks=np.argsort(-deg,kind='stable');r=np.zeros(n,dtype=int);slots=np.repeat([1,2,3],counts[1:]);r[ranks[:len(slots)]]=slots;return r
    if design=='task_swap':
        rng.shuffle(base);r=base.copy();edges=np.asarray(g.edges(),dtype=int)
        def score(z):return np.count_nonzero(z[edges[:,0]]!=z[edges[:,1]])
        old=score(r)
        for _ in range(180):
            i,j=rng.choice(n,2,replace=False)
            if r[i]==r[j]:continue
            r[i],r[j]=r[j],r[i];new=score(r)
            if new>=old:old=new
            else:r[i],r[j]=r[j],r[i]
        return r
    raise ValueError(design)

def task_of(roles,seed,noise):
    rng=np.random.default_rng(seed);n=len(roles);theta=rng.normal(size=D); theta=theta/np.linalg.norm(theta)*np.sqrt(D)
    A=np.zeros((n,D));
    for i,k in enumerate(roles):
        A[i,2*k:2*k+2]=rng.normal(size=2)
        # modest cross-specialty coupling, creating partial information about other coordinates
        A[i,2*((k+1)%4) : 2*((k+1)%4)+2]+=.12*rng.normal(size=2)
    A/=np.maximum(np.linalg.norm(A,axis=1,keepdims=True),1e-10)
    sigma=np.full(n,.10)
    if noise=='heterogeneous':sigma=np.where(rng.random(n)<.25,.35,.07)
    y=np.einsum('ij,j->i',A,theta)+rng.normal(size=n)*sigma
    return A,y,theta

def grad(z,A,y):
    return ((np.sum(A*z,axis=1)-y)[:,None]*A+REG*z)

def run(w,A,y,theta,algo,eta=.22):
    n=len(y); z=np.zeros((n,D));tracker=grad(z,A,y); snapshots=[]
    # Objective uses mean individual residuals plus regularization
    optimum=np.linalg.solve((A.T@A)/n+REG*np.eye(D),(A.T@y)/n)
    def record(t):
        mse=np.mean((z-theta)**2)
        oracle=np.mean((optimum-theta)**2)
        subopt=np.mean(np.sum((z-optimum)**2,axis=1))/D
        consensus=np.mean((z-z.mean(axis=0))**2)
        # normalized global objective gap (objective, not state error)
        residual=A@optimum-y
        objstar=np.mean(residual**2)/2+REG*np.linalg.norm(optimum)**2/2
        local_loss=np.mean((z@A.T-y[None,:])**2,axis=1)/2+REG*np.sum(z*z,axis=1)/2
        gap=max(0,float(np.mean(local_loss)-objstar))
        snapshots.append((t,mse,oracle,subopt,consensus,gap))
    for t in range(1,max(HORIZONS)+1):
        if algo=='dgd':z=w@z-eta*grad(z,A,y)
        else:
            znew=w@z-eta*tracker
            tracker=w@tracker+grad(znew,A,y)-grad(z,A,y)
            z=znew
        if t in HORIZONS:record(t)
    return snapshots

def main():
    families=['ring','small_world','random_regular','erdos_renyi','preferential','modular','geometric','bipartite','core_periphery','torus']
    placements=['balanced_random','clustered','interleaved','degree_mixed','task_swap','imbalance','rare_hubs']
    rows=[];taskrows=[]
    for n in [48,96]:
      for fi,f in enumerate(families):
       for rep in range(3):
        seed=1031+100*n+fi*31+rep;g=graph_of(f,n,seed);w=mixing(g)
        for p in placements:
          roles=roles_of(g,p,seed+99)
          for noise in ['homogeneous','heterogeneous']:
            # Task seed fixed across algorithms; sensor matrix conditional on roles
            A,y,theta=task_of(roles,seed+761+(noise=='heterogeneous')*100003,noise)
            rank=np.linalg.matrix_rank(A)
            central=np.linalg.solve(A.T@A/n+REG*np.eye(D),A.T@y/n)
            solo=np.mean((y[:,None]*A/(np.sum(A*A,axis=1)[:,None]+REG)-theta[None,:])**2)
            # All methods share exactly same graph, sensors, noise and initialization
            for algo in ['dgd','gradient_tracking']:
              for t,mse,oracle,subopt,consensus,gap in run(w,A,y,theta,algo,eta=(.22 if algo=='dgd' else .10)):
                rows.append(dict(n=n,family=f,rep=rep,graph_seed=seed,placement=p,noise=noise,
                                 protocol=algo,rounds=t,mse=mse,oracle_mse=oracle,
                                 excess_state_error=subopt,consensus_error=consensus,
                                 objective_gap=gap,solo_mse=solo,rank=rank,edge_count=g.number_of_edges(),
                                 transmitted_scalars=2*g.number_of_edges()*D*t*(2 if algo=='gradient_tracking' else 1)))
        print('finished',n,f,rep,flush=True)
    df=pd.DataFrame(rows);df.to_csv(OUT/'collaborative_private_constraints_runs.csv',index=False)
    return df
if __name__=='__main__':main()
