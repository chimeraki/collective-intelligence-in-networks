from pathlib import Path
"""Independent small-system check of finite-time DGD risk formula.
Uses the paper's update, independent noise, and isotropic fixed-norm states.
"""
import numpy as np
rng=np.random.default_rng(21019)
N,d,T=5,3,8
W=np.zeros((N,N))
for i in range(N):
 W[i,i]=.5;W[i,(i-1)%N]+=.25;W[i,(i+1)%N]+=.25
A=rng.normal(size=(N,d)); A/=np.linalg.norm(A,axis=1,keepdims=True)
eta=.12;lam=.025
M=np.kron(W,np.eye(d))-eta*np.kron(np.eye(N),lam*np.eye(d))
for i in range(N):M[i*d:(i+1)*d,i*d:(i+1)*d]-=eta*np.outer(A[i],A[i])
C=np.vstack([np.outer(a,a) for a in A]);E=np.zeros((N*d,N))
for i in range(N):E[i*d:(i+1)*d,i]=A[i]
J=np.tile(np.eye(d),(N,1));sigma=np.array([.08,.1,.05,.12,.07]);Sigma=np.diag(sigma**2)
K=np.zeros_like(M);P=np.eye(N*d)
for _ in range(T):K+=eta*P;P=P@M
analytic=(np.linalg.norm(K@C-J,'fro')**2+np.linalg.norm(K@E@np.diag(sigma),'fro')**2)/(N*d)
B=60000
u=rng.normal(size=(B,d));theta=np.sqrt(d)*u/np.linalg.norm(u,axis=1,keepdims=True)
eps=rng.normal(size=(B,N))*sigma
Z=np.zeros((B,N*d))
F=theta@C.T+eps@E.T
for _ in range(T):Z=Z@M.T+eta*F
emp=np.mean(np.sum((Z-theta@J.T)**2,axis=1)/(N*d))
print('exact risk',analytic,'Monte Carlo risk',emp,'abs error',abs(emp-analytic))
assert abs(emp-analytic)<0.015
# check explicit block recursion equals propagation, one realization
u=theta[0];y=A@u+eps[0];zi=np.zeros((N,d))
for _ in range(T):
 zi=W@zi-eta*(A*(np.sum(A*zi,axis=1)-y)[:,None]+lam*zi)
assert np.max(np.abs(zi.ravel()-Z[0]))<1.e-12
print('recurrence match max abs error',np.max(np.abs(zi.ravel()-Z[0])))
