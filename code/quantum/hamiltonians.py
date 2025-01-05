import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import numpy as np
import numpy.linalg as la 
import scipy
from tensornetwork.MPO import MPO
from tensornetwork.MPS import MPS
import time 
from tensornetwork.contraction import *
from tensornetwork.stopping import *
from tqdm import tqdm
import matplotlib.pyplot as plt
import itertools

seed_value = 42

np.random.seed(seed_value)

I2 = np.eye(2)
#pauli matrices
X = np.array([[0,1.],[1,0]])
Y = np.array([[0,-1.j],[1j,0]])
Z = np.array([[1, 0], [0, -1]])

# spin-1/2 operators (bottom-top)
## σ^z_l
sz = np.zeros((2,2))
sz[0,0] = 1/2
sz[1,1] = -1/2
## σ^+_l
sp = np.zeros((2,2))
sp[0,1] = 1
## σ^-_l
sm = np.zeros((2,2))
sm[1,0] = 1

# spin-1 operators (bottom-top)
## S^z_l
Sz = np.zeros((3,3))
Sz[0,0] = 1
Sz[2,2] = -1
## S^+_l
Sp = np.zeros((3,3))
Sp[0,1] = np.sqrt(2)
Sp[1,2] = np.sqrt(2)
## S^-_l
Sm = np.zeros((3,3))
Sm[1,0] = np.sqrt(2)
Sm[2,1] = np.sqrt(2)
## I_l
I3 = np.eye(3)

def mpo_from_fsm(graph, k, n, source = -1, target = 0):
    assert(len(graph) > 0)
    d = graph[list(graph.keys())[0]].shape[0]
    
    A = np.zeros((k,d,k,d), dtype=complex)
    for j, i in graph.keys():
        A[i,:,j,:] = graph[(j,i)]
        
    
    return [A[target,:,:,:]] + (n-2)*[np.copy(A)] + [A[:,:,source,:]] 


#$H_{\mathrm{TC}}=-J \sum_v A_v-J \sum_p B_p, \quad J>0$.
def toric_code(N):
    pass    

#======================== ALKT model ======================================
#(https://doi.org/10.1103/PhysRevLett.59.799)
def ALKT(N,local_ops=False):    
    # MPO Hamiltonian (left-bottom-right-top)
    ## H[l]
    Hl = np.zeros((14,3,14,3))
    Hl[0,:,0,:] = I3
    Hl[1,:,0,:] = Sz
    Hl[2,:,0,:] = Sm
    Hl[3,:,0,:] = Sp
    Hl[4,:,0,:] = np.matmul(Sz,Sz)
    Hl[5,:,0,:] = np.matmul(Sz,Sm)
    Hl[6,:,0,:] = np.matmul(Sz,Sp)
    Hl[7,:,0,:] = np.matmul(Sm,Sz)
    Hl[8,:,0,:] = np.matmul(Sm,Sm)
    Hl[9,:,0,:] = np.matmul(Sm,Sp)
    Hl[10,:,0,:] = np.matmul(Sp,Sz)
    Hl[11,:,0,:] = np.matmul(Sp,Sm)
    Hl[12,:,0,:] = np.matmul(Sp,Sp)
    Hl[13,:,1,:] = Sz
    Hl[13,:,2,:] = 0.5*Sp
    Hl[13,:,3,:] = 0.5*Sm
    Hl[13,:,4,:] = 1/3*np.matmul(Sz,Sz)
    Hl[13,:,5,:] = 1/6*np.matmul(Sz,Sp)
    Hl[13,:,6,:] = 1/6*np.matmul(Sz,Sm)
    Hl[13,:,7,:] = 1/6*np.matmul(Sp,Sz)
    Hl[13,:,8,:] = 1/12*np.matmul(Sp,Sp)
    Hl[13,:,9,:] = 1/12*np.matmul(Sp,Sm)
    Hl[13,:,10,:] = 1/6*np.matmul(Sm,Sz)
    Hl[13,:,11,:] = 1/12*np.matmul(Sm,Sp)
    Hl[13,:,12,:] = 1/12*np.matmul(Sm,Sm)
    Hl[13,:,13,:] = I3
    ## H
    H = [Hl for l in range(N)]
    H[0] = Hl[-1:np.shape(Hl)[0],:,:,:]
    H[0]=H[0].reshape(H[0].shape[1],H[0].shape[2],H[0].shape[3])

    H[N-1] = Hl[:,:,0:1,:]
    H[N-1]= H[N-1].reshape(H[N-1].shape[0],H[N-1].shape[1],H[N-1].shape[3])
    return MPO(H)

#======================== Majumdar-Ghosh model ======================================
# (https://doi.org/10.1063/1.1664979) 

def Madjumdar_Gosh(N,local_ops=False):
    Hl = np.zeros((8,2,8,2))
    Hl[0,:,0,:] = I2
    Hl[1,:,0,:] = sz
    Hl[2,:,0,:] = sm
    Hl[3,:,0,:] = sp
    Hl[4,:,1,:] = I2
    Hl[5,:,2,:] = I2
    Hl[6,:,3,:] = I2
    Hl[7,:,1,:] = sz
    Hl[7,:,2,:] = 0.5*sp
    Hl[7,:,3,:] = 0.5*sm
    Hl[7,:,4,:] = 0.5*sz
    Hl[7,:,5,:] = 1/4*sp
    Hl[7,:,6,:] = 1/4*sm
    Hl[7,:,7,:] = I2
    ## H
    H = [Hl for l in range(N)]
    H[0] = Hl[-1:np.shape(Hl)[0],:,:,:]
    H[0]=H[0].reshape(H[0].shape[1],H[0].shape[2],H[0].shape[3])

    H[N-1] = Hl[:,:,0:1,:]
    H[N-1]= H[N-1].reshape(H[N-1].shape[0],H[N-1].shape[1],H[N-1].shape[3])
    
    if local_ops:
        return H 
    
    return MPO(H)


#======================== Spin-1 Heisenberg model with Zeeman term ======================================

def Heis_zeeman(N,J=1,h=1,local_ops=False):
    Hl = np.zeros((5,3,5,3))
    Hl[0,:,0,:] = I3
    Hl[1,:,0,:] = Sz
    Hl[2,:,0,:] = Sm
    Hl[3,:,0,:] = Sp
    Hl[4,:,0,:] = -h*Sz
    Hl[4,:,1,:] = J*Sz
    Hl[4,:,2,:] = J/2*Sp
    Hl[4,:,3,:] = J/2*Sm
    Hl[4,:,4,:] = I3
    ## H
    H = [Hl for l in range(N)]
    H[0] = Hl[-1:np.shape(Hl)[0],:,:,:]
    H[0]=H[0].reshape(H[0].shape[1],H[0].shape[2],H[0].shape[3])

    H[N-1] = Hl[:,:,0:1,:]
    H[N-1]= H[N-1].reshape(H[N-1].shape[0],H[N-1].shape[1],H[N-1].shape[3])

    if local_ops:
        return H
    
    return MPO(H)


#======================== XY model ======================================

# def XY_model(N,Jx=1,Jy=1,local_ops=False):
#     Hl = np.zeros((4,2,4,2))
#     Hl[0,:,0,:] = I2
#     Hl[1,:,0,:] = sm
#     Hl[2,:,0,:] = sp
#     Hl[3,:,1,:] = -0.5*sp
#     Hl[3,:,2,:] = -0.5*sm
#     Hl[3,:,3,:] = I2
    
#     H = [Hl for l in range(N)]
#     H[0] = Hl[-1:np.shape(Hl)[0],:,:,:]
#     H[0]=H[0].reshape(H[0].shape[1],H[0].shape[2],H[0].shape[3])
#     H[N-1] = Hl[:,:,0:1,:]
#     H[N-1]= H[N-1].reshape(H[N-1].shape[0],H[N-1].shape[1],H[N-1].shape[3])
    
#     if local_ops:
#         local_ops = []
#         local_H_template = np.zeros((2, 2, 2, 2))
#         for _ in range(N-1):
#             H_local = local_H_template.copy()
#             H_local += np.reshape(-Jx*np.kron(sm, sm.T) - Jy*np.kron(sp, sp.T), (2, 2, 2, 2))
#             local_ops.append(H_local)
            
#         return local_ops
#     return MPO(H)

#======================== Isotropic Heisenberg/ X X X ======================================

def isotropic_heisenberg(N,Jx,Jy,Jz,local_ops=False):
    H =  [np.reshape(-Jx*np.kron(X, X) - Jy*np.kron(Y, Y) - Jz*np.kron(Z, Z), (2, 2, 2, 2)) for _ in range(N-1)] 
    if local_ops:
        return H
    return MPO(H)

#======================== Transverse Field Ising Model ======================================

def transverse_ising(N, J=1, h=1, local_ops=False):
    fsm = { (0,0): I2,
            (0,1): -J*X,
            (1,2): X,
            (0,2): -h*Z,
            (2,2): I2
          }
    H =mpo_from_fsm(fsm,3,N,source=0,target=2)
    if local_ops:
        return H 
    return MPO(H)

#======================== Heisenberg Hamiltonian with next and next next interactions ======================================
#(FSM representation)
def heisenberg(N,variant="heis",Jx=1,Jy=1,Jz=1,alpha=1,beta=1.1,gamma=1.2,local_ops=False):
    X = np.array([[0,1],[1,0]])
    Y = np.array([[0,-1j],[1j,0]])
    Z =np.array([[1,0],[0,-1]])
    
    if variant=="heis":
        graph_heis = {
        (0, 0): np.eye(2),
        (0, 1): Jx * X,
        (0, 2): Jy * Y,
        (0, 3): Jz * Z,
        (1, 4): Jx * X,
        (2, 4): Jy * Y,
        (3, 4): Jz * Z,
        (4, 4): np.eye(2),
    }
        graph_max = 4
        H = mpo_from_fsm(graph_heis, graph_max+1, N, source=0, target=graph_max)
        if local_ops:
            return H
        return MPO(H)
    
    elif variant =="heis_next":
        graph_neighbors={
        (0,0) : np.eye(2),
        (0,1) : X ,
        (0,2) : Y,
        (0,3) : Z,
        (1,7) : alpha * X,
        (2,7) : alpha * Y,
        (3,7) : alpha * Z,
        (1,4) : beta * np.eye(2),
        (2,5) : beta * np.eye(2),
        (3,6) : beta * np.eye(2),
        (4,7) : X ,
        (5,7) : Y,
        (6,7) : Z,
        (7,7) : np.eye(2),    
        }
        graph_max = 7
        H = mpo_from_fsm(graph_neighbors, graph_max+1, N, source=0, target=graph_max)
        
        if local_ops:
            return H
        
        return MPO(H)
    
    elif variant== "heis_next_next":
        graph_neighbors_neighbors={
        (0,0) : np.eye(2),
        
        (0,1) : X ,
        (0,2) : Y,
        (0,3) : Z,
        
        (1,10): alpha * X,
        (2,10): alpha * Y,
        (3,10): alpha * Z,
        
        (1,4): np.eye(2),
        (2,5): np.eye(2) ,
        (3,6): np.eye(2) ,
        
        (4,7): gamma * np.eye(2),
        (5,8): gamma * np.eye(2) ,
        (6,9): gamma * np.eye(2) ,
        
        (7,10): X ,
        (8,10): Y ,
        (9,10): Z ,
        
        (4,10): beta * X ,
        (5,10): beta *Y ,
        (6,10): beta *Z ,

        (10,10) : np.eye(2),
        }
        graph_max = 10
        H = mpo_from_fsm(graph_neighbors_neighbors, graph_max+1, N, source=0, target=graph_max)
        if local_ops:
            return H
        
        return MPO(H)

def esprit(N, interaction, stop=Cutoff(1e-8), small_eig_cutoff=1e-10):
    r = stop.outputdim if (not (stop.outputdim is None)) else stop.mindim
    r = min(r, (N-1)//2)
    while True:
        # Find bases
        F = np.zeros((N-r, r))
        for i, j in itertools.product(range(N-r), range(r)):
            F[i,j] = interaction(i+j+1)
        W, _, _, _ = la.lstsq(F[:-1,:], F[1:,:])
        bases, _ = la.eig(W)
        bases = np.real(bases)
        bases = bases[bases > small_eig_cutoff * la.norm(bases, np.inf)]

        # Find coefficients
        F = np.zeros((N-1, len(bases)))
        b = np.zeros(N-1)
        for i in range(N-1):
            for j in range(len(bases)):
                F[i,j] = np.power(bases[j],i)
            b[i] = interaction(i+1)
        coeffs, _, _, _ = la.lstsq(F, b)

        if (not (stop.outputdim is None) and r >= stop.outputdim) or r >= stop.maxdim or r == (N-1)//2:
            return bases, coeffs

        error = la.norm(b - F@coeffs, np.inf) / la.norm(b, np.inf)
        if error <= stop.cutoff:
            return bases, coeffs

        r += 1
    
def long_range_tfim(N, h=1.0, interaction=lambda dist: dist**-2, stop=Cutoff(1e-8)):
    bases, coeffs = esprit(N, interaction, stop=stop)
    fsm = { (0,0): I2,
            **{(0,i+1): coeffs[i]*X for i in range(len(bases))},
            **{(i+1,i+1): bases[i]*I2 for i in range(len(bases))},
            **{(i+1,len(bases)+1): X for i in range(len(bases))},
            (0,len(bases)+1): h*Z,
            (len(bases)+1,len(bases)+1): I2 }
    return MPO(mpo_from_fsm(fsm, len(bases)+2, N, source=0, target=len(bases)+1))

def cluster(N,K=1,h=1,local_ops=False):
    fsm = { 
        (0,0): I2,
        (0,1): K*X,
        (1,2): X,
        (2,3): X,
        (0,3): h*Z,
        (3,3): I2
        }
    H = mpo_from_fsm(fsm,4,N,source=0,target=3)
    if local_ops:
        return H 
    return MPO(H)

def long_range_XY_model(N,J=1,alpha=1,local_ops=False,stop=Cutoff(1e-10)):
    bases, coeffs = esprit(N, lambda dist: dist**-alpha, stop=stop)
    r = len(bases)
    start = 2*r
    stp = 2*r+1
    fsm = { (start,start): I2,
            **{(start,i): J/2*coeffs[i]*X for i in range(r)},
            **{(i,i): bases[i]*I2 for i in range(r)},
            **{(i,stp): X for i in range(r)},
            **{(start,i+r): J/2*coeffs[i]*Y for i in range(r)},
            **{(i+r,i+r): bases[i]*I2 for i in range(r)},
            **{(i+r,stp): Y for i in range(r)},
            (stp,stp): I2 }
    H = mpo_from_fsm(fsm,2*r+2,N,source=start,target=stp)
    if local_ops:
        return H 
    return MPO(H)