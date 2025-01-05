import numpy as np
import numpy.linalg as la 
import scipy
from tensornetwork.MPO import MPO
from tensornetwork.MPS import MPS
import time 
from itertools import product
from tensornetwork.contraction import *
# from tensornetwork.util import *
from tensornetwork.stopping import *
from quantum.hamiltonians import *
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import time
seed_value = 42

np.random.seed(seed_value)

#pauli matrices
X = np.array([[0,1.],[1,0]])
Y = np.array([[0,-1.j],[1j,0]])
Z = np.array([[1, 0], [0, -1]])
I = np.eye(2, dtype=np.complex64)


def random_tensor(*args, a=-.5, b=1):
    output = a + (b-a)*np.random.rand(*args)
    output /= np.linalg.norm(output)
    return output


#======================Initialization Functions=======================
def initialize_system(N, m, cutoff):
    localops = isotropic_heisenberg(N, 1, 2, 1)
    psi = rand_MPS(2, m, N)
    psi.canonize_right()
    psi[-1] /= np.linalg.norm(psi[-1], 'fro')
    return localops, psi

#==================== Trotterized Time Evolution Preparation========================
def prepare_trotterized_operators(localops, total_time, num_steps):
    delta = total_time / num_steps
    ell = num_steps - 1
    U, H1, H2_half = trotterize(localops, delta)
    return U, H1, H2_half, ell

def compute_expectation(psi, operator, site_index):
    #TODO: FIx for non boundary cores
    """
    Compute the expectation value of an operator at a specific site of an MPS.
    :param psi: MPS representing the quantum state.
    :param operator: The operator for which the expectation value is computed.
    :param site_index: Index of the site where the operator is applied.
    :return: Expectation value at the specified site.
    """
    # Access the tensor at the specified site
    modified_tensor = np.tensordot(operator,psi[site_index], axes=[1,1])
    #inner product over matrices 
    expectation_value = np.tensordot(psi[site_index].conj(), modified_tensor, axes=[(0,1),(1,0)])
    return expectation_value

#======================Performance Measurement=======================
def perform_time_evolution(psi, U, H1, H2_half, ell, contraction_types, baseline_states, stop):
    times = []
    norms = []
    for name in contraction_types:
        newpsi = psi.copy()
        print(f"Preforming Time evolution: Method: [{name}] ...")
        start_time = time.time()
        newpsi = time_evo_step(newpsi, U, H1, H2_half, ell, contraction_type=name, stop=stop)
        execution_time = time.time() - start_time
        newpsi.canonize_right()
        newpsi[-1] /= np.linalg.norm(newpsi[-1], 'fro')
        norm = (newpsi - baseline_states).norm() / baseline_states.norm()
        times.append(execution_time)
        norms.append(norm)
    return norms, times

def _time_evolution_helper(local_ops, dt, parity, cutoff=1e-14):
    '''
    Each local operation is a d * d * d * d tensor, interpreted
    as a Hamiltonian term acting on a pair of sites. We use
    the following index convention

    | 0    | 2
    _________
    |       |
    _________
    | 1    | 3
    '''

    d = local_ops[0].shape[0]
    n = len(local_ops)+1
    mpo = []

    if parity == 1:
        site = np.eye(d,d)
        site = np.reshape(site, (d,1,d))
        mpo.append(site)
    
    if cutoff is None:
        cutoff = 0
        
    for i in range(parity, len(local_ops), 2):
        assert(local_ops[i].shape == (d,d,d,d))
        op = np.transpose(local_ops[i], (0,2,1,3)) # op(0,2,1,3)
        op = np.reshape(op, (d*d, d*d)) # op(0 2,1 3)
        op = scipy.linalg.expm(-dt * 1j * op)
        op = np.reshape(op, (d,d,d,d)) # op(0,2,1,3)
        op = np.transpose(op, (0,2,1,3)) # op(0,1,2,3)
        op = np.reshape(op, (d*d,d*d)) # op(0 1,2 3)
        U, s, Vh = np.linalg.svd(op)
        idx = np.nonzero(s > (max(s) * cutoff))[0]
        left = U[:,idx] * s[idx]
        right = Vh[idx, :]
        k = len(idx)
        
        if i == 0:
            left = np.reshape(left, (d,d,k)) # left(0,1,bond)
            left = np.transpose(left, (0,2,1)) # left(0,bond,1)
        else:
            left = np.reshape(left, (1,d,d,k)) # left(lbond,0,1,bond)
            left = np.transpose(left, (0,1,3,2)) # left(lbond,0,bond,1)
        mpo.append(left)
        
        if i == n-2:
            right = np.reshape(right, (k,d,d)) # right(bond,2,3)
        else:
            right = np.reshape(right, (k,d,1,d)) # right(bond,2,rbond,3)
        mpo.append(right)

    if (parity == 1 and n % 2 == 0) or (parity == 0 and n % 2 == 1):
        site = np.eye(d,d)
        site = np.reshape(site, (1,d,d))
        mpo.append(site)

    return MPO(mpo)


def trotterize(local_ops, dt, cutoff=1e-14):
    H1 = _time_evolution_helper(local_ops, dt, 0, cutoff=cutoff)
    H2 = _time_evolution_helper(local_ops, dt, 1, cutoff=cutoff)
    H2_half = _time_evolution_helper(local_ops, dt, 1, cutoff=cutoff)
    return MPO(H1 * H2), H1, H2_half


def time_evo_step(psi, U, H1, H2_half,ell,finalround=None, stop= Cutoff(1e-14),contraction_type="classical",**kwargs):
    if contraction_type == "boundary_trick":
        psi.canonize_left()
        psi.canonize_right()
        #print(f"Time step starting MPS bond dimension:{psi.max_bond_dim()}\n")
        psi = mps_mpo_blas(psi,H2_half,stop=stop,round_type="dass_blas")    #e^idt/2H2
        psi = mps_mpo_blas(psi,H1,stop=stop,round_type="dass_blas")         #e^idtH1
        for l in ((range(ell))):
            psi = mps_mpo_blas(psi,U,stop=stop,round_type="dass_blas") 
            #(U(dt)^l-1) 
        #print("Performing last time evolution step e^idt/2H2 \n")
        psi= mps_mpo_blas(psi,H2_half,stop=stop,round_type="dass_blas")     #e^idt/2H2 
        
    if contraction_type == "classical":
        #print(f"Time step starting MPS bond dimension:{psi.max_bond_dim()}\n")
        psi = mps_mpo_blas(psi,H2_half,stop=stop,round_type="dass_blas")    #e^idt/2H2
        psi = mps_mpo_blas(psi,H1,stop=stop,round_type="dass_blas")         #e^idtH1
        for l in ((range(ell))):
            # print("exponentiation step:",l+1)
            psi = mps_mpo_blas(psi,U,stop=stop,round_type="dass_blas")      #(U(dt)^l-1) 
        #print("Performing last time evolution step e^idt/2H2 \n")
        psi = mps_mpo_blas(psi,H2_half,stop=stop,round_type="dass_blas")     #e^idt/2H2

    elif contraction_type == "random":
        psi = rand_apply(H2_half, psi,stop=stop, **kwargs)    #e^idt/2H2
        psi = rand_apply(H1, psi, stop=stop, **kwargs)        #e^idtH1
        for l in tqdm(range(ell)):
            psi = rand_apply(U, psi, stop=stop, **kwargs)     #(U(dt)^l-1)
        #print("Performing last time evolution step e^idt/2H2 \n")
        psi = rand_apply(H2_half, psi,stop=stop, **kwargs)    #e^idt/2H2

    elif contraction_type == "Blas2.0":
        psi = rand_apply_blas2(H2_half, psi,stop=stop, **kwargs)    #e^idt/2H2
        psi = rand_apply_blas2(H1, psi, stop=stop, **kwargs)        #e^idtH1
        for l in tqdm(range(ell)):
            psi = rand_apply_blas2(U, psi, stop=stop)     #(U(dt)^l-1)
        #print("Performing last time evolution step e^idt/2H2 \n")
        psi = rand_apply_blas2(H2_half, psi,stop=stop, **kwargs)    #e^idt/2H2
        
    elif contraction_type == "Blas2.0_Heuristic":
        
        sketchdim = psi.max_bond_dim()
        sketchincrement=int(np.floor(0.5*psi.max_bond_dim()))
        psi = rand_apply_blas2(H2_half,  psi, stop=stop,sketchdim=sketchdim,finalround=finalround,sketchincrement=sketchincrement)    #e^idt/2H2
    
        sketchdim = psi.max_bond_dim()
        sketchincrement=int(np.floor(0.5*psi.max_bond_dim()))
        psi = rand_apply_blas2(H1, psi, stop=stop,sketchdim=sketchdim,finalround=finalround,sketchincrement=sketchincrement)       #e^idtH1
        
        for l in (range(ell)):
            # print("exponentiation step:",l+1)
            sketchdim = psi.max_bond_dim()
            sketchincrement=int(np.floor(0.5*maxlinkdim(psi)))
            psi = rand_apply_blas2(U, psi, stop=stop,sketchdim=sketchdim,finalround=finalround,sketchincrement=sketchincrement)     #(U(dt)^l-1)
        
        # print("Performing last time evolution step e^idt/2H2 \n")
        
        sketchdim = maxlinkdim(psi)
        sketchincrement=int(np.floor(0.5*maxlinkdim(psi)))
        psi = rand_apply_blas2(H2_half,  psi, stop=stop,sketchdim=sketchdim,finalround=None,sketchincrement=sketchincrement)   #e^idt/2H2  
           
    return psi

def generate_swap_gate(d):
    swap = np.zeros((d,d,d,d))
    for i,j,k,l in product(range(d),range(d),range(d),range(d)):
        if i == l and j == k:
            swap[i,j,k,l] = 1
    return swap

def apply_gates(mpo, gates):
    N = mpo.N
    assert N > 2
    mpo.canonize_left()
    mpo_out = mpo.copy()
    i = 0
    left_sweep = True
    
    for gate in gates:        
        # ============ Left Sweep ============
        if left_sweep:
            if i == 0:
                site = np.einsum("ijdk,dDl->lijkD",gate,mpo_out[i]) #(l,i,j,k,d)
                site = np.einsum("lijkD,DkEp->lijEp",site,mpo_out[i+1]) #(l,i,d,E,p)

                U,S,Vt = np.linalg.svd(site.reshape(site.shape[0]*site.shape[1],-1),full_matrices=False) #SVD((li,jEp))
                
                U = U.reshape(site.shape[0],site.shape[1],U.shape[-1]) #(li,r)->(l,i,r)
                mpo_out[i] = U.transpose(1,2,0) #(l,i,r)-> (i,r,l)
                
                SV = np.diag(S) @ Vt #(r,jEp)
                mpo_out[i+1] = SV.reshape(S.shape[-1],site.shape[2],site.shape[3],site.shape[4]) #(rj,Ep)->(r,j,E,p)
                    
            elif i == N-2:
                site = np.einsum("ijdk,Dkp->dijkpD",gate,mpo_out[i+1])
                site = np.einsum("dijkpD,rdDl->riljp",site,mpo_out[i])
                
                U,S,Vt= np.linalg.svd(site.reshape(site.shape[0]*site.shape[1]*site.shape[2],-1),full_matrices=False) #SVD(ril,jp)
                
                # Move orthogonalization pivot in Consider optional pivot selection in the case when
                # specifially n-1 gates are given and no right sweep ocurs so that the orthogonalization  center ends at site N 
                
                US = U @ np.diag(S)
                US = US.reshape(site.shape[0],site.shape[1],gate.shape[1],U.shape[-1]) # (r,i,l,x)
                mpo_out[i]=US.transpose(0,1,3,2) #(r,i,x,l)
                
                Vt = Vt.reshape(S.shape[-1],gate.shape[2],mpo_out[i+1].shape[2]) #(x,jp)->(x,j,p)
                mpo_out[i+1] = Vt  
                left_sweep = False
                continue 
                
            else:
                site = np.einsum("ijdk,rdDl->lrijkD", gate,mpo_out[i]) 
                site = np.einsum("lrijkD,DkEp->lrijEp",site,mpo_out[i+1])
                
                U,S,Vt= np.linalg.svd(site.reshape(site.shape[0]*site.shape[1]*site.shape[2],-1),full_matrices=False) #SVD(lri,jEp)
                
                U = U.reshape(site.shape[0],site.shape[1],site.shape[2],U.shape[-1]) # (l,r,i,x)
                mpo_out[i]=U.transpose(1,2,3,0) #(r,i,x,l)
                
                SV = np.diag(S) @ Vt #(x,jEp)
                mpo_out[i+1] = SV.reshape(S.shape[-1],site.shape[3],site.shape[4],site.shape[5]) #(r,jFp)->(r,j,F,p)
            
            i += 1
            
        # ============ Right Sweep ============
        else:
            if i == 1:
                site = np.einsum("ijdk,Dkxp->dijxpD", gate,mpo_out[i]) 
                    
                site = np.einsum("dijxpD,dDl->lijxp",site,mpo_out[i-1])
                
                U,S,Vt= np.linalg.svd(site.reshape(site.shape[0]*site.shape[1],-1),full_matrices=False) #SVD(li,jxp)
                mpo_out[i] = (np.diag(S) @ Vt).reshape(Vt.shape[0],gate.shape[2],site.shape[3],site.shape[4]) # (r,j,x,p)
        
                U = U.reshape(site.shape[0],site.shape[1],U.shape[-1])# (l,i,r)
                mpo_out[0] = U.transpose(1,2,0) # (r,i,l) 
                left_sweep = True
                continue
            else:
                site = np.einsum("ijdk,rdDl->lrijkD", gate,mpo_out[i-1]) 
                site = np.einsum("lrijkD,DkEp->lrijEp",site,mpo_out[i])
                
                U,S,Vt= np.linalg.svd(site.reshape(site.shape[0]*site.shape[1]*site.shape[2],-1),full_matrices=False) #SVD(lri,jEp)
                
                US = (U @ np.diag(S)).reshape(site.shape[0],site.shape[1],site.shape[2],U.shape[-1]) # (l,r,i,x)
                mpo_out[i-1] = US.transpose(1,2,3,0) #(r,i,x,l)
                
                mpo_out[i] = Vt.reshape(S.shape[-1],site.shape[3],site.shape[4],site.shape[5]) #(r,jFp)->(r,j,F,p)    
            i-=1
        
    return mpo_out
        
        
#======================Plotting=======================
def plot_execution_times(labels, times):
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, times, color=['blue', 'green', 'red', 'purple'])

    for bar in bars:
        plt.hlines(bar.get_height(), bar.get_x() + bar.get_width(), plt.gca().get_xlim()[1], colors='grey', linestyles='-', lw=2, alpha=0.5)

    plt.ylabel('Execution Time (s)')
    plt.title('Comparison of Execution Times for Different Algorithms')
    plt.show()

