
import numpy as np 
from tensornetwork.linalg import truncated_svd,lq
from tensornetwork.stopping import Cutoff
from tensornetwork.MPS import MPS
import time
from tensornetwork.contraction import * 

def expand(states, cutoff=1e-14):
    N = states[0].N
    # if len(states) == 1:
    #     raise RuntimeError ("Only a single state was passed to basis enrichment")
    assert all(state.N == N for state in states), "All expansion states must have the same number of sites as the MPS."
    
    for i in range(len(states)):
            if states[i].canform != "Right":
                    states[i].canonize_right()
            
    mps_out = [None] * states[0].N
    
    #============ SVD site N ============
    boundary = np.vstack([state[-1] for state in states])
    chi0 = states[0][-1].shape[0]
    L,Qt = lq(boundary)
    L11 = L[chi0:,chi0:] 
    U, S, Vt = truncated_svd(L11, stop=Cutoff(cutoff), abstol=cutoff * np.linalg.norm(L))
    B = np.hstack([L[:,:chi0],
                   np.vstack([
                           np.zeros((chi0,U.shape[1])),
                           U @ np.diag(S)
                           ]) ])  
    
    mps_out[-1] = np.vstack((Qt[:chi0,:],Vt @ Qt[chi0:,:]))
    
    for i in reversed(range(1,N-1)):
        d = states[0][i].shape[1]
        left_dimension = sum([state[i].shape[0] for state in states])
        left_idx = 0
        right_idx = 0
        Bnew = np.zeros((left_dimension*d, B.shape[1]),states[0][0].dtype)
        for state in states: #Avoids redudundant zero multiplications 
                reshaped_state = np.reshape(state[i], (state[i].shape[0]*state[i].shape[1], -1))
                Bnew[right_idx:right_idx+reshaped_state.shape[0],:] = reshaped_state @ B[left_idx:left_idx+reshaped_state.shape[1],:]
                left_idx  += reshaped_state.shape[1]
                right_idx += reshaped_state.shape[0]
        
        idx = 0
        blocks = []
        for state in states:
                dim = state[i].shape[0] * state[i].shape[1] 
                block = Bnew[idx:idx+dim,:]
                blocks.append(block.reshape(state[i].shape[0],d,Bnew.shape[-1]))
                idx += dim 
                
        B = np.concatenate(blocks, axis=0)
        B = np.reshape(B, (B.shape[0], B.shape[1]*B.shape[2]))
                
        L, Qt = lq(B)
        chi0 = states[0][i].shape[0]
        L11 = L[chi0:,chi0:]
        U, S, Vt = truncated_svd(L11, stop=Cutoff(cutoff), abstol=cutoff * np.linalg.norm(L))
        B = np.hstack([L[:,:chi0],
                np.vstack([
                        np.zeros((chi0,U.shape[1])),
                        U @ np.diag(S)
                        ]) ])  
        mps_out[i] = np.vstack((Qt[:chi0,:],Vt @ Qt[chi0:,:]))
        mps_out[i] = mps_out[i].reshape(mps_out[i].shape[0],d,-1)
    
    mps_out[0] = states[0][0] @ B[:states[0][0].shape[1],:]
    return MPS(mps_out)

def MPS_krylov(mps, mpo, basis_size=3, stop=Cutoff(1e-14), algorithm_name='random', bond_dim=None, sketch_increment=5, sketch_dim=5, fit_sweeps=1, baseline_basis=None):
    basis = [mps]
    k_vec = mps
    total_time = 0
    
    if algorithm_name == 'naive':
        for t in range(basis_size-1):
            start = time.time()
            k_vec = mps_mpo_blas(k_vec, mpo, stop=stop, round_type="dass_blas")
            total_time += time.time() - start
            k_vec.normalize()
            basis.append(k_vec)
            
    elif algorithm_name == 'rand_then_orth':
        for t in range(basis_size-1):
            start = time.time()
            k_vec = mps_mpo_blas(k_vec, mpo, stop=stop, round_type="rand_then_orth_blas", r=bond_dim)
            total_time += time.time() - start
            k_vec.normalize()
            basis.append(k_vec)
            
    elif algorithm_name == 'Nyst':
        for t in range(basis_size-1):
            start = time.time()
            k_vec = mps_mpo_blas(k_vec, mpo, stop=stop, round_type="nystrom_blas", l=bond_dim)
            total_time += time.time() - start
            k_vec.normalize()
            basis.append(k_vec)
            
    elif algorithm_name == 'random':
        for t in range(basis_size-1):
            start = time.time()
            k_vec = random_contraction_inc(mpo, k_vec, stop=stop, accuracychecks=False, 
                                           finalround=None, sketchincrement=sketch_increment, sketchdim=sketch_dim)
            total_time += time.time() - start
            k_vec.normalize()
            basis.append(k_vec)
            
    elif algorithm_name == 'density':
        for t in range(basis_size-1):
            start = time.time()
            k_vec = density_matrix(mpo, k_vec, stop=stop)
            total_time += time.time() - start
            k_vec.normalize()
            basis.append(k_vec)
            
    elif algorithm_name == 'zipup':
        for t in range(basis_size-1):
            start = time.time()
            k_vec = zipup(mpo, k_vec, stop=stop, finalround=None)
            total_time += time.time() - start
            k_vec.normalize()
            basis.append(k_vec)
            
    elif algorithm_name == 'fit':
        sweeps = fit_sweeps
        #while True:
        # basis = []
        # k_vec = mps
        # basis.append(k_vec)
        # total_time = 0
        for t in range(basis_size-1):
            start = time.time()
            k_vec = fit(mpo, k_vec, max_sweeps=sweeps, stop=stop)
            total_time += time.time() - start
            k_vec.normalize()
            basis.append(k_vec)
                
            # # Adaptive Fitting dropout
            # relative_errors = []
            # for k_vec, baseline_vec in zip(basis, baseline_basis):
            #     error = (baseline_vec - k_vec).norm() / baseline_vec.norm()
            #     relative_errors.append(error)
            # avg_relative_error = sum(relative_errors) / len(relative_errors)
            # print(f"Fit stats: Avg Relative Error: {avg_relative_error:.4f}, Sweeps: {sweeps}")
            # if avg_relative_error <= 5 * avg_relative_error or sweeps == 10:
            #     break
            # else:
            #     sweeps += 1
    
    else:
        print("Invalid algorithm choice for", algorithm_name, "review your inputted algorithm names")
        return
    
    return basis, total_time