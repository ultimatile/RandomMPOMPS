from tensornetwork.stopping import Cutoff
from tensornetwork.linalg import truncated_svd
from tensornetwork.MPS import MPS 
import scipy 
from tqdm import tqdm 

import numpy as np 

def dmrg2(mpo,mps, sweeps=10, stop=Cutoff(1e-14), maxit=10,eig_tol=1e-8,eigensolver="Lan"):
    
    """
    Einsum Indexing convention L->R:
     ______                 ______
    |      |    _______    |      |   
    |      |-Z-|mps_c_j|-W-|      |
    |      |   |_______|   |      |
    |      |    __|d___    |      |
    | L[j] |-D-| mpo_j |-E-|R[N-j]|
    |      |   |_______|   |      |
    |      |    __|l___    |      |
    |      |-X-| mps_j |-Y-|      |
    |______|   |_______|   |______|
    """
        
    def compute_right_envs(mps, mpo):
        N = len(mps)  # Assuming mps is a list or array-like object
        R = [None] * N
        
        # -------- Right environments --------
        R[-1] = np.einsum("Ddl,xd->xDl", mpo[-1], np.conj(mps[-1]))
        R[-1] = np.einsum("xDl,Xl->xDX", R[-1], mps[-1])
        
        for i in range(N-2,1,-1):
            R[i] = np.einsum("ydx,xDX->ydDX", np.conj(mps[i]), R[i+1])            
            R[i] = np.einsum("EdDl,ydDX->yElX", mpo[i], R[i])
            R[i] = np.einsum("YlX,yElX->yEY", mps[i], R[i])
        
        return R
        
    mps.canonize_left()
    LR = compute_right_envs(mps, mpo)
    Es = []
    
    # H_eff = None
    approx_solution = mps.copy()
    
    def eigensolve(H_eff,psi,*args):
        # if eigensolver == "srr":
        #     psi =psi.reshape(-1)
        #     num_krylov = H_eff.shape[0]
        #     tol = 1e-2 #literally all of them
        #     partial = H_eff.shape[0]
        #     stabilize = 150
        #     V_srr, D_srr, X_srr = sRR(H_eff, num_krylov, tol, partial,b=psi, stabilize=stabilize)
            
        #     return D_srr[-1],V_srr[-1]
        if eigensolver == "Lan":
            energy, psivec = scipy.sparse.linalg.eigsh(H_eff,k=1, which='SA', maxiter=maxit, v0=psi.reshape(-1),tol=eig_tol)
            return energy, psivec

        elif eigensolver == "Dense":
            evals, evecs = eigh(H_eff)
            energy = evals[0]
            psivec = evecs[:,0]
            return energy, psivec
    
    def sweep_scheduler(n, sweeps):
        sequence = np.concatenate([np.arange(n-1), np.arange(n-3, 0, -1)])
        repeated_sequence = np.tile(sequence, sweeps)
        repeated_sequence = np.append(repeated_sequence, 0)
        return list(repeated_sequence)
    
    for k in tqdm(sweep_scheduler(mps.N,sweeps)): #Using 1 indexing
        #================ First Site ================ 
        if k == 0:  
            H_eff = np.einsum("DkEj,yEY->ykDjY",mpo[1],LR[2])
            H_eff = np.einsum("dDl,ykDjY->dkyljY",mpo[0],H_eff)
            H_eff = H_eff.reshape(H_eff.shape[0]*H_eff.shape[1]*H_eff.shape[2], -1)# (ykd,ljy)
            
            psi =  np.einsum("lr,rjY->ljY",approx_solution[0],approx_solution[1]) # (l,j,Y)
            
            energy, psivec = eigensolve(H_eff,psi,eigensolver) 
            Es.append(energy)      

            U, S, Vt = truncated_svd(psivec.reshape(psi.shape[0],psi.shape[1]*psi.shape[2]), stop=stop)
            S = np.diag(S)/np.linalg.norm(np.diag(S))
            approx_solution[0] = U # l x r
            approx_solution[1] = (S @ Vt).reshape(Vt.shape[0],psi.shape[1],psi.shape[2]) #(r,j,Y)
            # approx_solution.show()

            LR[0] = np.einsum("dx,dDl->xDl", np.conj(U), mpo[0])
            LR[0] = np.einsum("xDl,lX->xDX", LR[0], U)
            
            left_sweep = True

        #================ Middle Sites ================ 
        elif 0 < k and k < mps.N-2:
            H_eff = np.einsum("xDX,DdEl->xdElX",LR[k-1],mpo[k])
            H_eff = np.einsum("xdElX,EkFj->xdkFjlX",H_eff,mpo[k+1])  
            H_eff  = np.einsum("xdkFjlX,yFY->xdkyXljY",H_eff,LR[k+2])
            H_eff = H_eff.reshape(H_eff.shape[0]*H_eff.shape[1]*H_eff.shape[2]*H_eff.shape[3],-1)
            
            psi =  np.einsum("Xlr,rjY->XljY",approx_solution[k],approx_solution[k+1])
            energy, psivec = eigensolve(H_eff,psi,eigensolver) 
            Es.append(energy)        

            U, S, Vt = truncated_svd(psivec.reshape(psi.shape[0] * psi.shape[1], psi.shape[2] * psi.shape[3]), stop=stop)
            S = np.diag(S)/np.linalg.norm(np.diag(S))
            
            if left_sweep:
                approx_solution[k] = U.reshape(psi.shape[0], mpo[k].shape[1], U.shape[1])
                approx_solution[k+1] = (S @ Vt).reshape(Vt.shape[0],psi.shape[2],psi.shape[3])
                
                LR[k] = np.einsum("xDX,xdz->zdDX", LR[k - 1], np.conj(approx_solution[k]))
                LR[k] = np.einsum("zdDX,DdEl->zElX", LR[k], mpo[k])
                LR[k] = np.einsum("zElX,XlZ->zEZ", LR[k], approx_solution[k])
            else:
                approx_solution[k] = (U @ S).reshape(psi.shape[0], mpo[k].shape[1], U.shape[1])
                approx_solution[k+1] = Vt.reshape(Vt.shape[0],psi.shape[2],psi.shape[3])
                
                LR[k+1] = np.einsum("yEY,zdy->zdEY", LR[k+2], np.conj(approx_solution[k+1]))
                LR[k+1] = np.einsum("zdEY,DdEl->zDlY", LR[k+1], mpo[k+1])
                LR[k+1] = np.einsum("zDlY,ZlY->zDZ", LR[k+1], approx_solution[k+1])
            # approx_solution.show()


        #================ Last Site ================  Index =  N -1  and N  
        else: # k == mps.N-2
            H_eff = np.einsum("xDX,DdEl->xdElX",LR[-3],mpo[-2])
            H_eff = np.einsum("xdElX,Ekj->xdkXlj",H_eff,mpo[-1])
            H_eff= H_eff.reshape(H_eff.shape[0]*H_eff.shape[1]*H_eff.shape[2],-1)
            
            psi = np.einsum("Xlr,rj->Xlj",approx_solution[-2],approx_solution[-1])
            energy, psivec = eigensolve(H_eff,psi,eigensolver)  
            Es.append(energy)        

            U, S, Vt = truncated_svd(psivec.reshape(psi.shape[0] * psi.shape[1], psi.shape[2]), stop=stop)
            S = np.diag(S)/np.linalg.norm(np.diag(S))
            approx_solution[-2] = (U @ S).reshape(psi.shape[0], approx_solution[-2].shape[1], -1)
            approx_solution[-1] = Vt.reshape(-1, approx_solution[-1].shape[1])
            # approx_solution.show()

            LR[-1] = np.einsum("Ddl,yd->yDl", mpo[-1], np.conj(approx_solution[-1]))
            LR[-1] = np.einsum("yDl,Yl->yDY", LR[-1], approx_solution[-1])
            
            left_sweep = False

    approx_solution[0] = U @ S
    approx_solution[1] = Vt.reshape(Vt.shape[0],psi.shape[1],psi.shape[2])
    mps = MPS(approx_solution)
    mps.canform="Left"
            
    return mps, Es

