
import numpy as np 
from tqdm import tqdm 
from tensornetwork.linalg import lq, lan_exp
from scipy.linalg import expm

def tdvp_implicit(mps, mpo,dt,sweeps,krylov_depth = 10,verbose=False,use_lanczos=True,measure=True):
     
    def compute_right_envs(mps, mpo):
        N = len(mps)  # Assuming mps is a list or array-like object
        R = [None] * N
        
        # -------- Right environments --------
        # R[-1] = np.einsum("Ddl,xd->xDl", mpo[-1], np.conj(mps[-1]))
        mpo_transposed = mpo[-1].transpose(0,2,1) #Transpose to (Dld)
        product  = mpo_transposed.reshape(-1,mpo_transposed.shape[-1]) @ (np.conj(mps[-1])).T #Resulting shape (Dl,x)
        product = product.reshape(mpo[-1].shape[0],mpo[-1].shape[2],mps[-1].shape[0]) #Reshape to (D,l,x)
        R[-1] =  product.transpose(2,0,1)
        
        # R[-1] = np.einsum("xDl,Xl->xDX", R[-1], mps[-1])
        product =  R[-1].reshape(-1,R[-1].shape[-1]) @ mps[-1].T #Resulting shape (xD,X)
        R[-1] = product.reshape(R[-1].shape[0],R[-1].shape[1],mps[-1].shape[0])
        
        for i in range(N-2,0,-1):
            #R[i] = np.einsum("ydx,xDX->ydDX", np.conj(mps[i]), R[i+1])       
            product =  np.conj(mps[i].reshape(-1,mps[i].shape[-1])) @ R[i+1].reshape(R[i+1].shape[0],-1) #Resulting shape (yd,DX)
            R[i] =  product.reshape(mps[i].shape[0],mps[i].shape[1],R[i+1].shape[1],R[i+1].shape[2])
                 
            # R[i] = np.einsum("EdDl,ydDX->yElX", mpo[i], R[i])
            mpo_transposed =  mpo[i].transpose(0,3,1,2) #Transpose to (E,l,d,D)
            R_tranposed = R[i].transpose(1,2,0,3) #Transpose to (d,D,y,X)
            product = mpo_transposed.reshape(-1,mpo_transposed.shape[2]*mpo_transposed.shape[3]) @ R_tranposed.reshape(-1,R_tranposed.shape[2]*R_tranposed.shape[3]) #Resulting shape (El,yX)
            product = product.reshape(mpo[i].shape[0],mpo[i].shape[3],R[i].shape[0],R[i].shape[-1]) #(E,l,y,X)
            R[i] = product.transpose(2,0,1,3) #(y,E,l,X)
            
            R[i] = np.einsum("YlX,yElX->yEY", mps[i], R[i]) #TODO seems like a strange reshaping bug at line 2 for the case of a product state: ie (x ,1,1,x ->x^2,1 ?)
            # R_transposed = R[i].transpose(2,3,0,1) #Transpose to (l,X,y,E)
            # R_reshaped = R_transposed.reshape(-1,R_tranposed.shape[2]*R_tranposed.shape[3])
            # product = mps[i].reshape(mps[i].shape[0],-1) @ R_reshaped #Resulting shape (Y,yE)
            # product = product.reshape(mps[i].shape[0],R[i].shape[0],R[i].shape[1]) #(Y,y,E))
            # R[i]= product.transpose(1,2,0) # (y,E,Y)
        
        return R
    
    def update_A_left(H_eff,approx_solution,sweep_index): #TODO:lingering einsum
        
        if sweep_index == 0:
            approx_solution[0], R = np.linalg.qr(approx_solution[0])
            if H_eff is None:
                K = None
            else:
                # K = np.einsum("dylY,dk->kylY",H_eff,np.conj(approx_solution[0]))
                product = H_eff.T.reshape(-1,H_eff.shape[0]) @ np.conj(approx_solution[0]) #Resulting shape (Yly,k)
                K = product.T.reshape(product.shape[1],H_eff.shape[1],H_eff.shape[2],H_eff.shape[3]) #Reshape to (k,y,l,Y)
                
                # K = np.einsum("kylY,lj->kyjY",K,approx_solution[0]) 
                K_transposed = K.transpose(0,1,3,2) # Reshape to (k,y,Y,l)
                product = K_transposed.reshape(-1,K_transposed.shape[-1]) @ approx_solution[0] #Resulting shape (kyY,j)
                product = product.reshape(K.shape[0],K.shape[1],K.shape[3],approx_solution[0].shape[-1]) #Reshape to (k,y,Y,j)
                K = product.transpose(0,1,3,2) #Transpose to (k,y,j,Y)
        else:
            Q, R = np.linalg.qr(approx_solution[sweep_index].reshape(-1,approx_solution[sweep_index].shape[-1]))
            approx_solution[sweep_index] = Q.reshape(approx_solution[sweep_index].shape[0], approx_solution[sweep_index].shape[1], -1)
            if H_eff is None:
                K = None
            else:
                #K = np.einsum("xdyXlY,xdk->kyXlY",H_eff,np.conj(approx_solution[sweep_index]))
                H_transposed = H_eff.transpose(2,3,4,5,0,1) #Transpose to (y,X,l,Y,x,d) 
                product = H_transposed.reshape(-1,H_transposed.shape[-2]*H_transposed.shape[-1]) @  np.conj(approx_solution[sweep_index]).reshape(-1,approx_solution[sweep_index].shape[-1]) #Resulting shape (yXlY,k)
                product = product.reshape(H_eff.shape[2],H_eff.shape[3],H_eff.shape[4],H_eff.shape[5],product.shape[1],) #Reshape to (y,X,l,Y,k)
                K= product.transpose(4,0,1,2,3) #Tranpose to (k,y,X,l,Y)
                
                #K = np.einsum("kyXlY,Xlj->kyjY",K,approx_solution[sweep_index])
                K_tranposed = K.transpose(0,1,4,2,3) #Transpose to (k,y,Y,X,l)
                product = K_tranposed.reshape(-1,K_tranposed.shape[-2]*K_tranposed.shape[-1]) @ approx_solution[sweep_index].reshape(-1,approx_solution[sweep_index].shape[-1]) #Resulting shape (kyY,j)
                product = product.reshape(K.shape[0],K.shape[1],K.shape[-1],approx_solution[sweep_index].shape[-1]) #(k,y,Y,j)
                K = product.transpose(0,1,3,2)
            
         # Update left environment
        if sweep_index == 0:
            # LR[0] = np.einsum("dx,dDl->xDl", np.conj(approx_solution[0]), mpo[0])
            product = np.conj(approx_solution[0]).T @ mpo[0].reshape(mps[0].shape[0],-1) #resulting shape (x,DL)
            LR[0] = product.reshape(product.shape[0],mpo[0].shape[1],mpo[0].shape[2])
            
            # LR[0] = np.einsum("xDl,lX->xDX", LR[0], approx_solution[0]) 
            product = LR[0].reshape(-1,LR[0].shape[-1]) @ approx_solution[0] #resulting shape (xD,X)
            LR[0] = product.reshape(LR[0].shape[0],LR[0].shape[1],product.shape[-1]) #reshape to (x,D,X)
        else:   
            # LR[sweep_index] = np.einsum("xDX,xdz->zdDX", LR[sweep_index-1], np.conj(approx_solution[sweep_index]))
            LR_tranposed = LR[sweep_index-1].transpose(1,2,0) #transpose to (D,X,x)
            product = LR_tranposed.reshape(-1,LR_tranposed.shape[-1]) @ np.conj(approx_solution[sweep_index]).reshape(approx_solution[sweep_index].shape[0],-1) #resulting shape (DX,dz)
            product = product.reshape(LR[sweep_index-1].shape[-2],LR[sweep_index-1].shape[-1],approx_solution[sweep_index].shape[1],approx_solution[sweep_index].shape[2]) #reshape to (D,X,d,z)
            LR[sweep_index]= product.transpose(3,2,0,1) #Transpose to (z,d,D,X)
            
            # LR[sweep_index] = np.einsum("zdDX,DdEl->zElX", LR[sweep_index], mpo[sweep_index])
            LR_transposed = LR[sweep_index].transpose(0,3,2,1) #Transpose to (z,X,D,d)
            product = LR_transposed.reshape(-1,LR_transposed.shape[-2]*LR_transposed.shape[-1]) @ mpo[sweep_index].reshape(mpo[sweep_index].shape[0]*mpo[sweep_index].shape[1],-1) #resulting shape (zX,El)
            product = product.reshape(LR[sweep_index].shape[0],LR[sweep_index].shape[-1],mpo[sweep_index].shape[2],mpo[sweep_index].shape[3]) #reshape to (z,X,E,l)
            LR[sweep_index] = product.transpose(0,2,3,1) #tranpose to (z,E,l,X)
            
            # LR[sweep_index] = np.einsum("zElX,XlZ->zEZ", LR[sweep_index], approx_solution[sweep_index])
            LR_transposed =  LR[sweep_index].transpose(0,1,3,2) #transpose to (z,E,X,l)
            product = LR_transposed.reshape(-1,LR_transposed.shape[-2]*LR_transposed.shape[-1]) @ approx_solution[sweep_index].reshape(-1,approx_solution[sweep_index].shape[-1]) #Resulting shape (zE,Z)
            LR[sweep_index] = product.reshape(LR[sweep_index].shape[0],LR[sweep_index].shape[1],approx_solution[sweep_index].shape[-1])
        return R, K
    
    def update_A_right(H_eff,approx_solution,sweep_index):   #TODO:lingering einsum
                   
        if sweep_index == approx_solution.N-1:
            L, approx_solution[-1] = lq(approx_solution[-1])
            
            if H_eff is None:
                K = None
                
            else:
                # K = np.einsum("ydYl,kd->ykYl",H_eff,np.conj(approx_solution[-1]))
                H_tranposed = H_eff.transpose(0,2,3,1) #Transpose to (y,Y,l,d)
                product = H_tranposed.reshape(-1,H_tranposed.shape[-1]) @np.conj(approx_solution[-1]).T #Resulting shape (yYl,k)
                product = product.reshape(H_eff.shape[0],H_eff.shape[2],H_eff.shape[3],approx_solution[-1].shape[0]) #Reshape to (y,Y,l,k)
                K = product.transpose(0,3,1,2) # Transpose to (y,k,Y,l)
                
                # K = np.einsum("ykYl,jl->ykYj",K,approx_solution[-1]) 
                product = K.reshape(-1,K.shape[-1]) @ approx_solution[-1].T #Resulting shape (ykY,j)
                K = product.reshape(K.shape[0],K.shape[1],K.shape[2],approx_solution[-1].shape[0]) #reshape to (y,k,Y,j)
            
        else:
            L, Qt = lq(approx_solution[sweep_index].reshape(approx_solution[sweep_index].shape[0],-1))
            approx_solution[sweep_index] = Qt.reshape(approx_solution[sweep_index].shape[0], approx_solution[sweep_index].shape[1], -1)
            
            if H_eff is None:
                K = None
                
            else:
                #The correct einsum is given below the blased version is incorrect will fix 
                K = np.einsum("xdyXlY,kdy->xkXlY",H_eff,np.conj(approx_solution[sweep_index]))
                K = np.einsum("xkXlY,jlY->xkXj",K,approx_solution[sweep_index]) #<<<-fix!!!!
            
                # # K = np.einsum("xdyXlY,klY->xdyXk",H_eff,np.conj(approx_solution[sweep_index]))
                # approx_transposed = approx_solution[sweep_index].transpose(1,2,0)
                # product = H_eff.reshape(-1,H_eff.shape[-2]*H_eff.shape[-1]) @ approx_transposed.reshape(-1,approx_transposed.shape[-1]) #resulting shape (xdyX,k)
                # K = product.reshape(H_eff.shape[0],H_eff.shape[1],H_eff.shape[2],H_eff.shape[3],approx_solution[sweep_index].shape[0])# (x,d,y,X,k)
                
                # # K = np.einsum("xdyXk,jdy->xjXk",K,approx_solution[sweep_index])
                # K_tranposed = K.transpose(0,3,4,1,2) #Transpose to (x,X,k,d,y)
                # approx_transposed = approx_solution[sweep_index].transpose(1,2,0) # Transpose to (d,y,j)
                # product = K_tranposed.reshape(-1,K_tranposed.shape[-2]*K_tranposed.shape[-1])@ approx_transposed.reshape(-1,approx_transposed.shape[-1]) #Resulting shape (xXk,j)
                # product = product.reshape(K.shape[0],K.shape[3],K.shape[4],approx_solution[sweep_index].shape[0]) # Reshape to (x,X,k,j)
                # K = product.transpose(0,3,1,2) #Transpose to (x,j,X,k)
        
        if sweep_index == mpo.N-1:
            # LR[-1] = np.einsum("xd,Ddl->xDl", np.conj(approx_solution[-1]), mpo[-1])
            mpo_transposed = mpo[-1].transpose(1,0,2) #transpose to (d,D,l)
            product = np.conj(approx_solution[-1]) @ mpo_transposed.reshape(mpo_transposed.shape[0],-1) #resulting shape (x,Dl)
            LR[-1] = product.reshape(product.shape[0],mpo[-1].shape[0],mpo[-1].shape[2])
            
            # LR[-1] = np.einsum("xDl,Xl->xDX", LR[-1], approx_solution[-1]) 
            product = LR[-1].reshape(-1,LR[-1].shape[-1]) @ approx_solution[-1].T #resulting shape (xD,X)
            LR[-1] = product.reshape(LR[-1].shape[0],LR[-1].shape[1],approx_solution[-1].shape[0])
        else:   
            # LR[sweep_index] = np.einsum("yEY,zdy->zdEY", LR[sweep_index+1], np.conj(approx_solution[sweep_index]))
            product = LR[sweep_index+1].T.reshape(-1,LR[sweep_index+1].shape[0]) @ np.conj(approx_solution[sweep_index]).T.reshape(approx_solution[sweep_index].shape[-1],-1) #resulting shape (YE,dz)
            product = product.reshape(LR[sweep_index+1].shape[-1],LR[sweep_index+1].shape[-2],approx_solution[sweep_index].shape[1],approx_solution[sweep_index].shape[0]) #reshape to (Y,E,d,z)
            LR[sweep_index]= product.transpose(3,2,1,0) #tranpose to (z,d,E,Y)
            
            # LR[sweep_index] = np.einsum("zdEY,DdEl->zDlY", LR[sweep_index], mpo[sweep_index])
            LR_transposed = LR[sweep_index].transpose(0,3,1,2) #Transpose to (z,Y,d,E)
            mpo_transposed = mpo[sweep_index].transpose(1,2,0,3) # Transpose to (d,E,D,l)
            product = LR_transposed.reshape(-1,LR_transposed.shape[-2]*LR_transposed.shape[-1]) @ mpo_transposed.reshape(-1,mpo_transposed.shape[-2]*mpo_transposed.shape[-1]) #resulting shape (zY,Dl)
            product = product.reshape(LR[sweep_index].shape[0],LR[sweep_index].shape[3],mpo[sweep_index].shape[0],mpo[sweep_index].shape[3]) #reshape to (z,Y,D,l)
            LR[sweep_index] = product.transpose(0,2,3,1) #transpose to (z,D,l,Y)
            
            # LR[sweep_index] = np.einsum("zDlY,ZlY->zDZ", LR[sweep_index], approx_solution[sweep_index])
            approx_transposed = approx_solution[sweep_index].transpose(1,2,0) #transpose to (l,Y,Z)
            product = LR[sweep_index].reshape(-1,LR[sweep_index].shape[-2]*LR[sweep_index].shape[-1]) @ approx_transposed.reshape(-1,approx_transposed.shape[-1]) # resulting shape (zD,Z)
            LR[sweep_index] = product.reshape(LR[sweep_index].shape[0],LR[sweep_index].shape[1],product.shape[1])
     
        return L, K

    def update_C_left(LR,C,mpo,approx_solution,sweep_index):  
        if sweep_index == mpo.N-2:
            # approx_solution[-1] = np.einsum("dX,Xl->dl",C,approx_solution[-1])
            approx_solution[-1] = C @ approx_solution[-1]
        else:
            # approx_solution[sweep_index+1] = np.einsum("dX,XlY->dlY",C,approx_solution[sweep_index+1])
            product = C @ approx_solution[sweep_index+1].reshape(approx_solution[sweep_index+1].shape[0],-1) #Rsulting shape (d,lY)
            approx_solution[sweep_index+1] =  product.reshape(product.shape[0],approx_solution[sweep_index+1].shape[1],approx_solution[sweep_index+1].shape[2] ) #Reshape to (d,l,Y)
                   
    def update_C_right(LR,C,mpo,approx_solution,sweep_index):
        if sweep_index == 1:
            # approx_solution[0] = np.einsum("dx,xy->dy",approx_solution[0],C)
            approx_solution[0] =  approx_solution[0] @ C
        else:
            # approx_solution[sweep_index-1] = np.einsum("xly,yd->xld",approx_solution[sweep_index-1],C)
            product = approx_solution[sweep_index-1].reshape(-1,approx_solution[sweep_index-1].shape[-1]) @ C 
            approx_solution[sweep_index-1] = product.reshape(approx_solution[sweep_index-1].shape[0],approx_solution[sweep_index-1].shape[1],C.shape[1])
    
    def time_evolve_forward(LR,mpo,approx_solution,krylov_depth,sweep_index,timestep): 
        if sweep_index == 0: # Left boundary case 
            if use_lanczos:
                def matvec(psi):
                    tmp = psi.reshape(mpo[0].shape[2], LR[1].shape[2])
                    # tmp = np.einsum("lX,xDX->xDl", tmp, LR[1])
                    product = LR[1].reshape(-1,LR[1].shape[-1]) @ tmp.T #resulting shape xD,l
                    tmp = product.reshape(LR[1].shape[0],LR[1].shape[1],tmp.shape[0])
                    
                    # tmp = np.einsum("xDl,dDl->dx", tmp, mpo[0])
                    mpo_transposed = mpo[0].transpose(1,2,0) #Transpose to D,l,d
                    product = tmp.reshape(tmp.shape[0],-1) @ mpo_transposed.reshape(-1,mpo_transposed.shape[-1]) #resulting shape xd
                    tmp = product.T
                    
                    return tmp.reshape(-1)
                approx_solution[sweep_index] = lan_exp(approx_solution[sweep_index].reshape(-1), matvec, t=-1j * timestep, k=krylov_depth).reshape(approx_solution[sweep_index].shape)
                return None
            else:
                # H_eff = np.einsum("dDl,yDY->dylY",mpo[0],LR[1])
                LR_transposed = LR[1].transpose(1,0,2) #transpose to (D,y,Y)
                mpo_transposed = mpo[0].transpose(0,2,1) #transpose to (d,l,D)
                product = mpo_transposed.reshape(-1,mpo_transposed.shape[-1]) @ LR_transposed.reshape(LR_transposed.shape[0],-1) #resulting shape (dl,yY)
                product= product.reshape(mpo[0].shape[0],mpo[0].shape[2],LR[1].shape[0],LR[1].shape[2])# reshape to (d,l,y,Y)
                H_eff = product.transpose(0,2,1,3) #Transpose to (d,y,l,Y)
                H_flat = H_eff.reshape(H_eff.shape[0]*H_eff.shape[1], -1)
                expH = expm(-1j * timestep * H_flat)
                approx_solution[sweep_index] = np.dot(expH, approx_solution[sweep_index].reshape(-1))
            approx_solution[sweep_index] = approx_solution[sweep_index].reshape(H_eff.shape[0],H_eff.shape[1])
        
        elif 0 < sweep_index < mpo.N - 1:
            if use_lanczos:
                def matvec(psi):
                    tmp = psi.reshape((LR[sweep_index-1].shape[2],mpo[sweep_index].shape[3], LR[sweep_index+1].shape[2]))
                    # tmp = np.einsum("xDX,XlY->xDlY", LR[sweep_index-1], tmp) 
                    product = LR[sweep_index-1].reshape(-1,LR[sweep_index-1].shape[-1]) @ tmp.reshape(tmp.shape[0],-1) #Resulting shape (xD,lY)
                    tmp = product.reshape(LR[sweep_index-1].shape[0],LR[sweep_index-1].shape[1],tmp.shape[1],tmp.shape[2])
                    
                    # tmp = np.einsum("xDlY,DdEl->xdEY",tmp,mpo[sweep_index])
                    tmp_transposed = tmp.transpose (0,3,1,2) #Transpose to (x,Y,D,l)
                    mpo_transposed = mpo[sweep_index].transpose(0,3,1,2) #Transpose to (D,l,d,E)
                    product = tmp_transposed.reshape(-1,tmp_transposed.shape[-2]*tmp_transposed.shape[-1]) @ mpo_transposed.reshape(-1,mpo_transposed.shape[-2]*mpo_transposed.shape[-1]) #Resulting shape (xY,dE)
                    product = product.reshape(tmp.shape[0],tmp.shape[3],mpo[sweep_index].shape[1],mpo[sweep_index].shape[2]) #Reshape to (x,D,d,E)
                    tmp = product.transpose(0,2,3,1) #Transpose to (x,d,E,Y)
                    
                    # tmp = np.einsum("xdEY,yEY->xdy",tmp,LR[sweep_index+1])
                    LR_transposed =  LR[sweep_index+1].transpose (1,2,0) #Transpose to (E,Y,y))
                    tmp = tmp.reshape(-1,tmp.shape[-2]*tmp.shape[-1]) @ LR_transposed.reshape(-1,LR_transposed.shape[-1]) #Resulting shape (xd,y)
                    # tmp = product.
                    return tmp.reshape(-1)
                approx_solution[sweep_index] = lan_exp(approx_solution[sweep_index].reshape(-1), matvec, t=-1j * timestep, k=krylov_depth).reshape(approx_solution[sweep_index].shape)
                return None
            else:
                # H_eff = np.einsum("xDX,DdEl->xdElX",LR[sweep_index-1],mpo[sweep_index])
                LR_transposed = LR[sweep_index-1].transpose(0,2,1) #transpose to (x,X,D)
                product = LR_transposed.reshape(-1,LR_transposed.shape[-1]) @ mpo[sweep_index].reshape(mpo[sweep_index].shape[0],-1) #resulting shape (xX,dEl)
                product = product.reshape(LR[sweep_index-1].shape[0],LR[sweep_index-1].shape[2],mpo[sweep_index].shape[1],mpo[sweep_index].shape[2],mpo[sweep_index].shape[3]) #reshape to (x,X,d,E,l)
                H_eff = product.transpose(0,2,3,4,1) #transpose to (x,d,E,l,X)
                
                # H_eff  = np.einsum("xdElX,yEY->xdyXlY",H_eff,LR[sweep_index+1])
                H_tranposed = H_eff.transpose(0,1,3,4,2) # transpose to (x,d,l,X,E)
                LR_transposed =  LR[sweep_index+1].transpose(1,0,2) #Transpose to (E,y,Y)
                product = H_tranposed.reshape(-1,H_tranposed.shape[-1])@ LR_transposed.reshape(LR_transposed.shape[0],-1)#resulting shape (xdlX,yY)
                product = product.reshape(H_eff.shape[0],H_eff.shape[1],H_eff.shape[3],H_eff.shape[4],LR[sweep_index+1].shape[0],LR[sweep_index+1].shape[2]) #reshape to (x,d,l,X,y,Y)
                H_eff = product.transpose(0,1,4,3,2,5)
                
                H_flat = H_eff.reshape(H_eff.shape[0]*H_eff.shape[1]*H_eff.shape[2],-1)
                #approx_solution[sweep_index] = expm_multiply(-1j * timestep * H_flat, approx_solution[sweep_index].reshape(-1))
                expH = expm(-1j * timestep * H_flat)
                approx_solution[sweep_index] = np.dot(expH, approx_solution[sweep_index].reshape(-1)) 
            approx_solution[sweep_index] = approx_solution[sweep_index].reshape(H_eff.shape[0],H_eff.shape[1],H_eff.shape[2]).reshape(approx_solution[sweep_index].shape)
        
        else: # Right boundary case (N-1)
            if use_lanczos:    
                def matvec(psi):
                    tmp = psi.reshape(LR[-2].shape[2], mpo[-1].shape[2])
                    # tmp = np.einsum("xDX,Xl->xDl", LR[-2], tmp)
                    product = LR[-2].reshape(-1,LR[-2].shape[-1]) @ tmp #Resulting shape (xD,l)
                    tmp = product.reshape(LR[-2].shape[0],LR[-2].shape[1],tmp.shape[1]) 
                    
                    # tmp = np.einsum("xDl,Ddl->xd", tmp, mpo[-1])
                    mpo_transposed = mpo[-1].transpose(0,2,1) #Tranpose to (D,l,d))
                    product = tmp.reshape(tmp.shape[0],-1) @ mpo_transposed.reshape(-1,mpo_transposed.shape[-1]) #Resulting shape x,d
                    return product.reshape(-1)
                approx_solution[sweep_index] = lan_exp(approx_solution[sweep_index].reshape(-1), matvec, t=-1j * timestep, k=krylov_depth).reshape(approx_solution[sweep_index].shape)
                return None
            else:
                # H_eff = np.einsum("Ddl,yDY->ydYl",mpo[-1],LR[sweep_index-1])
                mpo_transposed = mpo[-1].transpose(1,2,0) #transpose to (d,l,D)
                LR_transposed = LR[sweep_index-1].transpose(1,0,2) #transpose to (D,y,Y)
                product = mpo_transposed.reshape(-1,mpo_transposed.shape[-1]) @ LR_transposed.reshape(LR_transposed.shape[0],-1) #resulting shape (dl,yY)
                product = product.reshape(mpo[-1].shape[1],mpo[-1].shape[2],LR[sweep_index-1].shape[0],LR[sweep_index-1].shape[2]) #reshape to (d,l,y,Y)
                H_eff = product.transpose(2,0,3,1) #transpose to (y,d,Y,l)
                
                H_flat = H_eff.reshape(H_eff.shape[0]*H_eff.shape[1],-1)   
                expH = expm(-1j * timestep * H_flat)
                approx_solution[sweep_index] = np.dot(expH, approx_solution[sweep_index].reshape(-1))
            approx_solution[sweep_index] = approx_solution[sweep_index].reshape(H_eff.shape[0],H_eff.shape[1])
        return H_eff
    
    def time_evolve_backwards(R,K,k,krylov_depth): #TODO:lingering einsum
        if use_lanczos:
            def matvec(C):
                tmp = C.reshape(R.shape)
                # tmp = np.einsum("xDX,XY->xDY",LR[k],tmp)
                product = LR[k].reshape(-1,LR[k].shape[-1]) @ tmp #resulting shape (xD,Y)
                tmp = product.reshape(LR[k].shape[0],LR[k].shape[1],tmp.shape[-1])
               
                # tmp = np.einsum("xDY,yDY->xy",tmp,LR[k+1])
                LR_transposed = LR[k+1].transpose(1,2,0) #Transpose to DY,y
                tmp = tmp.reshape(tmp.shape[0],-1) @ LR_transposed.reshape(-1,LR_transposed.shape[-1]) #Resulting shape x,y
                return tmp.reshape(-1)
            C_evolved = lan_exp(R.reshape(-1), matvec, t=1j * dt/2, k=krylov_depth) 
        else:
            Kflat = K.reshape(K.shape[0]*K.shape[1],-1) #Flatten to Hermitian Matrix 
            expK = expm(1j * dt/2 * Kflat)
            C_evolved = np.dot(expK, R.reshape(-1))
        C_evolved = C_evolved.reshape(R.shape)
        return C_evolved
    
    def sweep_scheduler(n, sweeps):
        sequence = np.concatenate([np.arange(n), np.arange(n-2, 0, -1)])
        repeated_sequence = np.tile(sequence, sweeps)
        repeated_sequence = np.append(repeated_sequence, 0)
        is_first_last = np.zeros(repeated_sequence.shape)
        is_first_last[0] = 1
        is_first_last[-1] = -1
        return zip(list(repeated_sequence), list(is_first_last))
    
    def measure_energy_C(C,LR,sweep_index): 
        # E = np.einsum("XY,xDX->xDY",C,LR[sweep_index])
        product = LR[sweep_index].reshape(-1,LR[sweep_index].shape[-1]) @ C #resulting shape (x,D,Y)
        E = product.reshape(LR[sweep_index].shape[0],LR[sweep_index].shape[1],C.shape[1])
        
        # E = np.einsum("xDY,yDY->xy",E,LR[sweep_index+1])
        LR_transposed = LR[sweep_index+1].transpose(1,2,0) #resulting shape (D,Y,y)
        E = E.reshape(E.shape[0],-1) @ LR_transposed.reshape(-1,LR_transposed.shape[-1]) #resulting shape xy
        
        E = np.dot(C.conj().reshape(-1),E.reshape(-1))
        return E
    

    if mps.canform != "Left":
        mps.canonize_left()
    approx_solution = mps.copy()
    LR = compute_right_envs(mps, mpo)
    Es_C = []
    Es_A = []
    sweep_schedule = sweep_scheduler(mps.N,sweeps)
    for k, is_first_last in (sweep_schedule): #Using 1 indexing
        if verbose:
            print("-------New epoch at site:",k)
            approx_solution.show()
        #================ First Site ================ 
        if k == 0:  
            if is_first_last:
                H_eff = time_evolve_forward(LR,mpo,approx_solution,krylov_depth,k,dt/2)
            else:
                H_eff = time_evolve_forward(LR,mpo,approx_solution,krylov_depth,k,dt)
            #Es_A.append(measure_energy_A(approx_solution,mpo,LR,k))
            if is_first_last != -1: # not last
                R, K = update_A_left(H_eff, approx_solution,k)
                C = time_evolve_backwards(R,K,k,krylov_depth)
                update_C_left(LR,C,mpo,approx_solution,k)
                if measure:
                    Es_C.append(measure_energy_C(C,LR,k))
            left_sweep = True
            
        #================ Middle Sites ================ 
        elif 0 < k and k < mps.N-1: 
            #Evolve AC forward in time 
            H_eff = time_evolve_forward(LR,mpo,approx_solution,krylov_depth,k,dt/2)
            #Es_A.append(measure_energy_A(approx_solution,mpo,LR,k))

            if left_sweep:
                R, K = update_A_left(H_eff, approx_solution, k)
                C = time_evolve_backwards(R,K,k,krylov_depth)
                update_C_left(LR,C,mpo,approx_solution,k)
                    
            else:
                R, K = update_A_right(H_eff, approx_solution, k)
                C = time_evolve_backwards(R,K,k-1,krylov_depth)
                update_C_right(LR,C,mpo,approx_solution,k)
                    
            if measure: 
                Es_C.append(measure_energy_C(C,LR,k if left_sweep else k-1))

        #================ Last Site ================  Index =  N -1  and N  
        else: # k == mps.N-1
            H_eff = time_evolve_forward(LR,mpo,approx_solution,krylov_depth,k,dt)
            #Es_A.append(measure_energy_A(approx_solution,mpo,LR,k))

            # Setup for right sweep
            R, K = update_A_right(H_eff, approx_solution, k)
            C = time_evolve_backwards(R,K,k-1,krylov_depth)
            #print(measure_energy_C(C,LR,k))

            update_C_right(LR,C,mpo,approx_solution,k)
            if measure:
                Es_C.append(measure_energy_C(C,LR,k-1))
            left_sweep = False
            
    approx_solution.canform = "Left"
    return approx_solution, np.array(Es_C), np.array(Es_A)