import robot as rb
import numpy as np
import numpy.matlib as npmat

#Concatenates transforms
def rel2abs(pred):
    # relative to absolute transform
    data_size = pred.shape[0]
    #print(pred.shape)
    #print(pred)
    R = rb.rpy2tr(pred[:, 0:3])
    #print(R[0])
    t = rb.transl(pred[:, 3:6])
    Tl= []
    Tl.append( npmat.eye(4) )  # T0
    Tl.append( npmat.mat( np.concatenate( (np.concatenate( ( R[0][0:3,0:3], t[0][0:3,3] ), 1 ), npmat.mat([0, 0, 0, 1])), 0 ) ) ) #T1
    T = np.zeros((data_size+1, 4, 4))
    T[0,:,:] = npmat.eye(4)
    T[1,:,:] = npmat.mat( np.concatenate( (np.concatenate( ( R[0][0:3,0:3], t[0][0:3,3] ), 1 ), npmat.mat([0, 0, 0, 1])), 0 ) )
    for k in range(2,data_size+1):
        Tn = npmat.mat( np.concatenate( (np.concatenate( ( R[k-1][0:3,0:3], t[k-1][0:3,3] ), 1 ), npmat.mat([0, 0, 0, 1])), 0 ) ) #relative transform from k-1 to k
        Tl.append( Tl[k-1].dot(Tn) )
        T[k,:,:] = Tl[k]
    return Tl, T
