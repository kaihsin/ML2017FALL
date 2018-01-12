import numpy as np 
"""
def PCA2(Raw,keep_dim=None):
    ""
        Raw      : 2D array with [Ndata x Ndim]
        keep_dim : the # of eigen_vector to be returned. if set as None, all eigen_vectors will be returned
    ""
    if keep_dim is None:
        keep_dim = len(Raw[0])

    mean = np.mean(Raw,axis=0)
    B = Raw - mean 
    w,v = np.linalg.eig(np.dot(B.T,B))

    return mean, v.T[:keep_dim]
    #print (Raw)
    #print (mean)
    #print (B)
"""

def PCA(Raw,full_vec=False,keep_dim=None):
    """
        Raw      : 2D array with [Ndata x Ndim]
        keep_dim : the # of eigen_vector to be returned. if set as None, all eigen_vectors will be returned
        full_vec : if True, all the v will be computed. if False, only min[dim(u),dim(v)] will be computed by SVD. 
    """

    mean = np.mean(Raw,axis=0)
    B = Raw - mean 
    u,s,v = np.linalg.svd(B,full_matrices=full_vec)
    if keep_dim is None:
        keep_dim = len(v)
    
    return mean, s, v[:keep_dim]


if __name__ == '__main__':
    A = np.arange(0,15,1).reshape((3,5))
    #mean, eigV = PCA2(A)
    _ ,s, eigsvd = PCA(A)
    print (eigsvd)
    #print (eigV)




