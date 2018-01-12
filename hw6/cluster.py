import numpy as np 
from sklearn import cluster 

def K_means_batch(data,k,batch_size=128,verbose=False):
    """
        data      : A 2D array with [Ndata x Ndim]
        k         : the # of cluster
        criterion : convergence  
    """
    categorizer = cluster.MiniBatchKMeans(n_clusters=k,batch_size=128,max_no_improvement=100000,tol=1.0e-10,verbose=verbose)
    categorizer = categorizer.fit(data)
    if hasattr(categorizer,'labels_'):
        y_pred = categorizer.labels_.astype(np.int)
    else:
        y_pred = categorizer.predict(data)

    weights = categorizer.get_params()

    return y_pred, weights
    

def K_means(data,k,verbose=False):
    """
        data      : A 2D array with [Ndata x Ndim]
        k         : the # of cluster
        criterion : convergence  
    """
    categorizer = cluster.KMeans(n_clusters=k,verbose=verbose)
    categorizer = categorizer.fit(data)
    if hasattr(categorizer,'labels_'):
        y_pred = categorizer.labels_.astype(np.int)
    else:
        y_pred = categorizer.predict(data)

    weights = categorizer.get_params()

    return y_pred, weights

def Agg(data,k):
    """
        data      : A 2D array with [Ndata x Ndim]
        k         : the # of cluster
        criterion : convergence  
    """
    categorizer = cluster.AgglomerativeClustering(n_clusters=k)
    categorizer = categorizer.fit(data)
    if hasattr(categorizer,'labels_'):
        y_pred = categorizer.labels_.astype(np.int)
    else:
        y_pred = categorizer.predict(data)

    weights = categorizer.get_params()

    return y_pred, weights


def Birch(data,k):
    """
        data      : A 2D array with [Ndata x Ndim]
        k         : the # of cluster
        criterion : convergence  
    """
    categorizer = cluster.Birch(n_clusters=k)
    categorizer.fit(data)
    if hasattr(categorizer,'labels_'):
        y_pred = categorizer.labels_.astype(np.int)
    else:
        y_pred = categorizer.predict(data)

    weights = categorizer.get_params()

    return y_pred, weights

def Birch_batch(data,k,batch_size=128,verbose=False):
    """
        data      : A 2D array with [Ndata x Ndim]
        k         : the # of cluster
        criterion : convergence  
    """
    categorizer = cluster.Birch(n_clusters=k)
    for i in np.arange(0,len(data),batch_size):
        if verbose:
            print ("[Batch] elem : %d"%(i))

        endl = i + batch_size
        if endl > len(data):
            endl = len(data)
        categorizer.partial_fit(data[i:endl])
    

    if hasattr(categorizer,'labels_'):
        y_pred = categorizer.labels_.astype(np.int)
    else:
        y_pred = categorizer.predict(data)

    weights = categorizer.get_params()

    return y_pred, weights

def SpCluster(data):
    categorizer = cluster.SpectralClustering(n_clusters=2,eigen_solver='arpack',affinity="nearest_neighbors",n_jobs=8)
    categorizer.fit(data)
    if hasattr(categorizer,'labels_'):
        y_pred = categorizer.labels_.astype(np.int)
    else:
        y_pred = categorizer.predict(data)
    
    weights = categorizer.get_params()

    return y_pred , weights 


