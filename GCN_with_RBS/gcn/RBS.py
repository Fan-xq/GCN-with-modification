import networkx as nx
import numpy as np
import scipy as sc



def RBS_matrix(G, Kmax, alpha):
    """
    create the RBS matrix
    Kmax: lenght of longest path
    alpha: RBS parameter, 0 < alpha < 1

    Input: networkX graph
    Output: sparse lil matrix
    """

    if Kmax>len(G):
        print('Error!! Kmax must be smaller then number of nodes!')
        return 0
    
    A = nx.adjacency_matrix(G) #adjacency matrix 
    w,v = sc.sparse.linalg.eigs(A.asfptype(), k=1) #largest eigenvalue
    beta = alpha/np.abs(w[0]) #beta parameter

    X = sc.sparse.lil_matrix((len(G), 2*Kmax))
    
    AT = A.transpose()
    one = np.ones(len(G))
    
    Ak = A.copy()
    ATk = AT.copy()
    for k in range(Kmax):
        X[:,k] = sc.sparse.lil_matrix(ATk.dot(one)).transpose()
        ATk = beta * AT.dot(ATk)

        X[:,Kmax+k] = sc.sparse.lil_matrix(Ak.dot(one)).transpose()
        Ak = beta * A.dot(Ak)
            
    from sklearn.metrics.pairwise import cosine_similarity
    Y = cosine_similarity(X,dense_output=False) #compute the cosine similarity
    
    return Y
