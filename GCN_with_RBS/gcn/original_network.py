import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy import sparse
from scipy.sparse.linalg.eigen.arpack import eigsh
import pandas as pd
import sys
sys.path.append('/Users/fanxiaoqing/Downloads/gcn-performance/gcn/RBS-master')
sys.path.append('/Users/fanxiaoqing/Downloads/gcn-performance/gcn/RMST-master')


import RBS as RBS
import RMST as RMST
from os import listdir
from os.path import isfile, join




#G = nx.Graph()
#labels = np.ones((5*(50)+1))
#feature = np.eye(5*(50)+1)
#for j in range(50):
#    nx.add_star(G, range(5*j,5*(j+1)))
#    labels[5*j] = 0
#G.add_node(5*(j+1))    
#for k in range(25):
#    G.add_edge(10*k+1,10*k+5)
#    G.add_edge(10*k+1,5*(j+1))
nx.Graph()
k = 100
g1 = nx.cycle_graph(k)
mapping = {old_label:new_label for old_label, new_label in enumerate(np.arange(0, 3*k, 3))}
R1 = nx.relabel_nodes(g1, mapping)
g2 = nx.cycle_graph(k)
mapping = {old_label:new_label for old_label, new_label in enumerate(np.arange(2, 3*k+2, 3))}
R2 = nx.relabel_nodes(g2, mapping)
h = nx.empty_graph(k)
mapping = {old_label:new_label for old_label, new_label in enumerate(np.arange(1, 3*k+1, 3))}
R3 = nx.relabel_nodes(h, mapping)
G = nx.compose(R1,R2)
G.add_nodes_from(R3)
for j in range(k):
    G.add_edges_from([(3*j,3*j+1),(3*j+1,3*j+2)])
#g.add_edges_from([(0,9),(1,8),(2,7),(3,6),(4,5)])
features = np.eye(3*k)
lists = ['1','0','1']*k
labels = np.array(lists)




def load_dataset1(features, G, labels):
    """Save the features in a pickle"""
    features = sp.csr_matrix(features, dtype=np.float32)
    labels = encode_onehot(labels)
    
    adj = nx.adjacency_matrix(G)
    return adj, features.tolil(), labels




def get_splits(y):    
    idx_train = range(16)
    idx_val = range(16, 48)
    idx_test = range(48, 300)
    y_train = np.zeros(y.shape, dtype=np.int32)
    y_val = np.zeros(y.shape, dtype=np.int32)
    y_test = np.zeros(y.shape, dtype=np.int32)
    y_train[idx_train] = y[idx_train]
    y_val[idx_val] = y[idx_val]
    y_test[idx_test] = y[idx_test]
    train_mask = sample_mask(idx_train, y.shape[0])
    val_mask = sample_mask(idx_val, y.shape[0])
    test_mask = sample_mask(idx_test, y.shape[0])

    return y_train, y_val, y_test, train_mask, val_mask, test_mask, idx_train, idx_val, idx_test




def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)




def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx




def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    from sklearn.preprocessing import normalize
    features = normalize(features, norm='l1', axis=1)
    return sparse_to_tuple(features)




def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()




def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)




def construct_feed_dict(features, support, labels, labels_mask, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    return feed_dict




def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k+1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return sparse_to_tuple(t_k)




def encode_onehot(labels):
    classes = sorted(list(set(labels)))
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    
    return labels_onehot






