import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy import sparse
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
import numpy as np
import pandas as pd


from os import listdir
from os.path import isfile, join

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

def build_mst(X):
    from sklearn.metrics.pairwise import euclidean_distances
    from scipy.sparse.csgraph import minimum_spanning_tree
    """
    """
    D = euclidean_distances(X, X)
    adj_directed = minimum_spanning_tree(D).toarray()
    adj = adj_directed + adj_directed.T
    adj[adj > 0] = 1
    np.fill_diagonal(adj,0)

    return sparse.csr_matrix(adj)

def load_data(dataset,case):
    """Load citation network dataset"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("./data/{}/{}.content".format(dataset,dataset), dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    if case == "knn":
        sparsity_parameters_knn_k ={
            "constructive": 9,
            "cora":12,
            "aminer":12,
            "digits":6,
            "fma":1,
            "cell":3,
            "segmentation":7,
        }
        edges_unordered = np.genfromtxt("./data/{}/{}_{}_k_{}.cites".format(dataset,dataset,case,sparsity_parameters_knn_k[dataset]), dtype=np.int32)
    if case == "mknn":
        sparsity_parameters_mknn_k ={
            "constructive": 104,
            "cora":39,
            "aminer":171,
            "digits":39,
            "fma":1,
            "cell":7,
            "segmentation":17,
        }
        edges_unordered = np.genfromtxt("./data/{}/{}_{}_k_{}.cites".format(dataset,dataset,case,sparsity_parameters_mknn_k[dataset]), dtype=np.int32)
    if case == "cknn":
        sparsity_parameters_cknn_k ={
            "constructive": 29,
            "cora":74,
            "aminer":199,
            "digits":33,
            "fma":13,
            "cell":35,
            "segmentation":5,
        }
        edges_unordered = np.genfromtxt("./data/{}/{}_{}_delta_1_k_{}.cites".format(dataset,dataset,case,sparsity_parameters_cknn_k[dataset]), dtype=np.int32)
    if case == "rmst":
        sparsity_parameters_rmst_gamma ={
            "constructive": 0.07421,
            "cora":0.02924,
            "aminer":0.02317,
            "digits":0.00296,
            "fma":0.01435,
            "cell":0.00159,
            "segmentation":0.03423,
        }
        edges_unordered = np.genfromtxt("./data/{}/{}_{}_gamma_{}_k_1.cites".format(dataset,dataset,case,sparsity_parameters_rmst_gamma[dataset]), dtype=np.int32)
    
    # Spielman sparsification from dense CkNN
    if case == "sssa":
        sparsity_parameters_sssa_dense_sigma = {
            "constructive":0.2293,
            "cora":0.3401,
            "aminer":0.2216,
            "digits":0.2229,
            "fma":0.3816,
            "cell":0.7806,
            "segmentation":0.7003,
        }
        sparsity_parameters_cknn_k = {
            "constructive":120,
            "cora":266,
            "aminer":509,
            "digits":286,
            "fma":56,
            "cell":48,
            "segmentation":52,
        }
        edges_unordered = np.genfromtxt("./data/{}/{}_{}_epsilon_{:.4f}_cknn_delta_1_k_{}_mst.cites".format(dataset,dataset,case,sparsity_parameters_sssa_dense_sigma[dataset],sparsity_parameters_cknn_k[dataset]), dtype=np.int32)

    # Spielman sparsification from optimal CkNN
    if case == "sssa_optimalCkNN":
        sparsity_parameters_sssa_optimal_sigma = {
            "constructive":0.2491,
            # "cora":,
            "aminer":0.1618,
            "digits":0.1034,
            # "fma":,
            "cell":0.7407,
            # "segmentation":,
        }
        sparsity_parameters_cknn_k ={
            "constructive": 29,
            "cora":74,
            "aminer":199,
            "digits":33,
            "fma":13,
            "cell":35,
            "segmentation":5,
        }
        if dataset not in ["cora","digits","segmentation"]:
            edges_unordered = np.genfromtxt("./data/{}/{}_{}_epsilon_{:.4f}_cknn_delta_1_k_{}_mst.cites".format(dataset,dataset,case,sparsity_parameters_sssa_optimal_sigma[dataset],sparsity_parameters_cknn_k[dataset]), dtype=np.int32)
        else:
            case = "cknn"
            edges_unordered = np.genfromtxt("./data/{}/{}_{}_delta_1_k_{}.cites".format(dataset,dataset,case,sparsity_parameters_cknn_k[dataset]), dtype=np.int32)   
    
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)

    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    # print('Dataset has {} nodes, {} edges, {} features.'.format(adj.shape[0], edges.shape[0], features.shape[1]))

    if case in ["knn","mknn","cknn"]:
            print("Adding mst graph")
            from sklearn import preprocessing
            adj_mst = build_mst(preprocessing.normalize(features.toarray(), norm='l1', axis=1))
            adj = adj + adj_mst
            adj[adj > 0] = 1

    print('Dataset has {} nodes, {} edges, {} features.'.format(adj.shape[0], edges.shape[0], features.shape[1]))

    return features.tolil(), adj, labels

def get_splits(y, dataset):
    if dataset == "constructive":
        idx_train = range(50)
        idx_val = range(50, 150)
        idx_test = range(150, 1000)
    elif dataset == "cora":
        idx_train = range(119)
        idx_val = range(119, 372)
        idx_test = range(372, 2485)
    elif dataset == "aminer":
        idx_train = range(98)
        idx_val = range(98, 310)
        idx_test = range(310, 2072)
    elif dataset == "digits":
        idx_train = range(80)
        idx_val = range(80, 269)
        idx_test = range(269, 1797)
    elif dataset == "fma":
        idx_train = range(96)
        idx_val = range(96, 300)
        idx_test = range(300, 2000)
    elif dataset == "cell":
        idx_train = range(100)
        idx_val = range(100, 300)
        idx_test = range(300, 2000)
    elif dataset == "cell_b":
        idx_train = range(100)
        idx_val = range(100, 300)
        idx_test = range(300, 2000)
    elif dataset == "segmentation":
        idx_train = range(112)
        idx_val = range(112, 346)
        idx_test = range(346, 2310)
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