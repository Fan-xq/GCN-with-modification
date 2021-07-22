from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf

from gcn.original_network import *
from gcn.models import GCN, MLP

from pathlib import Path
import os

# from sklearn.neighbors import kneighbors_graph
from sklearn import preprocessing
import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
import scipy.io




# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)




# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
for name in list(flags.FLAGS):
    delattr(flags.FLAGS,name)
#flags.DEFINE_string('dataset', 'constructive', 'Dataset string.') # 'constructive', 'cora', 'aminer', 'digits', 'fma', 'cell', 'segmentation'
flags.DEFINE_string('model', 'gcn', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 2000, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 16, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 200, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')
# 'sssa': spielman sparsification from dense CkNN for which its performance is equal to or worse than MLP.
# 'sssa_optimalCkNN': spielman sparsification from optimal CkNN
#flags.DEFINE_string('case', 'knn', 'Case string.') # 'knn', 'mknn', 'cknn', 'rmst', 'sssa', 'sssa_optimalCkNN'
tf.app.flags.DEFINE_string('f', '', 'kernel')




G = nx.Graph()
labels = np.ones(50*(50)+1)
features = np.eye(50*(50)+1)
for j in range(50):
    nx.add_star(G, range(50*j,50*(j+1)))
    labels[50*j] = 0
G.add_node(50*(j+1))    
for k in range(25):
    G.add_edge(100*k+1,100*k+50)
    G.add_edge(100*k+1,50*(j+1))




# Load data

adj, features, y = load_dataset1(features, G, labels)
#features, adj, y = load_data(FLAGS.dataset,FLAGS.case)
y_train, y_val, y_test, train_mask, val_mask, test_mask, idx_train, idx_val, idx_test = get_splits(y)




# Some preprocessing
features = preprocess_features(features)
if FLAGS.model == 'gcn':
    support = [preprocess_adj(adj)]
    num_supports = 1
    model_func = GCN
elif FLAGS.model == 'gcn_cheby':
    support = chebyshev_polynomials(adj, FLAGS.max_degree)
    num_supports = 1 + FLAGS.max_degree
    model_func = GCN
elif FLAGS.model == 'dense':
    support = [preprocess_adj(adj)]  # Not used
    num_supports = 1
    model_func = MLP
else:
    raise ValueError('Invalid argument for model: ' + str(FLAGS.model))




# Define placeholders
placeholders = {
    'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
    'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
    'labels_mask': tf.placeholder(tf.int32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
}




# Create modele
model = model_func(placeholders, input_dim=features[2][1], logging=True)

# Initialize session
sess = tf.Session()




# Define model evaluation function
def evaluate(features, support, labels, mask, placeholders):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(features, support, labels, mask, placeholders)
    outs_val = sess.run([model.loss, model.accuracy, model.outputs, model.outputs], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], outs_val[2], (time.time() - t_test)




# Init variables
sess.run(tf.global_variables_initializer())

cost_val = []




# Train model
for epoch in range(FLAGS.epochs):

    t = time.time()
    # Construct feed dictionary
    feed_dict = construct_feed_dict(features, support, y_train, train_mask, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})

    # Training step
    outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)

    # Validation
    cost, acc, outputs, duration = evaluate(features, support, y_val, val_mask, placeholders)
    cost_val.append(cost)

    # Print results
    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
          "train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(cost),
          "val_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t))

    if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
        print("Early stopping...")
        break

print("Optimization Finished!")




# Testing
test_cost, test_acc, test_outputs, test_duration = evaluate(features, support, y_test, test_mask, placeholders)
print("Test set results:", "cost=", "{:.5f}".format(test_cost),
      "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))

#from scipy.special import softmax
#np.save("{}_{}_gcn_output.npy".format(FLAGS.dataset,FLAGS.case),softmax(test_outputs,axis=1))






