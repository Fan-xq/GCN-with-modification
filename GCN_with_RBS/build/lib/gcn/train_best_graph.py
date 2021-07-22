from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf

from gcn.utils import *
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
flags.DEFINE_string('dataset', 'constructive', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
flags.DEFINE_string('model', 'gcn', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 400, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 16, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 100, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')
flags.DEFINE_string('case', 'knn', 'Case string.')  #

if Path('performance_{}_{}_best_graph.csv'.format(FLAGS.dataset,FLAGS.case)).is_file():
    os.remove('performance_{}_{}_best_graph.csv'.format(FLAGS.dataset,FLAGS.case))

file = open('performance_{}_{}_best_graph.csv'.format(FLAGS.dataset,FLAGS.case),'w')
file.write("Sparsity_parameter" + ',' + "id" + ',' + "Epoches_stopped" + ','+ 'Test_accuracy' + ',' + 'Density'+ ',' + 'Time(second)' + '\n')

sparsity_parameter = best_graph(dataset=FLAGS.dataset, case=FLAGS.case)
sparsity_parameter_list = []
sparsity_parameter_list.append(sparsity_parameter)

id_file = list(range(1,101,1))

for iii in sparsity_parameter_list:
    for jjj in id_file:
        print("Current sparsity_parameter is {}".format(iii),"Current Id is {}".format(jjj))
        t0 = time.time()
        # Load data using keras input
        features, adj, y = load_data(FLAGS.dataset,FLAGS.case,iii)
        if jjj == 1:
            density = count_density(adj.toarray())
        y_train, y_val, y_test, train_mask, val_mask, test_mask, idx_train, idx_val, idx_test = get_splits(y, dataset=FLAGS.dataset)

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

        # Create model
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

        y_test_output = list(np.argmax(test_outputs[idx_test], 1))

        y_test_list = list(y_test)
        y_test_value = []
        for i in idx_test:
            y_test_value.append(list(np.nonzero(list(y_test_list[i]))[0])[0])

        import pandas as pd
        df = pd.DataFrame(
            {'paper_groundTruth_id': y_test_value,
             'paper_gcn': y_test_output,
            })
        df.index.rename('paper_id',inplace=True)
        if FLAGS.case == "knn":
            df.to_csv('../../data/{}/{}_{}/output/performance_{}_output_{}_k_{}_id_{}.csv'.format(FLAGS.dataset,FLAGS.dataset,FLAGS.case,FLAGS.dataset,FLAGS.case,iii,jjj),index=True,sep=',')
        if FLAGS.case == "cknn":
            df.to_csv('../../data/{}/{}_{}/output/performance_{}_output_{}_delta_1_k_{}_id_{}.csv'.format(FLAGS.dataset,FLAGS.dataset,FLAGS.case,FLAGS.dataset,FLAGS.case,iii,jjj),index=True,sep=',')
        if FLAGS.case == "rmst":
            df.to_csv('../../data/{}/{}_{}/output/performance_{}_output_{}_gamma_{:.5f}_k_1_id_{}.csv'.format(FLAGS.dataset,FLAGS.dataset,FLAGS.case,FLAGS.dataset,FLAGS.case,iii,jjj),index=True,sep=',')
        
        file = open('performance_{}_{}_best_graph.csv'.format(FLAGS.dataset,FLAGS.case),'a')
        file.write(str(iii) + ',' + str(jjj) + ','  + str(epoch+1) + ',' + str(test_acc) + ',' + str(density) + ',' + str(round(time.time() - t0,4)) + '\n')
        file.close()
