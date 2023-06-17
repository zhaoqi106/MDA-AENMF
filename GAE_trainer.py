import tensorflow._api.v2.compat.v1 as tf
from GAE import GATE
tf.disable_eager_execution()
from clr import cyclic_learning_rate
import tensorflow._api.v2.compat.v1 as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import networkx as nx
import scipy.sparse as sp
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc
import argparse




class GATETrainer():

    def __init__(self, args):

        self.args = args
        self.build_placeholders()
        gate = GATE(args.hidden_dims, args.lambda_)
        self.loss, self.H, self.C = gate(self.A, self.X, self.R, self.S)
        self.optimize(self.loss)
        self.build_session()

    def build_placeholders(self):
        self.A = tf.sparse_placeholder(dtype=tf.float32)
        self.X = tf.placeholder(dtype=tf.float32)
        self.S = tf.placeholder(tf.int64)
        self.R = tf.placeholder(tf.int64)

    def build_session(self, gpu= True):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        if gpu == False:
            config.intra_op_parallelism_threads = 0
            config.inter_op_parallelism_threads = 0
        self.session = tf.Session(config=config)
        self.session.run([tf.global_variables_initializer(), tf.local_variables_initializer()])


    def optimize(self, loss):

        optimizer = tf.train.AdamOptimizer(learning_rate=cyclic_learning_rate(global_step=0, learning_rate=0.0005,
                                            max_lr=0.005, mode='exp_range', gamma=.995)
      )
        gradients, variables = zip(*optimizer.compute_gradients(loss))
        gradients, _ = tf.clip_by_global_norm(gradients, self.args.gradient_clipping)
        self.train_op = optimizer.apply_gradients(zip(gradients, variables))


    def __call__(self, A, X, S, R):
        for epoch in range(self.args.n_epochs):
            self.run_epoch(epoch, A, X, S, R)


    def run_epoch(self, epoch, A, X, S, R):

        loss, _ = self.session.run([self.loss, self.train_op],
                                         feed_dict={self.A: A,
                                                    self.X: X,
                                                    self.S: S,
                                                    self.R: R})

        return loss

    def infer(self, A, X, S, R):
        H, C = self.session.run([self.H, self.C],
                           feed_dict={self.A: A,
                                      self.X: X,
                                      self.S: S,
                                      self.R: R})


        return H, conver_sparse_tf2np(C)


def get_gae_feature(adj, features, epochs, l):
    args = parse_args(epochs=epochs,l=l)
    feature_dim = features.shape[1]
    args.hidden_dims = [feature_dim] + args.hidden_dims

    G, S, R = prepare_graph_data(adj)
    gate_trainer = GATETrainer(args)  # 初始化__init__完成
    gate_trainer(G, features, S, R)
    embeddings, attention = gate_trainer.infer(G, features, S, R)
    tf.reset_default_graph()
    return embeddings


def prepare_graph_data(adj):

    num_nodes = adj.shape[0]
    adj = adj + sp.eye(num_nodes)
    data = adj.tocoo().data
    if not sp.isspmatrix_coo(adj):
        adj = adj.tocoo()
    adj = adj.astype(np.float32)
    indices = np.vstack((adj.col, adj.row)).transpose()
    return (indices, adj.data, adj.shape), adj.row, adj.col


def conver_sparse_tf2np(input):

    return [sp.coo_matrix((input[layer][1], (input[layer][0][:, 0], input[layer][0][:, 1])),
                          shape=(input[layer][2][0], input[layer][2][1])) for layer in input]


def parse_args(epochs,l):

    parser = argparse.ArgumentParser(description="Run gate.")

    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate. Default is 0.001.')

    parser.add_argument('--n-epochs', default=epochs, type=int,
                        help='Number of epochs')

    parser.add_argument('--hidden-dims', type=list, nargs='+', default=[128,64],
                        help='Number of dimensions.')

    parser.add_argument('--lambda-', default=l, type=float,
                        help='Parameter controlling the contribution of graph structure reconstruction in the loss function.')

    parser.add_argument('--dropout', default=0.5, type=float,
                        help='Dropout.')

    parser.add_argument('--gradient_clipping', default=5.0, type=float,
                        help='gradient clipping')

    return parser.parse_args()


def aupr(label, prob):
    precision, recall, _thresholds = precision_recall_curve(label, prob)
    area = auc(recall, precision)
    return area


def adj_show(th):
    seq_sim_matrix = pd.read_csv("mydata/gene_seq_sim.csv", index_col=0, dtype=np.float32).to_numpy()
    seq_sim_matrix[seq_sim_matrix <= th] = 0
    seq_sim_matrix[seq_sim_matrix > th] = 1
    graph = nx.from_numpy_matrix(seq_sim_matrix)
    adj = nx.adjacency_matrix(graph)
    adj = adj.todense()
    plt.imshow(adj)
    plt.show()


def generate_graph_adj_and_feature(c_network, d_network, c_feature, d_feature):
    c_features = sp.csr_matrix(c_feature).tolil().todense()
    d_features = sp.csr_matrix(d_feature).tolil().todense()

    c_graph = nx.from_numpy_matrix(c_network)
    c_adj = nx.adjacency_matrix(c_graph)
    c_adj = sp.coo_matrix(c_adj)

    d_graph = nx.from_numpy_matrix(d_network)
    d_adj = nx.adjacency_matrix(d_graph)
    d_adj = sp.coo_matrix(d_adj)

    return c_adj, d_adj, c_features, d_features

def generate_adj_and_feature(network, feature):
    features = sp.csr_matrix(feature).tolil().todense()

    graph = nx.from_numpy_matrix(network)
    adj = nx.adjacency_matrix(graph)
    adj = sp.coo_matrix(adj)


    return adj, features

