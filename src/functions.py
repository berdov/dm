import numpy as np
import networkx as nx
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, ks_2samp
from sklearn.metrics import roc_auc_score, roc_curve
import pandas as pd


def sample_exp(n, lam):
    return np.random.exponential(1 / lam, n)

def sample_gamma(n, shape, lam):
    return np.random.gamma(shape, 1 / lam, n)

def sample_normal(n, sigma):
    return np.random.normal(0, sigma, n)

def sample_t(n, df):
    return np.random.standard_t(df, n)


def build_knn_graph(X, k):
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X.reshape(-1, 1))
    distances, indices = nbrs.kneighbors(X.reshape(-1, 1))
    G = nx.Graph()
    n = len(X)
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in indices[i][1:]:
            G.add_edge(i, j)
    return G


def count_triangles(G):
    return sum(nx.triangles(G).values()) // 3


def monte_carlo_characteristic(sample_func, graph_func, char_func, *sample_args, n_sim=300, n=1000):
    results = []
    for _ in range(n_sim):
        X = sample_func(n, *sample_args)
        G = graph_func(X)
        T = char_func(G)
        results.append(T)
    return np.array(results)


def build_dist_graph(X, d):
    G = nx.Graph()
    n = len(X)
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            if abs(X[i] - X[j]) <= d:
                G.add_edge(i, j)
    return G


def chromatic_number(G):
    coloring = nx.coloring.greedy_color(G, strategy='largest_first')
    return max(coloring.values()) + 1