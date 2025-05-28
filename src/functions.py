import numpy as np
import networkx as nx
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, ks_2samp
from sklearn.metrics import roc_auc_score, roc_curve
import pandas as pd
from networkx.algorithms import approximation as approx


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



def max_degree(G):
    return max(dict(G.degree()).values())

def min_degree(G):
    return min(dict(G.degree()).values())

def count_components(G):
    return nx.number_connected_components(G)

def count_articulation_points(G):
    return len(list(nx.articulation_points(G)))

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

def clique_number(G):
    return len(max(nx.find_cliques(G), key=len))

def max_independent_set_size(G):
    indep_set = nx.algorithms.approximation.maximum_independent_set(G)
    return len(indep_set)

def domination_number(G):
    dom_set = nx.algorithms.approximation.min_weighted_dominating_set(G)
    return len(dom_set)

from networkx.algorithms.approximation.clique import clique_removal

def clique_cover_number(G):
    complement = nx.complement(G)
    independent_sets, _ = clique_removal(complement)
    return len(independent_sets)

import pandas as pd

def generate_dataset(n, d, dist_H0, args_H0, dist_H1, args_H1, 
                     feature_funcs, graph_func, n_sim=300):
    X = []
    y = []

    for _ in range(n_sim):
        data = dist_H0(n, *args_H0)
        G = graph_func(data)
        features = [f(G) for f in feature_funcs]
        X.append(features)
        y.append(0)

    for _ in range(n_sim):
        data = dist_H1(n, *args_H1)
        G = graph_func(data)
        features = [f(G) for f in feature_funcs]
        X.append(features)
        y.append(1)

    df = pd.DataFrame(X, columns=[f.__name__ for f in feature_funcs])
    df["label"] = y
    return df