
import pytest
import numpy as np
from functions import (
    sample_normal, sample_t, sample_exp, sample_gamma,
    build_knn_graph, build_dist_graph,
    max_degree, min_degree, count_components,
    count_articulation_points, count_triangles,
    max_independent_set_size, domination_number,
    clique_number, chromatic_number, clique_cover_number
)

def test_sample_normal_shape():
    X = sample_normal(100, 1.0)
    assert X.shape == (100,)  # <== было (100, 2)

def test_sample_t_shape():
    X = sample_t(100, 3.0)
    assert X.shape == (100,)

def test_sample_exp_shape():
    X = sample_exp(100, 1.0)
    assert X.shape == (100,)

def test_sample_gamma_shape():
    X = sample_gamma(100, 0.5, 0.5)
    assert X.shape == (100,)

def test_build_knn_graph_runs():
    X = sample_normal(30, 1.0)
    G = build_knn_graph(X, 5)
    assert len(G.nodes) == 30

def test_build_dist_graph_runs():
    X = sample_normal(30, 1.0)
    G = build_dist_graph(X, 0.3)
    assert len(G.nodes) == 30

def test_graph_characteristics_run():
    X = sample_normal(30, 1.0)
    G = build_knn_graph(X, 5)
    assert isinstance(max_degree(G), int)
    assert isinstance(min_degree(G), int)
    assert isinstance(count_components(G), int)
    assert isinstance(count_articulation_points(G), int)
    assert isinstance(count_triangles(G), int)
    assert isinstance(max_independent_set_size(G), int)
    assert isinstance(domination_number(G), int)
    assert isinstance(clique_number(G), int)
    assert isinstance(chromatic_number(G), int)
    assert isinstance(clique_cover_number(G), int)
