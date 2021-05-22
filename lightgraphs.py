"""
This module provides simple implementations of graphs and hypergraphs.

The intention is to give a lightweight, minimal structure which should allow fast construction of the objects.
"""
import scipy as sp
import scipy.sparse
import numpy as np
import hypernetx as hnx
import math


class LightHypergraph(object):
    """
    A light implementation of hypergraphs. Provides many methods shadowing the hypernetx interface so that we can
    sneakily re-use some code.
    """

    def __init__(self, edges):
        """
        Construct the hypergraph from a list of edges. Nodes should be zero-indexed. All edges are unweighted.

        Nodes are numbered sequentially from 0. Nodes and edges do not have names other than their indices.

        :param edges: a list of lists.
        """
        self.edges = edges

        self.num_edges = len(self.edges)
        self.num_vertices = max(max(self.edges, key=max)) + 1 if len(self.edges) > 0 else 0
        self.nodes = list(range(self.num_vertices))
        self.neighbours = {node: set() for node in self.nodes}

        self.degrees = [0] * self.num_vertices
        for edge in self.edges:
            for vertex in edge:
                self.neighbours[vertex] = self.neighbours[vertex].union(edge)
                self.neighbours[vertex].remove(vertex)
                self.degrees[vertex] += 1

        self.inv_degrees = list(map(lambda x: 1 / x if x > 0 else 0, self.degrees))
        self.sqrt_degrees = list(map(math.sqrt, self.degrees))
        self.inv_sqrt_degrees = list(map(lambda x: 1 / x if x > 0 else 0, self.sqrt_degrees))

    def degree(self, vertex):
        return self.degrees[vertex]

    def average_rank(self):
        return np.mean(list(map(len, self.edges)))

    def number_of_nodes(self):
        return self.num_vertices

    def number_of_edges(self):
        return self.num_edges

    def to_hypernetx(self):
        """Return a hypernetx graph which is equivalent to this hypergraph."""
        return hnx.Hypergraph(self.edges)

    def induced_hypergraph(self, node_list, remove_edges=False):
        """
        Construct the hypergraph induced by the given nodes. The node indices in the induced graph will equal their
        index in the given list.

        :param node_list:
        :param remove_edges: Whether to remove edges containing any of the removed nodes completely from the hypergraph
        :return:
        """
        # The given list should have unique elements
        assert(len(set(node_list)) == len(node_list))
        node_list_set = set(node_list)

        # In order to construct the induced graph, we need to construct a list of edges in the new hypergraph.
        new_edges = []
        for edge in self.edges:
            # We keep the subset of each edge which is incident on the new vertex set
            if not remove_edges and len(node_list_set.intersection(edge)) > 1:
                new_edges.append([node_list.index(v) for v in node_list_set.intersection(edge)])

            # Otherwise, we keep only edges entirely inside the remaining node list
            elif remove_edges and len(node_list_set.intersection(edge)) == len(edge):
                new_edges.append([node_list.index(v) for v in edge])

        # Construct the new hypergraph and return it
        return LightHypergraph(new_edges)


class LightGraph(object):
    """What is a graph, but an adjacency matrix?"""

    def __init__(self, adj_mat):
        """Initialise the graph with an adjacency matrix. This should be a sparse scipy matrix."""
        self.adj_mat = adj_mat
        self.lil_adj_mat = adj_mat.tolil()

        self.degrees = adj_mat.sum(axis=0).tolist()[0]
        self.inv_degrees = list(map(lambda x: 1 / x if x != 0 else 0, self.degrees))
        self.sqrt_degrees = list(map(math.sqrt, self.degrees))
        self.inv_sqrt_degrees = list(map(lambda x: 1 / x if x > 0 else 0, self.sqrt_degrees))

        self.num_vertices = self.adj_mat.shape[0]
        self.num_edges = sum(self.degrees) / 2

    def degree_matrix(self):
        return sp.sparse.spdiags(self.degrees, [0], self.num_vertices, self.num_vertices, format="csr")

    def inverse_degree_matrix(self):
        return sp.sparse.spdiags(self.inv_degrees, [0], self.num_vertices, self.num_vertices, format="csr")

    def inverse_sqrt_degree_matrix(self):
        return sp.sparse.spdiags(self.inv_sqrt_degrees, [0], self.num_vertices, self.num_vertices, format="csr")

    def volume(self, vertex_set):
        return sum([self.degrees[v] for v in vertex_set])

    def bipartiteness(self, vertex_set_l, vertex_set_r):
        w_l_r = self.lil_adj_mat[vertex_set_l][:, vertex_set_r].sum()
        return 1 - 2 * w_l_r / self.volume(vertex_set_l + vertex_set_r)
