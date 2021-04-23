"""
This module provides simple implementations of graphs and hypergraphs.

The intention is to give a lightweight, minimal structure which should allow fast construction of the objects.
"""
import scipy as sp
import scipy.sparse
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
        self.num_vertices = max(max(self.edges, key=max)) + 1
        self.nodes = list(range(self.num_vertices))

        self.degrees = [0] * self.num_vertices
        for edge in self.edges:
            for vertex in edge:
                self.degrees[vertex] += 1

        self.inv_degrees = list(map(lambda x: 1 / x if x > 0 else 0, self.degrees))
        self.sqrt_degrees = list(map(math.sqrt, self.degrees))
        self.inv_sqrt_degrees = list(map(lambda x: 1 / x if x > 0 else 0, self.sqrt_degrees))

    def degree(self, vertex):
        return self.degrees[vertex]

    def number_of_nodes(self):
        return self.num_vertices

    def number_of_edges(self):
        return self.num_edges

    def induced_hypergraph(self, node_list):
        """
        Construct the hypergraph induced by the given nodes. The node indices in the induced graph will equal their
        index in the given list.

        :param node_list:
        :return:
        """
        # The given list should have unique elements
        assert(len(set(node_list)) == len(node_list))
        node_list_set = set(node_list)

        # In order to construct the induced graph, we need to construct a list of edges in the new hypergraph.
        new_edges = []
        for edge in self.edges:
            if len(node_list_set.intersection(edge)) == len(edge):
                # This edge should be included in the induced hypergraph.
                new_edges.append([node_list.index(v) for v in edge])

        # Construct the new hypergraph and return it
        return LightHypergraph(new_edges)


class LightGraph(object):
    """What is a graph, but an adjacency matrix?"""

    def __init__(self, adj_mat):
        """Initialise the graph with an adjacency matrix. This should be a sparse scipy matrix."""
        self.adj_mat = adj_mat

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
