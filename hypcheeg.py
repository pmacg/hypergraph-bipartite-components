"""
This file implements various methods for studying the spectral properties of hypergraphs and their related graphs.
"""
import numpy as np


def graph_cut_size(graph, vertex_set):
    """
    Compute the cut size of a set S in a graph
    :param graph: The graph as a networkx object
    :param vertex_set: The vertex set in question, as a list
    :return: The size of cut(S, V - S)
    """
    cut_size = 0

    # Iterate through the edges to find those in the cut
    for edge in graph.edges():
        edge_intersects_vertex_set = len([v for v in edge if v in vertex_set]) > 0
        edge_entirely_inside_vertex_set = len([v for v in edge if v not in vertex_set]) == 0

        # If this edge is on the cut, add one to the total
        if edge_intersects_vertex_set and not edge_entirely_inside_vertex_set:
            cut_size += 1

    return cut_size


def hypergraph_cut_size(hypergraph, vertex_set):
    """
    Compute the cut size of a set S in a hypergraph.
    :param hypergraph: The hypergraph as a hypernetx object.
    :param vertex_set: The vertex set in question, as a list
    :return: The size of cut(S, V - S)
    """
    cut_size = 0

    # Iterate through the edges to find those in the cut
    for edge in hypergraph.edges():
        edge_intersects_vertex_set = len([v for v in edge.elements if v in vertex_set]) > 0
        edge_entirely_inside_vertex_set = len([v for v in edge.elements if v not in vertex_set]) == 0

        # If this edge is on the cut, add one to the total
        if edge_intersects_vertex_set and not edge_entirely_inside_vertex_set:
            cut_size += 1

    return cut_size


def hypergraph_volume(hypergraph, vertex_set, complement=False):
    """
    Compute the volume of a set S in a hypergraph.
    :param hypergraph: The hypergraph as a hypernetx object
    :param vertex_set: The vertex set as a list
    :param complement: Whether to find the volume of the set complement instead
    :return: vol(S)
    """
    total_vol = 0
    for vertex in hypergraph.nodes:
        if vertex in vertex_set and not complement:
            total_vol += hypergraph.degree(vertex)
        if vertex not in vertex_set and complement:
            total_vol += hypergraph.degree(vertex)

    return total_vol


def hypergraph_conductance(hypergraph, vertex_set):
    """
    Given a hypergraph H and a set of vertices S, compute the conductance of the set S given by
        phi(S) = cut(S, V - S) / vol(S)
    :param hypergraph: The hypergraph as a hypernetx object
    :param vertex_set: A subset of the vertices, as a list
    :return: The conductance of the set S in the graph H
    """
    cut = hypergraph_cut_size(hypergraph, vertex_set)
    vol_s = hypergraph_volume(hypergraph, vertex_set)
    vol_s_complement = hypergraph_volume(hypergraph, vertex_set, complement=True)
    if min(vol_s, vol_s_complement) > 0:
        return cut / min(vol_s, vol_s_complement)
    else:
        return 1


def hypergraph_sweep_set(x, hypergraph):
    """
    Perform a sweep set procedure for a hypergraph in order to find a cut with low conductance.
    :param x: The vector to sweep over
    :param hypergraph: The underlying hypergraph, as a hypernetx object
    :return: The set with the smallest conductance found by sweeping over the vector x.
    """
    all_vertices = [v.uid for v in hypergraph.nodes()]
    best_set = []
    best_conductance = 1
    current_set = []

    # Get the sorted indices of x, in order highest to lowest
    ordering = reversed(np.argsort(x))

    # Perform the sweep
    for vertex_index in ordering:
        # Add the next vertex to the candidate set S
        current_set.append(all_vertices[vertex_index])
        phi = hypergraph_conductance(hypergraph, current_set)
        if phi < best_conductance:
            best_conductance = phi
            best_set = np.copy(current_set)

    # Return the best set we found
    return best_set


def hypergraph_common_edges(u, v, hypergraph):
    """
    Return the number of common edges between the vertices u and v in the hypergraph H.
    :param u: A vertex
    :param v: A vertex
    :param hypergraph: The hypergraph
    :return: The number of edges in H containing both u and v
    """
    total = 0
    for e in hypergraph.edges():
        if u in e.elements and v in e.elements:
            total += 1
    return total


def graph_self_loop_conductance(vertex_set, graph, hypergraph):
    """
    Given a graph, assuming that each vertex is given a self loop in order to give the vertex the same degree as it has
    in H, return the conductance of a set.
    :param vertex_set: The vertex set to find the conductance of
    :param graph: The simple graph
    :param hypergraph: The hypergraph
    :return: The conductance of the set S in G assuming each vertex has appropriate self-loops
    """
    vol_s = hypergraph_volume(hypergraph, vertex_set)
    vol_s_complement = hypergraph_volume(hypergraph, vertex_set, complement=True)
    cut_s = graph_cut_size(graph, vertex_set)

    if min(vol_s, vol_s_complement) == 0:
        return 1
    else:
        return cut_s / min(vol_s, vol_s_complement)


def hypergraph_weighted_degree(vertex, hypergraph):
    """
    Compute a 'weighted' degree for v in the hypergraph H.
    d_v(e) = w(e) / r(e)
    :param vertex: the vertex of interest
    :param hypergraph: The underlying hypergraph
    :return: the degree of the vertex
    """
    total_degree = 0
    for e in hypergraph.edges():
        if vertex in e.elements:
            total_degree += (1 / len(e.elements))
    return total_degree
