"""
This file implements various methods for studying the spectral properties of hypergraphs and their related graphs.
"""
import networkx as nx
import numpy as np
import hyplogging


def networkx_directed_cut_size(graph, vertex_set_l, vertex_set_r=None):
    """
    Compute the cut size between sets S and T in a directed networkx graph.

    If T is not given, assume it is the complement of S.

    :param graph: The graph as a networkx object
    :param vertex_set_l: The vertex set S
    :param vertex_set_r: The vertex set T, or None
    :return: The size of cut(S, T)
    """
    edges = nx.edge_boundary(graph, vertex_set_l, vertex_set_r, data="weight", default=1)
    return sum(weight for u, v, weight in edges)


def networkx_volume_in(graph, vertex_set):
    """
    Compute the 'volume' of in-degrees to the given vertex set in a directed graph. Closely based on the networkx code
    for ordinary volume.

    :param graph:
    :param vertex_set:
    :return: the 'in-volume' of the given set.
    """
    degree = graph.in_degree
    return sum(d for v, d in degree(vertex_set, weight="weight"))


def clsz_cut_imbalance(graph, vertex_set_l, vertex_set_r):
    """
    Compute the cut imbalance between the sets L and R as specified in CLSZ. This is defined to be

        CI(S, T) = 1/2 * abs[ (w(S, T) - w(T, S)) / (w(S, T) + w(T, S)) ]

    :param graph: the directed graph on which to operate
    :param vertex_set_l: the set S
    :param vertex_set_r: the set T
    :return: the cut imbalance ratio CI(S, T)
    """
    hyplogging.logger.debug("Computing cut imbalance.")
    hyplogging.logger.debug(f"    Left set size: {len(vertex_set_l)}")
    hyplogging.logger.debug(f"   Right set size: {len(vertex_set_r)}")
    w_s_t = networkx_directed_cut_size(graph, vertex_set_l, vertex_set_r)
    w_t_s = networkx_directed_cut_size(graph, vertex_set_r, vertex_set_l)
    return (1/2) * abs((w_s_t - w_t_s) / (w_s_t + w_t_s))


def ms_flow_ratio(graph, vertex_set_l, vertex_set_r):
    """
    Compute the flow ratio between the sets L and R as specified by Macgregor and Sun. This is defined to be

        FR(S, T) = 1 - 2 w(S, T) / (vol_out(S) + vol_in(T))

    :param graph:
    :param vertex_set_l:
    :param vertex_set_r:
    :return: the flow ratio FR(S, T)
    """
    hyplogging.logger.debug("Computing flow ratio.")
    hyplogging.logger.debug(f"    Left set size: {len(vertex_set_l)}")
    hyplogging.logger.debug(f"   Right set size: {len(vertex_set_r)}")
    w_s_t = networkx_directed_cut_size(graph, vertex_set_l, vertex_set_r)
    vol_out_s = nx.volume(graph, vertex_set_l, weight="weight")
    vol_in_t = networkx_volume_in(graph, vertex_set_r)
    return 1 - 2 * w_s_t / (vol_out_s + vol_in_t)


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
    vertex_set = set(vertex_set)
    if not complement:
        return sum(hypergraph.degrees[v] for v in vertex_set)
    else:
        return sum(hypergraph.degrees[v] for v in hypergraph.nodes if v not in vertex_set)


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


def hypergraph_bipartiteness(hypergraph, vertex_set_l, vertex_set_r):
    """
    Given a hypergraph H and a set of vertices (L, R), compute the bipartiteness of the set L, R given by
        beta(L, R) = [ w(L | notL) + w(R | notR) + w(L | R) + w(R | L) ] / vol(L union R)
    :param hypergraph: the hypergraph on which to operate
    :param vertex_set_l: the vertex set L
    :param vertex_set_r: the vertex set R
    :return: the bipartiteness beta(L, R)
    """
    hyplogging.logger.debug("Computing hypergraph bipartiteness.")
    hyplogging.logger.debug(f"    Left set size: {len(vertex_set_l)}")
    hyplogging.logger.debug(f"   Right set size: {len(vertex_set_r)}")
    vol_s = hypergraph_volume(hypergraph, vertex_set_l + vertex_set_r)

    vertex_set_l = set(vertex_set_l)
    vertex_set_r = set(vertex_set_r)

    w_l_not_l = 0
    w_r_not_r = 0
    w_l_r = 0
    w_r_l = 0
    for edge in hypergraph.edges:
        edge_set = set(edge)
        edge_l_intersection = len(vertex_set_l.intersection(edge_set))
        edge_r_intersection = len(vertex_set_r.intersection(edge_set))
        edge_entirely_inside_l = edge_l_intersection == len(edge_set)
        edge_entirely_inside_r = edge_r_intersection == len(edge_set)

        if edge_entirely_inside_l:
            w_l_not_l += 1
        if edge_entirely_inside_r:
            w_r_not_r += 1
        if edge_l_intersection > 0 and not edge_r_intersection > 0:
            w_l_r += 1
        if edge_r_intersection > 0 and not edge_r_intersection > 0:
            w_r_l += 1

    # Compute the bipartiteness
    return (w_l_not_l + w_r_not_r + w_l_r + w_r_l) / vol_s


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


def hypergraph_two_sided_sweep(x, hypergraph):
    """
    Perform the two-sided sweep set procedure for a hypergraph in order to find a set with low bipartiteness.
    :param x: The vector to sweep over
    :param hypergraph: The underlying hypergraph, as a hypernetx object
    :return: L, R - the pair of sets with the smallest bipartiteness found by sweeping over the vector x
    """
    # Get useful data about the hypergraph that we will use repeatedly in the algorithm
    all_vertices = hypergraph.nodes
    all_edges = hypergraph.edges
    dict_vertices_to_adjacent_edges = {v: [] for v in all_vertices}
    hyplogging.logger.debug("Constructing dictionary of adjacent edges.")
    for edge in all_edges:
        for vertex in edge:
            dict_vertices_to_adjacent_edges[vertex].append(set(edge))

    best_l = set()
    best_r = set()
    best_bipartiteness = 1
    current_l = set()
    current_r = set()

    # We will update the computation of the bipartiteness as we go along.
    current_vol = 0
    current_numerator = 0

    # Get the sorted indices of x, in order of highest absolute value to lowest
    hyplogging.logger.debug("Sorting the vertices according to given vector.")
    ordering = reversed(np.argsort(abs(x)))

    # Perform the sweep
    hyplogging.logger.debug("Checking each sweep set.")
    for i, vertex_index in enumerate(ordering):
        if i % 1000 == 0:
            hyplogging.logger.debug(f"Checking sweep set number {i}/{hypergraph.num_vertices}.")

        # Add the next vertex to the candidate set
        added_to_l = False
        if x[vertex_index] >= 0:
            current_r.add(all_vertices[vertex_index])
        else:
            added_to_l = True
            current_l.add(all_vertices[vertex_index])

        # Update the bipartiteness values
        current_vol += hypergraph.degree(all_vertices[vertex_index])
        for edge in dict_vertices_to_adjacent_edges[all_vertices[vertex_index]]:
            edge_l_intersection = len(current_l.intersection(edge))
            edge_r_intersection = len(current_r.intersection(edge))
            edge_entirely_inside_l = edge_l_intersection == len(edge)
            edge_entirely_inside_r = edge_r_intersection == len(edge)

            if edge_entirely_inside_l:
                current_numerator += 1
            if edge_entirely_inside_r:
                current_numerator += 1
            if edge_l_intersection > 0 and edge_r_intersection == 0:
                current_numerator += 1
            if edge_r_intersection > 0 and edge_l_intersection == 0:
                current_numerator += 1

            # Remove 1 from the numerator if we were already counting one.
            if added_to_l:
                if edge_l_intersection >= 2 and edge_r_intersection == 0:
                    current_numerator -= 1
                if edge_l_intersection == 1 and edge_r_intersection > 0:
                    current_numerator -= 1
            else:
                if edge_r_intersection >= 2 and edge_l_intersection == 0:
                    current_numerator -= 1
                if edge_r_intersection == 1 and edge_l_intersection > 0:
                    current_numerator -= 1

        # Get the bipartiteness and check if it is best so far
        beta = current_numerator / current_vol if current_vol != 0 else 1
        if beta < best_bipartiteness:
            best_bipartiteness = beta
            best_l = current_l.copy()
            best_r = current_r.copy()

    hyplogging.logger.debug(f"Best bipartiteness in sweep set: {best_bipartiteness}")
    return list(best_l), list(best_r)


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
    cut_s = nx.cut_size(graph, vertex_set)

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
