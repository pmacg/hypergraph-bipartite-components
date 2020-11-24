"""
This file implements various methods for studying the spectral properties of hypergraphs and their related graphs.
"""
import networkx as nx
import numpy as np
import random


def graph_cutsize(G, S):
    """
    Compute the cut size of a set S in a graph
    :param G: The graph
    :param S: The vertex set in question
    :return: The size of cut(S, V \ S)
    """
    cutsize = 0

    # Iterate through the edges to find those in the cut
    for e in G.edges():
        e_intersect_S = [u for u in e if u in S]
        e_minus_S = [u for u in e if u not in S]

        # If this edge is on the cut, add one to the total
        if len(e_intersect_S) > 0 and len(e_minus_S) > 0:
            cutsize += 1

    return cutsize


def hyp_cutsize(H, S):
    """
    Compute the cut size of a set S in a hypergraph.
    :param H: The hypergraph
    :param S: The vertex set in question
    :return: The size of cut(S, V \ S)
    """
    cutsize = 0

    # Iterate through the edges to find those in the cut
    for e in H.edges():
        e_intersect_S = [u for u in e.elements if u in S]
        e_minus_S = [u for u in e.elements if u not in S]

        # If this edge is on the cut, add one to the total
        if len(e_intersect_S) > 0 and len(e_minus_S) > 0:
            cutsize += 1

    return cutsize


def hyp_volume(H, S, complement=False):
    """
    Compute the volume of a set S in a hypergraph.
    :param H: The hypergraph
    :param S: The vertex set
    :param complement: Whether to find the volume of the set complement instead
    :return: vol(S)
    """
    total_vol = 0
    for v in H.nodes:
        if v in S and not complement:
            total_vol += H.degree(v)
        if v not in S and complement:
            total_vol += H.degree(v)

    return total_vol


def hyp_conductance(H, S):
    """
    Given a hypergraph H and a set of vertices S, compute the conductance of the set S given by
        phi(S) = cut(S, V \ S) / vol(S)
    :param H: The hypergraph
    :param S: A subset of the vertices
    :return: The conductance of the set S in the graph H
    """
    cut = hyp_cutsize(H, S)
    vol_S = hyp_volume(H, S)
    vol_Sbar = hyp_volume(H, S, complement=True)
    if min(vol_S, vol_Sbar) > 0:
        return cut / min(vol_S, vol_Sbar)
    else:
        return 1


def hyp_sweep_set(x, H, debug=False):
    """
    Perform a sweep set procedure for a hypergraph in order to find a cut with low conductance.
    :param x: The vector to sweep over
    :param H: The underlying hypergraph
    :return: The set with the smallest conductance found by sweeping over the vector x.
    """
    V = [v.uid for v in H.nodes()]
    best_S = []
    best_cond = 1
    current_S = []

    # Get the sorted indices of x, in order highest to lowest
    ordering = reversed(np.argsort(x))

    # Perform the sweep
    for vidx in ordering:
        # Add the next vertex to the candidate set S
        current_S.append(V[vidx])
        if debug:
            print("Checking sweep set: ", current_S)
        phi = hyp_conductance(H, current_S)
        if debug:
            print("Conductance:", phi)
        if phi < best_cond:
            if debug:
                print("Updating best conductance.")
            best_cond = phi
            best_S = np.copy(current_S)

    # Return the best set we found
    return best_S


def hyp_common_edges(u, v, H):
    """
    Return the number of common edges between the vertices u and v in the hypergraph H.
    :param u: A vertex
    :param v: A vertex
    :param H: The hypergraph
    :return: The number of edges in H containing both u and v
    """
    total = 0
    for e in H.edges():
        if u in e.elements and v in e.elements:
            total += 1
    return total


def hyp_min_edges_graph(H):
    """
    Given a hypergraph, construct a graph by replacing each hyperedge with a single edge joining the vertices with the
    fewest overlapping edges.
    :param H: The hypergrpah in question
    :return:
    """
    G = nx.Graph()

    # Add the vertices
    n = 0
    for v in H.nodes:
        n += 1
        G.add_node(v)

    # Add the edges
    for e in H.edges():
        fewest = None
        edge = None
        for u in e.elements:
            for v in e.elements:
                if u != v:
                    if fewest is None or hyp_common_edges(u, v, H) < fewest:
                        fewest = hyp_common_edges(u, v, H)
                        edge = (u, v)
        G.add_edge(edge[0], edge[1])

    return G


def graph_self_loop_conductance(S, G, H):
    """
    Given a graph, assuming that each vertex is given a self loop in order to give the vertex the same degree as it has
    in H, return the conductance of a set.
    :param S: The vertex set to find the conductance of
    :param G: The simple graph
    :param H: The hypergraph
    :return: The conductance of the set S in G assuming each vertex has appropriate self-loops
    """
    vol_S = hyp_volume(H, S)
    vol_Sbar = hyp_volume(H, S, complement=True)
    cut_S = graph_cutsize(G, S)

    if min(vol_S, vol_Sbar) == 0:
        return 1
    else:
        return cut_S / min(vol_S, vol_Sbar)


def hyp_weighted_degree(v, H):
    """
    Compute a 'weighted' degree for v in the hypergraph H.
    d_v(e) = w(e) / r(e)
    :param v: the vertex of interest
    :param H: The underlying hypergraph
    :return: the degree of the vertex
    """
    total_degree = 0
    for e in H.edges():
        if v in e.elements:
            total_degree += (1 / len(e.elements))
    return total_degree


def hyp_degree_graph(H, c=1, debug=False):
    """
    Construct a simple graph by sampling based on vertex degrees.
    :param H: The hypergraph
    :param c: A parameter to the random process. Roughly the number of vertices to select a single hyperedge.
    :return: A graph G
    """
    G = nx.Graph()
    m = len([e for e in H.edges()])
    n = len([v for v in H.nodes()])

    random.seed()

    # Add the vertices
    i = 0
    for v in H.nodes:
        i += 1
        G.add_node(v)

    # Add the edges
    for e in H.edges():
        # For each vertex in e, check whether the vertex randomly selects e.
        chosen_vertices = {}
        for u in e.elements:
            # The probability of choosing this edge is
            p = c * (m / n) * (1 / len(e.elements)) * (1 / hyp_weighted_degree(u, H))
            if random.random() <= p:
                chosen_vertices[u] = p

        if debug:
            print(f"Vertices which chose edge {e}: {[v for v in chosen_vertices.keys()]}")

        # Add a clique on the chosen vertices
        processed = []
        for u, pu in chosen_vertices.items():
            processed.append(u)
            for v, pv in chosen_vertices.items():
                if v not in processed:
                    G.add_edge(u, v, weight=(pu + pv - (pu * pv)))

    return G
