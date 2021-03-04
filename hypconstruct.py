"""
Construct some hypergraphs with certain properties.
"""
import hypernetx as hnx
import networkx as nx
import random
import math


def construct_hyp_low_cond(n1, n2, m, r, p1, p2):
    """
    Construct a hypergraph with a low conductance cut.
    :param n1: The number of vertices on side A of the cut
    :param n2: The number of vertices on side B of the cut
    :param m: The number of edges in the graph
    :param r: The rank of each edge
    :param p1: The probability that a given edge 'may' cross the cut
    :param p2: Given that an edge doesn't cross the cut, the probability that it is on the left
    :return: The hypergraph object constructed.
    """
    # Create a list of edge and node names
    edges = ['e' + str(i) for i in range(1, m+1)]
    nodesA = ['a' + str(i) for i in range(1, n1 + 1)]
    nodesB = ['b' + str(i) for i in range(1, n2 + 1)]

    # Seed the RNG
    random.seed()

    # Store the edge information for constructing the graph
    hyp_dict = {}
    for e in edges:
        # Decide whether this edges will cross the cut
        if random.random() < p1:
            # This edge may cross the cut
            hyp_dict[e] = random.sample(nodesA + nodesB, r)
        else:
            # Decide whether it should be on the left or the right side of the cut
            if random.random() < p2:
                hyp_dict[e] = random.sample(nodesA, r)
            else:
                hyp_dict[e] = random.sample(nodesB, r)

    # Return the final hypergraph
    print(f"Constructed hypergraph: {hyp_dict}")
    return hnx.Hypergraph(hyp_dict)


def random_hypergraph(n, m, r):
    """
    Construct a completely random hypergraph.
    :param n: The number of nodes
    :param m: The number of edges
    :param r: The rank of each edge
    :return:
    """
    return construct_hyp_low_cond(n, 0, m, r, 0, 1)


def construct_hyp_2_colorable(n1, n2, m, r, attempt_limit=100):
    """
    Construct a hypergraph with a low conductance cut. Guarantees that the graph is connected.
    :param n1: The number of vertices on side A of the cut
    :param n2: The number of vertices on side B of the cut
    :param m: The number of edges in the graph
    :param r: Half the rank of each edge
    :param attempt_limit: How many times to try constructing a graph before giving up
    :return: The hypergraph object constructed.
    """
    # Seed the RNG
    random.seed()

    # Try until we are connected
    connected = False
    hyp_dict = None
    h = None
    attempts = 0
    while not connected:
        attempts += 1
        if attempts > attempt_limit:
            # We've failed to make a connected graph.
            print("WARNING: failed to make a connected graph. Try increasing the number or rank of the edges.")
            raise ValueError("m, r, or attempt_limit too small")

        # Create a list of edge and node names
        edges = ['e' + str(i) for i in range(1, m+1)]
        nodes_a = ['a' + str(i) for i in range(1, n1 + 1)]
        nodes_b = ['b' + str(i) for i in range(1, n2 + 1)]

        # Store the edge information for constructing the graph
        hyp_dict = {}
        for e in edges:
            side_a_nodes = random.sample(nodes_a, r)
            side_b_nodes = random.sample(nodes_b, r)
            hyp_dict[e] = side_a_nodes + side_b_nodes

        h = hnx.Hypergraph(hyp_dict)
        connected = h.is_connected() and len(h.nodes) == (n1 + n2)

    # Return the final hypergraph
    #print(f"Constructed hypergraph: {hyp_dict}")
    return h


def get_clique_graph(H):
    """
    Given a hypergraph, H, return the networkx graph corresponding to the clique graph of H.
    The clique graph is constructed by replacing each hyperedge e with a clique with edges of weight 1/(r(e) - 1).
    :param H:
    :return: A networkx graph G
    """
    G = nx.Graph()

    # Add the vertices to the graph
    for vertex in H.nodes:
        G.add_node(vertex)

    # Add the edges to the graph
    new_edges = {}
    for edge in H.edges():
        vertices = [vertex for vertex in edge]
        rank = len(vertices)
        for v1_idx in range(rank):
            for v2_idx in range(v1_idx + 1, rank):
                new_edge = (vertices[v1_idx], vertices[v2_idx])
                new_weight = 1 / (rank - 1)
                if new_edge in new_edges:
                    new_edges[new_edge] = new_edges[new_edge] + new_weight
                else:
                    new_edges[new_edge] = new_weight
    new_edges_list = []
    for new_edge in new_edges:
        new_edges_list.append((new_edge[0], new_edge[1], {'weight': new_edges[new_edge]}))
    G.add_edges_from(new_edges_list)

    return G
