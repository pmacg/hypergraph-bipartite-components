"""
Construct some hypergraphs with certain properties.
"""
import hypernetx as hnx
import random


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
    return hnx.Hypergraph(hyp_dict)


