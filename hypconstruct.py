"""
Various methods for constructing hypergraphs.
"""
import hypernetx as hnx
import random


def construct_low_conductance_hypergraph(n1, n2, m, r, p1, p2):
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
    nodes_a = ['a' + str(i) for i in range(1, n1 + 1)]
    nodes_b = ['b' + str(i) for i in range(1, n2 + 1)]

    # Seed the RNG
    random.seed()

    # Store the edge information for constructing the graph
    hyp_dict = {}
    for e in edges:
        # Decide whether this edges will cross the cut
        if random.random() < p1:
            # This edge may cross the cut
            hyp_dict[e] = random.sample(nodes_a + nodes_b, r)
        else:
            # Decide whether it should be on the left or the right side of the cut
            if random.random() < p2:
                hyp_dict[e] = random.sample(nodes_a, r)
            else:
                hyp_dict[e] = random.sample(nodes_b, r)

    # Return the final hypergraph
    print(f"Constructed hypergraph: {hyp_dict}")
    return hnx.Hypergraph(hyp_dict)


def construct_random_hypergraph(n, m, r):
    """
    Construct a completely random hypergraph.
    :param n: The number of nodes
    :param m: The number of edges
    :param r: The rank of each edge
    :return:
    """
    return construct_low_conductance_hypergraph(n, 0, m, r, 0, 1)


def construct_2_colorable_hypergraph(n1, n2, m, r, attempt_limit=100):
    """
    Construct a 2-colorable hypergraph. Guarantees that the graph is connected.
    :param n1: The number of vertices on side A of the cut
    :param n2: The number of vertices on side B of the cut
    :param m: The number of edges in the graph
    :param r: The rank of each edge
    :param attempt_limit: How many times to try constructing a graph before giving up
    :return: The hypergraph object constructed.
    """
    # Seed the RNG
    random.seed()

    # Try until we are connected
    connected = False
    new_hypergraph = None
    attempts = 0
    while not connected:
        attempts += 1
        if attempt_limit >= attempts > 1:
            print(f"Failed to create a connected graph. Attempts: {attempts - 1}.")
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
        i = 0   # Edge must contain node i
        for e in edges:
            # Add one node from each side of the cut to begin with
            new_edge = []
            if i < n1:
                new_edge.append(nodes_a[i])
                new_edge.append(random.choice(nodes_b))
            else:
                new_edge.append(nodes_b[i - n1])
                new_edge.append(random.choice(nodes_a))

            # Now, add remaining vertices from anywhere.
            new_edge = new_edge + random.sample([v for v in (nodes_a + nodes_b) if v not in new_edge], r - 2)
            hyp_dict[e] = new_edge

            # Next iteration must include the next vertex
            i += 1
            if i >= n1 + n2:
                i = 0

        new_hypergraph = hnx.Hypergraph(hyp_dict)
        connected = new_hypergraph.is_connected() and len(new_hypergraph.nodes) == (n1 + n2)

    # Return the final hypergraph
    return new_hypergraph


def simple_two_edge_hypergraph():
    """
    Construct and return a simple two-edge hypergraph.
    :return:
    """
    return hnx.Hypergraph({'e1': [1, 2, 3], 'e2': [2, 3]})


def simple_not_two_colorable_hypergraph():
    """
    Construct and return the simplest 3-uniform hypergraph that is not two colorable.
    :return:
    """
    new_hypergraph = hnx.Hypergraph(
        {
            'e1': [1, 2, 3],
            'e2': [1, 2, 4],
            'e3': [1, 2, 5],
            'e4': [3, 4, 5],
            'e5': [2, 6, 7],
            'e6': [2, 6, 8],
            'e7': [2, 6, 9],
            'e8': [7, 8, 9],
            'e9': [6, 1, 10],
            'e10': [6, 1, 11],
            'e11': [6, 1, 12],
            'e12': [10, 11, 12]
        })
    return new_hypergraph
