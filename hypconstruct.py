"""
Various methods for constructing hypergraphs.
"""
import hypernetx as hnx
import numpy as np
import numpy.random
import scipy.special
import random
import math
import hyplogging


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


def hypergraph_sbm_two_cluster(filename, n, r, p, q):
    """
    Generate a hypergraph from the hypergraph SBM and save the edgelist to the file specified.

    Generates a hypergraph with two clusters of size n. Every possible rank r edge inside each cluster is added with
    probability p. Every possible rank r edge between the two clusters is added with probability q.

    :param filename:
    :param n:
    :param p:
    :param q:
    :return:
    """
    hyplogging.logger.info(f"Generating two cluster SBM hypergraph.")
    hyplogging.logger.info(f"  filename = {filename}")
    hyplogging.logger.info(f"         n = {n}")
    hyplogging.logger.info(f"         r = {r}")
    hyplogging.logger.info(f"         p = {p}")
    hyplogging.logger.info(f"         q = {q}")

    with open(filename, 'w') as edgelist_out:
        # Start by generating the edges inside each cluster.
        # Work out how many of them there will be - this is a number drawn from the binomial distribution
        possible_edges_in_each_cluster = scipy.special.comb(n, r)
        num_edges_in_left_cluster = np.random.binomial(possible_edges_in_each_cluster, p)
        num_edges_in_right_cluster = np.random.binomial(possible_edges_in_each_cluster, p)
        hyplogging.logger.debug(f"Possible edges in each cluster: {possible_edges_in_each_cluster}")
        hyplogging.logger.debug(f"Edges in left cluster: {num_edges_in_left_cluster}")
        hyplogging.logger.debug(f"Edges in right cluster: {num_edges_in_right_cluster}")

        # Generate these edges
        for _ in range(num_edges_in_left_cluster):
            edge = random.sample(range(n), r)
            edgelist_out.write(' '.join(map(str, edge)))
            edgelist_out.write('\n')
        for _ in range(num_edges_in_right_cluster):
            edge = [v + n for v in random.sample(range(n), r)]
            edgelist_out.write(' '.join(map(str, edge)))
            edgelist_out.write('\n')

        # Now, generate the edges between the clusters.
        # We will consider each possible split between left and right clusters individually
        for r_prime in range(1, r):
            hyplogging.logger.debug(f"Considering r_prime = {r_prime}.")

            # Get the number of possible edges with this split.
            possible_edges_between_clusters = scipy.special.comb(n, r_prime) * scipy.special.comb(n, r - r_prime)
            num_edges_between_clusters = np.random.binomial(possible_edges_between_clusters, q)
            hyplogging.logger.debug(f"Possible edges between clusters: {possible_edges_between_clusters}")
            hyplogging.logger.debug(f"Edges between clusters: {num_edges_between_clusters}")

            # Generate the edges
            for _ in range(num_edges_between_clusters):
                edge = random.sample(range(n), r_prime) + [v + n for v in random.sample(range(n), r - r_prime)]
                edgelist_out.write(' '.join(map(str, edge)))
                edgelist_out.write('\n')


if __name__ == "__main__":
    n = 100
    r = 4
    p = 0.001
    q = 0.001
    hypergraph_sbm_two_cluster(f"data/sbm/two_cluster_sbm_{n}_{r}_{p}_{q}.edgelist", n, r, p, q)
