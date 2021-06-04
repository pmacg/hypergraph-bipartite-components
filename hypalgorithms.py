"""This file contains complete implementations of various hypergraph algorithms."""
import numpy as np
import scipy as sp
import scipy.sparse.linalg
import hypcheeg
import hypjop
import hypreductions
import hyplogging
import lightgraphs


def _internal_bipartite_diffusion(starting_vector, hypergraph, max_time, step_size, approximate,
                                  construct_induced=False):
    """
    Internal method for running the bipartite diffusion algorithm.
    :param starting_vector: the starting vector of the diffusion process
    :param hypergraph: the hypergraph on which to run
    :param max_time: the maximum diffusion time
    :param step_size: the step size for the diffusion
    :param approximate: whether to run the approximate version of the diffusion process
    :param construct_induced: Whether to construct and report the induced graph at each step of the diffusion
    :return: the sets L and R, and their bipartiteness
    """
    # Compute the diffusion process until convergence
    measure_vector, _, _ = hypjop.sim_mc_heat_diff(starting_vector, hypergraph, max_time=max_time, min_step=step_size,
                                                   plot_diff=False, check_converged=True, approximate=approximate,
                                                   construct_induced=construct_induced)

    # Perform the sweep set algorithm on the measure vector to find the almost-bipartite set
    vertex_set_l, vertex_set_r = hypcheeg.hypergraph_two_sided_sweep(measure_vector, hypergraph)
    beta = hypcheeg.hypergraph_bipartiteness(hypergraph, vertex_set_l, vertex_set_r)

    return vertex_set_l, vertex_set_r, beta


def induced_graph_demo():
    """Produce some example induced graphs from the diffusion process."""
    edges = [[0, 1, 2], [1, 3, 4], [2, 3, 5], [3, 4, 5]]
    hypergraph = lightgraphs.LightHypergraph(edges)
    s = [1, 0, 0, 0, 0, 0]
    _internal_bipartite_diffusion(s, hypergraph, 100, 0.1, False, construct_induced=True)


def find_bipartite_set_diffusion(hypergraph, max_time=100, step_size=0.1, use_random_initialisation=False,
                                 approximate=True, construct_induced=False):
    """
    Given a hypergraph, use the diffusion process to find an almost bipartite set.
    :param hypergraph: The hypergraph on which to find a bipartite set
    :param max_time: The maximum time to run the diffusion process
    :param step_size: The step size to use for the diffusion
    :param use_random_initialisation: By default, we will use the eigenvector of the clique graph to initialise. If this
                                      parameter is true, then we will use a random vector to initialise the diffusion.
    :param approximate: Whether to use the approximate, no-LP version of the diffusion operator
    :param construct_induced: Whether to construct and report the induced graph at each step of the diffusion process
    :return: the sets L, and R, and the bipartiteness value beta(L, R)
    """
    # If the hypergraph does not contain any nodes, then just return empty sets
    if len(hypergraph.nodes) == 0:
        hyplogging.logger.debug("Hypergraph is empty, returning empty sets.")
        return [], [], 1

    if use_random_initialisation:
        # We will run the diffusion process 5 times, starting from a different random starting vector each time.
        best_bipartiteness = 1
        best_vertex_set_l = []
        best_vertex_set_r = []

        n = hypergraph.number_of_nodes()
        for i in range(5):
            # Compute a random starting vector for the diffusion process
            s = 2 * (np.random.randint(2, size=n) - 1/2)

            # Run the diffusion process
            this_vertex_set_l, this_vertex_set_r, this_bipartiteness = _internal_bipartite_diffusion(
                s, hypergraph, max_time, step_size, approximate, construct_induced=construct_induced)

            # Check if this is the best one so far
            if this_bipartiteness < best_bipartiteness:
                best_bipartiteness = this_bipartiteness
                best_vertex_set_l = this_vertex_set_l
                best_vertex_set_r = this_vertex_set_r

        # Return the best set
        return best_vertex_set_l, best_vertex_set_r, best_bipartiteness
    else:
        # Construct the clique graph from the hypergraph
        weighted_clique_graph = hypreductions.hypergraph_clique_reduction(hypergraph)

        # Compute the operator L = (I + AD^-1) of the clique graph
        l_clique = hypjop.graph_diffusion_operator(weighted_clique_graph)

        # Compute the eigenvector corresponding to the smallest eigenvalue
        _, eigenvectors = sp.sparse.linalg.eigs(l_clique, k=1, which='SM')
        s = eigenvectors[:, 0]

        return _internal_bipartite_diffusion(s, hypergraph, max_time, step_size, approximate,
                                             construct_induced=construct_induced)


def find_max_cut(hypergraph, max_time=100, step_size=0.1, approximate=True, algorithm="diffusion",
                 return_each_pair=False):
    """
    Use the specified algorithm to find a partitioning of the vertices into two sets. Run the algorithm recursively
    until the vertices are fully partitioned (similar to trevisan's algorithm).

    :param hypergraph:
    :param max_time:
    :param step_size:
    :param approximate:
    :param algorithm: one of 'diffusion' or 'clique'
    :param return_each_pair: If True, yield each pair of left and right sets as the iterations goes, rather than the
                             complete cut.
    :return:
    """
    hyplogging.logger.info(f"Finding max cut. Method: {algorithm}. Returning each pair: {return_each_pair}.")
    left_set = []
    right_set = []
    unclassified_nodes = hypergraph.nodes
    n = len(unclassified_nodes)

    while len(unclassified_nodes) > 0:
        hyplogging.logger.debug(f"Max cut iteration. Nodes left: {len(unclassified_nodes)}/{n}")
        # Run the next iteration of the algorithm.
        induced_hypergraph = hypergraph.induced_hypergraph(unclassified_nodes)

        if algorithm == 'diffusion':
            new_l, new_r, _ = find_bipartite_set_diffusion(induced_hypergraph, max_time=max_time, step_size=step_size,
                                                           approximate=approximate)
        elif algorithm == 'clique':
            new_l, new_r, _ = find_bipartite_set_clique(induced_hypergraph)
        else:
            raise AssertionError("Algorithm must be diffusion or clique.")

        if return_each_pair and len(new_l) > 0 and len(new_r) > 0:
            yield [unclassified_nodes[i] for i in new_l], [unclassified_nodes[i] for i in new_r]

        # If the returned sets are empty, we cannot continue other than adding all remaining vertices to one of the sets
        if len(new_l) == 0 and len(new_r) == 0:
            left_set.extend(unclassified_nodes)
            break

        if return_each_pair:
            yield [unclassified_nodes[i] for i in new_l], [unclassified_nodes[i] for i in new_r]

        # Figure out which way round to put the new sets
        possible_left_set = left_set.copy()
        possible_left_set.extend([unclassified_nodes[i] for i in new_l])
        possible_right_set = right_set.copy()
        possible_right_set.extend([unclassified_nodes[i] for i in new_r])
        same_bipartiteness = hypcheeg.hypergraph_bipartiteness(hypergraph, possible_left_set, possible_right_set)

        possible_left_set = left_set.copy()
        possible_left_set.extend([unclassified_nodes[i] for i in new_r])
        possible_right_set = right_set.copy()
        possible_right_set.extend([unclassified_nodes[i] for i in new_l])
        opposite_bipartiteness = hypcheeg.hypergraph_bipartiteness(hypergraph, possible_left_set, possible_right_set)

        if same_bipartiteness < opposite_bipartiteness:
            left_set.extend([unclassified_nodes[i] for i in new_l])
            right_set.extend([unclassified_nodes[i] for i in new_r])
        else:
            left_set.extend([unclassified_nodes[i] for i in new_r])
            right_set.extend([unclassified_nodes[i] for i in new_l])

        # Update the list of nodes yet to be classified.
        unclassified_nodes = [i for i in hypergraph.nodes if i not in left_set and i not in right_set]

    if not return_each_pair:
        yield left_set, right_set


def check_cluster_pairs(clusters, hypergraph, max_time=100, step_size=0.1, use_random_initialisation=False,
                        approximate=True):
    """
    Given a set of clusters in the hypergraph, check whether it is possible to improve the clustering by re-clustering
    some pair of clusters.

    :param clusters:
    :param hypergraph:
    :param max_time:
    :param step_size:
    :param use_random_initialisation:
    :param approximate:
    :return: the optimised clusters
    """
    improvement_found = True
    max_iterations = 10
    iterations = 0
    while improvement_found or iterations >= max_iterations:
        improvement_found = False

        # The metric we will try to improve is the sum of the bipartiteness of every pair of clusters
        current_bipartiteness_sum = hypcheeg.compute_total_bipartiteness(hypergraph, clusters)

        # Look through each pair of clusters and try to find an improvement
        for i in range(len(clusters)):
            if improvement_found:
                break
            for j in range(i + 1, len(clusters)):
                # Check if re-clustering these sets improves the bipartiteness
                both_clusters = clusters[i] + clusters[j]
                induced_hypergraph = hypergraph.induced_hypergraph(both_clusters)

                # Run the diffusion
                if induced_hypergraph.num_vertices > 0:
                    candidate_induced_l, candidate_induced_r, _ = find_bipartite_set_diffusion(
                        induced_hypergraph, max_time=max_time, step_size=step_size,
                        use_random_initialisation=use_random_initialisation, approximate=approximate)
                    candidate_l = [both_clusters[v] for v in candidate_induced_l]
                    candidate_r = [both_clusters[v] for v in candidate_induced_r]

                    # Check if it is better
                    new_clusters = clusters.copy()
                    new_clusters[i] = candidate_l
                    new_clusters[j] = candidate_r
                    new_bipartiteness_sum = hypcheeg.compute_total_bipartiteness(hypergraph, new_clusters)
                    if new_bipartiteness_sum < current_bipartiteness_sum:
                        # This is better. Update the clusters
                        hyplogging.logger.debug(f"Found better clustering with clusters {i} and {j}.")
                        improvement_found = True
                        clusters[i] = candidate_l
                        clusters[j] = candidate_r
                        break

    return clusters


def recursive_bipartite_diffusion(hypergraph, iterations, max_time=100, step_size=0.1, use_random_initialisation=False,
                                  approximate=False, return_unclassified=False, mix_and_match=False,
                                  use_clique_alg=False):
    """
    Run the bipartite diffusion process recursively, to return 2^iterations clusters. The remaining arguments have the
    same meaning as in the find_bipartite_set_diffusion method.

    :param hypergraph:
    :param iterations:
    :param max_time:
    :param step_size:
    :param use_random_initialisation:
    :param approximate:
    :param return_unclassified: Whether to treat the 'unclassified' set as a cluster in each iteration (i.e. return 3
                                clusters per iteration.
    :param mix_and_match: Whether to check each pair of clusters at the end of the algorithm to see if they can be split
                          in such a way as to reduce the bipartiteness between them.
    :param use_clique_alg: Whether to use the clique algorithm instead of the diffusion algorithm.
    :return: a list of lists containing the final clusters
    """
    current_clusters = [hypergraph.nodes]

    for i in range(iterations):
        hyplogging.logger.info(f"Diffusion iteration {i + 1}/{iterations}.")
        new_clusters = []

        # For each existing cluster, perform the diffusion algorithm.
        for cluster in current_clusters:
            # Construct the hypergraph induced by this cluster
            induced_hypergraph = hypergraph.induced_hypergraph(cluster)

            # Run the diffusion on this hypergraph.
            if induced_hypergraph.num_vertices > 0:
                if use_clique_alg:
                    cluster_l, cluster_r, _ = find_bipartite_set_clique(induced_hypergraph)
                else:
                    cluster_l, cluster_r, _ = find_bipartite_set_diffusion(
                        induced_hypergraph, max_time=max_time, step_size=step_size,
                        use_random_initialisation=use_random_initialisation, approximate=approximate)

                # Add the found clusters to the new list. Recall that the vertex indices in the induced hypergraph are
                # equal to the vertex indices in the list 'cluster'.
                new_clusters.append([cluster[v] for v in cluster_l])
                new_clusters.append([cluster[v] for v in cluster_r])
                if return_unclassified:
                    new_clusters.append(
                        [cluster[v] for v in range(len(cluster)) if v not in cluster_l and v not in cluster_r])
            else:
                # If the induced hypergraph does not have any edges, then we do not try to run the algorithm.
                new_clusters.append(cluster)

        # Update the current clusters for the next iteration.
        current_clusters = new_clusters

    # Check for better clusters by the 'mix-and-match' algorithm.
    clusters = [cluster for cluster in current_clusters if len(cluster) > 0]
    if mix_and_match:
        clusters = check_cluster_pairs(clusters, hypergraph, max_time=max_time, step_size=step_size,
                                       use_random_initialisation=use_random_initialisation, approximate=approximate)

    # Return the non-empty clusters
    return clusters


def find_bipartite_set_clique(hypergraph):
    """
    Given a hypergraph, use the clique graph to compute an almost-bipartite set.
    :param hypergraph: The hypergraph on which to operate
    :return: the sets L and R, and the bipartiteness value beta(L, R)
    """
    # If the given hypergraph is empty, return empty sets
    if len(hypergraph.nodes) == 0:
        return [], [], 1

    # Construct the clique graph from the hypergraph
    hyplogging.logger.debug("Constructing the clique graph.")
    weighted_clique_graph = hypreductions.hypergraph_clique_reduction(hypergraph)

    # Compute the operator L = (I + AD^-1) of the clique graph
    hyplogging.logger.debug("Computing the clique graph diffusion operator.")
    l_clique = hypjop.graph_diffusion_operator(weighted_clique_graph)

    # Compute the eigenvector corresponding to the smallest eigenvalue
    hyplogging.logger.debug("Computing the eigenvalues and eigenvectors.")
    _, eigenvectors = sp.sparse.linalg.eigs(l_clique, k=1, which='SM')
    x = eigenvectors[:, 0]

    # Run the two-sided sweep set algorithm on this eigenvector
    hyplogging.logger.debug("Running the sweep-set procedure.")
    vertex_set_l, vertex_set_r = hypcheeg.hypergraph_two_sided_sweep(x, hypergraph)
    beta = hypcheeg.hypergraph_bipartiteness(hypergraph, vertex_set_l, vertex_set_r)

    return vertex_set_l, vertex_set_r, beta


def find_bipartite_set_random(hypergraph):
    """
    Given a hypergraph, find an almost bipartite set by randomly partitioning the vertices into two sets.
    :param hypergraph:
    :return: the sets L and R, and the bipartiteness value beta(L, R)
    """
    # We will run the algorithm 5 times, and return the best result
    best_bipartiteness = 1
    best_vertex_set_l = []
    best_vertex_set_r = []

    n = hypergraph.number_of_nodes()
    for i in range(5):
        # Compute a random starting vector for the diffusion process
        s = 2 * (np.random.randint(2, size=n) - 1 / 2)

        # Perform the sweep-set procedure on this vector
        this_vertex_set_l, this_vertex_set_r = hypcheeg.hypergraph_two_sided_sweep(s, hypergraph)
        this_bipartiteness = hypcheeg.hypergraph_bipartiteness(hypergraph, this_vertex_set_l, this_vertex_set_r)

        # Check if this is the best one so far
        if this_bipartiteness < best_bipartiteness:
            best_bipartiteness = this_bipartiteness
            best_vertex_set_l = this_vertex_set_l
            best_vertex_set_r = this_vertex_set_r

    # Return the best set
    return best_vertex_set_l, best_vertex_set_r, best_bipartiteness
