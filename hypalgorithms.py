"""This file contains complete implementations of various hypergraph algorithms."""
import numpy as np
import scipy as sp
import scipy.sparse.linalg
import hypcheeg
import hypmc
import hypreductions
import hyplogging


def _internal_bipartite_diffusion(starting_vector, hypergraph, max_time, step_size, approximate):
    """
    Internal method for running the bipartite diffusion algorithm.
    :param starting_vector: the starting vector of the diffusion process
    :param hypergraph: the hypergraph on which to run
    :param max_time: the maximum diffusion time
    :param step_size: the step size for the diffusion
    :param approximate: whether to run the approximate version of the diffusion process
    :return: the sets L and R, and their bipartiteness
    """
    # Compute the diffusion process until convergence
    measure_vector, _, _ = hypmc.sim_mc_heat_diff(
        starting_vector, hypergraph, max_time=max_time, step=step_size, check_converged=True, plot_diff=False,
        approximate=approximate)

    # Perform the sweep set algorithm on the measure vector to find the almost-bipartite set
    vertex_set_l, vertex_set_r = hypcheeg.hypergraph_two_sided_sweep(measure_vector, hypergraph)
    beta = hypcheeg.hypergraph_bipartiteness(hypergraph, vertex_set_l, vertex_set_r)

    return vertex_set_l, vertex_set_r, beta


def find_bipartite_set_diffusion(hypergraph, max_time=100, step_size=0.1, use_random_initialisation=False,
                                 approximate=False):
    """
    Given a hypergraph, use the diffusion process to find an almost bipartite set.
    :param hypergraph: The hypergraph on which to find a bipartite set
    :param max_time: The maximum time to run the diffusion process
    :param step_size: The step size to use for the diffusion
    :param use_random_initialisation: By default, we will use the eigenvector of the clique graph to initialise. If this
                                      parameter is true, then we will use a random vector to initialise the diffusion.
    :param approximate: Whether to use the approximate, no-LP version of the diffusion operator
    :return: the sets L, and R, and the bipartiteness value beta(L, R)
    """
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
                s, hypergraph, max_time, step_size, approximate)

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
        l_clique = hypmc.graph_diffusion_operator(weighted_clique_graph)

        # Compute the eigenvector corresponding to the smallest eigenvalue
        eigenvalues, eigenvectors = sp.sparse.linalg.eigs(l_clique, k=1, which='SM')
        s = eigenvectors[:, 0]

        return _internal_bipartite_diffusion(s, hypergraph, max_time, step_size, approximate)


def recursive_bipartite_diffusion(hypergraph, iterations, max_time=100, step_size=0.1, use_random_initialisation=False,
                                  approximate=False):
    """
    Run the bipartite diffusion process recursively, to return 2^iterations clusters. The remaining arguments have the
    same meaning as in the find_bipartite_set_diffusion method.

    :param hypergraph:
    :param iterations:
    :param max_time:
    :param step_size:
    :param use_random_initialisation:
    :param approximate:
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
                cluster_l, cluster_r, _ = find_bipartite_set_diffusion(induced_hypergraph, max_time=max_time,
                                                                       step_size=step_size,
                                                                       use_random_initialisation=use_random_initialisation,
                                                                       approximate=approximate)

                # Add the found clusters to the new list. Recall that the vertex indices in the induced hypergraph are
                # equal to the vertex indices in the list 'cluster'.
                new_clusters.append([cluster[v] for v in cluster_l])
                new_clusters.append([cluster[v] for v in cluster_r])
            else:
                # If the induced hypergraph does not have any edges, then we do not try to run the algorithm.
                new_clusters.append(cluster)

        # Update the current clusters for the next iteration.
        current_clusters = new_clusters

    return current_clusters


def find_bipartite_set_clique(hypergraph):
    """
    Given a hypergraph, use the clique graph to compute an almost-bipartite set.
    :param hypergraph: The hypergraph on which to operate
    :return: the sets L and R, and the bipartiteness value beta(L, R)
    """
    # Construct the clique graph from the hypergraph
    hyplogging.logger.debug("Constructing the clique graph.")
    weighted_clique_graph = hypreductions.hypergraph_clique_reduction(hypergraph)

    # Compute the operator L = (I + AD^-1) of the clique graph
    hyplogging.logger.debug("Computing the clique graph diffusion operator.")
    l_clique = hypmc.graph_diffusion_operator(weighted_clique_graph)

    # Compute the eigenvector corresponding to the smallest eigenvalue
    hyplogging.logger.debug("Computing the eigenvalues and eigenvectors.")
    eigenvalues, eigenvectors = sp.sparse.linalg.eigs(l_clique, k=1, which='SM')
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
