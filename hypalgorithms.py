"""This file contains complete implementations of various hypergraph algorithms."""
import numpy as np
import scipy as sp
import scipy.sparse.linalg
import hypcheeg
import hypmc
import hypreductions


def find_bipartite_set_diffusion(hypergraph, max_time=100, step_size=0.1):
    """
    Given a hypergraph, use the diffusion process to find an almost bipartite set.
    :param hypergraph: The hypergraph on which to find a bipartite set
    :param max_time: The maximum time to run the diffusion process
    :param step_size: The step size to use for the diffusion
    :return: the sets L, and R, and the bipartiteness value beta(L, R)
    """
    # Compute the starting vector for the diffusion process - use a random +/- 1 vector
    n = hypergraph.number_of_nodes()
    s = 2 * (np.random.randint(2, size=n) - 1/2)

    # Compute the diffusion process until convergence
    measure_vector, _, _ = hypmc.sim_mc_heat_diff(
        s, hypergraph, max_time=max_time, step=step_size, check_converged=True)

    # Perform the sweep set algorithm on the measure vector to find the almost-bipartite set
    # TODO - complete this function


def find_bipartite_set_clique(hypergraph):
    """
    Given a hypergraph, use the clique graph to compute an almost-bipartite set.
    :param hypergraph: The hypergraph on which to operate
    :return: the sets L and R, and the bipartiteness value beta(L, R)
    """
    # Construct the clique graph from the hypergraph
    weighted_clique_graph = hypreductions.hypergraph_clique_reduction(hypergraph)

    # Compute the graph L = (I + AD^-1) of the clique graph
    l_clique = hypmc.graph_diffusion_operator(weighted_clique_graph)

    # Compute the eigenvector corresponding to the smallest eigenvalue
    eigs, eigvecs = sp.sparse.linalg.eigsh(l_clique, k=1, which='SM')
    x = eigvecs[:, 0]

    # Run the two-sided sweep set algorithm on this eigenvector
    vertex_set_l, vertex_set_r = hypcheeg.hypergraph_two_sided_sweep(x, hypergraph)
    beta = hypcheeg.hypergraph_bipartiteness(hypergraph, vertex_set_l, vertex_set_r)

    return vertex_set_l, vertex_set_r, beta
