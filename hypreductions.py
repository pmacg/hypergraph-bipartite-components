"""
This file gives various methods for reducing a hypergraph to a simple 2-graph.
"""
import scipy as sp
import scipy.sparse
import hypjop
import lightgraphs
import hyplogging


def hypergraph_clique_reduction(hypergraph):
    """
    Given a light hypergraph, H, return a light graph corresponding to the clique graph of H.
    The clique graph is constructed by replacing each hyperedge e with a clique with edges of weight 1/(r(e) - 1).
    :param hypergraph:
    :return: A LightGraph G
    """
    hyplogging.logger.debug(f"Initialising the adjacency matrix.")
    hyplogging.logger.debug(f"Number of vertices: {hypergraph.num_vertices}.")
    adj_mat = sp.sparse.lil_matrix((hypergraph.num_vertices, hypergraph.num_vertices))

    # Add the edges to the graph.
    m = hypergraph.num_edges
    hyplogging.logger.debug("Adding the edges.")
    hyplogging.logger.debug(f"Number of edges: {m}.")
    for i, edge in enumerate(hypergraph.edges):
        if i % 1000 == 0:
            hyplogging.logger.debug(f"Adding edge {i}/{m}.")

        rank = len(edge)
        new_weight = 1 / (rank - 1) if rank > 1 else 0
        for vertex_index_1 in range(rank):
            for vertex_index_2 in range(vertex_index_1 + 1, rank):
                adj_mat[edge[vertex_index_1], edge[vertex_index_2]] += new_weight
                adj_mat[edge[vertex_index_2], edge[vertex_index_1]] += new_weight

    hyplogging.logger.debug("Constructing and returning the LightGraph object.")
    return lightgraphs.LightGraph(adj_mat.tocsr())


def hypergraph_approximate_diffusion_reduction(hypergraph, x):
    """
    Given a light hypergraph and a vector, compute the light graph 'induced' by the diffusion process.
    This computes an approximate version of the graph in which each edges weight is evenly distributed between
    the bipartite graph between I(e) and S(e).
    :param hypergraph: the hypergraph on which we are operating
    :param x: the vector which is inducing the graph
    :return: a light graph object
    """
    adj_mat = sp.sparse.lil_matrix((hypergraph.num_vertices, hypergraph.num_vertices))

    # Compute the maximum and minimum sets for each edge
    edge_info = hypjop.compute_jop_edge_info(x, hypergraph)

    # Add the edges to the graph.
    for this_edge_info in edge_info.values():
        min_vertices = this_edge_info[0]
        max_vertices = this_edge_info[1]
        new_edge_weight = 1 / (len(min_vertices) * len(max_vertices))

        for vertex_1 in min_vertices:
            for vertex_2 in max_vertices:
                adj_mat[vertex_1, vertex_2] += new_edge_weight
                adj_mat[vertex_2, vertex_1] += new_edge_weight

    return lightgraphs.LightGraph(adj_mat.tocsr())
