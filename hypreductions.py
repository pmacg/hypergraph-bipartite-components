"""
This file gives various methods for reducing a hypergraph to a simple 2-graph.
"""
import random
import networkx as nx
import scipy as sp
import scipy.sparse
import hypcheeg
import hypmc
import lightgraphs


def hypergraph_min_edges_graph_reduction(hypergraph):
    """
    Given a hypergraph, construct a graph by replacing each hyperedge with a single edge joining the vertices with the
    fewest overlapping edges.
    :param hypergraph: The hypergraph on which to operate.
    :return:
    """
    new_graph = nx.Graph()

    # Add the vertices
    n = 0
    for v in hypergraph.nodes:
        n += 1
        new_graph.add_node(v)

    # Add the edges
    for hyperedge in hypergraph.edges():
        fewest = None
        edge = None
        for u in hyperedge.elements:
            for v in hyperedge.elements:
                if u != v:
                    if fewest is None or hypcheeg.hypergraph_common_edges(u, v, hypergraph) < fewest:
                        fewest = hypcheeg.hypergraph_common_edges(u, v, hypergraph)
                        edge = (u, v)
        new_graph.add_edge(edge[0], edge[1])

    return new_graph


def hypergraph_degree_graph_reduction(hypergraph, c=1):
    """
    Construct a simple graph by sampling based on vertex degrees.
    :param hypergraph: The hypergraph
    :param c: A parameter to the random process. Roughly the number of vertices to select a single hyperedge.
    :return: A graph G
    """
    new_graph = nx.Graph()
    m = len([e for e in hypergraph.edges()])
    n = len([v for v in hypergraph.nodes()])

    random.seed()

    # Add the vertices
    i = 0
    for v in hypergraph.nodes:
        i += 1
        new_graph.add_node(v)

    # Add the edges
    for e in hypergraph.edges():
        # For each vertex in e, check whether the vertex randomly selects e.
        chosen_vertices = {}
        for u in e.elements:
            # The probability of choosing this edge is
            p = c * (m / n) * (1 / len(e.elements)) * (1 / hypcheeg.hypergraph_weighted_degree(u, hypergraph))
            if random.random() <= p:
                chosen_vertices[u] = p

        # Add a clique on the chosen vertices
        processed = []
        for u, pu in chosen_vertices.items():
            processed.append(u)
            for v, pv in chosen_vertices.items():
                if v not in processed:
                    new_graph.add_edge(u, v, weight=(pu + pv - (pu * pv)))

    return new_graph


def hypergraph_clique_reduction(hypergraph):
    """
    Given a light hypergraph, H, return a light graph corresponding to the clique graph of H.
    The clique graph is constructed by replacing each hyperedge e with a clique with edges of weight 1/(r(e) - 1).
    :param hypergraph:
    :return: A LightGraph G
    """
    adj_mat = sp.sparse.lil_matrix((hypergraph.num_vertices, hypergraph.num_vertices))

    # Add the edges to the graph.
    for edge in hypergraph.edges:
        rank = len(edge)
        new_weight = 1 / (rank - 1) if rank > 1 else 0
        for vertex_index_1 in range(rank):
            for vertex_index_2 in range(vertex_index_1, rank):
                adj_mat[edge[vertex_index_1], edge[vertex_index_2]] += new_weight
                adj_mat[edge[vertex_index_2], edge[vertex_index_1]] += new_weight

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
    edge_info = hypmc.compute_mc_edge_info(x, hypergraph)

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
