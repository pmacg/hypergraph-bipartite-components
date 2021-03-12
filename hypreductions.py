"""
This file gives various methods for reducing a hypergraph to a simple 2-graph.
"""
import random
import networkx as nx
import hypcheeg


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
    Given a hypergraph, H, return the networkx graph corresponding to the clique graph of H.
    The clique graph is constructed by replacing each hyperedge e with a clique with edges of weight 1/(r(e) - 1).
    :param hypergraph:
    :return: A networkx graph G
    """
    new_graph = nx.Graph()

    # Add the vertices to the graph
    for vertex in hypergraph.nodes:
        new_graph.add_node(vertex)

    # Add the edges to the graph. This dictionary will use tuples (u, v) as keys and store the total weight between u
    # and v as the value.
    new_edges = {}
    for edge in hypergraph.edges():
        edge_vertices = [vertex for vertex in edge]
        rank = len(edge_vertices)
        for vertex_index_1 in range(rank):
            for vertex_index_2 in range(vertex_index_1 + 1, rank):
                new_edge = (edge_vertices[vertex_index_1], edge_vertices[vertex_index_2])
                new_weight = 1 / (rank - 1)
                if new_edge in new_edges:
                    new_edges[new_edge] = new_edges[new_edge] + new_weight
                else:
                    new_edges[new_edge] = new_weight

    # Unroll this dictionary into a list of 3-tuples and add the edges to the new graph
    new_edges_list = []
    for new_edge in new_edges:
        new_edges_list.append((new_edge[0], new_edge[1], {'weight': new_edges[new_edge]}))
    new_graph.add_edges_from(new_edges_list)

    return new_graph
