"""
This file implements methods for using the hypergraph laplacian operator.
The terminology used in this file comes from the following paper: https://arxiv.org/abs/1605.01483
"""
import networkx as nx
import hypernetx as hnx
import numpy as np
import scipy as sp
import scipy.sparse.linalg
import matplotlib.pyplot as plt
import sys
from scipy.optimize import linprog


def hypergraph_degree_mat(hypergraph, inverse=False, sqrt=False):
    """
    Given a hypergraph H, compute the degree matrix containing the degree of each vertex on the diagonal.
    :param hypergraph:
    :param inverse: If true, return the inverse of the matrix
    :param sqrt: If true, return the sqrt of the matrix
    :return: A numpy matrix containing the degrees.
    """
    # Get the vertices from the hypergraph
    vertices = hypergraph.nodes
    n = len(vertices)

    if not inverse and not sqrt:
        return sp.sparse.spdiags(hypergraph.degrees, 0, n, n, format="csr")
    if inverse and not sqrt:
        return sp.sparse.spdiags(hypergraph.inv_degrees, 0, n, n, format="csr")
    if sqrt and not inverse:
        return sp.sparse.spdiags(hypergraph.sqrt_degrees, 0, n, n, format="csr")
    if inverse and sqrt:
        return sp.sparse.spdiags(hypergraph.inv_sqrt_degrees, 0, n, n, format="csr")


# =======================================================
# Convert between measure, weighted and normalised space
# =======================================================
def measure_to_weighted(phi, hypergraph):
    """
    Given a vector in the measure space, compute the corresponding vector in the weighted space.
            f = W^{-1} x
    :param phi: The vector in the measure space to be converted.
    :param hypergraph: The underlying hypergraph.
    :return: The vector f in the weighted space.
    """
    return hypergraph_degree_mat(hypergraph, inverse=True) @ phi


def weighted_to_normalized(f, hypergraph):
    return hypergraph_degree_mat(hypergraph, sqrt=True) @ f


def normalized_to_measure(x, hypergraph):
    return hypergraph_degree_mat(hypergraph, sqrt=True) @ x


def measure_to_normalized(phi, hypergraph):
    return weighted_to_normalized(measure_to_weighted(phi, hypergraph), hypergraph)


def normalized_to_weighted(x, hypergraph):
    return measure_to_weighted(normalized_to_measure(x, hypergraph), hypergraph)


def weighted_to_measure(f, hypergraph):
    return normalized_to_measure(weighted_to_normalized(f, hypergraph), hypergraph)


def compute_edge_info(f, hypergraph, debug=False):
    """
    Given a hypergraph H and weighted vector f, compute a dictionary with the following information for each edge in H:
        (I(e), S(e), max_{u, v \\in e} (f(u) - f(v), [vertices])
    :param f: A vector in the normalised space
    :param hypergraph: The underlying hypergraph
    :param debug: Whether to print debug info
    :return: a dictionary with the above information for each edge in the hypergraph.
    """
    max_f = max(f)
    min_f = min(f)
    if debug:
        print('f', f)
        print('max_f', max_f, 'min_f', min_f)
    edge_info = {}

    # Compute a dictionary of vertex names and vertex ids
    vertex_name_to_index = {}
    for vertex_index, vertex_name in enumerate(hypergraph.nodes):
        vertex_name_to_index[vertex_name] = vertex_index
    if debug:
        print('vertex_name_to_index', vertex_name_to_index)

    for edge in hypergraph.edges():
        if debug:
            print("Processing edge:", edge)
        # Compute the maximum and minimum sets for the edge.
        # Referred to as S_e and I_e in the analysis of the algorithm.
        min_vertices = []
        min_value = max_f
        max_vertices = []
        max_value = min_f
        for vertex in edge.elements:
            if debug:
                print("Considering vertex:", vertex)
            fv = f[vertex_name_to_index[vertex]]
            if debug:
                print("fv", fv)

            if fv < min_value:
                if debug:
                    print("New minimum.")
                min_value = fv
                min_vertices = [vertex]
            if fv > max_value:
                if debug:
                    print("New maximum.")
                max_value = fv
                max_vertices = [vertex]
            if fv == min_value and vertex not in min_vertices:
                if debug:
                    print("Updating minimum.")
                min_vertices.append(vertex)
            if fv == max_value and vertex not in max_vertices:
                if debug:
                    print("Updating maximum.")
                max_vertices.append(vertex)
            if debug:
                print("min_vertices", min_vertices, "max_vertices", max_vertices)
        edge_info[edge.uid] = (min_vertices, max_vertices, max_value - min_value, edge.elements)
    if debug:
        print("Returning:", edge_info)
    return edge_info


def delta(vertex_set, edge_info, hypergraph):
    """
    Given some set of vertices P and the hypergraph edge information, compute the delta value as defined in the paper.
        delta(P) = C(P) / w(P)
    where
        C(P) = c(I_P) - c(S_P)
    and w(P) is just the number of vertices in P.
    :param vertex_set: a set of vertices in the hypergraph.
    :param edge_info: the standard, precomputed information about the hypergraph edges.
    :param hypergraph: The underlying hypergraph on which we are operating.
    :return: the computed delta value.
    """
    # Compute the I and S edge sets
    i_edges = []
    s_edges = []
    for e, e_info in edge_info.items():
        # Check whether the edge maximum/minimum set has a non-zero intersection with U
        if len([vertex for vertex in vertex_set if vertex in e_info[1]]) > 0:
            s_edges.append(e)
        if len([vertex for vertex in vertex_set if vertex in e_info[0]]) == len(e_info[0]):
            i_edges.append(e)

    # Compute C(P) and w(P)
    c_p = 0
    for e in i_edges:
        c_p += edge_info[e][2]
    for e in s_edges:
        c_p -= edge_info[e][2]

    # w(P) is the sum of the original degrees of the vertices
    w_p = 0
    for vertex in vertex_set:
        w_p += hypergraph.degree(vertex)

    # Return the final value
    if c_p == 0:
        return 0
    else:
        return c_p / w_p


def power_set(s):
    """Return an iterator over the power set of s. Remember that this will introduce an exponential factor into your
    algorithm."""
    x = len(s)
    masks = [1 << i for i in range(x)]
    for i in range(1, 1 << x):
        yield [ss for mask, ss in zip(masks, s) if i & mask]


def find_densest_subset(vertex_set, hypergraph, edge_info, debug=False):
    """
    Given a set of vertices U, find the densest subset as per section 4.2 of the original paper.
    :param vertex_set: the vertex level-set
    :param hypergraph: the underlying hypergraph
    :param edge_info: the pre-computed information about every edge in the graph
    :param debug: whether to make debug logs
    :return:
    """
    if debug:
        print("Finding densest subgraph for", vertex_set)
    # We will solve this problem using the linear program given in the original paper.
    # First, we need to compute the sets I and S.
    i_edges = []
    s_edges = []
    for e, e_info in edge_info.items():
        # Check whether the edge maximum/minimum set has a non-zero intersection with U
        if len([vertex for vertex in vertex_set if vertex in e_info[1]]) > 0:
            s_edges.append(e)
        if len([vertex for vertex in vertex_set if vertex in e_info[0]]) > 0:
            i_edges.append(e)

    if debug:
        print('I', i_edges, 'S', s_edges)

    # If I and S are both empty, then we can return with zero weight
    if len(i_edges) + len(s_edges) == 0:
        best_p = np.copy(vertex_set)
        max_delta = 0
        if debug:
            print("Returning early since I and S are empty")
            print("best_p:", best_p)
            print("max_delta:", max_delta)
        return best_p, max_delta

    # Now, we construct the linear program.
    # The variables are as follows:
    #   x_e for each e in I \union S
    #   y_v for each v in U
    num_vars = len(i_edges) + len(s_edges) + len(vertex_set)

    # The objective function to be minimised is
    #    -c(x) = - sum_{I} c_e x_e + sum_{S} c_e x_e
    obj = np.zeros(num_vars)
    i = 0
    for e in i_edges:
        obj[i] = -edge_info[e][2]
        i += 1
    for e in s_edges:
        obj[i] = edge_info[e][2]
        i += 1
    if debug:
        print("obj", obj)

    # The equality constraints are
    #    sum_{v \in U} w_v y_v = 1
    lhs_eq = [np.zeros(num_vars)]
    rhs_eq = 1
    for i, vertex in enumerate(vertex_set):
        lhs_eq[0][len(i_edges) + len(s_edges) + i] = hypergraph.degree(vertex)
    if debug:
        print('lhs_eq', lhs_eq)

    # The inequality constraints are:
    #   x_e - y_v \leq 0    for all e \in I, v \in e
    #   y_v - x_e \leq 0    for all e \in S, v \in e
    inequalities_lhs = []
    inequalities_rhs = []
    edge_index = -1
    for e in i_edges:
        edge_index += 1
        for vertex in edge_info[e][3]:
            if vertex in vertex_set:
                # Construct the left hand side of the inequality in terms of the coefficients.
                lhs = np.zeros(num_vars)
                vert_index = len(i_edges) + len(s_edges) + vertex_set.index(vertex)
                lhs[edge_index] = 1
                lhs[vert_index] = -1

                # Add this inequality to the lists
                if debug:
                    print(f"Added inequality: {e} - {vertex} < 0")
                inequalities_lhs.append(lhs)
                inequalities_rhs.append(0)
    for e in s_edges:
        edge_index += 1
        for vertex in edge_info[e][3]:
            if vertex in vertex_set:
                # Construct the left hand side of the inequality in terms of the coefficients.
                lhs = np.zeros(num_vars)
                vert_index = len(i_edges) + len(s_edges) + vertex_set.index(vertex)
                lhs[edge_index] = -1
                lhs[vert_index] = 1

                # Add this inequality to the lists
                if debug:
                    print(f"Added inequality: {vertex} - {e} < 0")
                inequalities_lhs.append(lhs)
                inequalities_rhs.append(0)

    # Now, solve the linear program.
    opt = linprog(c=obj, A_ub=inequalities_lhs, b_ub=inequalities_rhs, A_eq=lhs_eq, b_eq=rhs_eq)
    if debug:
        print(opt)

    # Find a level set of the output of the linear program.
    max_delta = -np.round(opt.fun, decimals=5)
    lp_solution = np.round(opt.x, decimals=5)
    thresh = 0
    for i in range(len(vertex_set)):
        if lp_solution[len(i_edges) + len(s_edges) + i] > thresh:
            thresh = lp_solution[len(i_edges) + len(s_edges) + i]
    if debug:
        print("thresh:", thresh)
    best_p = []
    for i in range(len(vertex_set)):
        if lp_solution[len(i_edges) + len(s_edges) + i] >= thresh:
            best_p.append(vertex_set[i])

    if debug:
        print("best_P:", best_p)
        print("max_delta:", max_delta)
    return best_p, max_delta


# ====================================================
# Compute the hypergraph laplacian for a given vector
# ====================================================
def weighted_diffusion_gradient(f, hypergraph, debug=False):
    """
    Given a vector in the weighted space, compute the gradient
         r = df/dt
    according to the heat diffusion procedure. We assume that f contains only positive values.
    :param f:
    :param hypergraph:
    :param debug: Whether to print debug statements.
    :return:
    """
    # We will round the vector f to a small(ish) precision so as to
    # avoid vertices ending in the wrong equivalence classes due to numerical errors.
    f = np.round(f, decimals=5)

    # Compute some standard information about every edge in the hypergraph
    edge_info = compute_edge_info(f, hypergraph)

    # This is the eventual output of the algorithm
    r = np.zeros(len(hypergraph.nodes))

    # We will refer to the steps of the procedure in Figure 4.1 of the reference paper. Starting with...
    # STEP 1
    # Find the equivalence classes of the input vector.
    equiv_classes = {}
    vertex_name_to_index = {}
    for vertex_index, vertex_name in enumerate(hypergraph.nodes):
        vertex_name_to_index[vertex_name] = vertex_index
        # Check if we have seen this value yet
        if f[vertex_index] in equiv_classes:
            equiv_classes[f[vertex_index]].append(vertex_name)
        else:
            # This is the first time we've seen this value
            equiv_classes[f[vertex_index]] = [vertex_name]

    if debug:
        print("Equivalence classes:", equiv_classes)

    # We now iterate through the equivalence classes for the remainder of the algorithm
    for _, U in equiv_classes.items():
        u_temp = U

        # STEP 2 + 3
        # We need to find the subset P of U to maximise
        # \delta(P) = C(P) / w(P)
        # For now, we will iterate over all possible subsets (!!) - this is exponential though! For small graphs it
        # should at least work, and I can come back later to optimise.
        while len(u_temp) > 0:
            best_p, max_delta = find_densest_subset(u_temp, hypergraph, edge_info, debug=debug)

            # Update the r value for the best vertices and remove them from u_temp
            for vertex in best_p:
                r[vertex_name_to_index[vertex]] = max_delta
                u_temp.remove(vertex)
    if debug:
        print("Complete Gradient:", r)
    return r


def hypergraph_measure_laplacian(phi, hypergraph, debug=False):
    """
    Apply the hypergraph laplacian operator to a vector phi in the measure space.
    In the normal language of graphs, this laplacian would be written as:
       L = I - A D^{-1}
    It can be considered to be the 'random walk' laplacian.
    :param phi: The vector to apply the laplacian operator to.
    :param hypergraph: The underlying hypergraph
    :param debug: Whether to print debug statements.
    :return: L_H x
    """
    f = measure_to_weighted(phi, hypergraph)
    r = weighted_diffusion_gradient(f, hypergraph, debug=debug)
    return -hypergraph_degree_mat(hypergraph) @ r


def hypergraph_lap_conn_graph(phi, hypergraph):
    """
    Given a current vector in the measure space, return a graph object demonstrating the basic connectivity of the
    underlying laplacian graph. (For now, I don't actually compute the weights on this graph fully)
    :param phi: A vector in the measure space
    :param hypergraph: The underlying hypergraph
    :return: A networkx graph object
    """
    f = measure_to_weighted(phi, hypergraph)

    # We will round the vector f to a small(ish) precision so as to
    # avoid vertices ending in the wrong equivalence classes due to numerical errors.
    f = np.round(f, decimals=5)

    edge_info = compute_edge_info(f, hypergraph)
    new_graph = nx.Graph()

    # Add the vertices
    for vertex in hypergraph.nodes:
        new_graph.add_node(vertex)

    # Add the edges
    for e, e_info in edge_info.items():
        for u in e_info[0]:
            for vertex in e_info[1]:
                new_graph.add_edge(u, vertex)

    return new_graph


def hyp_plot_conn_graph(phi, hypergraph, show_hyperedges=True):
    """
    Plot the pagerank connectivity graph along with the hypergraph.
    :param phi:
    :param hypergraph:
    :param show_hyperedges: Whether to plot the hyperedges as well as the connection edges.
    :return:
    """
    connection_graph = hypergraph_lap_conn_graph(phi, hypergraph)
    hyp_plot_with_graph(connection_graph, hypergraph, show_hyperedges=show_hyperedges)


def hyp_plot_with_graph(graph, hypergraph, show_hyperedges=True, plot_graph_weights=False):
    """
    Plot a hypergraph with a simple graph on the same vertex set.
    :param graph: The simple graph
    :param hypergraph: The hypergraph
    :param show_hyperedges: Whether to plot the hyperedges of the hypergraph.
    :param plot_graph_weights: Whether to plot the weights of the edges in the graph
    :return:
    """
    # Get the positioning of the nodes for drawing the hypergraph
    pos = hnx.drawing.rubber_band.layout_node_link(hypergraph, layout=nx.spring_layout)
    ax = plt.gca()
    if show_hyperedges:
        hnx.drawing.rubber_band.draw(hypergraph, pos=pos)
    else:
        # Get the node radius (taken from hypernetx code)
        r0 = hnx.drawing.rubber_band.get_default_radius(hypergraph, pos)
        a0 = np.pi * r0 ** 2

        def get_node_radius(vertex):
            return np.sqrt(a0 * (len(vertex) if type(vertex) == frozenset else 1) / np.pi)

        node_radius = {
            vertex: get_node_radius(vertex)
            for vertex in hypergraph.nodes
        }

        # Draw stuff
        hnx.drawing.rubber_band.draw_hyper_nodes(hypergraph, pos=pos, ax=ax, node_radius=node_radius)
        hnx.drawing.rubber_band.draw_hyper_labels(hypergraph, pos=pos, ax=ax, node_radius=node_radius)

        # Set the axes (taken from hypernetx code)
        if len(hypergraph.nodes) == 1:
            x, y = pos[list(hypergraph.nodes)[0]]
            s = 20
            ax.axis([x - s, x + s, y - s, y + s])
        else:
            ax.axis('equal')
        ax.axis('off')

    # Draw the edges only from the connectivity graph.
    nx.draw_networkx_edges(graph, pos=pos)
    if plot_graph_weights:
        labels = nx.get_edge_attributes(graph, 'weight')
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=labels)
    plt.show()


# =======================================================
# Simulate the heat diffusion process
# =======================================================
def sim_lap_heat_diff(phi, hypergraph, max_time=1, step=0.1):
    """
    Simulate the heat diffusion process by the euler method.
    :param phi: The measure vector at the start of the process
    :param hypergraph: The underlying hypergraph
    :param max_time: The end time of the process
    :param step: The time interval to use for each step
    :return: A measure vector at the end of the process.
    """
    x_t = phi
    for _ in np.linspace(0, max_time, int(max_time / step)):
        print(x_t)
        grad = -hypergraph_measure_laplacian(x_t, hypergraph)
        x_t += step * grad
    return x_t


def sim_hyp_pagerank(alpha, s, phi0, hypergraph, max_iterations=1000, step=0.01, debug=False, check_converge=True):
    """
    Compute an approximation of the hypergraph pagerank. Note that the pagerank is defined with respect to the
    normalised vector.
    :param alpha: As in definition of pagerank. (the teleport probability)
    :param s: The teleport vector.
    :param phi0: The starting vector, in the measure space
    :param hypergraph: The underlying hypergraph
    :param max_iterations: The maximum number of iterations
    :param step: The step size of the diffusion process
    :param debug: Whether to print debug statements
    :param check_converge: Whether to check for convergence before the maximum number of iterations has passed
    :return: The pagerank vector
    """
    n = len(hypergraph.nodes)
    x_t = phi0
    beta = (2 * alpha) / (1 + alpha)
    total_iterations = 0
    converged = False
    print("Computing pagerank...")
    while not converged and total_iterations < max_iterations:
        total_iterations += 1
        if not debug:
            sys.stdout.write('.')
            if total_iterations % 10 == 0:
                sys.stdout.flush()
            if total_iterations % 100 == 0:
                sys.stdout.write('\n')
                sys.stdout.flush()
        if debug:
            print("iteration", total_iterations)
        grad = beta * (s - x_t) - (1 - beta) * hypergraph_measure_laplacian(x_t, hypergraph, debug=debug)
        if debug:
            print("Pagerank gradient:", grad)
        x_old = np.copy(x_t)
        x_t += step * grad

        # Check for convergence
        if check_converge and np.sum(np.abs(x_old - x_t)) < (0.00001 * n):
            converged = True
            print("\nPagerank converged. Iterations:", total_iterations)
    return x_t


def check_pagerank(alpha, pr, hypergraph):
    test = pr + ((1 - alpha) / (2 * alpha)) * hypergraph_measure_laplacian(pr, hypergraph)
    print("Pagerank test:", np.round(test, decimals=2))
