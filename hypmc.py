"""
This file contains code for computing the hypergraph max-cut laplacian operator.
"""
import math
import hyplap
import hypconstruct
import numpy as np
from scipy.optimize import linprog
import scipy as sp
import scipy.sparse
import hypernetx as hnx
import networkx as nx
import matplotlib.pyplot as plt


def find_densest_subset_mc(vertex_level_set, hypergraph, edge_info, debug=False):
    """
    Given a set of vertices U, find the densest subset as per the linear program operation for the signed diffusion
    process.
    :param vertex_level_set: the vertex level-set
    :param hypergraph: the underlying hypergraph
    :param edge_info: the pre-computed information about every edge in the graph
    :param debug: whether to make debug logs
    :return:
    """
    if debug:
        print("Finding densest subgraph for", vertex_level_set)
    # We will solve this problem using the linear program given in the original paper.
    # First, we need to compute the sets S+, S-, I+, I-.
    increasing_infimum_edges = []
    decreasing_infimum_edges = []
    increasing_supremum_edges = []
    decreasing_supremum_edges = []
    for e, e_info in edge_info.items():
        # Check whether the edge maximum/minimum set has a non-zero intersection with U
        if len([v for v in vertex_level_set if v in e_info[1]]) > 0:
            if e_info[2] < 0:
                increasing_supremum_edges.append(e)
            elif e_info[2] > 0:
                decreasing_supremum_edges.append(e)
        if len([v for v in vertex_level_set if v in e_info[0]]) > 0:
            if e_info[2] < 0:
                increasing_infimum_edges.append(e)
            elif e_info[2] > 0:
                decreasing_infimum_edges.append(e)

    if debug:
        print('Ip', increasing_infimum_edges, 'Im', decreasing_infimum_edges, 'Sp', increasing_supremum_edges, 'Sm',
              decreasing_supremum_edges)

    # If I and S are both empty, then we can return with zero weight
    if (len(increasing_infimum_edges) + len(decreasing_infimum_edges) + len(increasing_supremum_edges) +
       len(decreasing_supremum_edges)) == 0:
        best_set = np.copy(vertex_level_set)
        max_delta = 0
        if debug:
            print("Returning early since Ip, Im, Sp and Sm are empty")
            print("best_P:", best_set)
            print("max_delta:", max_delta)
        return best_set, max_delta

    # Now, we construct the linear program.
    # The variables are as follows:
    #   x_e for each e in Ip \union Im \union Sp \union Sm
    #   y_v for each v in U
    num_vars = len(increasing_infimum_edges) + len(decreasing_infimum_edges) + len(increasing_supremum_edges) + \
        len(decreasing_supremum_edges) + len(vertex_level_set)

    # The objective function to be minimised is
    #    -c(x) = - sum_{Sp} c_e x_e - sum_{Ip} c_e x_e + sum_{Sm} c_e x_e + sum_{Im} c_e x_e
    obj = np.zeros(num_vars)
    i = 0
    for e in increasing_supremum_edges:
        obj[i] = edge_info[e][2]
        i += 1
    for e in increasing_infimum_edges:
        obj[i] = edge_info[e][2]
        i += 1
    for e in decreasing_supremum_edges:
        obj[i] = edge_info[e][2]
        i += 1
    for e in decreasing_infimum_edges:
        obj[i] = edge_info[e][2]
        i += 1
    if debug:
        print("obj", obj)

    # The equality constraints are
    #    sum_{v \in U} w_v y_v = 1
    lhs_eq = [np.zeros(num_vars)]
    rhs_eq = [1]
    for i, v in enumerate(vertex_level_set):
        lhs_eq[0][len(increasing_supremum_edges) + len(increasing_infimum_edges) + len(decreasing_supremum_edges) + len(
            decreasing_infimum_edges) + i] = hypergraph.degree(v)
    if debug:
        print('lhs_eq', lhs_eq)

    #    x_e = y_u         for all e \in Sp, u \in e
    for e in increasing_supremum_edges:
        for v in edge_info[e][3]:
            if v in vertex_level_set:
                lhs = np.zeros(num_vars)
                vert_index = len(increasing_supremum_edges) + len(increasing_infimum_edges) + len(
                    decreasing_supremum_edges) + len(decreasing_infimum_edges) + vertex_level_set.index(v)
                edge_index = increasing_supremum_edges.index(e)
                lhs[edge_index] = 1
                lhs[vert_index] = -1

                # Add this inequality to the lists
                if debug:
                    print(f"Added inequality: {e} - {v} = 0")
                lhs_eq.append(lhs)
                rhs_eq.append(0)

    #    x_e = y_u         for all e \in Im, u \in e
    for e in decreasing_infimum_edges:
        for v in edge_info[e][3]:
            if v in vertex_level_set:
                lhs = np.zeros(num_vars)
                vert_index = len(increasing_supremum_edges) + len(increasing_infimum_edges) + len(
                    decreasing_supremum_edges) + len(decreasing_infimum_edges) + vertex_level_set.index(v)
                edge_index = len(increasing_supremum_edges) + len(increasing_infimum_edges) + len(
                    decreasing_supremum_edges) + decreasing_infimum_edges.index(e)
                lhs[edge_index] = 1
                lhs[vert_index] = -1

                # Add this inequality to the lists
                if debug:
                    print(f"Added inequality: {e} - {v} = 0")
                lhs_eq.append(lhs)
                rhs_eq.append(0)

    # The inequality constraints are:
    #   x_e - y_v \leq 0    for all e \in Ip, v \in e
    #   y_v - x_e \leq 0    for all e \in Sm, v \in e
    inequalities_lhs = []
    inequalities_rhs = []
    for e in increasing_infimum_edges:
        for v in edge_info[e][3]:
            if v in vertex_level_set:
                # Construct the left hand side of the inequality in terms of the coefficients.
                lhs = np.zeros(num_vars)
                vert_index = len(increasing_supremum_edges) + len(increasing_infimum_edges) + len(
                    decreasing_supremum_edges) + len(decreasing_infimum_edges) + vertex_level_set.index(v)
                edge_index = len(increasing_supremum_edges) + increasing_infimum_edges.index(e)
                lhs[edge_index] = 1
                lhs[vert_index] = -1

                # Add this inequality to the lists
                if debug:
                    print(f"Added inequality: {e} - {v} < 0")
                inequalities_lhs.append(lhs)
                inequalities_rhs.append(0)
    for e in decreasing_supremum_edges:
        for v in edge_info[e][3]:
            if v in vertex_level_set:
                # Construct the left hand side of the inequality in terms of the coefficients.
                lhs = np.zeros(num_vars)
                vert_index = len(increasing_supremum_edges) + len(increasing_infimum_edges) + len(
                    decreasing_supremum_edges) + len(decreasing_infimum_edges) + vertex_level_set.index(v)
                edge_index = len(increasing_supremum_edges) + len(
                    increasing_infimum_edges) + decreasing_supremum_edges.index(e)
                lhs[edge_index] = -1
                lhs[vert_index] = 1

                # Add this inequality to the lists
                if debug:
                    print(f"Added inequality: {v} - {e} < 0")
                inequalities_lhs.append(lhs)
                inequalities_rhs.append(0)

    # Now, solve the linear program.
    # Set unused constraints to none
    if len(inequalities_lhs) == 0:
        inequalities_lhs = None
        inequalities_rhs = None
    if len(lhs_eq) == 0:
        lhs_eq = None
        rhs_eq = None
    if debug:
        print(f"c: {obj}; A_ub: {inequalities_lhs}; b_ub: {inequalities_rhs}; A_eq: {lhs_eq}; b_eq: {rhs_eq}")
    opt = linprog(c=obj, A_ub=inequalities_lhs, b_ub=inequalities_rhs, A_eq=lhs_eq, b_eq=rhs_eq)
    if debug:
        print(opt)

    # Find a level set of the output of the linear program.
    max_delta = -np.round(opt.fun, decimals=5)
    lp_solution = np.round(opt.x, decimals=5)
    thresh = 0
    for i in range(len(vertex_level_set)):
        if lp_solution[len(increasing_supremum_edges) + len(increasing_infimum_edges) + len(decreasing_supremum_edges) +
                       len(decreasing_infimum_edges) + i] > thresh:
            thresh = lp_solution[
                len(increasing_supremum_edges) + len(increasing_infimum_edges) + len(decreasing_supremum_edges) + len(
                    decreasing_infimum_edges) + i]
    if debug:
        print("thresh:", thresh)
    best_set = []
    for i in range(len(vertex_level_set)):
        if lp_solution[len(increasing_supremum_edges) + len(increasing_infimum_edges) + len(decreasing_supremum_edges) +
                       len(decreasing_infimum_edges) + i] >= thresh:
            best_set.append(vertex_level_set[i])

    if debug:
        print("best_P:", best_set)
        print("max_delta:", max_delta)
    return best_set, max_delta


def compute_mc_edge_info(f, hypergraph, debug=False):
    """
    Given a hypergraph H and weighted vector f, compute a dictionary with the following information for each edge in H:
        (Ie, Se, max_{u in e} f(u) + min_{v in e} f(v), [vertices])
    :param f: A vector in the normalised space
    :param hypergraph: The underlying hypergraph
    :param debug: Whether to print debug information
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

    for e in hypergraph.edges():
        if debug:
            print("Processing edge:", e)
        # Compute the maximum and minimum sets for the edge.
        min_vertices = []
        min_e = max_f
        max_vertices = []
        max_e = min_f
        for v in e.elements:
            if debug:
                print("Considering vertex:", v)
            fv = f[vertex_name_to_index[v]]
            if debug:
                print("fv", fv)

            if fv < min_e:
                if debug:
                    print("New minimum.")
                min_e = fv
                min_vertices = [v]
            if fv > max_e:
                if debug:
                    print("New maximum.")
                max_e = fv
                max_vertices = [v]
            if fv == min_e and v not in min_vertices:
                if debug:
                    print("Updating minimum.")
                min_vertices.append(v)
            if fv == max_e and v not in max_vertices:
                if debug:
                    print("Updating maximum.")
                max_vertices.append(v)
            if debug:
                print("Ie", min_vertices, "Se", max_vertices)
        edge_info[e.uid] = (min_vertices, max_vertices, max_e + min_e, e.elements)
    if debug:
        print("Returning:", edge_info)
    return edge_info


# ==========================================================
# Compute the hypergraph max cut operator for a given vector
# ==========================================================
def weighted_mc_diffusion_gradient(f, hypergraph, debug=False):
    """
    Given a vector in the weighted space, compute the gradient
         r = df/dt
    according to the max cut heat diffusion procedure.
    :param f:
    :param hypergraph:
    :param debug: Whether to print debug information
    :return:
    """
    # We will round the vector f to a small(ish) precision so as to
    # avoid vertices ending in the wrong equivalence classes due to numerical errors.
    f = np.round(f, decimals=5)

    # Compute some standard information about every edge in the hypergraph
    edge_info = compute_mc_edge_info(f, hypergraph)

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
        while len(u_temp) > 0:
            best_p, max_delta = find_densest_subset_mc(u_temp, hypergraph, edge_info, debug=debug)

            # Update the r value for the best vertices and remove them from u_temp
            for v in best_p:
                r[vertex_name_to_index[v]] = max_delta
                u_temp.remove(v)
    if debug:
        print("Complete Gradient:", r)
    return r


def hypergraph_measure_mc_laplacian(phi, hypergraph, debug=False):
    """
    Apply the hypergraph max cut operator to a vector phi in the measure space.
    In the normal language of graphs, this operator would be written as:
       L = I + A D^{-1}
    :param phi: The vector to apply the laplacian operator to.
    :param hypergraph: The underlying hypergraph
    :param debug: Whether to print debug information
    :return: L_H x
    """
    f = hyplap.measure_to_weighted(phi, hypergraph)
    r = weighted_mc_diffusion_gradient(f, hypergraph, debug=debug)
    return -hyplap.hypergraph_degree_mat(hypergraph) @ r


def graph_diffusion_operator(graph):
    """
    Construct the operator (I + A D^{-1}) for the given graph.
    :param graph: A networkx graph
    :return: a scipy matrix
    """
    adjacency_matrix = nx.to_scipy_sparse_matrix(graph, format="csr")
    n, m = adjacency_matrix.shape
    degrees = adjacency_matrix.sum(axis=1)
    degree_matrix = sp.sparse.spdiags(degrees.flatten(), [0], m, n, format="csr")
    # noinspection PyUnresolvedReferences
    inverse_degree_matrix = sp.sparse.linalg.inv(degree_matrix)
    l_operator = sp.sparse.identity(n) + adjacency_matrix @ inverse_degree_matrix
    return l_operator


# =======================================================
# Simulate the max cut heat diffusion process
# =======================================================
def sim_mc_heat_diff(phi, hypergraph, max_time=1, step=0.1, debug=False, plot_diff=False,
                     print_time=False, check_converged=False):
    """
    Simulate the heat diffusion process for the hypergraph max cut operator.
    :param phi: The measure vector at the start of the process
    :param hypergraph: The underlying hypergraph
    :param max_time: The end time of the process
    :param step: The time interval to use for each step
    :param debug: Whether to print debug statements
    :param plot_diff: Whether to plot graphs showing the progression of the diffusion
    :param print_time: Whether to print the time steps
    :param check_converged: Whether to check for convergence of G(t)
    :return: A measure vector at the end of the process, the final time of the diffusion process, the sequence of G(T)
    """
    # If we are going to plot the diffusion process, we will show the following quantities:
    #  F(t) = \phi_t D^{-1} \phi_t
    #  - log F(t)
    #  G(t) = d/dt - log F(t)
    #  h(t) = - d/dt G(t)
    t_steps = np.linspace(0, max_time, int(max_time / step))
    f_t = []
    negative_log_ft = []
    g_t = []
    h_t = []

    x_t = phi
    final_t = 0
    for t in t_steps:
        final_t = t
        if print_time:
            print(f"Time {t:.2f}")

        # Apply the diffusion operator
        grad_hyp = -hypergraph_measure_mc_laplacian(x_t, hypergraph, debug=debug)
        x_t += step * grad_hyp

        # Add the graph points for this time step
        this_ft = x_t @ np.linalg.inv(hyplap.hypergraph_degree_mat(hypergraph)) @ x_t
        f_t.append(this_ft)
        negative_log_ft.append(- math.log(this_ft))
        x_tn = x_t / this_ft
        this_gt = (x_tn @ np.linalg.inv(hyplap.hypergraph_degree_mat(hypergraph)) @
                   hypergraph_measure_mc_laplacian(x_tn, hypergraph)) / (
                x_tn @ np.linalg.inv(hyplap.hypergraph_degree_mat(hypergraph)) @ x_tn)
        g_t.append(this_gt)
        if len(g_t) > 2:
            h_t.append((g_t[-2] - g_t[-1]) / step)
        elif len(g_t) == 2:
            h_t.append((g_t[-2] - g_t[-1]) / step)
            h_t.append((g_t[-2] - g_t[-1]) / step)

        # Check for convergence
        if check_converged:
            gt_len = len(g_t)
            if gt_len >= 30:
                prev_gt = g_t[int(gt_len / 3)]
                diff = prev_gt - this_gt
                if diff < 0.0000001:
                    # We have converged
                    break

    # Print the final value of G(t). If the process is fully converged, then this is an eigenvalue of the hypergraph
    # laplacian operator.
    this_ft = x_t @ np.linalg.inv(hyplap.hypergraph_degree_mat(hypergraph)) @ x_t
    x_tn = x_t / this_ft
    final_gt = (x_tn @ np.linalg.inv(hyplap.hypergraph_degree_mat(hypergraph)) @
                hypergraph_measure_mc_laplacian(x_tn, hypergraph)) / (
            x_tn @ np.linalg.inv(hyplap.hypergraph_degree_mat(hypergraph)) @ x_tn)
    if print_time:
        print(f"Final value of G(t): {final_gt:.5f}")

    # Now, plot the graphs
    if plot_diff:
        # Create the axes
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()

        # Label the axes
        ax1.set_xlabel("t")
        ax1.set_ylabel("F(t) or G(t)")
        ax2.set_ylabel("- log F(t)")

        # Plot the functions
        line1 = ax1.plot(t_steps[:len(g_t)], g_t)

        # Add legend and show plot
        ax2.legend(line1, "G(t) = d/dt - log F(t)")
        fig.tight_layout()
        plt.show()

    return x_t, final_t, g_t


def check_random_2_color_graph(n, m, r, t, eps):
    """
    Check the final eigenvalue of a random 2-colorable graph. Returns the following tuple:
        T, G(T), G(T/2), G(9T/10)
    :param n: Half the number of vertices in the graph
    :param m: The number of edges in the hypergraph
    :param r: The rank of each edge
    :param t: The maximum end time of the diffusion process
    :param eps: The update step size during the diffusion process
    :return: Tuple described above
    """
    random_hypergraph = hypconstruct.construct_2_colorable_hypergraph(n, n, m, r)

    # Create the starting vector
    s = np.zeros(2 * n)
    s[1] = 1

    # Run the heat diffusion process
    _, final_t, g_sequence = sim_mc_heat_diff(s, random_hypergraph, t,
                                              step=eps,
                                              check_converged=True)

    # Return the result
    g_length = len(g_sequence)
    return final_t, g_sequence[-1], g_sequence[int(g_length / 2)], g_sequence[int(9 * g_length / 10)]


def main():
    n = 10
    m = 2 * n
    r = 3
    show_hypergraph = True
    show_diffusion = True
    max_t = 20
    step_size = 0.1

    # Construct a hypergraph
    # hypergraph = hypconstruct.simple_two_edge_hypergraph()
    # hypergraph = hypconstruct.simple_not_two_colorable_hypergraph()
    hypergraph = hypconstruct.construct_2_colorable_hypergraph(int(n/2), int(n/2), m, r)

    # Optionally, plot the hypergraph
    if show_hypergraph:
        hyp_fig = plt.figure(1)
        hnx.draw(hypergraph)

        if not show_diffusion:
            hyp_fig.show()

    # Construct the starting vector
    s = np.zeros(n)
    s[0] = 1

    # Run the diffusion process
    _ = sim_mc_heat_diff(s, hypergraph, max_t, step=step_size, debug=False, plot_diff=show_diffusion,
                         check_converged=True, print_time=True)


if __name__ == "__main__":
    main()
