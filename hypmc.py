"""
This file contains code for computing the hypergraph max-cut laplacian operator.
"""
import math
import hyplap
import hypconstruct
import hypreductions
import numpy as np
from scipy.optimize import linprog
import scipy as sp
import scipy.sparse.linalg
import networkx as nx
import matplotlib.pyplot as plt
import hyplogging


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

    for edge_idx, e in enumerate(hypergraph.edges):
        if debug:
            print("Processing edge:", e)
        # Compute the maximum and minimum sets for the edge.
        min_vertices = []
        min_e = max_f
        max_vertices = []
        max_e = min_f
        for v in e:
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
        edge_info[edge_idx] = (min_vertices, max_vertices, max_e + min_e, e)
    if debug:
        print("Returning:", edge_info)
    return edge_info


# ==========================================================
# Compute the hypergraph max cut operator for a given vector
# ==========================================================
def weighted_mc_diffusion_gradient(f, hypergraph, debug=False, approximate=False):
    """
    Given a vector in the weighted space, compute the gradient
         r = df/dt = - L_H f
    according to the max cut heat diffusion procedure.
    :param f:
    :param hypergraph:
    :param debug: Whether to print debug information
    :param approximate: Do not use the LP to compute the gradient, instead use the approximate induced graph.
    :return:
    """
    # We will round the vector f to a small(ish) precision so as to
    # avoid vertices ending in the wrong equivalence classes due to numerical errors.
    f = np.round(f, decimals=5)

    # If we are computing the approximate gradient, do so and return from the function
    if approximate:
        approximate_induced_graph = hypreductions.hypergraph_approximate_diffusion_reduction(hypergraph, f)
        hypergraph_inverse_degree_matrix = hyplap.hypergraph_degree_mat(hypergraph, inverse=True)
        diffusion_operator = graph_diffusion_operator(approximate_induced_graph, unnormalised=True)
        r = - hypergraph_inverse_degree_matrix @ diffusion_operator @ f
        return r

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


def hypergraph_measure_mc_laplacian(phi, hypergraph, debug=False, approximate=False):
    """
    Apply the hypergraph max cut operator to a vector phi in the measure space.
    In the normal language of graphs, this operator would be written as:
       L = I + A D^{-1}
    :param phi: The vector to apply the laplacian operator to.
    :param hypergraph: The underlying hypergraph
    :param debug: Whether to print debug information
    :param approximate: Whether to use the approximate no-LP version
    :return: L_H x
    """
    f = hyplap.measure_to_weighted(phi, hypergraph)
    r = weighted_mc_diffusion_gradient(f, hypergraph, debug=debug, approximate=approximate)
    return -hyplap.hypergraph_degree_mat(hypergraph) @ r


def graph_degree_matrix(graph, inverse=False):
    """
    Compute the degree matrix of a given networkx graph.
    :param graph:
    :param inverse: If true, return the inverse of the degree matrix.
    :return: A scipy sparse matrix
    """
    # adjacency_matrix = nx.to_scipy_sparse_matrix(graph, format="csr")
    adjacency_matrix = nx.adjacency_matrix(graph)
    n, m = adjacency_matrix.shape
    degrees = adjacency_matrix.sum(axis=1)

    if inverse:
        inverse_degrees = np.array([1/x if x != 0 else 0 for x in degrees])
        return sp.sparse.spdiags(inverse_degrees.flatten(), [0], m, n, format="csr")
    else:
        return sp.sparse.spdiags(degrees.flatten(), [0], m, n, format="csr")


def graph_diffusion_operator(graph, normalised=False, weighted=False, unnormalised=False):
    """
    Construct the operator (I + A D^{-1}) for the given graph. This is the diffusion operator on the measure space.

    By default, this gives the operator on the measure space. By the command line arguments, it can also return the
    operator on the weighted or normalised spaces.

    :param graph: A lightgraph object
    :param normalised: Return the normalised operator (I + D^{-1/2} A D^{-1/2})
    :param weighted: Return the weighted operator (I + D^{-1} A)
    :param unnormalised: Return the raw graph operator (D + A)
    :return: a scipy matrix
    """
    adjacency_matrix = graph.adj_mat
    n, m = adjacency_matrix.shape

    if normalised:
        inverse_sqrt_degree_matrix = graph.inverse_sqrt_degree_matrix()
        return sp.sparse.identity(n) + inverse_sqrt_degree_matrix @ adjacency_matrix @ inverse_sqrt_degree_matrix
    elif weighted:
        inverse_degree_matrix = graph.inverse_degree_matrix()
        return sp.sparse.identity(n) + inverse_degree_matrix @ adjacency_matrix
    elif unnormalised:
        degree_matrix = graph.degree_matrix()
        return degree_matrix + adjacency_matrix
    else:
        inverse_degree_matrix = graph.inverse_degree_matrix()
        return sp.sparse.identity(n) + adjacency_matrix @ inverse_degree_matrix


def compute_gt(varphi, hypergraph, measure_diffusion_operator):
    """
    Compute the value of the function g(t) given the measure vector varphi, the hypergraph and the diffusion operator at
    this time step.

    :param varphi:
    :param hypergraph:
    :param measure_diffusion_operator:
    :return:
    """
    inverse_degree_matrix = hyplap.hypergraph_degree_mat(hypergraph, inverse=True)
    return (varphi @ inverse_degree_matrix @ measure_diffusion_operator @ varphi) /\
           (varphi @ measure_diffusion_operator @ varphi)


def compute_ht(gts, step_size):
    """Compute the value of h(t) = - d/dt g(t), given the list of gt values and the current step size."""
    if len(gts) >= 2:
        return (gts[-2] - gts[-1]) / step_size
    else:
        return 0


def plot_convergence_graphs(t_steps, g_t):
    """Plot the graphs illustrating the convergence of the diffusion process.

    :param t_steps: the list of time steps used in the diffusion
    :param g_t: the list of g_t values to go along with t_steps
    """
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


def diffusion_has_converged(t_steps, gts):
    """
    Determine whether the diffusion process has converged, given a list of time steps and g(t) values.
    :param t_steps:
    :param gts:
    :return: boolean indicating convergence
    """
    # If we have been running for more than 11 time steps
    current_time = t_steps[-1]
    if current_time >= 11:
        # Get the value of g(t) for 10 time steps ago
        t_minus_10_index = 0
        for i, t in enumerate(t_steps):
            if t < current_time - 10:
                t_minus_10_index = i
        previous_gt = gts[t_minus_10_index]
        diff = previous_gt - gts[-1]

        hyplogging.logger.debug(f"Convergence target = 0.001; actual diff = {diff}")

        if diff < 0.001:
            # We have converged
            hyplogging.logger.info(f"Diffusion process has converged at time {t}")
            return True
    return False


def approximate_measure_diffusion_operator(measure_vector, hypergraph):
    """
    Compute the approximate measure diffusion operator for the given measure vector and hypergraph.
    :param measure_vector:
    :param hypergraph:
    :return:
    """
    # Apply the approximate diffusion operator. We construct the approximate reduced graph using the weighted
    # vector f.
    f = hyplap.measure_to_weighted(measure_vector, hypergraph)
    approximate_diffusion_graph = hypreductions.hypergraph_approximate_diffusion_reduction(hypergraph, f)
    inverse_degree_matrix = hyplap.hypergraph_degree_mat(hypergraph, inverse=True)
    return graph_diffusion_operator(approximate_diffusion_graph, unnormalised=True) @ inverse_degree_matrix


def approximate_diffusion_update_step(measure_vector, hypergraph, step_size):
    """
    Given a current measure vector, hypergraph and step size, perform an update step of the approximate diffusion
    process.

    Returns the values
      - measure_vector
      - g(t)
      - f(t)
      - -log(f(t))
    :param measure_vector:
    :param hypergraph:
    :param step_size:
    :return:
    """
    # Apply the approximate diffusion operator. We construct the approximate reduced graph using the weighted
    # vector f.
    diffusion_operator_measure = approximate_measure_diffusion_operator(measure_vector, hypergraph)
    grad_hyp = - diffusion_operator_measure @ measure_vector
    new_measure_vector = measure_vector + step_size * grad_hyp

    # Add the graph points for this time step
    inverse_degree_matrix = hyplap.hypergraph_degree_mat(hypergraph, inverse=True)
    this_ft = new_measure_vector @ inverse_degree_matrix @ new_measure_vector
    negative_log_ft = - math.log(this_ft)
    x_tn = new_measure_vector / this_ft

    # Approximate the value of this_gt using the matrices of the approximate induced graph
    this_gt = compute_gt(x_tn, hypergraph, diffusion_operator_measure)

    # Return all the key values
    return new_measure_vector, this_gt, this_ft, negative_log_ft


def diffusion_update_step(measure_vector, hypergraph, step_size):
    """
    Given the current measure vector, hypergraph and step size, perform an update step of the diffusion process.

    Returns the values
      - measure_vector
      - g(t)
      - f(t)
      - -log(f(t))
    :param measure_vector:
    :param hypergraph:
    :param step_size:
    :return:
    """
    # Apply the diffusion operator
    grad_hyp = -hypergraph_measure_mc_laplacian(measure_vector, hypergraph)
    new_measure_vector = measure_vector + step_size * grad_hyp

    # Compute the graph points for this time step
    this_ft = new_measure_vector @ hyplap.hypergraph_degree_mat(hypergraph, inverse=True) @ new_measure_vector
    negative_log_ft = - math.log(this_ft)
    x_tn = new_measure_vector / this_ft
    this_gt = (x_tn @ np.linalg.inv(hyplap.hypergraph_degree_mat(hypergraph)) @
               hypergraph_measure_mc_laplacian(x_tn, hypergraph)) / (
                      x_tn @ np.linalg.inv(hyplap.hypergraph_degree_mat(hypergraph)) @ x_tn)

    return new_measure_vector, this_gt, this_ft, negative_log_ft


def choose_update_step_size(min_step, measure_vector, hypergraph):
    """
    Choose the optimal update step size for the given measure vector and hypergraph.
    :param min_step:
    :param measure_vector:
    :param hypergraph:
    :return:
    """
    hyplogging.logger.debug("Choosing a new step size.")
    # Which step size options we will consider
    options = [min_step, 2*min_step, 5*min_step, 10*min_step]

    # If all step sizes result in a small negative decrease in g(t), take the largest step size
    small_threshold = 0.0001
    all_small_negative = True

    # We will choose the step size which gives the best decrease in g(t) value for a single step.
    best_step_size = None
    best_gt_decrease = None
    current_gt = compute_gt(measure_vector, hypergraph,
                            approximate_measure_diffusion_operator(measure_vector, hypergraph))
    for step in options:
        # Perform an update step with this step size and check if it is the best
        _, this_gt, _, _ = approximate_diffusion_update_step(measure_vector, hypergraph, step)
        this_decrease = current_gt - this_gt
        hyplogging.logger.debug(f"Step size {step} has decrease {this_decrease}")

        if best_step_size is None or this_decrease > best_gt_decrease:
            hyplogging.logger.debug("... which is the best so far!")
            best_step_size = step
            best_gt_decrease = this_decrease

        if this_decrease > 0 or abs(this_decrease) > small_threshold:
            all_small_negative = False

    if all_small_negative:
        hyplogging.logger.debug("Since all decreases are small and negative, we will use the largest step size.")
        return options[-1]
    return best_step_size


# =======================================================
# Simulate the max cut heat diffusion process
# =======================================================
def sim_mc_heat_diff(phi, hypergraph, max_time=1, min_step=0.1, debug=False, plot_diff=False,
                     check_converged=False, approximate=False, adaptive_step_size=True):
    """
    Simulate the heat diffusion process for the hypergraph max cut operator.
    :param phi: The measure vector at the start of the process
    :param hypergraph: The underlying hypergraph
    :param max_time: The end time of the process
    :param min_step: The time interval to use for each step
    :param debug: Whether to print debug statements
    :param plot_diff: Whether to plot graphs showing the progression of the diffusion
    :param check_converged: Whether to check for convergence of G(t)
    :param approximate: Whether to use the approximate no-LP version of the diffusion operator
    :param adaptive_step_size: Whether to vary the step size to shorted convergence time
    :return: A measure vector at the end of the process, the final time of the diffusion process, the sequence of G(T)
    """
    hyplogging.logger.info(f"Beginning heat diffusion process.")
    hyplogging.logger.info(f"   max_time        = {max_time}")
    hyplogging.logger.info(f"   min_step        = {min_step}")
    hyplogging.logger.info(f"   approximate     = {approximate}")
    hyplogging.logger.info(f"   check_converged = {check_converged}")
    hyplogging.logger.info(f"adaptive_step_size = {adaptive_step_size}")

    # If we are going to plot the diffusion process, we will show the following quantities:
    #  F(t) = \phi_t D^{-1} \phi_t
    #  - log F(t)
    #  G(t) = d/dt - log F(t)
    #  h(t) = - d/dt G(t)
    t_steps = []
    f_t = []
    negative_log_ft = []
    g_t = []
    h_t = []

    # We may vary the step size, but we will start with the minimum one given
    step_size = min_step

    x_t = phi
    final_t = 0
    steps_since_choosing_size = 0
    steps_before_choosing_size = 1
    t = 0
    while t < max_time:
        steps_since_choosing_size += 1
        t_steps.append(t)

        # Update the step size every 10 steps
        if adaptive_step_size and steps_since_choosing_size > steps_before_choosing_size:
            old_step_size = step_size
            step_size = choose_update_step_size(min_step, x_t, hypergraph)
            steps_since_choosing_size = 0

            # If we have kept the same size, increase the steps to wait before checking
            if step_size == old_step_size:
                steps_before_choosing_size += max(2, int(0.4 * steps_before_choosing_size))
            else:
                steps_before_choosing_size = 1
            hyplogging.logger.debug(f"Number of steps before checking size again: {steps_before_choosing_size}")
        t += step_size

        # Apply an update step of the diffusion process
        if not approximate:
            new_xt, this_gt, this_ft, this_negative_log_ft = diffusion_update_step(x_t, hypergraph, step_size)
        else:
            new_xt, this_gt, this_ft, this_negative_log_ft = approximate_diffusion_update_step(
                x_t, hypergraph, step_size)

        x_t = new_xt
        f_t.append(this_ft)
        negative_log_ft.append(this_negative_log_ft)
        g_t.append(this_gt)
        h_t.append(compute_ht(g_t, step_size))
        hyplogging.logger.debug(f"time: {t}; g_t: {this_gt}")

        # Check for convergence
        if check_converged and diffusion_has_converged(t_steps, g_t):
            break

    # Now, plot the graphs
    if plot_diff:
        plot_convergence_graphs(t_steps, g_t)

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
    _, final_t, g_sequence = sim_mc_heat_diff(s, random_hypergraph, t, min_step=eps, check_converged=True)

    # Return the result
    g_length = len(g_sequence)
    return final_t, g_sequence[-1], g_sequence[int(g_length / 2)], g_sequence[int(9 * g_length / 10)]
