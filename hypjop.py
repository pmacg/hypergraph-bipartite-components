"""
This file contains code for computing the new hypergraph J_H operator.
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


def find_densest_subset_jop(vertex_level_set, hypergraph, edge_info, debug=False):
    """
    Given a set of vertices U, find the densest subset using the linear program described in Appendix A.

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


def compute_jop_edge_info(f, hypergraph, debug=False):
    """
    Given a hypergraph H and weighted vector f, compute a dictionary with the following information for each edge in H:
        (I_f(e), S_f(e), max_{u in e} f(u) + min_{v in e} f(v), [vertices])
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
# Compute the hypergraph J_H operator for a given vector
# ==========================================================
def weighted_jop_diffusion_gradient(f, hypergraph, debug=False, approximate=False, construct_induced=False):
    """
    Given a vector in the weighted space, compute the gradient
         r = df/dt = - D_H^{-1} J_H f
    according to the new heat diffusion procedure.
    :param f:
    :param hypergraph:
    :param debug: Whether to print debug information
    :param approximate: Do not use the LP to compute the gradient, instead use the approximate induced graph.
    :param construct_induced: Whether to construct the induced graph at each step.
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
    edge_info = compute_jop_edge_info(f, hypergraph)

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

    # We will keep track of the r_e(v) values. Keys are edges. Values are lists of (vertex, r_e(v)) tuples.
    edge_induced_gradients = {}

    # We now iterate through the equivalence classes for the remainder of the algorithm
    for _, U in equiv_classes.items():
        u_temp = U

        # STEP 2 + 3
        # We need to find the subset P of U to maximise
        # \delta(P) = C(P) / w(P)
        while len(u_temp) > 0:
            best_p, max_delta = find_densest_subset_jop(u_temp, hypergraph, edge_info, debug=debug)

            # Update the r value for the best vertices and remove them from u_temp
            for v in best_p:
                r[vertex_name_to_index[v]] = max_delta
                u_temp.remove(v)

            # Compute the r_e(v) values for the vertices in best_p
            if construct_induced:
                # Set up the flow problem
                flow_graph = nx.DiGraph()
                flow_graph.add_node('s')
                flow_graph.add_node('t')

                # Add the nodes in the set T
                for v in best_p:
                    flow_graph.add_node(v)
                    if max_delta >= 0:
                        flow_graph.add_edge(v, 't', capacity=(hypergraph.degree(v) * max_delta))
                    else:
                        flow_graph.add_edge('s', v, capacity=abs(hypergraph.degree(v) * max_delta))

                # Add the nodes in the set E_T^+ and E_T^-
                for e, e_info in edge_info.items():
                    if len(set(best_p).intersection(e_info[0])) > 0 or len(set(best_p).intersection(e_info[1])) > 0:
                        edge_name = 'e' + str(e)
                        flow_graph.add_node(edge_name)

                        # Calculate the weight to add for this edge to the flow graph
                        # Usually, this is the value of \Delta_f(e), but if the edge contains only vertices inside P,
                        # then we should double this since the edge is in both S_P and I_P.
                        edge_weight = abs(e_info[2])
                        if len(set(e_info[3]).difference(best_p)) == 0:
                            edge_weight *= 2

                        if e_info[2] < 0:
                            # This edge is in E_T^+
                            flow_graph.add_edge('s', edge_name, capacity=edge_weight)
                            for v in best_p:
                                if v in e_info[3]:
                                    # Infinite capacity
                                    flow_graph.add_edge(edge_name, v)
                        else:
                            # This edge is in E_T^-
                            flow_graph.add_edge(edge_name, 't', capacity=edge_weight)
                            for v in best_p:
                                if v in e_info[3]:
                                    # Infinite capacity
                                    flow_graph.add_edge(v, edge_name)

                # Solve the flow problem and add the gradients to the dictionary
                _, flow_result = nx.maximum_flow(flow_graph, 's', 't')
                for vertex, flow_dictionary in flow_result.items():
                    if vertex in ['s', 't']:
                        continue
                    for other_vertex, flow in flow_dictionary.items():
                        if other_vertex in ['s', 't'] or flow == 0:
                            continue

                        # This is a non-zero flow
                        if type(vertex) == str:
                            # This vertex is an edge, meaning there is a positive rate of flow from this edge to the
                            # vertex
                            if vertex not in edge_induced_gradients:
                                edge_induced_gradients[vertex] = {other_vertex: flow}
                            else:
                                edge_induced_gradients[vertex][other_vertex] = flow
                        else:
                            # This vertex is an edge, meaning there is a negative rate of flow from the edge to the
                            # vertex
                            if other_vertex not in edge_induced_gradients:
                                edge_induced_gradients[other_vertex] = {vertex: -flow}
                            else:
                                edge_induced_gradients[other_vertex][vertex] = -flow

    # Now, construct the induced graph
    if construct_induced:
        induced_graph = nx.Graph()
        induced_graph_edges = {}

        # Add the vertices
        for v in hypergraph.nodes:
            induced_graph.add_node(v)

        # Compute the edges for each hyperedge
        for edge_name, weights in edge_induced_gradients.items():
            # Solve the flow problem to compute the edge weights in this edge
            flow_graph = nx.DiGraph()
            flow_graph.add_node('s')
            flow_graph.add_node('t')

            # Get the edge info for this edge
            e_info = edge_info[int(edge_name[1:])]

            i_e = []
            s_e = []
            for vertex, weight in weights.items():
                # Get the weight of this edge to be added to the flow graph
                # ordinarily, this will be the full weight given by weight. However, if the vertex appears in both S(e)
                # and I(e), then we need to divide by two.
                flow_weight = abs(weight) if not (vertex in e_info[0] and vertex in e_info[1]) else abs(weight) / 2

                if vertex in e_info[0]:
                    # This vertex is in I(e)
                    flow_graph.add_node(f"{vertex}_i")
                    flow_graph.add_edge(f"{vertex}_i", 't', capacity=flow_weight)
                    i_e.append(vertex)
                if vertex in e_info[1]:
                    # This vertex is in S(e)
                    flow_graph.add_node(f"{vertex}_s")
                    flow_graph.add_edge('s', f"{vertex}_s", capacity=flow_weight)
                    s_e.append(vertex)
            for v in i_e:
                for u in s_e:
                    # Add an infinite capacity edge between the vertices in s(e), i(e)
                    flow_graph.add_edge(f"{u}_s", f"{v}_i")

            # Solve the flow problem, and add the weights to the induced graph
            flow_value, flow_solution = nx.maximum_flow(flow_graph, 's', 't')
            for v, edges in flow_solution.items():
                for u, weight in edges.items():
                    if v in ['s', 't'] or u in ['s', 't'] or round(flow_value, 4) == 0:
                        continue
                    if round(weight/flow_value, 4) > 0:
                        pair = tuple(sorted((int(v[:-2]), int(u[:-2]))))
                        if pair not in induced_graph_edges:
                            induced_graph_edges[pair] = weight/flow_value
                        else:
                            induced_graph_edges[pair] += weight/flow_value

        # Add all of the edges to the induced graph
        for pair, weight in induced_graph_edges.items():
            induced_graph.add_edge(pair[0], pair[1], weight=round(weight, 2))

        # Plot the induced graph
        hyplogging.logger.info(f"f: {f}")
        hyplap.hyp_plot_with_graph(induced_graph, hypergraph.to_hypernetx(), plot_graph_weights=True)

    return r


def hypergraph_measure_j_operator(phi, hypergraph, debug=False, approximate=False, construct_induced=False):
    """
    Apply the hypergraph J_H operator to a vector phi in the measure space.
    In the normal language of graphs, this operator would be written as:
       J = I + A D^{-1}
    :param phi: The vector to apply the J_H operator to.
    :param hypergraph: The underlying hypergraph
    :param debug: Whether to print debug information
    :param approximate: Whether to use the approximate no-LP version
    :param construct_induced: Whether to construct and report the induced graph at each step
    :return: L_H x
    """
    f = hyplap.measure_to_weighted(phi, hypergraph)
    r = weighted_jop_diffusion_gradient(f, hypergraph, debug=debug, approximate=approximate,
                                        construct_induced=construct_induced)
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


def compute_rayleigh_quotient(varphi, hypergraph, measure_diffusion_operator):
    """
    Compute the value of the Rayleigh quotient of the operator J_H, given the measure vector varphi,
    the hypergraph and the diffusion operator at this time step.

    :param varphi:
    :param hypergraph:
    :param measure_diffusion_operator:
    :return:
    """
    inverse_degree_matrix = hyplap.hypergraph_degree_mat(hypergraph, inverse=True)
    return (varphi @ inverse_degree_matrix @ measure_diffusion_operator @ varphi) /\
           (varphi @ inverse_degree_matrix @ varphi)


def compute_ht(rqs, step_size):
    """Compute the value of h(t) = - d/dt R_H(f_t), given the list of rayleigh quotient values and the current step
    size."""
    if len(rqs) >= 2:
        return (rqs[-2] - rqs[-1]) / step_size
    else:
        return 0


def plot_convergence_graphs(t_steps, rqs):
    """Plot the graphs illustrating the convergence of the diffusion process.

    :param t_steps: the list of time steps used in the diffusion
    :param rqs: the list of Rayleigh quotient values to go along with t_steps
    """
    # Create the axes
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    # Label the axes
    ax1.set_xlabel("t")
    ax1.set_ylabel("F(t) or G(t)")
    ax2.set_ylabel("- log F(t)")

    # Plot the functions
    line1 = ax1.plot(t_steps[:len(rqs)], rqs)

    # Add legend and show plot
    ax2.legend(line1, "G(t) = d/dt - log F(t)")
    fig.tight_layout()
    plt.show()


def diffusion_has_converged(t_steps, rqs):
    """
    Determine whether the diffusion process has converged, given a list of time steps and R_H(f_t) values.
    :param t_steps:
    :param rqs:
    :return: boolean indicating convergence
    """
    # If we have reached a g(t) value of 0, we have converged
    if rqs[-1] < 0.0000000001:
        hyplogging.logger.debug(f"Diffusion process has converged to 0 at time {t_steps[-1]}.")
        return True

    # If we have been running for more than 11 time steps
    current_time = t_steps[-1]
    if current_time >= 11:
        # Get the value of g(t) for 10 time steps ago
        t_minus_10_index = 0
        for i, t in enumerate(t_steps):
            if t < current_time - 10:
                t_minus_10_index = i
        previous_gt = rqs[t_minus_10_index]
        diff = previous_gt - rqs[-1]

        hyplogging.logger.debug(f"Convergence target = 0.001; actual diff = {diff}")

        if diff < 0.001:
            # We have converged
            hyplogging.logger.debug(f"Diffusion process has converged at time {current_time}")
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
      - R_H(f_t)
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
    hyplogging.logger.debug(f"Computed new value of this_ft: {this_ft}")
    negative_log_ft = - math.log(this_ft) if this_ft > 0 else 100
    x_tn = new_measure_vector / this_ft

    # Approximate the value of this_rq using the matrices of the approximate induced graph
    this_rq = compute_rayleigh_quotient(x_tn, hypergraph, diffusion_operator_measure)

    # Return all the key values
    return new_measure_vector, this_rq, this_ft, negative_log_ft


def diffusion_update_step(measure_vector, hypergraph, step_size, construct_induced=False):
    """
    Given the current measure vector, hypergraph and step size, perform an update step of the diffusion process.

    Returns the values
      - measure_vector
      - R_H(f_t)
      - f(t)
      - -log(f(t))
    :param measure_vector:
    :param hypergraph:
    :param step_size:
    :param construct_induced:
    :return:
    """
    # Apply the diffusion operator
    grad_hyp = -hypergraph_measure_j_operator(measure_vector, hypergraph, construct_induced=construct_induced)
    new_measure_vector = measure_vector + step_size * grad_hyp

    # Compute the graph points for this time step
    this_ft = new_measure_vector @ hyplap.hypergraph_degree_mat(hypergraph, inverse=True) @ new_measure_vector
    negative_log_ft = - math.log(this_ft) if this_ft > 0 else 100
    x_tn = new_measure_vector / this_ft
    this_rq = (x_tn @ hyplap.hypergraph_degree_mat(hypergraph, inverse=True) @
               hypergraph_measure_j_operator(x_tn, hypergraph)) / (
                      x_tn @ hyplap.hypergraph_degree_mat(hypergraph, inverse=True) @ x_tn)

    return new_measure_vector, this_rq, this_ft, negative_log_ft


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
    current_gt = compute_rayleigh_quotient(measure_vector, hypergraph,
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
def sim_mc_heat_diff(phi, hypergraph, max_time=1, min_step=0.1, plot_diff=False,
                     check_converged=False, approximate=False, adaptive_step_size=True, construct_induced=False):
    """
    Simulate the heat diffusion process for the hypergraph max cut operator.
    :param phi: The measure vector at the start of the process
    :param hypergraph: The underlying hypergraph
    :param max_time: The end time of the process
    :param min_step: The time interval to use for each step
    :param plot_diff: Whether to plot graphs showing the progression of the diffusion
    :param check_converged: Whether to check for convergence of G(t)
    :param approximate: Whether to use the approximate no-LP version of the diffusion operator
    :param adaptive_step_size: Whether to vary the step size to shorted convergence time
    :param construct_induced: Whether to construct and report the induced graph at each step of the diffusion process.
    :return: A measure vector at the end of the process, the final time of the diffusion process, the sequence of G(T)
    """
    hyplogging.logger.debug(f"Beginning heat diffusion process.")
    hyplogging.logger.debug(f"   max_time        = {max_time}")
    hyplogging.logger.debug(f"   min_step        = {min_step}")
    hyplogging.logger.debug(f"   approximate     = {approximate}")
    hyplogging.logger.debug(f"   check_converged = {check_converged}")
    hyplogging.logger.debug(f"adaptive_step_size = {adaptive_step_size}")

    # If we are going to plot the diffusion process, we will show the following quantities:
    #  F(t) = \phi_t D^{-1} \phi_t
    #  - log F(t)
    #  G(t) = d/dt - log F(t)
    #  h(t) = - d/dt G(t)
    t_steps = []
    f_t = []
    negative_log_ft = []
    rqt = []
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
            if construct_induced:
                hyplogging.logger.info(f"t: {round(t - step_size, 2)}")
            new_xt, this_rq, this_ft, this_negative_log_ft = diffusion_update_step(x_t, hypergraph, step_size,
                                                                                   construct_induced=construct_induced)
        else:
            new_xt, this_rq, this_ft, this_negative_log_ft = approximate_diffusion_update_step(
                x_t, hypergraph, step_size)

        x_t = new_xt
        f_t.append(this_ft)
        negative_log_ft.append(this_negative_log_ft)
        rqt.append(this_rq)
        h_t.append(compute_ht(rqt, step_size))
        hyplogging.logger.debug(f"time: {t}; rqt: {this_rq}")

        # Check for convergence
        if check_converged and diffusion_has_converged(t_steps, rqt):
            break

    # Now, plot the graphs
    if plot_diff:
        plot_convergence_graphs(t_steps, rqt)

    return x_t, final_t, rqt


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
