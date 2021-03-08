"""
This file contains code for computing the hypergraph max-cut laplacian operator.
"""
import math
import hyplap
import hypconstruct
import hypcheeg
import numpy as np
from scipy.optimize import linprog
import scipy as sp
import scipy.sparse
import hypernetx as hnx
import networkx as nx
import matplotlib.pyplot as plt
import random


def find_densest_subset_mc(U, H, edge_info, debug=False):
    """
    Given a set of vertices U, find the densest subset as per the new linear program operation
    :param U: the vertex level-set
    :param H: the underlying hypergraph
    :param edge_info: the pre-computed information about every edge in the graph
    :param debug: whether to make debug logs
    :return:
    """
    if debug:
        print("Finding densest subgraph for", U)
    # We will solve this problem using the linear program given in the original paper.
    # First, we need to compute the sets S+, S-, I+, I-.
    Ip = []
    Im = []
    Sp = []
    Sm = []
    for e, einfo in edge_info.items():
        # Check whether the edge maximum/minimum set has a non-zero intersection with U
        if len([v for v in U if v in einfo[1]]) > 0:
            if einfo[2] < 0:
                Sp.append(e)
            elif einfo[2] > 0:
                Sm.append(e)
        if len([v for v in U if v in einfo[0]]) > 0:
            if einfo[2] < 0:
                Ip.append(e)
            elif einfo[2] > 0:
                Im.append(e)

    if debug:
        print('Ip', Ip, 'Im', Im, 'Sp', Sp, 'Sm', Sm)

    # If I and S are both empty, then we can return with zero weight
    if len(Ip) + len(Im) + len(Sp) + len(Sm) == 0:
        best_P = np.copy(U)
        max_delta = 0
        if debug:
            print("Returning early since Ip, Im, Sp and Sm are empty")
            print("best_P:", best_P)
            print("max_delta:", max_delta)
        return best_P, max_delta

    # Now, we construct the linear program.
    # The variables are as follows:
    #   x_e for each e in Ip \union Im \union Sp \union Sm
    #   y_v for each v in U
    num_vars = len(Ip) + len(Im) + len(Sp) + len(Sm) + len(U)

    # The objective function to be minimised is
    #    -c(x) = - sum_{Sp} c_e x_e - sum_{Ip} c_e x_e + sum_{Sm} c_e x_e + sum_{Im} c_e x_e
    obj = np.zeros(num_vars)
    i = 0
    for e in Sp:
        obj[i] = edge_info[e][2]
        i += 1
    for e in Ip:
        obj[i] = edge_info[e][2]
        i += 1
    for e in Sm:
        obj[i] = edge_info[e][2]
        i += 1
    for e in Im:
        obj[i] = edge_info[e][2]
        i += 1
    if debug:
        print("obj", obj)

    # The equality constraints are
    #    sum_{v \in U} w_v y_v = 1
    lhs_eq = [np.zeros(num_vars)]
    rhs_eq = [1]
    for i, v in enumerate(U):
        lhs_eq[0][len(Sp) + len(Ip) + len(Sm) + len(Im) + i] = H.degree(v)
    if debug:
        print('lhs_eq', lhs_eq)

    #    x_e = y_u         for all e \in Sp, u \in e
    for e in Sp:
        for v in edge_info[e][3]:
            if v in U:
                lhs = np.zeros(num_vars)
                vert_index = len(Sp) + len(Ip) + len(Sm) + len(Im) + U.index(v)
                edge_index = Sp.index(e)
                lhs[edge_index] = 1
                lhs[vert_index] = -1

                # Add this inequality to the lists
                if debug:
                    print(f"Added inequality: {e} - {v} = 0")
                lhs_eq.append(lhs)
                rhs_eq.append(0)

    #    x_e = y_u         for all e \in Im, u \in e
    for e in Im:
        for v in edge_info[e][3]:
            if v in U:
                lhs = np.zeros(num_vars)
                vert_index = len(Sp) + len(Ip) + len(Sm) + len(Im) + U.index(v)
                edge_index = len(Sp) + len(Ip) + len(Sm) + Im.index(e)
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
    lhs_ineq = []
    rhs_ineq = []
    for e in Ip:
        for v in edge_info[e][3]:
            if v in U:
                # Construct the left hand side of the inequality in terms of the coefficients.
                lhs = np.zeros(num_vars)
                vert_index = len(Sp) + len(Ip) + len(Sm) + len(Im) + U.index(v)
                edge_index = len(Sp) + Ip.index(e)
                lhs[edge_index] = 1
                lhs[vert_index] = -1

                # Add this inequality to the lists
                if debug:
                    print(f"Added inequality: {e} - {v} < 0")
                lhs_ineq.append(lhs)
                rhs_ineq.append(0)
    for e in Sm:
        for v in edge_info[e][3]:
            if v in U:
                # Construct the left hand side of the inequality in terms of the coefficients.
                lhs = np.zeros(num_vars)
                vert_index = len(Sp) + len(Ip) + len(Sm) + len(Im) + U.index(v)
                edge_index = len(Sp) + len(Ip) + Sm.index(e)
                lhs[edge_index] = -1
                lhs[vert_index] = 1

                # Add this inequality to the lists
                if debug:
                    print(f"Added inequality: {v} - {e} < 0")
                lhs_ineq.append(lhs)
                rhs_ineq.append(0)

    # Now, solve the linear program.
    # Set unused constraints to none
    if len(lhs_ineq) == 0:
        lhs_ineq = None
        rhs_ineq = None
    if len(lhs_eq) == 0:
        lhs_eq = None
        rhs_eq = None
    if debug:
        print(f"c: {obj}; A_ub: {lhs_ineq}; b_ub: {rhs_ineq}; A_eq: {lhs_eq}; b_eq: {rhs_eq}")
    opt = linprog(c=obj, A_ub=lhs_ineq, b_ub=rhs_ineq, A_eq=lhs_eq, b_eq=rhs_eq)
    if debug:
        print(opt)

    # Find a level set of the output of the linear program.
    max_delta = -np.round(opt.fun, decimals=5)
    soln = np.round(opt.x, decimals=5)
    thresh = 0
    for i in range(len(U)):
        if soln[len(Sp) + len(Ip) + len(Sm) + len(Im) + i] > thresh:
            thresh = soln[len(Sp) + len(Ip) + len(Sm) + len(Im) + i]
    if debug:
        print("thresh:", thresh)
    best_P = []
    for i in range(len(U)):
        if soln[len(Sp) + len(Ip) + len(Sm) + len(Im) + i] >= thresh:
            best_P.append(U[i])

    if debug:
        print("best_P:", best_P)
        print("max_delta:", max_delta)
    return best_P, max_delta


def compute_mc_edge_info(f, H, debug=False):
    """
    Given a hypergraph H and weighted vector f, compute a dictionary with the following information for each edge in H:
        (Ie, Se, max_{u \in e} f(u) + min_{v \in e} f(v), [vertices])
    :param f: A vector in the normalised space
    :param H: The underlying hypergraph
    :return: a dictionary with the above information for each edge in the hypergraph.
    """
    maxf = max(f)
    minf = min(f)
    if debug:
        print('f', f)
        print('maxf', maxf, 'minf', minf)
    edge_info = {}

    # Compute a dictionary of vertex names and vertex ids
    vname_to_vidx = {}
    for vidx, vname in enumerate(H.nodes):
        vname_to_vidx[vname] = vidx
    if debug:
        print('vname_to_vidx', vname_to_vidx)

    for e in H.edges():
        if debug:
            print("Processing edge:", e)
        # Compute the maximum and minimum sets for the edge.
        Ie = []
        mine = maxf
        Se = []
        maxe = minf
        for v in e.elements:
            if debug:
                print("Considering vertex:", v)
            fv = f[vname_to_vidx[v]]
            if debug:
                print("fv", fv)

            if fv < mine:
                if debug:
                    print("New minimum.")
                mine = fv
                Ie = [v]
            if fv > maxe:
                if debug:
                    print("New maximum.")
                maxe = fv
                Se = [v]
            if fv == mine and v not in Ie:
                if debug:
                    print("Updating minimum.")
                Ie.append(v)
            if fv == maxe and v not in Se:
                if debug:
                    print("Updating maximum.")
                Se.append(v)
            if debug:
                print("Ie", Ie, "Se", Se)
        edge_info[e.uid] = (Ie, Se, maxe + mine, e.elements)
    if debug:
        print("Returning:", edge_info)
    return edge_info


# ==========================================================
# Compute the hypergraph max cut operator for a given vector
# ==========================================================
def weighted_mc_diffusion_gradient(f, H, debug=False):
    """
    Given a vector in the weighted space, compute the gradient
         r = df/dt
    according to the max cut heat diffusion procedure.
    :param f:
    :param H:
    :return:
    """
    # We will round the vector f to a small(ish) precision so as to
    # avoid vertices ending in the wrong equivalence classes due to numerical errors.
    f = np.round(f, decimals=5)

    # Compute some standard information about every edge in the hypergraph
    edge_info = compute_mc_edge_info(f, H)

    # This is the eventual output of the algorithm
    r = np.zeros(len(H.nodes))

    # We will refer to the steps of the procedure in Figure 4.1 of the reference paper. Starting with...
    # STEP 1
    # Find the equivalence classes of the input vector.
    equiv_classes = {}
    vname_to_vidx = {}
    for vidx, vname in enumerate(H.nodes):
        vname_to_vidx[vname] = vidx
        # Check if we have seen this value yet
        if f[vidx] in equiv_classes:
            equiv_classes[f[vidx]].append(vname)
        else:
            # This is the first time we've seen this value
            equiv_classes[f[vidx]] = [vname]

    if debug:
        print("Equivalence classes:", equiv_classes)

    # We now iterate through the equivalence classes for the remainder of the algorithm
    for _, U in equiv_classes.items():
        Utemp = U

        # STEP 2 + 3
        # We need to find the subset P of U to maximise
        # \delta(P) = C(P) / w(P)
        while len(Utemp) > 0:
            best_P, max_delta = find_densest_subset_mc(Utemp, H, edge_info, debug=debug)

            # Update the r value for the best vertices and remove them from Utemp
            for v in best_P:
                r[vname_to_vidx[v]] = max_delta
                Utemp.remove(v)
    if debug:
        print("Complete Gradient:", r)
    return r


def hypergraph_measure_mc_laplacian(phi, H, debug=False):
    """
    Apply the hypergraph max cut operator to a vector phi in the measure space.
    In the normal language of graphs, this operator would be written as:
       L = I + A D^{-1}
    :param phi: The vector to apply the laplacian operator to.
    :param H: The underlying hypergraph
    :return: L_H x
    """
    f = hyplap.measure_to_weighted(phi, H)
    r = weighted_mc_diffusion_gradient(f, H, debug=debug)
    return -hyplap.hypergraph_degree_mat(H) @ r


def graph_diffusion_operator(G):
    """
    Construct the operator (I + A D^{-1}) for the given graph.
    :param G: A networkx graph
    :return: a scipy matrix
    """
    A = nx.to_scipy_sparse_matrix(G, format="csr")
    n, m = A.shape
    degrees = A.sum(axis=1)
    D = sp.sparse.spdiags(degrees.flatten(), [0], m, n, format="csr")
    D_inv = sp.sparse.linalg.inv(D)
    L = sp.sparse.identity(n) + A @ D_inv
    return L


# =======================================================
# Simulate the max cut heat diffusion process
# =======================================================
def sim_mc_heat_diff(phi, H, T=1, step=0.1, debug=False, plot_diff=False, save_diffusion_data=False,
                     print_measure=False, normalise=False, print_time=False, check_converged=False, start_epsilon=0):
    """
    Simulate the heat diffusion process for the hypergraph max cut operator.
    :param phi: The measure vector at the start of the process
    :param H: The underlying hypergraph
    :param T: The end time of the process
    :param step: The time interval to use for each step
    :param debug: Whether to print debug logs
    :param plot_diff: Whether to plot graphs showing the progression of the diffusion
    :param save_diffusion_data: Whether to save a csv file containing the diffusion data
    :param print_measure: Whether to print the measure vector at each step
    :param normalise: Whether to normalise the measure vector at each step
    :param print_time: Whether to print the time steps
    :param check_converged: Whether to check for convergence of G(t)
    :param start_epsilon: At each iteration, we will add a small epsilon of the clique graph. This is the starting value
                          of epsilon. If it is equal to 0, then we will use the basic diffusion process.
    :return: A measure vector at the end of the process, the final time of the diffusion process, the full sequence of G(T)
    """
    # If we are going to plot the diffusion process, we will show the following quantities:
    #  F(t) = \phi_t D^{-1} \phi_t
    #  - log F(t)
    #  G(t) = d/dt - log F(t)
    #  h(t) = - d/dt G(t)
    t_steps = np.linspace(0, T, int(T/step))
    ft = []
    mlogft = []
    gt = []
    ht = []

    # Construct the clique graph and its laplacian operator
    n = H.number_of_nodes()
    G = hypconstruct.get_clique_graph(H)
    L_clique = graph_diffusion_operator(G)

    # Open the text file to write
    if save_diffusion_data:
        fout = open("diffusion_data.csv", 'w')
        foutw = open("diffusion_data_weighted.csv", 'w')
    else:
        fout = None
        foutw = None

    x_t = phi
    final_t = 0
    for t in t_steps:
        final_t = t
        eps_t = start_epsilon / (t + 1)
        if print_measure:
            print(f"Time {t:.2f}; eps_t = {eps_t:.2f}; rho_t = {x_t}")
        elif print_time:
            print(f"Time {t:.2f}; eps_t = {eps_t:.2f}")
        if save_diffusion_data:
            fout.write(f"{','.join([str(x) for x in x_t])}\n")
            foutw.write(f"{','.join([str(x) for x in hyplap.measure_to_weighted(x_t, H)])}\n")

        # Apply the diffusion operator
        grad_hyp = -hypergraph_measure_mc_laplacian(x_t, H, debug=debug)
        grad_clique = - L_clique @ x_t
        grad_combined = (1 - eps_t) * grad_hyp + eps_t * grad_clique
        x_t += step * grad_combined

        if normalise:
            x_t = x_t / (x_t @ x_t)

        # Add the graph points for this time step
        this_ft = x_t @ np.linalg.inv(hyplap.hypergraph_degree_mat(H)) @ x_t
        ft.append(this_ft)
        mlogft.append(- math.log(this_ft))
        x_tn = x_t / this_ft
        this_gt = (x_tn @ np.linalg.inv(hyplap.hypergraph_degree_mat(H)) @ hypergraph_measure_mc_laplacian(x_tn, H)) / (
                   x_tn @ np.linalg.inv(hyplap.hypergraph_degree_mat(H)) @ x_tn)
        gt.append(this_gt)
        if len(gt) > 2:
            ht.append((gt[-2] - gt[-1])/step)
        elif len(gt) == 2:
            ht.append((gt[-2] - gt[-1])/step)
            ht.append((gt[-2] - gt[-1])/step)

        # Check for convergence
        if check_converged:
            gt_len = len(gt)
            if gt_len >= 30:
                prev_gt = gt[int(gt_len / 3)]
                diff = prev_gt - this_gt
                if diff < 0.0000001:
                    # We have converged
                    break

    if save_diffusion_data:
        fout.close()
        foutw.close()

    # Print the final value of G(t). If the process is fully converged, then this is an eigenvalue of the hypergraph
    # laplacian operator.
    this_ft = x_t @ np.linalg.inv(hyplap.hypergraph_degree_mat(H)) @ x_t
    x_tn = x_t / this_ft
    final_gt = (x_tn @ np.linalg.inv(hyplap.hypergraph_degree_mat(H)) @ hypergraph_measure_mc_laplacian(x_tn, H)) / (
                x_tn @ np.linalg.inv(hyplap.hypergraph_degree_mat(H)) @ x_tn)
    if print_time or print_measure:
        print(f"Final value of G(t): {final_gt:.5f}")

    # Now, plot the graphs
    if plot_diff:
        # Create the axes
        fig, ax1 = plt.subplots()
        if not normalise:
            ax2 = ax1.twinx()

        # Label the axes
        ax1.set_xlabel("t")
        ax1.set_ylabel("F(t) or G(t)")
        if not normalise:
            ax2.set_ylabel("- log F(t)")

        # Plot the functions
        if not normalise:
            # line1 = ax1.plot(t_steps[:len(ft)], ft)
            # line2 = ax2.plot(t_steps[:len(mlogft)], mlogft, color='tab:green')
            line4 = ax1.plot(t_steps[:len(ht)], ht)
        line3 = ax1.plot(t_steps[:len(gt)], gt)

        # Add legend and show plot
        if not normalise:
            # ax2.legend(line1 + line2 + line3 + line4, ("F(t) = \\rho_t^T D^{-1} \\rho_t", "- log F(t)", "G(t) = d/dt - log F(t)", "h(t) = -d/dt G(t)"))
            # ax2.legend(line1 + line3 + line4,
            #            ("F(t) = \\rho_t^T D^{-1} \\rho_t", "G(t) = d/dt - log F(t)", "h(t) = -d/dt G(t)"))
            ax2.legend(line3 + line4,
                       ("G(t) = d/dt - log F(t)", "h(t) = -d/dt G(t)"))
        fig.tight_layout()
        plt.show()

    return x_t, final_t, gt


def check_random_2_color_graph(n, m, r, t, eps):
    """
    Check the final eigenvalue of a random 2-colorable graph. Returns the following tuple:
        T, G(T), G(T/2), G(9T/10)
    :param n: Half the number of vertices in the graph
    :param m: The number of edges in the hypergraph
    :param r: Half the rank of each edge
    :param t: The maximum end time of the diffusion process
    :param eps: The update step size during the diffusion process
    :return: Tuple described above
    """
    H = hypconstruct.construct_hyp_2_colorable(n, n, m, r)

    # Create the starting vector
    s = np.zeros(2 * n)
    s[1] = 1

    # Run the heat diffusion process
    _, final_t, g_sequence = sim_mc_heat_diff(s, H, t,
                                              step=eps,
                                              normalise=True,
                                              check_converged=True)

    # Return the result
    g_length = len(g_sequence)
    return final_t, g_sequence[-1], g_sequence[int(g_length/2)], g_sequence[int(9 * g_length / 10)]


def main():
    # Choose which example set-up to demonstrate
    #  1: Single 3-edge
    #  20s: 3-uniform not 2-colorable
    #  30s: 2-graph versions of the 20s
    #  40s: Example of hypergraph with two eigenvectors
    #  50s: Look at random 2-colorable graphs
    #  60s: Search (!) for 2-colorable graphs with bad algorithm results
    example = 50
    n = 300
    show_hypergraph = False
    show_diffusion = True

    if example == 1:
        # Construct a hypergraph
        H = hnx.Hypergraph({'e1': [1, 2, 3], 'e2': [2, 3]})

        # Draw the hypergraph
        if show_hypergraph:
            hyp_fig = plt.figure(1)
            hnx.draw(H)
            hyp_fig.show()

        # Simulate the heat diffusion
        s = [1, 0, 0]
        _ = sim_mc_heat_diff(s, H, 20, step=0.01, debug=False, plot_diff=True, check_converged=True, print_measure=True, normalise=True)

    if 20 <= example < 30:
        H = hnx.Hypergraph(
            {
                'e1': [1, 2, 3],
                'e2': [1, 2, 4],
                'e3': [1, 2, 5],
                'e4': [3, 4, 5],
                'e5': [2, 6, 7],
                'e6': [2, 6, 8],
                'e7': [2, 6, 9],
                'e8': [7, 8, 9],
                'e9': [6, 1, 10],
                'e10': [6, 1, 11],
                'e11': [6, 1, 12],
                'e12': [10, 11, 12]
            })

        # Draw the hypergraph
        if show_hypergraph:
            hyp_fig = plt.figure(1)
            hnx.draw(H)
            hyp_fig.show()

        # Simulate the max cut diffusion process
        s = np.zeros(12)
        if example == 21:
            s[0] = 1
        if example == 22:
            s[2] = 1
        if example == 23:
            s = [0.1 * x for x in [-1.4, -1.4, 0.22, 0.19, 0.19, 1.6, 0, 0, 0, 0, 0, 0]]
        h = sim_mc_heat_diff(s, H, 10, step=0.01, debug=False, plot_diff=True, save_diffusion_data=True)

    if 30 <= example < 40:
        if example == 30:
            H = hnx.Hypergraph(
                {
                    'e1': [1, 2],
                    'e2': [1, 2],
                    'e3': [1, 2],
                    'e41': [3, 4],
                    'e42': [4, 5],
                    'e43': [5, 3],
                    'e51': [2, 7],
                    'e52': [6, 7],
                    'e61': [2, 8],
                    'e62': [6, 8],
                    'e71': [2, 9],
                    'e72': [6, 9],
                    'e81': [7, 8],
                    'e82': [8, 9],
                    'e83': [9, 7],
                    'e9': [6, 1],
                    'e121': [10, 11],
                    'e122': [11, 12],
                    'e123': [12, 10],
                })
        elif example == 31:
            H = hypconstruct.construct_hyp_2_colorable(n, n, 6 * n, 1)

        # Draw the hypergraph
        if show_hypergraph:
            hyp_fig = plt.figure(1)
            hnx.draw(H)
            hyp_fig.show()

        # Simulate the max cut diffusion process
        if example == 30:
            s = np.zeros(12)
        elif example == 31:
            s = np.zeros(2 * n)
        s[0] = 1
        h = sim_mc_heat_diff(s, H, 100, step=0.1, debug=False, plot_diff=True, save_diffusion_data=True, print_measure=True, check_converged=True)

    if example == 40:
        # Construct a random 3-uniform hypergraph
        n = 10
        m = 20
        # H = hypconstruct.random_hypergraph(n, m, 3)
        # 'Random hypergraph'
        # vertex order: 6, 4, 8, 1, 5, 3, 7, 10, 2, 9
        H = hnx.Hypergraph({'e1': ['a6', 'a4', 'a8'],
                            'e2': ['a1', 'a5', 'a3'],
                            'e3': ['a6', 'a7', 'a10'],
                            'e4': ['a2', 'a4', 'a1'],
                            'e5': ['a9', 'a4', 'a6'],
                            'e6': ['a2', 'a4', 'a6'],
                            'e7': ['a10', 'a7', 'a3'],
                            'e8': ['a2', 'a1', 'a8'],
                            'e9': ['a1', 'a7', 'a2'],
                            'e10': ['a3', 'a6', 'a9'],
                            'e11': ['a7', 'a2', 'a6'],
                            'e12': ['a1', 'a9', 'a5'],
                            'e13': ['a8', 'a10', 'a9'],
                            'e14': ['a3', 'a9', 'a4'],
                            'e15': ['a8', 'a7', 'a4'],
                            'e16': ['a5', 'a10', 'a8'],
                            'e17': ['a4', 'a2', 'a3'],
                            'e18': ['a9', 'a5', 'a7'],
                            'e19': ['a2', 'a5', 'a6'],
                            'e20': ['a4', 'a9', 'a7']})

        # Plot the hypergraph
        if show_hypergraph:
            hyp_fig = plt.figure(1)
            hnx.draw(H)
            hyp_fig.show()
            plt.show()

        # First eigenvector - converged to from starting vertex index 4
        # s = hyplap.weighted_to_measure([0.185571, 0.316896, -0.358508, -0.348528, 0.2466, -0.348606, -0.348613, 0.394087, 0.185571, -0.348613], H)
        # s = np.zeros(n)
        # s[8] = 1

        # Second eigenvector - converged to from starting vertex index 8
        # s = hyplap.weighted_to_measure([-0.371296, 0.351462, -0.244721, -0.244721, -0.244721, -0.244721, -0.371296, 0.344727, 0.351462, 0.344727], H)

        # Start from the all-ones vector
        s = 0.2 * hyplap.weighted_to_measure(np.ones(n), H)

        # Run the heat diffusion process
        _ = sim_mc_heat_diff(s, H, 20,
                             step=0.01,
                             print_measure=True,
                             plot_diff=True,
                             save_diffusion_data=True,
                             normalise=False)

    if example == 50:
        H = hypconstruct.construct_hyp_2_colorable(n, n, 3*n, 2)

        # Plot the hypergraph
        if show_hypergraph:
            hyp_fig = plt.figure(1)
            hnx.draw(H)
            hyp_fig.show()
            if not show_diffusion:
                plt.show()

        # Create the starting vector - start at the minimum eigenvector of the clique graph operator
        L_clique = graph_diffusion_operator(hypconstruct.get_clique_graph(H))
        eigs, eigvecs = sp.sparse.linalg.eigsh(L_clique, k=1, which='SM')
        s = eigvecs[:, 0]
        # s = np.zeros(2*n)
        # s[1] = 1

        # Run the heat diffusion process
        _ = sim_mc_heat_diff(s, H, 30,
                             step=1,
                             print_measure=False,
                             print_time=True,
                             plot_diff=show_diffusion,
                             save_diffusion_data=False,
                             normalise=False,
                             start_epsilon=0)

    if example == 60:
        # Run some experiments
        # Which values of n to test
        values_of_n = [5, 10, 50, 100, 500, 1000]

        # Which values of r to test
        values_of_r = [2]

        # Which values of epsilon to test
        values_of_epsilon = [0.1]

        # How many of each combination to test
        num_to_check = 20

        # Iterate through all of the test cases
        with open("testresults.csv", 'w') as fout:
            # Save the header line
            to_print = "test, n, m, r, T, eps, G(T), G(T/2), G(9T/10)\n"
            fout.write(to_print)
            fout.flush()
            testcase = 0
            for n in values_of_n:
                for r in values_of_r:
                    for eps in values_of_epsilon:
                        for i in range(1, num_to_check + 1):
                            testcase += 1
                            T, G_T, G_T_2, G_9T_10 = check_random_2_color_graph(n, 6 * n, r, 4 * n, eps)
                            to_print = f"{testcase}, {n}, {6 * n}, {r}, {T}, {eps}, {G_T}, {G_T_2}, {G_9T_10}\n"
                            fout.write(to_print)
                            fout.flush()


if __name__ == "__main__":
    main()

