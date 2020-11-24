"""
This file contains code for computing the hypergraph max-cut laplacian operator.
"""
import hyplap
import hypcheeg
import numpy as np
from scipy.optimize import linprog
import hypernetx as hnx
import matplotlib.pyplot as plt


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
            else:
                Sm.append(e)
        if len([v for v in U if v in einfo[0]]) > 0:
            if einfo[2] < 0:
                Ip.append(e)
            else:
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


# =======================================================
# Simulate the max cut heat diffusion process
# =======================================================
def sim_mc_heat_diff(phi, H, T=1, step=0.1, debug=False):
    """
    Simulate the heat diffusion process for the hypergraph max cut operator.
    :param phi: The measure vector at the start of the process
    :param H: The underlying hypergraph
    :param T: The end time of the process
    :param step: The time interval to use for each step
    :return: A measure vector at the end of the process.
    """
    x_t = phi
    for t in np.linspace(0, T, int(T/step)):
        print(x_t)
        grad = -hypergraph_measure_mc_laplacian(x_t, H, debug=debug)
        x_t += step * grad
    return x_t


def main():
    # Construct a hypergraph
    H = hnx.Hypergraph({'e1': [1, 2, 3]})

    # Draw the hypergraph
    # hnx.draw(H)
    # plt.show()

    # Simulate the max cut diffusion process
    s = [1, 0, 0]
    h = sim_mc_heat_diff(s, H, 1, debug=False)
    print(h)


if __name__ == "__main__":
    main()

