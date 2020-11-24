"""
This file implements methods for using the hypergraph laplacian operator.
The terminology used in this file comes from the following paper: https://arxiv.org/abs/1605.01483
"""
import networkx as nx
import hypernetx as hnx
import numpy as np
import matplotlib.pyplot as plt
import hypconstruct
import hypcheeg
import sys
from scipy.optimize import linprog


def hypergraph_degree_mat(H):
    """
    Given a hypergraph H, compute the degree matrix containing the degree of each vertex on the diagonal.
    :param H:
    :return: A numpy matrix containing the degrees.
    """
    # Get the vertices from the hypergraph
    V = H.nodes
    n = len(V)

    # Construct the empty matrix whose diagonal we will fill.
    W = np.zeros((n, n))
    for vidx, vname in enumerate(V):
        W[vidx, vidx] = H.degree(vname)
    return W


# =======================================================
# Convert between measure, weighted and normalised space
# =======================================================
def measure_to_weighted(phi, H):
    """
    Givena vector in the measure space, compute the corresponding vector in the weighted space.
            f = W^{-1} x
    :param phi: The vector in the measure space to be converted.
    :param H: The underlying hypergraph.
    :return: The vector f in the weighted space.
    """
    return np.linalg.inv(hypergraph_degree_mat(H)) @ phi


def weighted_to_normalized(f, H):
    return np.sqrt(hypergraph_degree_mat(H)) @ f


def normalized_to_measure(x, H):
    return np.sqrt(hypergraph_degree_mat(H)) @ x


def measure_to_normalized(phi, H):
    return weighted_to_normalized(measure_to_weighted(phi, H), H)


def normalized_to_weighted(x, H):
    return measure_to_weighted(normalized_to_measure(x, H), H)


def weighted_to_measure(f, H):
    return normalized_to_measure(weighted_to_normalized(f, H), H)


def compute_edge_info(f, H, debug=False):
    """
    Given a hypergraph H and weighted vector f, compute a dictionary with the following information for each edge in H:
        (Ie, Se, max_{u, v \in e} (f(u) - f(v), [vertices])
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
        edge_info[e.uid] = (Ie, Se, maxe - mine, e.elements)
    if debug:
        print("Returning:", edge_info)
    return edge_info


def delta(P, edge_info, H):
    """
    Given some set of vertices P and the hypergraph edge information, compute the delta value as defined in the paper.
        \delta(P) = C(P) / w(P)
    where
        C(P) = c(I_P) - c(S_P)
    and w(P) is just the number of vertices in P.
    :param P: a set of vertices in the hypergraph.
    :param edge_info: the standard, precomputed information about the hypergraph edges.
    :return: the computed delta value.
    """
    # Compute the I and S edge sets
    I = []
    S = []
    for e, einfo in edge_info.items():
        # Check whether the edge maximum/minimum set has a non-zero intersection with U
        if len([v for v in P if v in einfo[1]]) > 0:
            S.append(e)
        if len([v for v in P if v in einfo[0]]) == len(einfo[0]):
            I.append(e)

    # Compute C(P) and w(P)
    CP = 0
    for e in I:
        CP += edge_info[e][2]
    for e in S:
        CP -= edge_info[e][2]

    # w(P) is the sum of the original degrees of the vertices
    wP = 0
    for v in P:
        wP += H.degree(v)

    # Return the final value
    if CP == 0:
        return 0
    else:
        return CP / wP


def powerset(s):
    x = len(s)
    masks = [1 << i for i in range(x)]
    for i in range(1, 1 << x):
        yield [ss for mask, ss in zip(masks, s) if i & mask]


def find_densest_subset(U, H, edge_info, debug=False):
    """
    Given a set of vertices U, find the densest subset as per section 4.2 of the original paper.
    :param U: the vertex level-set
    :param H: the underlying hypergraph
    :param edge_info: the pre-computed information about every edge in the graph
    :param debug: whether to make debug logs
    :return:
    """
    if debug:
        print("Finding densest subgraph for", U)
    # We will solve this problem using the linear program given in the original paper.
    # First, we need to compute the sets I and S.
    I = []
    S = []
    for e, einfo in edge_info.items():
        # Check whether the edge maximum/minimum set has a non-zero intersection with U
        if len([v for v in U if v in einfo[1]]) > 0:
            S.append(e)
        if len([v for v in U if v in einfo[0]]) > 0:
            I.append(e)

    if debug:
        print('I', I, 'S', S)

    # If I and S are both empty, then we can return with zero weight
    if len(I) + len(S) == 0:
        best_P = np.copy(U)
        max_delta = 0
        if debug:
            print("Returning early since I and S are empty")
            print("best_P:", best_P)
            print("max_delta:", max_delta)
        return best_P, max_delta

    # Now, we construct the linear program.
    # The variables are as follows:
    #   x_e for each e in I \union S
    #   y_v for each v in U
    num_vars = len(I) + len(S) + len(U)

    # The objective function to be minimised is
    #    -c(x) = - sum_{I} c_e x_e + sum_{S} c_e x_e
    obj = np.zeros(num_vars)
    i = 0
    for e in I:
        obj[i] = -edge_info[e][2]
        i += 1
    for e in S:
        obj[i] = edge_info[e][2]
        i += 1
    if debug:
        print("obj", obj)

    # The equality constraints are
    #    sum_{v \in U} w_v y_v = 1
    lhs_eq = [np.zeros(num_vars)]
    rhs_eq = 1
    for i, v in enumerate(U):
        lhs_eq[0][len(I) + len(S) + i] = H.degree(v)
    if debug:
        print('lhs_eq', lhs_eq)

    # The inequality constraints are:
    #   x_e - y_v \leq 0    for all e \in I, v \in e
    #   y_v - x_e \leq 0    for all e \in S, v \in e
    lhs_ineq = []
    rhs_ineq = []
    edge_index = -1
    for e in I:
        edge_index += 1
        for v in edge_info[e][3]:
            if v in U:
                # Construct the left hand side of the inequality in terms of the coefficients.
                lhs = np.zeros(num_vars)
                vert_index = len(I) + len(S) + U.index(v)
                lhs[edge_index] = 1
                lhs[vert_index] = -1

                # Add this inequality to the lists
                if debug:
                    print(f"Added inequality: {e} - {v} < 0")
                lhs_ineq.append(lhs)
                rhs_ineq.append(0)
    for e in S:
        edge_index += 1
        for v in edge_info[e][3]:
            if v in U:
                # Construct the left hand side of the inequality in terms of the coefficients.
                lhs = np.zeros(num_vars)
                vert_index = len(I) + len(S) + U.index(v)
                lhs[edge_index] = -1
                lhs[vert_index] = 1

                # Add this inequality to the lists
                if debug:
                    print(f"Added inequality: {v} - {e} < 0")
                lhs_ineq.append(lhs)
                rhs_ineq.append(0)

    # Now, solve the linear program.
    opt = linprog(c=obj, A_ub=lhs_ineq, b_ub=rhs_ineq, A_eq=lhs_eq, b_eq=rhs_eq)
    if debug:
        print(opt)

    # Find a level set of the output of the linear program.
    max_delta = -np.round(opt.fun, decimals=5)
    soln = np.round(opt.x, decimals=5)
    thresh = 0
    for i in range(len(U)):
        if soln[len(I) + len(S) + i] > thresh:
            thresh = soln[len(I) + len(S) + i]
    if debug:
        print("thresh:", thresh)
    best_P = []
    for i in range(len(U)):
        if soln[len(I) + len(S) + i] >= thresh:
            best_P.append(U[i])

    if debug:
        print("best_P:", best_P)
        print("max_delta:", max_delta)
    return best_P, max_delta


# ====================================================
# Compute the hypergraph laplacian for a given vector
# ====================================================
def weighted_diffusion_gradient(f, H, debug=False):
    """
    Given a vector in the weighted space, compute the gradient
         r = df/dt
    according to the heat diffusion procedure. We assume that f contains only positive values.
    :param f:
    :param H:
    :return:
    """
    # We will round the vector f to a small(ish) precision so as to
    # avoid vertices ending in the wrong equivalence classes due to numerical errors.
    f = np.round(f, decimals=5)

    # Compute some standard information about every edge in the hypergraph
    edge_info = compute_edge_info(f, H)

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
        # For now, we will iterate over all possible subsets (!!) - this is exponential though! For small graphs it
        # should at least work, and I can come back later to optimise.
        while len(Utemp) > 0:
            best_P, max_delta = find_densest_subset(Utemp, H, edge_info, debug=debug)

            # Update the r value for the best vertices and remove them from Utemp
            for v in best_P:
                r[vname_to_vidx[v]] = max_delta
                Utemp.remove(v)
    if debug:
        print("Complete Gradient:", r)
    return r


def hypergraph_measure_laplacian(phi, H, debug=False):
    """
    Apply the hypergraph laplacian operator to a vector phi in the measure space.
    In the normal language of graphs, this laplacian would be written as:
       L = I - A D^{-1}
    It can be considered to be the 'random walk' laplacian.
    :param phi: The vector to apply the laplacian operator to.
    :param H: The underlying hypergraph
    :return: L_H x
    """
    f = measure_to_weighted(phi, H)
    r = weighted_diffusion_gradient(f, H, debug=debug)
    return -hypergraph_degree_mat(H) @ r


def hypergraph_lap_conn_graph(phi, H):
    """
    Given a current vector in the measure space, return a graph object demonstrating the basic connectivity of the
    underlying laplacian graph. (For now, I don't actually compute the weights on this graph fully)
    :param phi: A vector in the measure space
    :param H: The underlying hypergraph
    :return: A networkx graph object
    """
    f = measure_to_weighted(phi, H)

    # We will round the vector f to a small(ish) precision so as to
    # avoid vertices ending in the wrong equivalence classes due to numerical errors.
    f = np.round(f, decimals=5)

    edge_info = compute_edge_info(f, H)
    G = nx.Graph()

    # Add the vertices
    for v in H.nodes:
        G.add_node(v)

    # Add the edges
    for e, einfo in edge_info.items():
        for u in einfo[0]:
            for v in einfo[1]:
                G.add_edge(u, v)

    return G


def hypergraph_clique_graph(H):
    """
    Given a hypergraph H, construct the 'clique graph' formed by replacing each hyperedge with a clique.
    :param H:
    :return:
    """
    edge_info = compute_edge_info(f, H)
    G = nx.Graph()

    # Add the vertices
    for v in H.nodes:
        G.add_node(v)

    # Add the edges
    for e, einfo in edge_info.items():
        for u in einfo[3]:
            for v in einfo[3]:
                if u != v:
                    G.add_edge(u, v)

    return G


def hyp_plot_conn_graph(phi, H, show_hyperedges=True):
    """
    Plot the pagerank connectivity graph along with the hypergraph.
    :param phi:
    :param H:
    :return:
    """
    G = hypergraph_lap_conn_graph(phi, H)
    hyp_plot_with_graph(G, H, show_hyperedges=show_hyperedges)


def hyp_plot_with_graph(G, H, show_hyperedges=True):
    """
    Plot a hypergraph with a simple graph on the same vertex set.
    :param G: The simple graph
    :param H: The hypergraph
    :return:
    """
    # Get the positioning of the nodes for drawing the hypergrpah
    pos = hnx.drawing.rubber_band.layout_node_link(H, layout=nx.spring_layout)
    ax = plt.gca()
    if show_hyperedges:
        hnx.drawing.rubber_band.draw(H, pos=pos)
    else:
        # Get the node radius (taken from hypernetx code)
        r0 = hnx.drawing.rubber_band.get_default_radius(H, pos)
        a0 = np.pi * r0 ** 2
        def get_node_radius(v):
            return np.sqrt(a0 * (len(v) if type(v) == frozenset else 1) / np.pi)
        node_radius = {
            v: get_node_radius(v)
            for v in H.nodes
        }

        # Draw stuff
        hnx.drawing.rubber_band.draw_hyper_nodes(H, pos=pos, ax=ax, node_radius=node_radius)
        hnx.drawing.rubber_band.draw_hyper_labels(H, pos=pos, ax=ax, node_radius=node_radius)

        # Set the axes (taken from hypernetx code)
        if len(H.nodes) == 1:
            x, y = pos[list(H.nodes)[0]]
            s = 20
            ax.axis([x - s, x + s, y - s, y + s])
        else:
            ax.axis('equal')
        ax.axis('off')

    # Draw the edges only from the connectivity graph.
    nx.draw_networkx_edges(G, pos=pos)
    plt.show()


# =======================================================
# Simulate the heat diffusion process
# =======================================================
def sim_lap_heat_diff(phi, H, T=1, step=0.1):
    """
    Simulate the heat diffusion process by the euler method.
    :param phi: The measure vector at the start of the process
    :param H: The underlying hypergraph
    :param T: The end time of the process
    :param step: The time interval to use for each step
    :return: A measure vector at the end of the process.
    """
    x_t = phi
    for t in np.linspace(0, T, int(T/step)):
        print(x_t)
        grad = -hypergraph_measure_laplacian(x_t, H)
        x_t += step * grad
    return x_t


def sim_hyp_pagerank(alpha, s, phi0, H, max_iters=1000, step=0.01, debug=False, check_converge=True):
    """
    Compute an approximation of the hypergraph pagerank. Note that the pagerank is defined with respect to the
    normalised vector.
    :param alpha: As in definition of pagerank. (the teleport probability)
    :param s: The teleport vector.
    :param phi0: The starting vector, in the measure space
    :param H: The underlying hypergraph
    :return: The pagerank vector
    """
    n = len(H.nodes)
    x_t = phi0
    beta = (2 * alpha) / (1 + alpha)
    total_iterations = 0
    converged = False
    print("Computing pagerank...")
    while not converged and total_iterations < max_iters:
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
        grad = beta * (s - x_t) - (1 - beta) * hypergraph_measure_laplacian(x_t, H, debug=debug)
        if debug:
            print("Pagerank gradient:", grad)
        x_old = np.copy(x_t)
        x_t += step * grad

        # Check for convergence
        if check_converge and np.sum(np.abs(x_old - x_t)) < (0.00001 * n):
            converged = True
            print("\nPagerank converged. Iterations:", total_iterations)
    return x_t


def check_pagerank(alpha, pr, H):
    test = pr + ((1 - alpha) / (2 * alpha)) * hypergraph_measure_laplacian(pr, H)
    print("Pagerank test:", np.round(test, decimals=2))


if __name__ == "__main__":
    # Construct an example hypergraph
    # H = hnx.Hypergraph({'e1': [1, 2, 3], 'e2': [1, 3, 4, 5, 6], 'e3': [3, 6]})
    H = hnx.Hypergraph({'e1': [1], 'e2': [1, 2], 'e3': [1, 2, 3], 'e4': [1, 2, 3, 4], 'e5': [1, 2, 3, 4, 5]})
    # n = 200
    # r = 10
    # m = int(n*2)
    # H = hypconstruct.construct_hyp_low_cond(int(n/2), int(n/2), m, r, 0.1, 0.5)
    # H = hnx.Hypergraph({'e1': [1, 2, 3], 'e2': [4, 5, 6]})

    # Counter-example for the first edge-sampling technique
    # n = 4
    # H = hnx.Hypergraph({
    #     'e1': ['a1', 'a2', 'a3'],
    #     'e2': ['a1', 'a3', 'a4'],
    #     'e3': ['a1', 'a4', 'a2'],
    # })
    # S = ['a1']

    # Get the mapping from node names to indices
    vname_to_vidx = {}
    for vidx, vname in enumerate(H.nodes):
        vname_to_vidx[vname] = vidx

    # Check whether a vector is an eigenvector of the hypergraph laplacian
    # v = [4, -5, -5, 5, 5]
    # v = [1, 1, 1, 1, 1]
    # v = [1, 1, 1, -4, -4]
    v = [3/2, 3/2, -1, -7/2, -7/2]
    Lv = - weighted_diffusion_gradient(v, H)
    possible_eigenvalue = Lv[0] / v[0]
    eigenvec_check = [possible_eigenvalue * x for x in v]
    print(f"Lv: {Lv}")
    print(f"lambda: {possible_eigenvalue}")
    print(f"lambda v: {eigenvec_check}")

    # Create a starting vector
    # s = np.zeros(n)
    # s[vname_to_vidx['a1']] = 1
    # phi = np.copy(s)
    # f = measure_to_weighted(phi, H)
    # x = measure_to_normalized(phi, H)

    # Compute the pagerank
    # pr = sim_hyp_pagerank(0.8, s, phi, H, debug=False, max_iters=1000, check_converge=True)
    # weighted_pr = measure_to_weighted(pr, H)

    # Output the pagerank vector
    # for i in range(1, int(n/2) + 1):
    #     print(f"pr(a{i}) = {pr[vname_to_vidx['a' + str(i)]]}, prw(a{i}) = {weighted_pr[vname_to_vidx['a' + str(i)]]}")
    # for i in range(1, int(n/2) + 1):
    #     print(f"pr(b{i}) = {pr[vname_to_vidx['b' + str(i)]]}, prw(b{i}) = {weighted_pr[vname_to_vidx['b' + str(i)]]}")
    # print("pagerank", pr)

    # Check that the pagerank vector is correct
    # check_pagerank(0.8, pr, H)

    # Compute an approximate graph
    # G = hypcheeg.hyp_min_edges_graph(H)
    # G = hypcheeg.hyp_degree_graph(H, c=2, debug=True)

    # Get the conductance of the target set
    # S = ['a' + str(i) for i in range(1, int(n/2) + 1)]
    # print(f"phi(S) = {hypcheeg.hyp_conductance(H, S)}")

    # Plot the conductance of the set in the constructed graph
    # print(f"phi_G(S) = {hypcheeg.graph_self_loop_conductance(S, G, H)}")
    # print(f"phi_G(S) = {nx.algorithms.cuts.conductance(G, S, weight='weight')}")

    # List the spectrum of the constructed graph G
    # print(f"# Connected Components of G: {nx.number_connected_components(G)}")
    # print(f"Connected Components of G: {[c for c in nx.connected_components(G)]}")
    # print(f"Spectrum of G: {nx.laplacian_spectrum(G, weight='weight')}")

    # Get the best sweep set
    # S_star = hypcheeg.hyp_sweep_set(pr, H, debug=False)
    # print(f"phi(S*) = {hypcheeg.hyp_conductance(H, S_star)}, S* = {S_star}")

    # Plot the hypergraph
    # hnx.draw(H)
    # plt.show()
    # Plot the hypergraph and the pagerank graph
    # hyp_plot_with_graph(G, H, show_hyperedges=False)
    # hyp_plot_conn_graph(pr, H, show_hyperedges=True)

