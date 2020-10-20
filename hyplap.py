"""
This file implements methods for using the hypergraph laplacian operator.
The terminology used in this file comes from the following paper: https://arxiv.org/abs/1605.01483
"""
import networkx as nx
import hypernetx as hnx
import numpy as np
import matplotlib.pyplot as plt


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
        (Ie, Se, max_{u, v \in e} (f(u) - f(v))
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
        edge_info[e.uid] = (Ie, Se, maxe - mine)
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


# ====================================================
# Compute the hypergraph laplacian for a given vector
# ====================================================
def weighted_diffusion_gradient(f, H):
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

    # We now iterate through the equivalence classes for the remainder of the algorithm
    for _, U in equiv_classes.items():
        Utemp = U

        # STEP 2 + 3
        # We need to find the subset P of U to maximise
        # \delta(P) = C(P) / w(P)
        # For now, we will iterate over all possible subsets (!!) - this is exponential though! For small graphs it
        # should at least work, and I can come back later to optimise.
        while len(Utemp) > 0:
            max_delta = -len(edge_info)
            best_P = []
            for P in powerset(U):
                if delta(U, edge_info, H) > max_delta:
                    max_delta = delta(P, edge_info, H)
                    best_P = P

            # Update the r value for the best vertices and remove them from Utemp
            for v in best_P:
                r[vname_to_vidx[v]] = max_delta
                Utemp.remove(v)
    return r


def hypergraph_laplacian(phi, H):
    """
    Apply the hypergraph laplacian operator to a vector phi in the measure space
    :param phi: The vector to apply the laplacian operator to.
    :param H: The underlying hypergraph
    :return: L_H x
    """
    f = measure_to_weighted(phi, H)
    r = weighted_diffusion_gradient(f, H)
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
        grad = -hypergraph_laplacian(x_t, H)
        x_t += step * grad
    return x_t


def sim_hyp_pagerank(alpha, s, phi0, H, num_iters=1000, step=0.01):
    """
    Compute an approximation of the hypergraph pagerank.
    :param alpha: As in definition of pagerank. (the teleport probability)
    :param s: The teleport vector.
    :param phi0: The starting vector, in the measure space
    :param H: The underlying hypergraph
    :return: The pagerank vector
    """
    x_t = phi0
    beta = (2 * alpha) / (1 + alpha)
    for t in np.linspace(0, int(num_iters * step), num_iters):
        print(x_t)
        grad = beta * (s - x_t) - (1 - beta) * hypergraph_laplacian(x_t, H)
        x_t += step * grad
    return x_t


if __name__ == "__main__":
    # Construct an example hypergraph
    H = hnx.Hypergraph({'e1': [1, 2, 3], 'e2': [1, 3, 4, 5, 6], 'e3': [3, 6]})

    # Create a starting vector
    phi = np.ones(6)
    f = measure_to_weighted(phi, H)
    x = measure_to_normalized(phi, H)

    phi_end = sim_hyp_pagerank(0.8, np.array([1, 0, 0, 0, 0, 0]), phi, H)
    G = hypergraph_lap_conn_graph(phi_end, H)

    # options = {
    #     'node_color': 'black',
    #     'node_size': 100,
    #     'width': 3,
    # }
    nx.draw_networkx(G, with_labels=True)
    plt.show()

