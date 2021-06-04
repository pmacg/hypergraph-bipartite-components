"""
This file implements some experiments on the hypergraph diffusion operator.
"""
import time
import statistics
import hypconstruct
import hypalgorithms
import hypcheeg
import datasets
import hyplogging
import hypreductions


def transpose_lists(lists):
    """
    Return the transpose of a list of lists.
    """
    return list(map(list, zip(*lists)))


def run_single_sbm_experiment(sbm_dataset, run_lp=False):
    """Given an SBM dataset, run the clique and diffusion algorithms, and report their performance."""
    # We will repeatedly use the clique graph to compute vitalstatistix
    clique_graph = hypreductions.hypergraph_clique_reduction(sbm_dataset.hypergraph)
    results = []

    # Run the clique algorithm
    hyplogging.logger.info("Running the clique algorithm.")
    start_time = time.time()
    vertex_set_l, vertex_set_r, clique_hyp_bipart = hypalgorithms.find_bipartite_set_clique(sbm_dataset.hypergraph)
    clique_execution_time = time.time() - start_time
    clique_hyp_vol = hypcheeg.hypergraph_volume(sbm_dataset.hypergraph, vertex_set_l + vertex_set_r)
    clique_clique_bipart = clique_graph.bipartiteness(vertex_set_l, vertex_set_r)
    clique_clique_vol = clique_graph.volume(vertex_set_l + vertex_set_r)
    results.extend([clique_hyp_bipart, clique_hyp_vol, clique_clique_bipart, clique_clique_vol, clique_execution_time])

    # Run the exact diffusion algorithm if required
    if run_lp:
        hyplogging.logger.info("Running the exact diffusion algorithm.")
        start_time = time.time()
        vertex_set_l, vertex_set_r, diff_hyp_bipart = hypalgorithms.find_bipartite_set_diffusion(sbm_dataset.hypergraph,
                                                                                                 approximate=False)
        diff_execution_time = time.time() - start_time
        diff_hyp_vol = hypcheeg.hypergraph_volume(sbm_dataset.hypergraph, vertex_set_l + vertex_set_r)
        diff_clique_bipart = clique_graph.bipartiteness(vertex_set_l, vertex_set_r)
        diff_clique_vol = clique_graph.volume(vertex_set_l + vertex_set_r)
        results.extend([diff_hyp_bipart, diff_hyp_vol, diff_clique_bipart, diff_clique_vol, diff_execution_time])

    # Run the approximate diffusion algorithm
    hyplogging.logger.info("Running the approximate diffusion algorithm.")
    start_time = time.time()
    vertex_set_l, vertex_set_r, approx_diff_hyp_bipart = hypalgorithms.find_bipartite_set_diffusion(
        sbm_dataset.hypergraph, approximate=True)
    approx_diff_execution_time = time.time() - start_time
    approx_diff_hyp_vol = hypcheeg.hypergraph_volume(sbm_dataset.hypergraph, vertex_set_l + vertex_set_r)
    approx_diff_clique_bipart = clique_graph.bipartiteness(vertex_set_l, vertex_set_r)
    approx_diff_clique_vol = clique_graph.volume(vertex_set_l + vertex_set_r)
    results.extend([approx_diff_hyp_bipart, approx_diff_hyp_vol, approx_diff_clique_bipart, approx_diff_clique_vol,
                    approx_diff_execution_time])

    return results


def sbm_runtime_experiment(n, r, p):
    """
    Compute the average runtime for the given parameters of graph. Print the output to screen - no need for CSV
    fancy-ness.
    """
    # Fix the ratio q = 2 * p
    q = 2 * p

    # Compute the average performance over 10 runs
    all_results = []
    for i in range(10):
        # Generate the hypergraph
        sbm_dataset = datasets.SbmDataset(n, r, p, q, graph_num=i)
        edges = sbm_dataset.hypergraph.num_edges
        results = run_single_sbm_experiment(sbm_dataset)
        results.append(edges)
        all_results.append(results)

    # Display the average results
    average_results = list(map(statistics.mean, transpose_lists(all_results)))
    hyplogging.logger.info(f"n = {n}, r = {r}, p = {p}, edges = {average_results[-1]}, "
                           f"diff_runtime = {average_results[9]}, clique_runtime = {average_results[4]}")


def sbm_experiments():
    """Run experiments with the stochastic block model."""
    # The key experiment is to fix n, r, and p and vary the ratio q/p.
    n = 1000
    average_over = 10

    # We will run a few experiments
    rs = [6]
    ps = [1e-14]
    run_lps = [False]

    # Use the same ratios for each experiment
    ratios = [0.5 * x for x in range(13, 61)]

    # Whether to append results to the results files
    append_results = True

    for index in range(len(rs)):
        sbm_experiment_internal(n, rs[index], ps[index], average_over, run_lps[index], ratios, append_results)


def sbm_experiment_internal(n, r, p, average_over, run_lp, ratios, append_results):
    mode = 'a' if append_results else 'w'
    with open(f"data/sbm/results/two_cluster_average_results_{n}_{r}_{p}_{average_over}.csv", mode) as average_file:
        # Write the header line of the average results file
        if not append_results:
            average_file.write("n, r, p, q, ratio, clique_hyp_bipart, clique_hyp_vol, clique_clique_bipart, "
                               "clique_clique_vol, clique_runtime, "
                               f"{'diff_hyp_bipart, diff_hyp_vol, diff_clique_bipart, ' if run_lp else ''}"
                               f"{'diff_clique_vol, diff_runtime, ' if run_lp else ''}"
                               "approx_diff_hyp_bipart, approx_diff_hyp_vol, "
                               "approx_diff_clique_bipart, approx_diff_clique_vol, approx_diff_runtime\n")

        with open(f"data/sbm/results/two_cluster_results_{n}_{r}_{p}.csv", mode) as results_file:
            # Write the header line of the file
            if not append_results:
                results_file.write(
                    "run_id, n, r, p, q, ratio, clique_hyp_bipart, clique_hyp_vol, clique_clique_bipart, "
                    "clique_clique_vol, clique_runtime, "
                    f"{'diff_hyp_bipart, diff_hyp_vol, diff_clique_bipart, ' if run_lp else ''}"
                    f"{'diff_clique_vol, diff_runtime, ' if run_lp else ''}"
                    "approx_diff_hyp_bipart, approx_diff_hyp_vol, "
                    "approx_diff_clique_bipart, approx_diff_clique_vol, approx_diff_runtime\n")

            # We will consider the following ratios of q/p
            run_id = 0
            for ratio in ratios:
                all_results = []
                for graph_index in range(average_over):
                    run_id += 1

                    # Load the dataset and run the required algorithms
                    q = ratio * p
                    sbm_dataset = datasets.SbmDataset(n, r, p, q, graph_num=graph_index)
                    results = run_single_sbm_experiment(sbm_dataset, run_lp=run_lp)
                    results_string = ', '.join(map(str, results))

                    # Update the list of results to be averaged
                    all_results.append(results)

                    # Write the results to file
                    results_file.write(f"{run_id}, {n}, {r}, {p}, {q}, {ratio}, ")
                    results_file.write(results_string)
                    results_file.write('\n')
                    results_file.flush()

                # Write out the average result values
                average_results = map(statistics.mean, transpose_lists(all_results))
                average_results_string = ', '.join(map(str, average_results))
                average_file.write(f"{n}, {r}, {p}, {q}, {ratio}, ")
                average_file.write(average_results_string)
                average_file.write('\n')
                average_file.flush()


def test_step_sizes():
    # We will run the diffusion algorithm and the approximate diffusion algorithm on a variety of graphs
    graph_sizes = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    ranks = [3]
    edge_multipliers = [2 ** (3 - 2), 2 ** (3 - 1)]  # See Duraj et al. 2021
    configuration_runs = 10
    step_sizes = [0.1, 0.5, 1, 2, 5]

    total_runs = len(graph_sizes) * len(ranks) * len(edge_multipliers) * len(step_sizes) * configuration_runs

    with open("results.csv", "w") as f_out:
        f_out.write("run_id,configuration_id,vertices,edge_multiplier,edges,rank,step_size,diffusion_bipartiteness,"
                    "diffusion_volume,diffusion_runtime,approx_diffusion_bipartiteness,approx_diffusion_volume,"
                    "approx_diffusion_runtime\n")
        current_run_id = 0
        configuration_id = 0
        for r in ranks:
            for n in graph_sizes:
                for edge_multiplier in edge_multipliers:
                    m = edge_multiplier * n
                    for step_size in step_sizes:
                        configuration_id += 1
                        for i in range(configuration_runs):
                            current_run_id += 1
                            print(f"Run ID: {current_run_id}/{total_runs}; Configuration ID: {configuration_id};"
                                  f"Rank: {r}; Vertices: {n}; Edges: {m}; Step size: {step_size};")

                            # Generate a random hypergraph
                            hypergraph = hypconstruct.construct_random_hypergraph(n, m, r)
                            f_out.write(
                                f"{current_run_id},{configuration_id},{n},{edge_multiplier},{m},{r},{step_size},")
                            f_out.flush()

                            # Run the clique algorithm
                            start_time = time.time()
                            vertex_set_l, vertex_set_r, bipartiteness = hypalgorithms.find_bipartite_set_diffusion(
                                hypergraph, step_size=step_size)
                            execution_time = time.time() - start_time
                            volume_l_r = hypcheeg.hypergraph_volume(hypergraph, vertex_set_l + vertex_set_r)
                            f_out.write(f"{bipartiteness},{volume_l_r},{execution_time},")
                            f_out.flush()

                            # Run the diffusion algorithm
                            start_time = time.time()
                            vertex_set_l, vertex_set_r, bipartiteness = hypalgorithms.find_bipartite_set_diffusion(
                                hypergraph, step_size=step_size, approximate=True)
                            execution_time = time.time() - start_time
                            volume_l_r = hypcheeg.hypergraph_volume(hypergraph, vertex_set_l + vertex_set_r)
                            f_out.write(f"{bipartiteness},{volume_l_r},{execution_time}\n")
                            f_out.flush()


def actor_director_experiment():
    """
    Run experiments on the smaller IMDB dataset, looking to distinguish actors and directors.
    :return:
    """
    # We only consider the case where each edge has one director and two actors
    for num_actors in [2]:
        hyplogging.logger.info(f"Using {num_actors} actors.")
        dataset = datasets.ActorDirectorDataset(num_actors=num_actors)

        # Run the diffusion algorithm
        for left, right in hypalgorithms.find_max_cut(dataset.hypergraph):
            dataset.log_confusion_matrix([left, right])
            dataset.show_clustering_stats([left, right])

        # Run the clique algorithm
        for left, right in hypalgorithms.find_max_cut(dataset.hypergraph, algorithm='clique'):
            dataset.log_confusion_matrix([left, right])
            dataset.show_clustering_stats([left, right])


def wikipedia_categories_experiment():
    """Run experiments on the wikipedia categories dataset."""
    # Our algorithm is capable of recovering the known underlying structure in the following graph.
    categories_dataset = datasets.WikipediaCategoriesDataset(
        "faculty-Computer_scientists_by_field_of_research")

    # Run the diffusion algorithm
    for left_set, right_set in hypalgorithms.find_max_cut(categories_dataset.hypergraph, return_each_pair=False,
                                                          algorithm='diffusion'):
        categories_dataset.log_confusion_matrix([left_set, right_set])
        categories_dataset.show_clustering_stats([left_set, right_set])

    # Run the clique algorithm
    for left_set, right_set in hypalgorithms.find_max_cut(categories_dataset.hypergraph, return_each_pair=False,
                                                          algorithm='clique'):
        categories_dataset.log_confusion_matrix([left_set, right_set])
        categories_dataset.show_clustering_stats([left_set, right_set])


def dblp_experiment():
    """Run experiments with the DBLP dataset."""
    # First, run the experiment with the author-conference hypergraph
    dblp_dataset = datasets.DblpDataset()

    # Run the diffusion algorithm
    clusters = hypalgorithms.recursive_bipartite_diffusion(dblp_dataset.hypergraph, iterations=2, approximate=True)
    dblp_dataset.log_confusion_matrix(clusters)
    dblp_dataset.show_clustering_stats(clusters)

    # Run the clique algorithm
    clusters = hypalgorithms.recursive_bipartite_diffusion(dblp_dataset.hypergraph, iterations=2, approximate=True,
                                                           use_clique_alg=True)
    dblp_dataset.log_confusion_matrix(clusters)
    dblp_dataset.show_clustering_stats(clusters)


def treebank_experiment():
    """Run experiments with the Penn-Treebank dataset."""
    # Start by loading the dataset
    treebank_dataset = datasets.PennTreebankDataset(n=4, min_degree=10, max_degree=float('inf'),
                                                    categories_to_use=["Verb", "Adverb", "Adjective"],
                                                    allow_proper_nouns=True)

    # Combine the non-verbs
    treebank_dataset.combine_clusters(1, 2, "Non-Verbs")

    gt_1 = treebank_dataset.get_cluster(0)
    gt_2 = treebank_dataset.get_cluster(1)
    print(
        f"Bipartiteness of ground truth: {hypcheeg.hypergraph_bipartiteness(treebank_dataset.hypergraph, gt_1, gt_2)}")
    print(f"Number of vertices in hypergraph: {treebank_dataset.hypergraph.number_of_nodes()}")
    print(f"Number of edges in hypergraph: {treebank_dataset.hypergraph.number_of_edges()}")
    print(f"Average rank of hypergraph: {treebank_dataset.hypergraph.average_rank()}")

    # Run the approximate diffusion algorithm
    for left, right in hypalgorithms.find_max_cut(treebank_dataset.hypergraph,
                                                  return_each_pair=False,
                                                  algorithm='diffusion'):
        treebank_dataset.log_confusion_matrix([left, right])
        treebank_dataset.show_clustering_stats([left, right])
        print(f"Bipartiteness: {hypcheeg.hypergraph_bipartiteness(treebank_dataset.hypergraph, left, right)}")

    # Run the clique algorithm
    for left, right in hypalgorithms.find_max_cut(treebank_dataset.hypergraph,
                                                  return_each_pair=False,
                                                  algorithm='clique'):
        treebank_dataset.log_confusion_matrix([left, right])
        treebank_dataset.show_clustering_stats([left, right])
        print(f"Bipartiteness: {hypcheeg.hypergraph_bipartiteness(treebank_dataset.hypergraph, left, right)}")


if __name__ == "__main__":
    # Real-world experiments
    treebank_experiment()
    dblp_experiment()
    wikipedia_categories_experiment()
    actor_director_experiment()

    # Synthetic experiments
    sbm_experiments()
    sbm_runtime_experiment(1000, 5, 1e-9)

    # Demonstration to help build the figures in the paper.
    hypalgorithms.induced_graph_demo()
