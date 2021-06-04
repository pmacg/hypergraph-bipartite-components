"""
This file implements some experiments on the hypergraph diffusion operator.
"""
import scipy as sp
import scipy.sparse.linalg
import time
import statistics
import clsz.cluster
import clsz.metrics
import hypconstruct
import hypalgorithms
import hypcheeg
import datasets
import hyplogging
import hypmc
import hypreductions
import lightgraphs


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


def random_hypergraph_experiments():
    """
    Run main experiments. Compare the four algorithms: clique, approx_diffusion_clique, random,
    approx_diffusion_random
    """
    # We will run the clique algorithm and the approximate diffusion algorithm on a variety of graphs
    graph_sizes = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    ranks = [3]
    edge_multipliers = [2 ** (3 - 2), 2 ** (3 - 1)]  # See Duraj et al. 2021
    configuration_runs = 10

    total_runs = len(graph_sizes) * len(ranks) * len(edge_multipliers) * configuration_runs

    with open("results.csv", "w") as f_out:
        f_out.write("run_id,configuration_id,vertices,edge_multiplier,edges,rank,"
                    "clique_bipartiteness,clique_volume,clique_runtime,"
                    "diffusion_clique_bipartiteness,diffusion_clique_volume,diffusion_clique_runtime,"
                    "random_bipartiteness,random_volume,random_runtime,"
                    "diffusion_random_bipartiteness,diffusion_random_volume,diffusion_random_runtime\n")
        current_run_id = 0
        configuration_id = 0
        for r in ranks:
            for n in graph_sizes:
                for edge_multiplier in edge_multipliers:
                    m = edge_multiplier * n
                    configuration_id += 1
                    for i in range(configuration_runs):
                        current_run_id += 1
                        print(f"Run ID: {current_run_id}/{total_runs}; Configuration ID: {configuration_id}; Rank: {r};"
                              f" Vertices: {n}; Edges: {m}")

                        # Generate a random hypergraph
                        hypergraph = hypconstruct.construct_random_hypergraph(n, m, r)
                        f_out.write(f"{current_run_id},{configuration_id},{n},{edge_multiplier},{m},{r},")
                        f_out.flush()

                        # Run the clique algorithm
                        start_time = time.time()
                        vertex_set_l, vertex_set_r, bipartiteness = hypalgorithms.find_bipartite_set_clique(hypergraph)
                        execution_time = time.time() - start_time
                        volume_l_r = hypcheeg.hypergraph_volume(hypergraph, vertex_set_l + vertex_set_r)
                        f_out.write(f"{bipartiteness},{volume_l_r},{execution_time},")
                        f_out.flush()

                        # Run the diffusion algorithm - clique
                        start_time = time.time()
                        vertex_set_l, vertex_set_r, bipartiteness = hypalgorithms.find_bipartite_set_diffusion(
                            hypergraph, step_size=1, approximate=True)
                        execution_time = time.time() - start_time
                        volume_l_r = hypcheeg.hypergraph_volume(hypergraph, vertex_set_l + vertex_set_r)
                        f_out.write(f"{bipartiteness},{volume_l_r},{execution_time},")
                        f_out.flush()

                        # Run the random algorithm
                        start_time = time.time()
                        vertex_set_l, vertex_set_r, bipartiteness = hypalgorithms.find_bipartite_set_random(hypergraph)
                        execution_time = time.time() - start_time
                        volume_l_r = hypcheeg.hypergraph_volume(hypergraph, vertex_set_l + vertex_set_r)
                        f_out.write(f"{bipartiteness},{volume_l_r},{execution_time},")
                        f_out.flush()

                        # Run the diffusion algorithm - random
                        start_time = time.time()
                        vertex_set_l, vertex_set_r, bipartiteness = hypalgorithms.find_bipartite_set_diffusion(
                            hypergraph, step_size=1, use_random_initialisation=True, approximate=True)
                        execution_time = time.time() - start_time
                        volume_l_r = hypcheeg.hypergraph_volume(hypergraph, vertex_set_l + vertex_set_r)
                        f_out.write(f"{bipartiteness},{volume_l_r},{execution_time}\n")
                        f_out.flush()


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


def simple_experiment(hypergraph, step_size=0.1, max_time=100):
    """
    Run a test of our algorithm on a given hypergraph.
    """
    # Run the clique algorithm
    hyplogging.logger.info("Running the clique algorithm.")
    clique_alg_l, clique_alg_r, clique_bipart = hypalgorithms.find_bipartite_set_clique(hypergraph)

    # Run the random algorithm
    hyplogging.logger.info("Running the random algorithm.")
    rand_alg_l, rand_alg_r, rand_bipart = hypalgorithms.find_bipartite_set_random(hypergraph)

    # Run the diffusion algorithm
    hyplogging.logger.info("Running the diffusion algorithm.")
    diff_alg_l, diff_alg_r, diff_bipart = hypalgorithms.find_bipartite_set_diffusion(hypergraph, max_time=max_time,
                                                                                     step_size=step_size,
                                                                                     use_random_initialisation=False,
                                                                                     approximate=True)

    hyplogging.logger.info("Running the random diffusion algorithm.")
    rand_diff_alg_l, rand_diff_alg_r, rand_diff_bipart = hypalgorithms.find_bipartite_set_diffusion(
        hypergraph, max_time=max_time, step_size=step_size, use_random_initialisation=True, approximate=True)

    hyplogging.logger.info(f"Clique algorithm bipartiteness: {clique_bipart}")
    hyplogging.logger.info(f"Random algorithm bipartiteness: {rand_bipart}")
    hyplogging.logger.info(f"Diffusion algorithm bipartiteness: {diff_bipart}")
    hyplogging.logger.info(f"Random diffusion algorithm bipartiteness: {rand_diff_bipart}")


def foodweb_experiment():
    print("Loading dataset...")
    hyplogging.logger.info("Loading the foodweb dataset.")
    foodweb_dataset = datasets.FoodWebHFDDataset()

    print("Running diffusion....")
    hyplogging.logger.info("Running the diffusion process on the foodweb graph.")
    clusters = hypalgorithms.recursive_bipartite_diffusion(foodweb_dataset.hypergraph, 4,
                                                           step_size=1, max_time=100,
                                                           approximate=True, return_unclassified=True)

    for i, cluster in enumerate(clusters):
        hyplogging.logger.info("")
        hyplogging.logger.info(f"Cluster {i + 1}")
        for index in cluster:
            vertex_name = foodweb_dataset.vertex_labels[index]
            vertex_cluster = foodweb_dataset.cluster_labels[foodweb_dataset.gt_clusters[index]] if \
                foodweb_dataset.gt_clusters[index] is not None else 'missing'
            hyplogging.logger.info(f"{vertex_name}\t\t{vertex_cluster}")
        hyplogging.logger.info("")


def imdb_experiment():
    hyplogging.logger.info("Loading the imdb dataset.")
    imdb_dataset = datasets.ImdbDataset()
    imdb_dataset.use_subgraph("Hugh Grant", degrees_of_separation=2)

    hyplogging.logger.info("Running the diffusion algorithm.")
    diff_alg_l, diff_alg_r, diff_bipart = hypalgorithms.find_bipartite_set_diffusion(imdb_dataset.hypergraph,
                                                                                     max_time=100, step_size=1,
                                                                                     approximate=True)
    hyplogging.logger.info(f"Diffusion algorithm bipartiteness: {diff_bipart}\n")
    hyplogging.logger.info(f"   SET 1")
    hyplogging.logger.info(str([imdb_dataset.vertex_labels[v] for v in diff_alg_l]))
    hyplogging.logger.info(f"   SET 2")
    hyplogging.logger.info(str([imdb_dataset.vertex_labels[v] for v in diff_alg_r]))

    imdb_dataset.simple_cluster_check("Left Set", diff_alg_l)
    imdb_dataset.simple_cluster_check("Right Set", diff_alg_r)


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


def log_migration_result(filename, migration_dataset, title, left_set, right_set):
    """
    Given a pair of clusters in the migration dataset, display their evaluation, and write it to the output csv file.

    :param filename:
    :param migration_dataset:
    :param title: The title of this result
    :param left_set:
    :param right_set:
    :return: nothing
    """
    hyplogging.logger.debug("Computing objectives for migration result.")
    hyplogging.logger.debug(f"    Left set: {str(left_set)}")
    hyplogging.logger.debug(f"   Right set: {str(right_set)}")
    bipartiteness = hypcheeg.hypergraph_bipartiteness(migration_dataset.hypergraph, left_set, right_set)
    cut_imbalance = clsz.metrics.networkx_cut_imbalance(migration_dataset.directed_graph, left_set, right_set)
    flow_ratio_left_right = hypcheeg.ms_flow_ratio(migration_dataset.directed_graph, left_set, right_set)
    flow_ratio_right_left = hypcheeg.ms_flow_ratio(migration_dataset.directed_graph, right_set, left_set)
    hyplogging.logger.info(f"{title}")
    hyplogging.logger.info(f"  Hypergraph Bipartiteness: {bipartiteness}")
    hyplogging.logger.info(f"             Cut Imbalance: {cut_imbalance}")
    hyplogging.logger.info(f"                Flow Ratio: {flow_ratio_left_right}")
    hyplogging.logger.info(f"       Reversed Flow Ratio: {flow_ratio_right_left}")
    with open(filename, 'a') as f_out:
        f_out.write(f"{title}, {bipartiteness}, {cut_imbalance}, {flow_ratio_left_right}, {flow_ratio_right_left}\n")


def log_migration_pairwise_results(filename, migration_dataset, title, clusters):
    """
    Given an array of cluster labels (clusters), log the objective values of every pairwise cluster in the set.

    :param filename: the file to which the results will be written.
    :param migration_dataset: the migration dataset object.
    :param title: the overall title of the results
    :param clusters: either a list of 0-indexed cluster labels, or a list of lists representing clusters
    :return:
    """
    if type(clusters[0]) == list:
        # Clusters is a list of lists
        k = len(clusters)
    else:
        # Clusters is a list of cluster labels
        k = max(clusters) + 1

    for cluster_idx_1 in range(k):
        for cluster_idx_2 in range(cluster_idx_1 + 1, k):
            if type(clusters[0]) == list:
                # Clusters is a list of lists
                cluster_1 = clusters[cluster_idx_1]
                cluster_2 = clusters[cluster_idx_2]
            else:
                # Clusters is a list of cluster labels
                cluster_1 = [i for (i, label) in enumerate(clusters) if label == cluster_idx_1]
                cluster_2 = [i for (i, label) in enumerate(clusters) if label == cluster_idx_2]

            log_migration_result(filename, migration_dataset,
                                 f"{title} Clusters {cluster_idx_1} and {cluster_idx_2}",
                                 cluster_1, cluster_2)


def migration_experiment():
    """Compare our algorithm to the CLSZ algorithm on the migration dataset. We will compare them on the following
    three metrics:
      - Hypergraph bipartiteness
      - Graph bipartiteness
      - Graph flow ratio (as described in CLSZ)
    """
    migration_dataset = datasets.MigrationDataset()

    hyplogging.logger.info("Running CLSZ algorithm.")
    k = 10
    clsz_labels = clsz.cluster.cluster_networkx(migration_dataset.directed_graph, k)
    hyplogging.logger.info(f'CLSZ labels: {" ".join(map(str, clsz_labels))}')

    # Now, run the hypergraph clustering algorithm
    hyplogging.logger.info("Running diffusion algorithm.")
    i = 4
    diff_clusters = hypalgorithms.recursive_bipartite_diffusion(migration_dataset.hypergraph, iterations=i,
                                                                max_time=100, step_size=0.5,
                                                                approximate=True)

    # Now, we will display the vitalstatistix of both algorithm.
    output_csv_filename = f"results/migration_experiment_motif_{k}_{i}.csv"
    with open(output_csv_filename, 'w') as f_out:
        f_out.write("name, bipartiteness, ci, fr1, fr2\n")

    # Now, print the results
    log_migration_pairwise_results(output_csv_filename, migration_dataset, "CLSZ", clsz_labels)
    log_migration_pairwise_results(output_csv_filename, migration_dataset, "Diffusion", diff_clusters)


def wikipedia_experiment():
    """Run the diffusion and related hypergraph algorithms on the three different wikipedia datasets."""
    for animal in ["chameleon", "crocodile", "squirrel"]:
        hyplogging.logger.info(f"NOW PROCESSING: {animal}")
        wikipedia_dataset = datasets.WikipediaDataset(animal)
        simple_experiment(wikipedia_dataset.hypergraph, step_size=0.1)


def mid_experiment():
    """Run experiments with the MID dataset."""
    mid_dataset = datasets.MidDataset(1950, 1990)

    # For this experiment, we will compare the performance of the diffusion algorithm on the motif-hypergraph to the
    # clique algorithm (trevisan's algorithm) on the original graph.
    clique_l, clique_r = hypalgorithms.find_max_cut(mid_dataset.graph_hypergraph, algorithm='clique')
    diff_l, diff_r = hypalgorithms.find_max_cut(mid_dataset.hypergraph, approximate=True)

    # Compute the objectives
    clique_bipartiteness = hypcheeg.hypergraph_bipartiteness(mid_dataset.graph_hypergraph, clique_l, clique_r)
    diff_bipartiteness = hypcheeg.hypergraph_bipartiteness(mid_dataset.graph_hypergraph, diff_l, diff_r)
    hyplogging.logger.info(f"Trevisan's algorithm bipartiteness: {clique_bipartiteness}")
    hyplogging.logger.info(f"Diffusion algorithm bipartiteness: {diff_bipartiteness}")


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


def nlp_experiment():
    """Run the experiment on the NLP dataset."""
    nlp_dataset = datasets.NlpDataset()

    # Run the diffusion algorithm, and report the result
    for left_set, right_set in hypalgorithms.find_max_cut(nlp_dataset.hypergraph, approximate=True,
                                                          return_each_pair=True):
        nlp_dataset.log_two_sets(left_set, right_set)


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


def induced_graph_demo():
    """Produce some example induced graphs from the diffusion process."""
    edges = [[0, 1, 2], [1, 3, 4], [2, 3, 5], [3, 4, 5]]
    hypergraph = lightgraphs.LightHypergraph(edges)
    s = [1, 0, 0, 0, 0, 0]
    hypalgorithms._internal_bipartite_diffusion(s, hypergraph, 100, 0.1, False, construct_induced=True)


if __name__ == "__main__":
    # Real-world experiments
    # treebank_experiment()
    # dblp_experiment()
    # wikipedia_categories_experiment()
    # actor_director_experiment()

    # Synthetic experiments
    # sbm_experiments()
    # sbm_runtime_experiment(1000, 5, 1e-9)

    # Demonstration to help build the figures in the paper.
    induced_graph_demo()
