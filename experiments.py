"""
This file implements some experiments on the hypergraph diffusion operator.
"""
import time
import clsz.cluster
import clsz.metrics
import hypconstruct
import hypalgorithms
import hypcheeg
import datasets
import hyplogging


def random_hypergraph_experiments():
    """
    Run main experiments. Compare the four algorithms: clique, approx_diffusion_clique, random,
    approx_diffusion_random
    """
    # We will run the clique algorithm and the approximate diffusion algorithm on a variety of graphs
    graph_sizes = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    ranks = [3]
    edge_multipliers = [2**(3-2), 2**(3-1)]  # See Duraj et al. 2021
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
                            hypergraph, step_size=1, approximate=True, use_random_initialisation=True)
                        execution_time = time.time() - start_time
                        volume_l_r = hypcheeg.hypergraph_volume(hypergraph, vertex_set_l + vertex_set_r)
                        f_out.write(f"{bipartiteness},{volume_l_r},{execution_time}\n")
                        f_out.flush()


def test_step_sizes():
    # We will run the diffusion algorithm and the approximate diffusion algorithm on a variety of graphs
    graph_sizes = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    ranks = [3]
    edge_multipliers = [2**(3-2), 2**(3-1)]  # See Duraj et al. 2021
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
    diff_alg_l, diff_alg_r, diff_bipart = hypalgorithms.find_bipartite_set_diffusion(hypergraph,
                                                                                     step_size=step_size,
                                                                                     max_time=max_time,
                                                                                     approximate=True,
                                                                                     use_random_initialisation=False)

    hyplogging.logger.info("Running the random diffusion algorithm.")
    rand_diff_alg_l, rand_diff_alg_r, rand_diff_bipart = hypalgorithms.find_bipartite_set_diffusion(
        hypergraph, step_size=step_size, max_time=max_time, approximate=True, use_random_initialisation=True)

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
    clusters = hypalgorithms.recursive_bipartite_diffusion(foodweb_dataset.hypergraph, 2,
                                                           step_size=1, max_time=100,
                                                           approximate=True)

    for i, cluster in enumerate(clusters):
        print()
        print(f"Cluster {i + 1}")
        for index in cluster:
            vertex_name = foodweb_dataset.vertex_labels[index]
            vertex_cluster = foodweb_dataset.cluster_labels[foodweb_dataset.gt_clusters[index]] if \
                foodweb_dataset.gt_clusters[index] is not None else 'missing'
            print(f"{vertex_name}\t\t{vertex_cluster}")
        print()


def imdb_experiment():
    hyplogging.logger.info("Loading the imdb dataset.")
    imdb_dataset = datasets.ImdbDataset()
    imdb_dataset.use_subgraph("Hugh Grant", degrees_of_separation=2)

    hyplogging.logger.info("Running the diffusion algorithm.")
    diff_alg_l, diff_alg_r, diff_bipart = hypalgorithms.find_bipartite_set_diffusion(imdb_dataset.hypergraph,
                                                                                     step_size=1, max_time=100,
                                                                                     approximate=True)
    hyplogging.logger.info(f"Diffusion algorithm bipartiteness: {diff_bipart}\n")
    hyplogging.logger.info(f"   SET 1")
    hyplogging.logger.info(str([imdb_dataset.vertex_labels[v] for v in diff_alg_l]))
    hyplogging.logger.info(f"   SET 2")
    hyplogging.logger.info(str([imdb_dataset.vertex_labels[v] for v in diff_alg_r]))

    imdb_dataset.simple_cluster_check("Left Set", diff_alg_l)
    imdb_dataset.simple_cluster_check("Right Set", diff_alg_r)


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
    categories_dataset = datasets.WikipediaCategoriesDataset()

    # Run the diffusion algorithm
    left_set, right_set, bipartiteness = hypalgorithms.find_bipartite_set_diffusion(categories_dataset.hypergraph,
                                                                                    approximate=True)

    # Show the left and right set.
    categories_dataset.log_two_sets(left_set, right_set)


if __name__ == "__main__":
    wikipedia_experiment()
