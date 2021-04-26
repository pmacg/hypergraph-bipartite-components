"""
This file implements some experiments on the hypergraph diffusion operator.
"""
import time
import clsz.cluster
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


def dataset_experiment(dataset):
    """
    Run a test of our algorithm on a given dataset.
    """
    # Run the clique algorithm
    hyplogging.logger.info("Running the clique algorithm.")
    clique_alg_l, clique_alg_r, clique_bipart = hypalgorithms.find_bipartite_set_clique(dataset.hypergraph)
    hyplogging.logger.info(f"Clique algorithm bipartiteness: {clique_bipart}\n")
    print(f"Clique algorithm bipartiteness: {clique_bipart}\n")

    # Run the diffusion algorithm
    hyplogging.logger.info("Running the diffusion algorithm.")
    diff_alg_l, diff_alg_r, diff_bipart = hypalgorithms.find_bipartite_set_diffusion(dataset.hypergraph,
                                                                                     step_size=1, max_time=100,
                                                                                     approximate=True)
    hyplogging.logger.info(f"Diffusion algorithm bipartiteness: {diff_bipart}\n")
    print(f"Diffusion algorithm bipartiteness: {diff_bipart}\n")


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
    print("Loading dataset...")
    hyplogging.logger.info("Loading the imdb dataset.")
    imdb_dataset = datasets.ImdbDataset()

    print("Running algorithms...")
    dataset_experiment(imdb_dataset)


def log_migration_result(migration_dataset, title, left_set, right_set):
    """
    Given a pair of clusters in the migration dataset, display their evaluation.

    :param migration_dataset:
    :param title: The title of this result
    :param left_set:
    :param right_set:
    :return: nothing
    """
    bipartiteness = hypcheeg.hypergraph_bipartiteness(migration_dataset.hypergraph, left_set, right_set)
    cut_imbalance = hypcheeg.clsz_cut_imbalance(migration_dataset.directed_graph, left_set, right_set)
    hyplogging.logger.info(f"{title}")
    hyplogging.logger.info(f"  Hypergraph Bipartiteness: {bipartiteness}")
    hyplogging.logger.info(f"             Cut Imbalance: {cut_imbalance}")
    print(f"{title}")
    print(f"  Hypergraph Bipartiteness: {bipartiteness}")
    print(f"             Cut Imbalance: {cut_imbalance}")


def migration_experiment():
    """Compare our algorithm to the CLSZ algorithm on the migration dataset. We will compare them on the following
    three metrics:
      - Hypergraph bipartiteness
      - Graph bipartiteness
      - Graph flow ratio (as described in CLSZ)
    """
    migration_dataset = datasets.MigrationDataset()

    hyplogging.logger.info("Running CLSZ algorithm.")
    clsz_labels = clsz.cluster.cluster_networkx(migration_dataset.directed_graph, 10)
    hyplogging.logger.info(f'CLSZ labels: {" ".join(map(str, clsz_labels))}')
    print("CLSZ labels:", " ".join(map(str, clsz_labels)))

    # Now, run the hypergraph clustering algorithm
    hyplogging.logger.info("Running diffusion algorithm.")
    diff_left, diff_right, diff_bipart = hypalgorithms.find_bipartite_set_diffusion(migration_dataset.hypergraph,
                                                                                    max_time=1, step_size=0.1,
                                                                                    approximate=True)
    hyplogging.logger.info(f"Diffusion left: {str(diff_left)}")
    hyplogging.logger.info(f"Diffusion right: {str(diff_left)}")
    print("Diffusion left:", diff_left)
    print("Diffusion right:", diff_right)
    print()

    # Now, we will display the vitalstatistix of both algorithm.
    # For each pair of clusters in the CLSZ results, display the key results
    for cluster_idx_1 in range(10):
        for cluster_idx_2 in range(cluster_idx_1 + 1, 10):
            cluster_1 = [i for (i, label) in enumerate(clsz_labels) if label == cluster_idx_1]
            cluster_2 = [i for (i, label) in enumerate(clsz_labels) if label == cluster_idx_2]
            log_migration_result(migration_dataset, f"CLSZ Clusters {cluster_idx_1} and {cluster_idx_2}",
                                 cluster_1, cluster_2)

    # Now, print the results for the hypergraph diffusion algorithm
    log_migration_result(migration_dataset, f"Diffusion Algorithm", diff_left, diff_right)


if __name__ == "__main__":
    migration_experiment()
