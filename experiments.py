"""
This file implements some experiments on the hypergraph diffusion operator.
"""
import time
import hypconstruct
import hypalgorithms
import hypcheeg


def main():
    """Run experiments."""
    # We will run the clique algorithm and the approximate diffusion algorithm on a variety of graphs
    graph_sizes = [100, 200, 300]
    ranks = [3]
    edge_multipliers = [2, 2**(3-1) - 1, 2**(3-1)]
    configuration_runs = 10

    total_runs = len(graph_sizes) * len(ranks) * len(edge_multipliers) * configuration_runs

    with open("results.csv", "w") as f_out:
        f_out.write("run_id,configuration_id,vertices,edge_multiplier,edges,rank,clique_bipartiteness,clique_volume,"
                    "clique_runtime,approx_diffusion_bipartiteness,approx_diffusion_volume,approx_diffusion_runtime\n")
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

                        # Run the clique algorithm
                        start_time = time.time()
                        vertex_set_l, vertex_set_r, bipartiteness = hypalgorithms.find_bipartite_set_clique(hypergraph)
                        execution_time = time.time() - start_time
                        volume_l_r = hypcheeg.hypergraph_volume(hypergraph, vertex_set_l + vertex_set_r)
                        f_out.write(f"{bipartiteness},{volume_l_r},{execution_time},")

                        # Run the diffusion algorithm
                        start_time = time.time()
                        vertex_set_l, vertex_set_r, bipartiteness = hypalgorithms.find_bipartite_set_diffusion(
                            hypergraph, approximate=True)
                        execution_time = time.time() - start_time
                        volume_l_r = hypcheeg.hypergraph_volume(hypergraph, vertex_set_l + vertex_set_r)
                        f_out.write(f"{bipartiteness},{volume_l_r},{execution_time}\n")


if __name__ == "__main__":
    main()
