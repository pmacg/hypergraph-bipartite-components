"""
The main entry point to the code. Accepts various parameters to run different experiments.
"""
import argparse
import datasets
import experiments
import hyplogging


def parse_args():
    parser = argparse.ArgumentParser(description='Run experiments on synthetic hypergraphs.')
    parser.add_argument('n', type=int, help="the number of vertices in the generated hypergraph")
    parser.add_argument('r', type=int, help="the rank of the edges in the generated hypergraph")
    parser.add_argument('p', type=float, help="the probability of an edge inside a cluster")
    parser.add_argument('q', type=float, help="the probability of an edge between clusters")
    return parser.parse_args()


def main():
    args = parse_args()
    sbm_dataset = datasets.SbmDataset(args.n, args.r, args.p, args.q)
    results = experiments.run_single_sbm_experiment(sbm_dataset)

    hyplogging.logger.info("")
    hyplogging.logger.info(f"Diffusion Algorithm Performance\n"
                           f"    Hypergraph Bipartiteness: {results[-5]}\n"
                           f"  Clique Graph Bipartiteness: {results[-3]}\n"
                           f"              Execution Time: {results[-1]}\n"
                           )
    hyplogging.logger.info(f"Clique Algorithm Performance\n"
                           f"    Hypergraph Bipartiteness: {results[0]}\n"
                           f"  Clique Graph Bipartiteness: {results[2]}\n"
                           f"              Execution Time: {results[4]}"
                           )


if __name__ == "__main__":
    main()
