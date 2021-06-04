"""
The main entry point to the code. Accepts various parameters to run different experiments.
"""
import argparse
import experiments


def parse_args():
    parser = argparse.ArgumentParser(description='Run experiments on real-world data.')
    parser.add_argument('experiment', type=str, choices=['ptb', 'dblp', 'imdb', 'wikipedia'],
                        help="which experiment to perform")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.experiment == 'ptb':
        experiments.treebank_experiment()
    elif args.experiment == 'dblp':
        experiments.dblp_experiment()
    elif args.experiment == 'imdb':
        experiments.actor_director_experiment()
    elif args.experiment == 'wikipedia':
        experiments.wikipedia_categories_experiment()


if __name__ == "__main__":
    main()
