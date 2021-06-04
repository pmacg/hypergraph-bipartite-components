# Finding Bipartite Components in Hypergraphs
This repository contains code to accompany the paper "Finding Bipartite Components in Hypergraph".
It provides an implementation of the proposed algorithm based on the new hypergraph diffusion process,
as well as the baseline algorithm based on the clique reduction.

Below, you can find instructions for running the code which will reproduce the results reported
in the paper.

## Set-up
The code was written to work with Python 3.6, although other versions of Python 3 
should also work.
We recommend that you run inside a virtual environment.

To install the dependencies of this project, run
```bash
pip install -r requirements.txt
```

## Viewing the visualisation
In order to demonstrate our algorithm, you can view the visualisation of the 2-graph
constructed at each step by running
```bash
python show_visualisation.py
```
This example was used to create Figure 1 in the main paper.

## Experiments
In this section, we give instructions for running the experiments reported in the paper.

### Penn Treebank Preprocessing
We are unfortunately not able to share the data used for the Penn Treebank experiment,
and so we give instructions here for how to preprocess this
data for use with our code. 
You will need to have your own access to the Penn Treebank corpus.

Follow the instructions in [this repository](https://github.com/hankcs/TreebankPreprocessing), passing
the ```--task pos``` command line option to
generate the files ```train.tsv```, ```test.tsv```, and ```dev.tsv```.
Copy these three files to the ```data/nlp/penn-treebank``` directory.

### Running the real-world experiments
To run the experiments on real-world data, you should run
```bash
python run_experiment.py {experiment_name}
```
where ```{experiment_name}``` is one of 'ptb', 'dblp', 'imdb', or 'wikipedia' to run the
Penn Treebank, DBLP, IMDB and Wikipedia experiments respectively.

### Running the synthetic experiments
To run an experiment on a single synthetic hypergraph, run
```bash
python run_experiment_synthetic.py {n} {r} {p} {q}
```
where ```{n}``` is the number of vertices in the hypergraph, ```{r}``` is the rank of
the hypergraph, ```{p}``` is the probability of an edge inside a cluster, and
```{q}``` is the probability of an edge between clusters.
Be careful not to set ```p``` or ```q``` to be too large.
See the main paper for more information about the random hypergraph model.
This will construct the hypergraph if needed, and report the performance of
the diffusion algorithm and
the clique algorithm on the constructed hypergraph.

#### Results
The full results from our experiments on synthetic hypergraphs are provided in the ```data/sbm/results```
directory, along with a Mathematica notebook for viewing them, and plotting the figures
shown in the paper.