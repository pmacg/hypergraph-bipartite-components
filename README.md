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
In order to view the visualisation

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

### Running the experiments