"""
This file provides an interface to each dataset we will use for our experiments.
"""
import hypernetx as hnx


class Dataset(object):
    """A generic interface for hypergraph datasets."""

    def __init__(self):
        # Track whether the data has been loaded yet
        self.is_loaded = False
        self.hypergraph = None
        self.num_vertices = 0
        self.num_edges = 0

        # The ground truth cluster for each vertex in the hypergraph
        # Should be 0-indexed.
        self.gt_clusters = []

        # The string labels for each vertex and cluster
        self.vertex_labels = []
        self.cluster_labels = []

        # Load the data at initialisation time
        self.load_data()

    @staticmethod
    def load_hypergraph_from_edgelist(filename, zero_indexed=True):
        """
        Construct a hypernetx hypergraph from a hypergraph edgelist file.
        :param filename: the edgelist file to load
        :param zero_indexed: whether the vertices in the given file are zero-indexed
        :return: the hypergraph object
        """
        # We construct a hypergraph with vertices numbered from 0.
        offset = 0 if zero_indexed else -1

        # Construct a dictionary of edges to construct the hypergraph
        hypergraph_edges = {}

        with open(filename, 'r') as f_in:
            edge_num = 0
            for line in f_in.readlines():
                vertices = [int(x) + offset for x in line.strip().split(',')]
                hypergraph_edges[f"e{edge_num}"] = vertices
                edge_num += 1

        # Construct and return the hypergraph
        return hnx.Hypergraph(hypergraph_edges)

    @staticmethod
    def load_list_from_file(filename):
        """
        Construct a python list with one entry per line in the given file.
        :param filename:
        :return: A python list
        """
        with open(filename, 'r') as f_in:
            new_list = []
            for line in f_in.readlines():
                new_list.append(line.strip())
        return new_list

    def load_data(self):
        """
        Load the dataset.
        :return: Nothing
        """
        # For the generic class, this does nothing.
        pass


class CongressCommitteesDataset(Dataset):
    """The congress committees dataset."""

    def load_data(self):
        # Start by constructing the hnx hypergraph object
        self.hypergraph = self.load_hypergraph_from_edgelist("data/senate-committees/hyperedges-senate-committees.txt",
                                                             zero_indexed=False)
        self.num_vertices = self.hypergraph.number_of_nodes()
        self.num_edges = self.hypergraph.number_of_edges()

        # Load the ground truth clusters and all labels
        self.gt_clusters =\
            [int(x) - 1 for x in self.load_list_from_file("data/senate-committees/node-labels-senate-committees.txt")]
        self.cluster_labels = self.load_list_from_file("data/senate-committees/label-names-senate-committees.txt")
        self.vertex_labels = self.load_list_from_file("data/senate-committees/node-names-senate-committees.txt")

        self.is_loaded = True
