"""
This file provides an interface to each dataset we will use for our experiments.
"""
import re
import lightgraphs


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
        self.edge_labels = []
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
        hypergraph_edges = []

        with open(filename, 'r') as f_in:
            for line in f_in.readlines():
                vertices = [int(x) + offset for x in re.split("[ ,]", line.strip())]
                hypergraph_edges.append(vertices)

        # Construct and return the hypergraph
        return lightgraphs.LightHypergraph(hypergraph_edges)

    @staticmethod
    def load_list_from_file(filename):
        """
        Construct a python list with one entry per line in the given file.
        :param filename:
        :return: A python list
        """
        if filename is None:
            return []

        with open(filename, 'r') as f_in:
            new_list = []
            for line in f_in.readlines():
                new_list.append(line.strip())
        return new_list

    def load_edgelist_and_labels(self, edgelist, vertex_labels, edge_labels, clusters, cluster_labels,
                                 vertex_zero_indexed=True, clusters_zero_indexed=True):
        """
        Load the hypergraph from the given files.
          - edgelist should have one hypergraph edge per line, with a list of vertices in the edge
          - vertex_labels: one label per line
          - edge_labels: one label per line, corresponding to the edgelist file
          - clusters: a cluster index per line, corresponding to vertex_labels
          - cluster_labels: a cluster label per line

        :param edgelist:
        :param vertex_labels:
        :param edge_labels:
        :param clusters:
        :param cluster_labels:
        :param vertex_zero_indexed:
        :param clusters_zero_indexed:
        :return:
        """
        # Start by constructing the hnx hypergraph object
        self.hypergraph = self.load_hypergraph_from_edgelist(edgelist, zero_indexed=vertex_zero_indexed)
        self.num_vertices = self.hypergraph.number_of_nodes()
        self.num_edges = self.hypergraph.number_of_edges()

        # Load the ground truth clusters and all labels
        offset = 0 if clusters_zero_indexed else -1
        self.gt_clusters = [(int(x) + offset if int(x) + offset >= 0 else None)
                            for x in self.load_list_from_file(clusters)]
        self.cluster_labels = self.load_list_from_file(cluster_labels)
        self.vertex_labels = self.load_list_from_file(vertex_labels)
        self.edge_labels = self.load_list_from_file(edge_labels)

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
        self.load_edgelist_and_labels("data/senate-committees/hyperedges-senate-committees.txt",
                                      "data/senate-committees/node-names-senate-committees.txt",
                                      None,
                                      "data/senate-committees/node-labels-senate-committees.txt",
                                      "data/senate-committees/label-names-senate-committees.txt",
                                      vertex_zero_indexed=False,
                                      clusters_zero_indexed=False)
        self.is_loaded = True


class ImdbDataset(Dataset):
    """The IMDB credit dataset."""

    def load_data(self):
        self.load_edgelist_and_labels("data/imdb/credit.edgelist",
                                      "data/imdb/credit.vertices",
                                      "data/imdb/credit.edges",
                                      None,
                                      None,
                                      vertex_zero_indexed=False)
        self.is_loaded = True


class FoodWebDataset(Dataset):
    """The foodweb dataset."""

    def load_data(self):
        self.load_edgelist_and_labels("data/foodweb/foodweb_hypergraph.edgelist",
                                      "data/foodweb/foodweb.vertices",
                                      None,
                                      "data/foodweb/foodweb.gt",
                                      "data/foodweb/foodweb.clusters",
                                      vertex_zero_indexed=False,
                                      clusters_zero_indexed=False)
        self.is_loaded = True
