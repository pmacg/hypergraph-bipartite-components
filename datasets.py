"""
This file provides an interface to each dataset we will use for our experiments.
"""
import re

import nltk.corpus
from scipy.io import loadmat
from uszipcode import SearchEngine
import numpy as np
import networkx as nx
import random
import itertools
import hyplogging
import lightgraphs
from nltk.corpus import stopwords


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
                try:
                    vertices = [int(x) + offset for x in re.split("[ ,\t]", line.strip())]
                    hypergraph_edges.append(vertices)
                except ValueError:
                    # Assume that this is a header line which cannot be parsed. If you see lots of the following log
                    # message then something has gone wrong.
                    hyplogging.logger.info("Ignoring header line in edgelist.")

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

    def log_two_sets(self, left_set, right_set):
        """
        Print two sets of vertices, using the vertex name if available.
        :param left_set:
        :param right_set:
        :return:
        """
        self.log_multiple_clusters([left_set, right_set])

    def log_multiple_clusters(self, clusters, max_rows=None):
        """
        Nicely format a list of clusters using the name of the vertices if available.
        :param clusters:
        :param max_rows: Optionally, specify the maximum number of rows to show
        :return:
        """
        if self.vertex_labels is None:
            for cluster_id in range(len(clusters)):
                hyplogging.logger.info(f"  Cluster {cluster_id}: {clusters[cluster_id]}")
        else:
            max_items = max(map(len, clusters))
            max_item_length =\
                max(map(len, [self.vertex_labels[i] for i in itertools.chain.from_iterable(clusters)])) + 2
            hyplogging.logger.info(
                '|'.join([f"{'Cluster ' + str(c_id): ^{max_item_length}}" for c_id in range(len(clusters))]))
            hyplogging.logger.info('|'.join([f"{'-' * max_item_length}" for i in range(len(clusters))]))

            for i in range(max_items):
                if max_rows is not None and i > max_rows:
                    break
                these_names = []
                for cluster_id in range(len(clusters)):
                    these_names.append(
                        self.vertex_labels[clusters[cluster_id][i]] if i < len(clusters[cluster_id]) else '')
                hyplogging.logger.info(
                    '|'.join([f"{these_names[cluster_id]: ^{max_item_length}}" for cluster_id in range(len(clusters))]))

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
                                      "data/imdb/credit.gt",
                                      "data/imdb/credit.clusters",
                                      vertex_zero_indexed=False,
                                      clusters_zero_indexed=False)
        self.is_loaded = True

    def use_subgraph(self, seed_actor, degrees_of_separation=2):
        """
        Given a seed actor, truncate the credit hypergraph to use only the nodes with the given degrees of separation
        of this actor.

        :param seed_actor:
        :param degrees_of_separation:
        :return: nothing, updates internal data structure
        """
        hyplogging.logger.info(f"Using IMDB with seed {seed_actor} and {degrees_of_separation} degrees of separation.")

        # Keep a record of the previous indices used for each node in the new hypergraph
        old_indices = {}  # new: old
        new_indices = {}  # old: new

        # Add the seed actor to the index dictionaries
        seed_old_index = self.vertex_labels.index(seed_actor)
        old_indices[0] = seed_old_index
        new_indices[seed_old_index] = 0

        # Find all of the nodes in the new graph first
        next_new_index = 1
        old_indices_in_new_graph = {seed_old_index}
        for step in range(degrees_of_separation):
            old_indices_at_start_iter = list(old_indices_in_new_graph)
            for node in old_indices_at_start_iter:
                for u in self.hypergraph.neighbours[node]:
                    if u not in old_indices_in_new_graph:
                        old_indices_in_new_graph.add(u)
                        old_indices[next_new_index] = u
                        new_indices[u] = next_new_index
                        next_new_index += 1

        # Now, find all of the edges in the hypergraph which will still be in the new hypergraph
        new_hyperedges = []
        old_edges_indices = {}
        next_edge_index = 0
        for i, edge in enumerate(self.hypergraph.edges):
            if len(set(edge).intersection(old_indices_in_new_graph)) > 1:
                # This edge will be in the new graph
                new_hyperedges.append([new_indices[u] for u in edge if u in old_indices_in_new_graph])
                old_edges_indices[next_edge_index] = i
                next_edge_index += 1

        # Set the new hypergraph
        self.hypergraph = lightgraphs.LightHypergraph(new_hyperedges)
        self.num_vertices = self.hypergraph.number_of_nodes()
        self.num_edges = self.hypergraph.number_of_edges()
        hyplogging.logger.debug(f"Now using {self.num_vertices} vertices and {self.num_edges} edges.")

        # Update the labels
        self.vertex_labels = [self.vertex_labels[old_indices[u]] for u in range(self.num_vertices)]
        self.gt_clusters = [self.gt_clusters[old_indices[u]] for u in range(self.num_vertices)]
        self.edge_labels = [self.edge_labels[old_edges_indices[u]] for u in range(self.num_edges)]

    def simple_cluster_check(self, cluster_name, cluster):
        """
        Given some set of vertices, break it down by percentages of each class.

        :param cluster_name:
        :param cluster:
        """
        cluster_size = len(cluster)
        gts = [self.gt_clusters[u] for u in cluster]

        hyplogging.logger.info(f"{cluster_name} breakdown:")
        for gt_id, cluster_label in enumerate(self.cluster_labels):
            proportion = gts.count(gt_id) / cluster_size
            hyplogging.logger.info(f"   {cluster_label}: {proportion}")


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


class FoodWebHFDDataset(Dataset):
    """The foodweb dataset, as used in the paper "Local HyperFlow Diffusion".
    See the github project here: https://github.com/s-h-yang/HFD
    """

    def load_data(self):
        self.load_edgelist_and_labels("data/foodweb-hfd/foodweb_edges.txt",
                                      "data/foodweb-hfd/foodweb_species.txt",
                                      None,
                                      "data/foodweb-hfd/foodweb_labels.txt",
                                      None,
                                      vertex_zero_indexed=False,
                                      clusters_zero_indexed=False)
        self.cluster_labels = ["Producers", "Low-level consumers", "High-level consumers"]
        self.is_loaded = True


class MigrationDataset(Dataset):
    """
    The Migration dataset. This dataset includes both a hypergraph and the original directed graph for comparison
    of hypergraph algorithms with directed graph algorithms.
    """

    def __init__(self):
        hyplogging.logger.info("Loading migration dataset.")
        self.directed_graph = None
        self.zipcodes = None
        super().__init__()

    def construct_simple_motif_hypergraph(self, directed_adjacency_matrix):
        """
        Given the adjacency matrix of the directed migration graph, construct a hypergraph in the following way:
          - for each vertex, we consider its 'outgoing' neighbours in turn, starting with the largest weight.
          - we bundle them into groups such that the total weight in each group is at least 0.8
          - we add a hyperedge for each bundle
        We throw away small-weighted edges where we cannot sum them to 0.8.
        """
        hyplogging.logger.debug("Constructing simple migration hypergraph.")
        hyperedges = []
        for vertex in self.directed_graph.nodes:
            neighbours = list(nx.neighbors(self.directed_graph, vertex))
            sorted_neighbours = sorted(neighbours, key=(lambda x: directed_adjacency_matrix[vertex, x]))

            # Construct the hyperedges for this vertex
            new_edge = [vertex]
            new_edge_weight = 0
            while len(sorted_neighbours) > 0:
                u = sorted_neighbours.pop()
                new_edge.append(u)
                new_edge_weight += directed_adjacency_matrix[vertex, u]

                if new_edge_weight > 0.8:
                    hyperedges.append(new_edge)
                    new_edge = [vertex]
                    new_edge_weight = 0
        self.hypergraph = lightgraphs.LightHypergraph(hyperedges)
        self.num_vertices = self.hypergraph.num_vertices
        self.num_edges = self.hypergraph.num_edges

    def construct_motif_hypergraph(self, directed_adjacency_matrix):
        """
        Given that the directed graph has been loaded already, construct a hypergraph based on the following motif:
            o --> o
            \    ^
             \  /
              X
             / \
            /   v
            o -> o

        Each of the edges must have weight greater than 0.8.
        """
        hyplogging.logger.debug("Constructing migration hypergraph.")

        # We start by constructing the directed graph, ignoring edges with small weight
        t = 0.8
        directed_adjacency_matrix[directed_adjacency_matrix < t] = 0
        thresholded_digraph = nx.to_networkx_graph(directed_adjacency_matrix, create_using=nx.DiGraph)

        # Search for the motif above
        hyperedges = []
        for node in thresholded_digraph.nodes:
            # Look at pairs of out-neighbours of node
            successors = list(thresholded_digraph.successors(node))
            used_successors = set()
            for i in range(len(successors)):
                if i in used_successors:
                    continue

                for j in range(i + 1, len(successors)):
                    if j in used_successors:
                        continue
                    u = successors[i]
                    v = successors[j]

                    # Look for nodes which have yet to be considered and appear as predecessors to both u and v
                    pred_u = set(thresholded_digraph.predecessors(u))
                    pred_v = set(thresholded_digraph.predecessors(v))
                    pred_common = pred_u.intersection(pred_v)
                    pred_common.remove(node)

                    # If we've found a valid configuration, add it to the hypergraph and remove the relevant edges
                    if len(pred_common) > 0:
                        chosen_pred = random.choice(list(pred_common))
                        hyperedges.append([node, chosen_pred, u, v])
                        thresholded_digraph.remove_edge(node, u)
                        thresholded_digraph.remove_edge(node, v)
                        thresholded_digraph.remove_edge(chosen_pred, u)
                        thresholded_digraph.remove_edge(chosen_pred, v)
                        used_successors.add(i)
                        used_successors.add(j)
                        break

        # Construct the final hypergraph
        self.hypergraph = lightgraphs.LightHypergraph(hyperedges)
        self.num_vertices = self.hypergraph.num_vertices
        self.num_edges = self.hypergraph.num_edges

    def load_data(self):
        # We construct the directed graph from the migration data as described in the CLSZ paper.
        hyplogging.logger.debug("Reading the adjacency matrix into a graph.")
        all_data = loadmat('data/migration/ALL_CENSUS_DATA_FEB_2015.mat')
        adjacency_matrix = all_data['MIG']
        normalised_adj_mat = np.divide(adjacency_matrix, (adjacency_matrix + adjacency_matrix.T))
        normalised_adj_mat[np.isnan(normalised_adj_mat)] = 0
        normalised_adj_mat[np.isinf(normalised_adj_mat)] = 0
        directed_adjacency_matrix = normalised_adj_mat - normalised_adj_mat.T
        directed_adjacency_matrix[directed_adjacency_matrix < 0] = 0
        self.directed_graph = nx.to_networkx_graph(directed_adjacency_matrix, create_using=nx.DiGraph)

        # Construct the hypergraph
        self.construct_motif_hypergraph(directed_adjacency_matrix)

        hyplogging.logger.debug("Generating zipcodes.")
        self.zipcodes = self.get_migration_zipcodes()

    @staticmethod
    def get_migration_zipcodes():
        m = loadmat('data/migration/ALL_CENSUS_DATA_FEB_2015.mat')

        # This is the array of latitude and longitude values for each vertex in the graph
        lat_long = list(map(list, zip(*m['A'])))
        long = lat_long[0]
        lat = lat_long[1]

        search = SearchEngine()

        zip_codes = []
        last_zipcode = 0
        for v in range(len(long)):
            result = search.by_coordinates(lat[v], long[v], returns=1)
            if len(result) > 0:
                zipcode = result[0].to_dict()["zipcode"]
            else:
                zipcode = last_zipcode
            last_zipcode = zipcode
            zip_codes.append(zipcode)

        return zip_codes


class WikipediaDataset(Dataset):
    """
    The wikipedia dataset.
    """

    def __init__(self, animal):
        """
        Load the dataset. Animal should be a string in ["chameleon", "crocodile", "squirrel"].
        """
        if animal not in ["chameleon", "crocodile", "squirrel"]:
            raise AssertionError("Animal must be chameleon, crocodile or squirrel.")
        self.animal = animal
        super().__init__()

    def load_data(self):
        hyplogging.logger.info(f"Loading the wikipedia {self.animal} dataset.")
        self.load_edgelist_and_labels(f"data/wikipedia/{self.animal}/musae_{self.animal}_edges.csv",
                                      None, None, None, None)
        self.is_loaded = True


class WikipediaCategoriesDataset(Dataset):
    """
    The wikipedia categories dataset.
    """

    def __init__(self, category_name):
        self.category_name = category_name
        super().__init__()

    def load_data(self):
        hyplogging.logger.info(f"Loading the wikipedia categories dataset.")
        self.load_edgelist_and_labels(f"data/wikipedia-categories/{self.category_name}.edgelist",
                                      f"data/wikipedia-categories/{self.category_name}.vertices",
                                      f"data/wikipedia-categories/{self.category_name}.edges",
                                      None, None)
        self.is_loaded = True


class MidDataset(Dataset):
    """
    The dataset object representing the dyadic MID dataset.
    """

    def __init__(self, start_date, end_date):
        """
        When constructing, pass the start and end dates for the data.

        :param start_date:
        :param end_date:
        """
        self.start_date = start_date
        self.end_date = end_date
        self.edgelist_filename = f"data/mid/dyadic_mid_{start_date}_{end_date}.edgelist"
        self.hypergraph_edgelist_filename = f"data/mid/dyadic_mid_{start_date}_{end_date}_hypergraph.edgelist"
        self.hypergraph_vertices_filename = f"data/mid/dyadic_mid_{start_date}_{end_date}_hypergraph.vertices"

        # The graph, graph_hypergraph and hypergraph objects will all have the same vertex set.
        self.graph = None
        self.graph_hypergraph = None

        # Store the mapping from the original vertex numbering to the number in the new vertex set
        self.new_vertex_to_original = {}
        self.original_vertex_to_new = {}

        super().__init__()

    def construct_simple_hypergraph(self):
        """Using the networkx graph object constructed in self.graph, construct a corresponding simple hypergraph."""
        if self.graph is None:
            raise AssertionError("Must construct networkx graph before simple hypergraph.")

        # Add a hyperedge for every edge
        hyperedges = []
        next_new_index = 0
        for u, v, weight in self.graph.edges.data("weight"):
            if weight is not None:
                # Check that we have assigned indexes for both u and v
                if u not in self.original_vertex_to_new:
                    self.original_vertex_to_new[u] = next_new_index
                    self.new_vertex_to_original[next_new_index] = u
                    next_new_index += 1
                if v not in self.original_vertex_to_new:
                    self.original_vertex_to_new[v] = next_new_index
                    self.new_vertex_to_original[next_new_index] = v
                    next_new_index += 1

                # Now, add the hyperedges with the new indices.
                for j in range(int(weight)):
                    hyperedges.append([self.original_vertex_to_new[u], self.original_vertex_to_new[v]])
        return lightgraphs.LightHypergraph(hyperedges)

    def construct_triangle_hypergraph(self):
        """Assuming that the simple graph hypergraph has been constructed, construct the triangle motif hypergraph."""
        if self.graph_hypergraph is None:
            raise AssertionError("Must construct simple hypergraph before constructing triangle hypergraph.")

        # Now, add edges for each triangle
        hyperedges = self.graph_hypergraph.edges.copy()
        all_nodes = list(self.graph.nodes)
        for a in range(len(all_nodes)):
            for b in range(a + 1, len(all_nodes)):
                for c in range(b + 1, len(all_nodes)):
                    u = all_nodes[a]
                    v = all_nodes[b]
                    w = all_nodes[c]

                    # Check for a triangle
                    try:
                        weight = min(self.graph.adj[u][v]['weight'],
                                     self.graph.adj[u][w]['weight'],
                                     self.graph.adj[v][w]['weight'])
                        for i in range(int(weight)):
                            hyperedges.append([self.original_vertex_to_new[u],
                                               self.original_vertex_to_new[v],
                                               self.original_vertex_to_new[w]])
                    except KeyError:
                        pass
        return lightgraphs.LightHypergraph(hyperedges)

    def get_motif_hyperedges(self, u, v, w, x):
        """Helper method for constructing the motif hypergraph. Given the vertices u, v, w, x, return a list of the
        hyperedges that should be added to the graph."""
        new_hyperedges = []

        def add_split_hyperedges(a, b, c, d):
            if b not in self.graph.adj[a] and d not in self.graph.adj[c]:
                try:
                    weight = min(self.graph.adj[a][c]['weight'],
                                 self.graph.adj[a][d]['weight'],
                                 self.graph.adj[b][c]['weight'],
                                 self.graph.adj[b][d]['weight'])
                    for i in range(int(weight)):
                        new_hyperedges.append([self.original_vertex_to_new[a],
                                               self.original_vertex_to_new[b],
                                               self.original_vertex_to_new[c],
                                               self.original_vertex_to_new[d]])
                except KeyError:
                    pass

        # There are 3 partitions of (u, v, w, x) to consider
        add_split_hyperedges(u, v, w, x)
        add_split_hyperedges(u, w, v, x)
        add_split_hyperedges(u, x, v, w)

        return new_hyperedges

    def check_motif_3_vertices(self, u, v, w):
        """Given three vertices, check whether they could possibly form part of a motif for the motif hypergraph."""
        # The key thing to check is that there are exactly 2 non-zero edges between these vertices.
        num_edges = 0
        if v in self.graph.adj[u]:
            num_edges += 1
        if w in self.graph.adj[u]:
            num_edges += 1
        if w in self.graph.adj[v]:
            num_edges += 1
        return num_edges == 2

    def construct_bipartite_motif_hypergraph(self):
        """Look for the following motif in the graph, and add corresponding hyperedges:
            - vertices u, v, w, x
            - edges u-w, u-x, v-w, v-x
            - no edges u-v, w-x
        """
        if self.graph_hypergraph is None:
            raise AssertionError("Must construct simple hypergraph before constructing motif hypergraph.")

        # Now, add edges for each motif
        hyperedges = self.graph_hypergraph.edges.copy()
        all_nodes = list(self.graph.nodes)
        for a in range(len(all_nodes)):
            hyplogging.logger.debug(f"Checking edges from vertex {a}/{len(all_nodes)}")
            u = all_nodes[a]
            for b in range(a + 1, len(all_nodes)):
                v = all_nodes[b]
                for c in range(b + 1, len(all_nodes)):
                    w = all_nodes[c]
                    if self.check_motif_3_vertices(u, v, w):
                        for d in range(c + 1, len(all_nodes)):
                            x = all_nodes[d]
                            hyperedges.extend(self.get_motif_hyperedges(u, v, w, x))
        return lightgraphs.LightHypergraph(hyperedges)

    def load_motif_hypergraph(self):
        """Load the simple graph and construct a hypergraph from the motifs."""
        hyplogging.logger.info(f"Loading MID dataset from {self.edgelist_filename}")

        # Construct a networkx graph from the edgelist, and extract the largest connected component.
        full_graph = nx.read_edgelist(self.edgelist_filename, data=[("weight", int)])
        connected_components = sorted(nx.connected_components(full_graph), key=len, reverse=True)
        self.graph = full_graph.subgraph(connected_components[0])

        # Now construct the hypergraphs from this graph
        self.graph_hypergraph = self.construct_simple_hypergraph()
        self.hypergraph = self.construct_bipartite_motif_hypergraph()

    def load_hypergraph(self):
        """
        Load the hypergraph directly constructed from the MID data.
        :return:
        """
        hyplogging.logger.info(f"Loading MID dataset from {self.hypergraph_edgelist_filename}")
        self.load_edgelist_and_labels(self.hypergraph_edgelist_filename, self.hypergraph_vertices_filename, None,
                                      None, None)
        self.is_loaded = True

    def load_data(self):
        self.load_hypergraph()


class DblpDataset(Dataset):
    """DBLP and ACM datasets from https://github.com/Jhy1993/HAN"""

    def __init__(self, nodes=None, max_authors=None):
        """
        We will always use the papers to define the edges in the hypergraph, However, the nodes can consist of
        different items.

        :param nodes: What object types to use as nodes. Can be a list with any combination of 'author', 'paper', and
                      'conf'.
        :param max_authors: The maximum number of authors to include for a single paper. Set this to some number to
                            limit the rank of the hypergraph.
        """
        self.max_authors = max_authors
        self.nodes = nodes
        if self.nodes is None:
            # By default, construct a hypergraph with two node types: authors and conferences.
            self.nodes = ["author", "conf"]

        super().__init__()

    @staticmethod
    def load_nodes_from(filename, next_node_index):
        """
        Given a file containing node labels, and the next node index to use, construct a list of node labels
        and a dictionary of file indices to the node index we will use.

        Return the list, dictionary, and the next unused node index.

        :param filename:
        :param next_node_index:
        :return:
        """
        new_node_labels = []
        file_index_dictionary = {}
        with open(filename, 'r') as f_in:
            for line in f_in.readlines():
                split_line = line.strip().split()
                node_name = ' '.join(split_line[1:])
                new_node_labels.append(node_name)
                file_index_dictionary[split_line[0]] = next_node_index
                next_node_index += 1

        return new_node_labels, file_index_dictionary, next_node_index

    @staticmethod
    def load_adjacencies_from(filename):
        """
        Given a filename with the adjacencies between a paper and other node types, load and return the tuples
        (paper ID, file_node_index)

        :param filename:
        :return:
        """
        with open(filename, 'r') as f_in:
            for line in f_in.readlines():
                yield tuple(line.strip().split())

    def load_data(self):
        hyplogging.logger.info(f"Loading DBLP dataset with nodes {self.nodes}")
        # Start by constructing the vertex set of the hypergraph.
        # We will store this as a dictionary of dictionaries. The top level is the different configured node types.
        # The second level is the name of the node to the node index.
        #
        # We will similarly store the mapping from the index given in the file, to the index we use internally for the
        # node. The indices in the files do not start at 0.
        file_index_to_internal = {node_type: {} for node_type in self.nodes}
        next_node_index = 0
        for node_type in self.nodes:
            new_vertex_labels, file_index_to_internal[node_type], next_node_index =\
                self.load_nodes_from(f"data/dblp/{node_type}.txt", next_node_index)

            # Shorten the labels for the papers!
            if node_type == 'paper':
                self.vertex_labels.extend([f"Paper: {label[:15]}..." for label in new_vertex_labels])
            else:
                self.vertex_labels.extend(new_vertex_labels)

        # Now, we construct the hyperedges. We will build a dictionary from the ID of the paper to the node of every
        # vertex in the corresponding edge.
        adjacency_list = {}
        for node_type in self.nodes:
            if node_type != 'paper':
                for paper_id, file_node_index in self.load_adjacencies_from(f"data/dblp/paper_{node_type}.txt"):
                    if paper_id not in adjacency_list:
                        adjacency_list[paper_id] = [file_index_to_internal[node_type][file_node_index]]
                        if 'paper' in self.nodes:
                            adjacency_list[paper_id].append(file_index_to_internal['paper'][paper_id])
                    else:
                        if (node_type != 'author' or
                                self.max_authors is None or
                                len(adjacency_list[paper_id]) < self.max_authors):
                            adjacency_list[paper_id].append(file_index_to_internal[node_type][file_node_index])

        self.hypergraph = lightgraphs.LightHypergraph(list(adjacency_list.values()))
        self.is_loaded = True


class NlpDataset(Dataset):

    def load_data(self):
        """Load the NLP dataset from file"""
        hyplogging.logger.info("Loading the NLP dataset.")
        word_to_node_index = {}
        next_node_index = 0
        ignore_words = set(stopwords.words('english'))
        edges = []

        with open("data/nlp/toy-dataset-processed.txt") as f_in:
            for line in f_in:
                words_in_line = [word for word in line.strip().split() if word not in ignore_words]
                new_edge = []

                for word in words_in_line:
                    if word not in word_to_node_index:
                        word_to_node_index[word] = next_node_index
                        self.vertex_labels.append(word)
                        next_node_index += 1
                    new_edge.append(word_to_node_index[word])
                edges.append(new_edge)

        # Construct the hypergraph object
        self.hypergraph = lightgraphs.LightHypergraph(edges)
        self.is_loaded = True
