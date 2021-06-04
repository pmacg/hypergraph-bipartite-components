"""
This file provides an interface to each dataset we will use for our experiments.
"""
import re

import numpy as np
import networkx as nx
import os.path
import itertools
import pandas as pd
import pickle
import hyplogging
import hypconstruct
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

    def combine_clusters(self, cluster_id1, cluster_id2, new_cluster_name):
        """
        Combine two existing clusters into a single cluster with the given name.
        """
        # We will need to re-map all clusters just to make sure the cluster indices still work
        old_cluster_to_new = {}
        new_cluster_labels = []
        new_cluster_id = None
        next_id = 0
        for cluster_id in range(len(self.cluster_labels)):
            if cluster_id not in [cluster_id1, cluster_id2]:
                # This is a cluster not to be renamed - give it a new index
                old_cluster_to_new[cluster_id] = next_id
                next_id += 1
                new_cluster_labels.append(self.cluster_labels[cluster_id])
            else:
                # This cluster is being renamed
                if new_cluster_id is None:
                    new_cluster_id = next_id
                    next_id += 1
                    new_cluster_labels.append(new_cluster_name)
                old_cluster_to_new[cluster_id] = new_cluster_id

        # Now, update the cluster labels
        self.cluster_labels = new_cluster_labels

        # Update the ground truth clusters, and we're done
        new_gt = [old_cluster_to_new[i] for i in self.gt_clusters]
        self.gt_clusters = new_gt

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

    def get_cluster(self, cluster_idx):
        """Return a list of the vertices in the dataset hypergraph which are in the given cluster."""
        return [i for i in range(len(self.gt_clusters)) if self.gt_clusters[i] == cluster_idx]

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

    def log_two_sets(self, left_set, right_set, show_clusters=False):
        """
        Print two sets of vertices, using the vertex name if available.
        :param left_set:
        :param right_set:
        :param show_clusters: Whether to show the cluster label of each item
        :return:
        """
        self.log_multiple_clusters([left_set, right_set], show_clusters=show_clusters)

    def log_multiple_clusters(self, clusters, max_rows=None, show_clusters=False):
        """
        Nicely format a list of clusters using the name of the vertices if available.
        :param clusters:
        :param max_rows: Optionally, specify the maximum number of rows to show
        :param show_clusters: Whether to show the cluster label of each item
        :return:
        """
        # Helper to construct the string to show for a given vertex index
        def to_print_for_item(ind):
            if not show_clusters:
                return self.vertex_labels[ind]
            else:
                return f"{self.vertex_labels[ind]} ({self.cluster_labels[self.gt_clusters[ind]]})"

        if self.vertex_labels is None:
            for cluster_id in range(len(clusters)):
                hyplogging.logger.info(f"  Cluster {cluster_id}: {clusters[cluster_id]}")
        else:
            max_items = max(map(len, clusters))
            max_item_length =\
                max(11, max(map(len, [to_print_for_item(i) for i in itertools.chain.from_iterable(clusters)])) + 2)
            hyplogging.logger.info(
                '|'.join([f"{'Cluster ' + str(c_id): ^{max_item_length}}" for c_id in range(len(clusters))]))
            hyplogging.logger.info('|'.join([f"{'-' * max_item_length}" for _ in range(len(clusters))]))

            for i in range(max_items):
                if max_rows is not None and i > max_rows:
                    break
                these_names = []
                for cluster_id in range(len(clusters)):
                    these_names.append(
                        to_print_for_item(clusters[cluster_id][i]) if i < len(clusters[cluster_id]) else '')
                hyplogging.logger.info(
                    '|'.join([f"{these_names[cluster_id]: ^{max_item_length}}" for cluster_id in range(len(clusters))]))

    def log_confusion_matrix(self, clusters):
        """Given a list of clusters, log the number in each cluster which corresponds to each ground truth value."""
        # Work out the cell size to use when printing the table
        cell_size = max(10, 2 + max(map(len, self.cluster_labels)))

        # Given a list of the strings to print on a row, construct the full row string
        def construct_row_string(list_of_contents):
            return '|'.join([f"{item: ^{cell_size}}" for item in list_of_contents])

        # Work out the horizontal line string
        horizontal_line = construct_row_string(['-' * cell_size] * (len(self.cluster_labels) + 2))

        # Print the header row with the cluster names
        header_row = construct_row_string([''] + self.cluster_labels + [''])
        hyplogging.logger.info(header_row)
        hyplogging.logger.info(horizontal_line)

        # Given the id of a ground truth cluster and a list of vertex indices, count the number of vertices in the
        # cluster which are in the corresponding ground truth cluster.
        def cluster_overlap(candidate_cluster, gt_id):
            gt_cluster = set([node for node in range(len(self.gt_clusters)) if self.gt_clusters[node] == gt_id])
            return len([node for node in candidate_cluster if node in gt_cluster])

        # Print each of the cluster rows
        gt_cluster_totals = [0] * len(self.cluster_labels)
        for cluster_id, cluster in enumerate(clusters):
            overlaps = [cluster_overlap(cluster, gt) for gt in range(len(self.cluster_labels))]
            total = sum(overlaps)
            cluster_row = construct_row_string([f"Cluster {cluster_id}"] + [str(o) for o in overlaps] + [str(total)])
            hyplogging.logger.info(cluster_row)

            # Update the cluster totals
            gt_cluster_totals = [gt_cluster_totals[i] + overlaps[i] for i in range(len(self.cluster_labels))]

        # Finally, print the bottom totals row
        hyplogging.logger.info(horizontal_line)
        bottom_row = construct_row_string([''] + gt_cluster_totals + [''])
        hyplogging.logger.info(bottom_row)

    def show_clustering_stats(self, clusters):
        """
        Given a list of clusters produced by some algorithm, report the key statistics for reporting.
        """
        # Given the id of a ground truth cluster and a list of vertex indices, count the number of vertices in the
        # cluster which are in the corresponding ground truth cluster.
        def cluster_overlap(candidate_cluster, gt_id):
            gt_cluster = set([node for node in range(len(self.gt_clusters)) if self.gt_clusters[node] == gt_id])
            return len([node for node in candidate_cluster if node in gt_cluster])

        # Construct the confusion matrix
        confusion_matrix = []
        gt_cluster_totals = [0] * len(self.cluster_labels)
        total_correctly_classified = 0
        for cluster_id, cluster in enumerate(clusters):
            overlaps = [cluster_overlap(cluster, gt) for gt in range(len(self.cluster_labels))]
            confusion_matrix.append(overlaps)
            total_correctly_classified += max(overlaps)

            # Update the cluster totals
            gt_cluster_totals = [gt_cluster_totals[i] + overlaps[i] for i in range(len(self.cluster_labels))]

        # For each GT cluster, print the precision and recall
        for cluster_id in range(len(self.cluster_labels)):
            hyplogging.logger.info(f"Cluster: {self.cluster_labels[cluster_id]}")

            # Check the precision and recall given by each found cluster, and report the best
            best_precision = 0
            best_recall = 0
            for found_cluster_id in range(len(clusters)):
                # Compute the precision
                if sum(confusion_matrix[found_cluster_id]) > 0:
                    precision = confusion_matrix[found_cluster_id][cluster_id] / sum(confusion_matrix[found_cluster_id])
                else:
                    precision = 0

                # Compute the recall
                if gt_cluster_totals[cluster_id] > 0:
                    recall = confusion_matrix[found_cluster_id][cluster_id] / gt_cluster_totals[cluster_id]
                else:
                    recall = 0

                # Update the best stats
                if precision > best_precision:
                    best_precision = precision
                if recall > best_recall:
                    best_recall = recall

            # Compute the f1 score
            f1_score = 2 / ((1/best_recall) + (1/best_precision))

            hyplogging.logger.info(f"    Precision: {best_precision}")
            hyplogging.logger.info(f"       Recall: {best_recall}")
            hyplogging.logger.info(f"           F1: {f1_score}")

        # Print the overall Accuracy
        accuracy = total_correctly_classified / sum(gt_cluster_totals)
        hyplogging.logger.info(f"Accuracy:")
        hyplogging.logger.info(f"  {total_correctly_classified} / {sum(gt_cluster_totals)} = {accuracy}")

    def show_large_and_small_degree_vertices(self):
        """
        Print a report of the vertices in the hypergraph with the largest and smallest degrees.
        :return:
        """
        hyplogging.logger.info("Showing degree distribution of hypergraph.")
        sorted_degrees = np.argsort(self.hypergraph.degrees)
        max_label_size = max(map(len, self.vertex_labels)) + 1
        for i in sorted_degrees:
            hyplogging.logger.info(f"{self.vertex_labels[i]: >{max_label_size}}: {self.hypergraph.degree(i)}")

    def remove_nodes(self, nodes_to_remove, remove_edges=False):
        """
        Given a list of node indices, remove them from the dataset
        :param nodes_to_remove:
        :param remove_edges: Whether to remove edges containing these nodes completely from the graph
        :return:
        """
        # Update the hypergraph, num_vertices and num_edges
        old_node_number = self.hypergraph.number_of_nodes()
        remaining_nodes = [node for node in range(old_node_number) if node not in nodes_to_remove]
        self.hypergraph = self.hypergraph.induced_hypergraph(remaining_nodes, remove_edges=remove_edges)
        self.num_edges = self.hypergraph.num_edges
        self.num_vertices = self.hypergraph.num_vertices

        # Update the node labels and gt clusters
        if len(self.vertex_labels) > 0:
            self.vertex_labels = [
                self.vertex_labels[node] for node in range(old_node_number) if node in remaining_nodes]
        if len(self.gt_clusters) > 0:
            self.gt_clusters = [self.gt_clusters[node] for node in range(old_node_number) if node in remaining_nodes]

        # At the moment, we are not able to update the edge labels.
        if len(self.edge_labels) > 0:
            hyplogging.logger.error("NOT ABLE TO UPDATE EDGE LABELS")

    def load_data(self):
        """
        Load the dataset.
        :return: Nothing
        """
        # For the generic class, this does nothing.
        pass


class ActorDirectorDataset(Dataset):
    """Create a simple dataset with actor and director information."""

    def __init__(self, num_actors=2):
        """Optionally pass the number of actors to include in each hyperedge. Can be a maximum of 3."""
        self.num_actors = num_actors
        super().__init__()

    @staticmethod
    def get_nx_giant_component(graph):
        """
        Given a networkx graph, extract the giant component of the graph.
        :param graph:
        :return:
        """
        gcc = sorted(nx.connected_components(graph), key=len, reverse=True)
        return graph.subgraph(gcc[0])

    def load_data(self):
        """Construct the hypergraph from the CSV of movie information."""
        movie_filename = "data/imdb/movie_metadata.csv"
        approved_filename = "data/imdb/connected_people.pickle"
        df = pd.read_csv(movie_filename)

        # We will include people only from the approved list - people in a connected subgraph.
        with open(approved_filename, 'rb') as f_in:
            approved_people = set(pickle.load(f_in))

        # We will go through the csv line by line, building the hypergraph.
        edge_labels = []
        person_ids = {}
        vertex_labels = []
        vertex_gt = []
        next_vertex_id = 0
        edges = []
        for row_ind in range(len(df)):
            edge_labels.append(df.loc[row_ind, "movie_title"])

            # For now, we only consider the director and one actor - proof of concept!
            director_name = df.loc[row_ind, "director_name"]
            actor1_name = df.loc[row_ind, "actor_1_name"]
            actor2_name = df.loc[row_ind, "actor_2_name"]
            actor3_name = df.loc[row_ind, "actor_3_name"]
            actors = [actor for actor in [actor1_name, actor2_name, actor3_name] if type(actor) is str]
            actors = actors[:self.num_actors]

            # If any of the data does not exist, ignore this row
            if type(director_name) is not str:
                continue

            this_edge = []
            for person in [director_name] + actors:
                if person in approved_people:
                    # Add the vertex id for this person if necessary
                    if person not in person_ids:
                        person_ids[person] = next_vertex_id
                        vertex_labels.append(person)
                        next_vertex_id += 1

                        # Update the ground truth clusters
                        if person == director_name:
                            vertex_gt.append(0)
                        else:
                            vertex_gt.append(1)
                    this_edge.append(person_ids[person])

            # Update the hypergraph edges
            if len(set(this_edge)) >= 2:
                edges.append(this_edge)

        # Fill out all of the dataset information
        self.hypergraph = lightgraphs.LightHypergraph(edges)
        self.vertex_labels = vertex_labels
        self.edge_labels = edge_labels
        self.gt_clusters = vertex_gt
        self.cluster_labels = ["director", "actor"]

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
                                      f"data/wikipedia-categories/{self.category_name}.gt",
                                      f"data/wikipedia-categories/{self.category_name}.clusters")
        self.is_loaded = True


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

        # The cluster labels are the names of the node types
        self.cluster_labels = self.nodes

        # Start by constructing the vertex set of the hypergraph.
        # We will store this as a dictionary of dictionaries. The top level is the different configured node types.
        # The second level is the name of the node to the node index.
        #
        # We will similarly store the mapping from the index given in the file, to the index we use internally for the
        # node. The indices in the files do not start at 0.
        file_index_to_internal = {node_type: {} for node_type in self.nodes}
        next_node_index = 0
        for cluster_idx, node_type in enumerate(self.nodes):
            new_vertex_labels, file_index_to_internal[node_type], next_node_index =\
                self.load_nodes_from(f"data/dblp/{node_type}.txt", next_node_index)

            # Shorten the labels for the papers!
            if node_type == 'paper':
                self.vertex_labels.extend([f"Paper: {label[:15]}..." for label in new_vertex_labels])
            else:
                self.vertex_labels.extend(new_vertex_labels)

            # Add the ground truth clusters
            self.gt_clusters.extend([cluster_idx] * len(new_vertex_labels))

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


class PennTreebankDataset(Dataset):
    """Load the part-of-speech tagging dataset. We assume that the dataset has already been pre-processed with
    https://github.com/hankcs/TreebankPreprocessing."""

    def __init__(self, n=float('inf'), min_degree=8, max_degree=float('inf'), categories_to_use=None,
                 allow_proper_nouns=True):
        """Take an argument specifying how many adjacent words to consider."""
        if categories_to_use is None:
            categories_to_use = ["Adverb", "Verb"]
        self.n = n
        self.min_degree = min_degree
        self.max_degree = max_degree

        noun_categories = ['NN', 'NNS']
        if allow_proper_nouns:
            noun_categories.extend(['NNP', 'NNPS'])
        category_poss = {"Adjective": ['JJ', 'JJR', 'JJS'],
                         "Noun": noun_categories,
                         "Adverb": ['RB', 'RBR', 'RBS'],
                         "Verb": ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']}

        # When we parse the dataset, we will ignore words which come from the list of POS.
        # We will additionally ignore any POS which are not in the categories to use.
        self.pos_to_ignore = {'CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'LS', 'MD', 'PDT', 'POS', 'PRP', 'PRP$', 'RP', 'SYM',
                              'TO', 'UH', 'WDT', 'WP', 'WP$', 'WRB', ',', '.', '$', '``', "''", ':', '-LRB-', '-RRB-',
                              '#'}
        if not allow_proper_nouns:
            self.pos_to_ignore.add('NNP')
            self.pos_to_ignore.add('NNPS')
        for category, poss in category_poss.items():
            if category not in categories_to_use:
                self.pos_to_ignore = self.pos_to_ignore.union(poss)

        # For the parts of speech we do not ignore, sort them into broader categories.
        self.pos_to_cluster = {}
        for category, poss in category_poss.items():
            if category in categories_to_use:
                for pos in poss:
                    self.pos_to_cluster[pos] = categories_to_use.index(category)

        # We will construct a list of readable edges in the hypergraph
        self.readable_edges = []

        # Load the hypergraph
        super().__init__()

        # Set the cluster labels
        self.cluster_labels = categories_to_use

    @staticmethod
    def get_sentences_from_processed_treebank(filenames):
        """
        Given a list of files containing the processed treebank data, return each sentence as a list of (word, pos)
        pairs.
        :param filenames:
        :return:
        """
        total_sentences = 0
        total_words = 0
        for filename in filenames:
            with open(filename, 'r') as f_in:
                current_sentence = []
                for line in f_in:
                    if len(line.strip()) == 0:
                        # If the line is empty, return the current sentence and reset
                        yield current_sentence
                        current_sentence = []
                        total_sentences += 1
                    else:
                        # Otherwise, add this word to the sentence.
                        word, pos = line.strip().split()
                        current_sentence.append((word, pos))
                        total_words += 1
        hyplogging.logger.info(
            f"There are {total_sentences} sentences and {total_words} words in the Treebank dataset.")

    def get_ngrams(self, sentence, stopwords=None, max_num=float('inf'), return_short_sentences=True):
        """
        Given a sentence, return the n-grams from the sentence.

        Combine adjacent proper nouns.

        :param sentence: The sentence on which to operate
        :param stopwords: If provided, will consider only words in the allowed parts of speech, and not in the stopwords
        :param max_num: The maximum number of n-grams to return, from the beginning of the sentence
        :param return_short_sentences: Whether we should return the sentence if it is shorter than n
        """
        current_ngram = []
        returned_any = False
        returned_so_far = 0
        for word, pos in sentence:
            # Combine proper nouns - get the 'word' to add to the ngram
            if stopwords is None or (pos not in self.pos_to_ignore and word not in stopwords):
                if pos in ['NNP', 'NNPS'] and len(current_ngram) > 0 and current_ngram[-1][1] in ['NNP', 'NNPS']:
                    new_word = current_ngram[-1][0] + ' ' + word
                    current_ngram.pop()
                    current_ngram.append((new_word, pos))
                else:
                    current_ngram.append((word, pos))

            if len(current_ngram) == self.n and returned_so_far < max_num:
                yield current_ngram
                current_ngram.pop(0)
                returned_any = True
                returned_so_far += 1

        # If we did not return any ngrams at all, return the whole sentence.
        if not returned_any and return_short_sentences:
            yield current_ngram

    def load_data(self):
        """Read in the dataset from the preprocessed file."""
        hyplogging.logger.info(
            f"Loading the Penn-Treebank dataset using {str(self.n) if self.n < 10000 else 'inf'}-grams.")
        treebank_filenames = ["data/nlp/penn-treebank/train.tsv"]

        # Construct the hypergraph
        edges = []
        words_to_indices = {}
        node_degrees = []
        next_index = 0
        for sentence in self.get_sentences_from_processed_treebank(treebank_filenames):
            for ngram in self.get_ngrams(sentence, stopwords=None, max_num=float('inf'), return_short_sentences=True):
                # Keep only the words of the allowed parts of speech and not in the list of stopwords
                filtered_ngram =\
                    [(word.lower(), pos) for word, pos in ngram if (pos not in self.pos_to_ignore)]

                if len(filtered_ngram) >= 2:
                    # Add each word to the graph if not there already, and construct the new edge list
                    self.readable_edges.append(filtered_ngram)
                    this_edge = []
                    for word, pos in filtered_ngram:
                        word_repr = f"{word}_{pos}"
                        if word_repr not in words_to_indices:
                            words_to_indices[word_repr] = next_index
                            next_index += 1
                            self.vertex_labels.append(word_repr)
                            self.gt_clusters.append(self.pos_to_cluster[pos])
                            node_degrees.append(0)
                        this_edge.append(words_to_indices[word_repr])
                        node_degrees[words_to_indices[word_repr]] += 1
                    edges.append(this_edge)

        # Now, construct the hypergraph, and we're done!
        self.hypergraph = lightgraphs.LightHypergraph(edges)

        # Remove all noes with small degrees
        nodes_to_remove = [node for node in range(len(node_degrees)) if (node_degrees[node] < self.min_degree or
                                                                         node_degrees[node] > self.max_degree)]
        self.remove_nodes(nodes_to_remove, remove_edges=True)
        self.is_loaded = True


class SbmDataset(Dataset):
    """Load an edgelist file generated with the stochastic block model."""

    def __init__(self, n, r, p, q, graph_num=None):
        self.n = n
        self.r = r
        self.p = p
        self.q = q
        self.graph_num = graph_num
        super().__init__()

    def load_data(self):
        """Load the hypergraph from the SBM edgelist file."""
        filename = f"data/sbm/two_cluster_sbm_{self.n}_{self.r}_{self.p}_{self.q}" \
                   f"{'_' + str(self.graph_num) if self.graph_num is not None else ''}.edgelist"
        hyplogging.logger.info(f"Loading SBM hypergraph from {filename}")

        # Check whether this graph has been generated with the SBM yet. If not, generate it.
        if not os.path.isfile(filename):
            hyplogging.logger.info("This hypergraph has not been generated from the SBM yet.")
            hypconstruct.hypergraph_sbm_two_cluster(filename, self.n, self.r, self.p, self.q)

        self.hypergraph = self.load_hypergraph_from_edgelist(filename)
        self.num_vertices = self.hypergraph.num_vertices
        self.num_edges = self.hypergraph.num_edges

        # Set the cluster information
        self.cluster_labels = ["Left Set", "Right Set"]
        self.gt_clusters = ([0] * self.n) + ([1] * self.n)

        self.is_loaded = True

