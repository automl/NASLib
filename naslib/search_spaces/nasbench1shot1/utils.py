import itertools
import os
import re
import collections
import glob
import json
import pickle

import networkx as nx
import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn
import torch.nn.functional as F


INPUT = 'input'
OUTPUT = 'output'
CONV1X1 = 'conv1x1-bn-relu'
CONV3X3 = 'conv3x3-bn-relu'
MAXPOOL3X3 = 'maxpool3x3'
OUTPUT_NODE = 6

PRIMITIVES = [
    'maxpool3x3',
    'conv3x3-bn-relu',
    'conv1x1-bn-relu'
]

def softmax(weights, axis=-1):
    return F.softmax(torch.Tensor(weights), axis).data.cpu().numpy()


def get_top_k(array, k):
    return list(np.argpartition(array[0], -k)[-k:])


def get_directory_list(path):
    """Find directory containing config.json files"""
    directory_list = []
    # return nothing if path is a file
    if os.path.isfile(path):
        return []
    # add dir to directorylist if it contains .json files
    if len([f for f in os.listdir(path) if f == 'config.json' or
            'sample_val_architecture' in f]) > 0:
        directory_list.append(path)
    for d in os.listdir(path):
        new_path = os.path.join(path, d)
        if os.path.isdir(new_path):
            directory_list += get_directory_list(new_path)
    return directory_list


def parent_combinations(adjacency_matrix, node, n_parents=2):
    """Get all possible parent combinations for the current node."""
    if node != 1:
        # Parents can only be nodes which have an index that is lower than the
        # current index, because of the upper triangular adjacency matrix and
        # because the index is also a topological ordering in our case.
        return itertools.combinations(
            np.argwhere(adjacency_matrix[:node, node] == 0).flatten(),
            n_parents
        )  # (e.g. (0, 1), (0, 2), (1, 2), ...
    else:
        return [[0]]


def draw_graph_to_adjacency_matrix(graph):
    """
    Draws the graph in circular format for easier debugging
    :param graph:
    :return:
    """
    dag = nx.DiGraph(graph)
    nx.draw_circular(dag, with_labels=True)


def upscale_to_nasbench_format(adjacency_matrix):
    """
    The search space uses only 4 intermediate nodes, rather than 5 as used in
    nasbench. This method adds a dummy node to the graph which is never used to
    be compatible with nasbench.

    :param adjacency_matrix:
    :return:
    """
    return np.insert(
        np.insert(adjacency_matrix,
                  5, [0, 0, 0, 0, 0, 0], axis=1),
        5, [0, 0, 0, 0, 0, 0, 0], axis=0)


def parse_log(path):
    f = open(os.path.join(path, 'log.txt'), 'r')
    # Read in the relevant information
    train_accuracies = []
    valid_accuracies = []
    for line in f:
        if 'train_acc' in line:
            train_accuracies.append(line)
        elif 'valid_acc' in line:
            valid_accuracies.append(line)

    valid_error = [
        [1 - 1 / 100 * float(re.search('valid_acc ([-+]?[0-9]*\.?[0-9]+)',
                                       line).group(1))] for line in
        valid_accuracies
    ]

    train_error = [
        [1 - 1 / 100 * float(re.search('train_acc ([-+]?[0-9]*\.?[0-9]+)',
                                       line).group(1))] for line in
        train_accuracies
    ]

    return valid_error, train_error


# https://stackoverflow.com/questions/5967500/how-to-correctly-sort-a-string-with-a-number-inside
def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def compute_spearman_correlation_top_1000(one_shot_test_error, nb_test_error):
    sort_by_one_shot = lambda os, nb: [[y, x] for (y, x) in sorted(zip(os, nb),
                                                                   key=lambda
                                                                   pair:
                                                                   pair[0])]
    correlation_at_epoch = []
    for one_shot_test_error_on_epoch in one_shot_test_error:
        sorted_by_os_error = np.array(
            sort_by_one_shot(one_shot_test_error_on_epoch[0],
                             nb_test_error)
        )
        correlation_at_epoch.append(
            stats.spearmanr(sorted_by_os_error[:, 0][:1000],
                            sorted_by_os_error[:, 1][:1000]).correlation
        )
    return correlation_at_epoch


def compute_spearman_correlation(one_shot_test_error, nb_test_error):
    correlation_at_epoch = []
    for one_shot_test_error_on_epoch in one_shot_test_error:
        correlation_at_epoch.append(
            stats.spearmanr(one_shot_test_error_on_epoch[0],
                            nb_test_error).correlation
        )
    return correlation_at_epoch


def read_in_correlation(path, config):
    correlation_files = glob.glob(os.path.join(path, 'correlation_*.obj'))
    # If no correlation files available
    if len(correlation_files) == 0:
        return None, None
    else:
        read_file_list_with_pickle = lambda file_list: [pickle.load(open(file,
                                                                         'rb'))
                                                        for file in file_list]
        correlation_files.sort(key=natural_keys)

        one_shot_test_errors = glob.glob(os.path.join(path,
                                                      'one_shot_test_errors_*'))
        one_shot_test_errors.sort(key=natural_keys)
        one_shot_test_errors = read_file_list_with_pickle(one_shot_test_errors)

        # TODO change paths
        if config['search_space'] == '1':
            nb_test_errors_per_epoch = pickle.load(
                open('experiments/analysis/data/test_errors_per_epoch_ss1.obj',
                     'rb'))
        elif config['search_space'] == '2':
            nb_test_errors_per_epoch = pickle.load(
                open('experiments/analysis/data/test_errors_per_epoch_ss2.obj',
                     'rb'))
        elif config['search_space'] == '3':
            nb_test_errors_per_epoch = pickle.load(
                open('experiments/analysis/data/test_errors_per_epoch_ss3.obj',
                     'rb'))
        else:
            raise ValueError('Unknown search space')
        correlation_per_epoch_total = {
            epoch: compute_spearman_correlation(one_shot_test_errors,
                                                nb_test_errors_at_epoch) for
            epoch, nb_test_errors_at_epoch in nb_test_errors_per_epoch.items()
        }

        correlation_per_epoch_top = {
            epoch: compute_spearman_correlation_top_1000(one_shot_test_errors,
                                                         nb_test_errors_at_epoch)
            for epoch, nb_test_errors_at_epoch in nb_test_errors_per_epoch.items()
        }

        return collections.OrderedDict(
            sorted(correlation_per_epoch_total.items())
        ), collections.OrderedDict(
            sorted(correlation_per_epoch_top.items())
        )


class ExperimentDatabase:
    def __init__(self, root_dir):
        """Load all directories with trainings."""
        self._load(root_dir=root_dir)

    def query(self, conditions):
        searched_config = []
        for config in self._database:
            # Only select config if all conditions are satisfied
            conds_satisfied = [config.get(cond_key, None) == cond_val for
                               cond_key, cond_val in conditions.items()]
            if all(conds_satisfied):
                searched_config.append(config)

        return searched_config

    def query_correlation(self, conditions):
        searched_config = []
        for config in self._database:
            # Only select config if all conditions are satisfied
            conds_satisfied = [config.get(cond_key, None) == cond_val for
                               cond_key, cond_val in conditions.items()]
            if all(conds_satisfied):
                if config['scalars']['correlation_total'] is not None:
                    searched_config.append(config)

        return searched_config

    def _load(self, root_dir):
        self._database = []
        for directory in get_directory_list(root_dir):
            try:
                self._database.append(self._get_run_dictionary(directory))
            except Exception as e:
                print('Error occurred in loading', directory, e)

    def _get_run_dictionary(self, path):
        with open(os.path.join(path, 'config.json')) as fp:
            config = json.load(fp)

        with open(os.path.join(path, 'one_shot_validation_errors.obj'), 'rb') as fp:
            validation_errors = pickle.load(fp)

        with open(os.path.join(path, 'one_shot_test_errors.obj'), 'rb') as fp:
            test_errors = pickle.load(fp)

        one_shot_validation_errors, one_shot_training_errors = parse_log(path)
        correlation_total, correlation_top = read_in_correlation(path, config)

        config['scalars'] = {}
        config['scalars']['validation_errors'] = validation_errors
        config['scalars']['test_errors'] = test_errors
        config['scalars']['one_shot_validation_errors'] = one_shot_validation_errors
        config['scalars']['one_shot_training_errors'] = one_shot_training_errors
        config['scalars']['correlation_total'] = correlation_total
        config['scalars']['correlation_top'] = correlation_top

        return config

