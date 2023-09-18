import os
import pickle
import numpy as np
import json
import copy
import random
import torch
import torch.nn as nn

from naslib.search_spaces.core.query_metrics import Metric
from naslib.search_spaces.core.graph import Graph
from naslib.search_spaces.nasbenchnlp.conversions import convert_recipe_to_compact, \
make_compact_mutable, convert_compact_to_recipe
from naslib.search_spaces.nasbenchnlp.encodings import encode_nlp
from naslib.utils.encodings import EncodingType
from naslib.utils import get_project_root


HIDDEN_TUPLE_SIZE = 2
INTERMEDIATE_VERTICES = 7
MAIN_OPERATIONS = ['linear', 'blend', 'elementwise_prod', 'elementwise_sum']
MAIN_WEIGHTS = [3., 1., 1., 1.]
MAIN_PROBABILITIES = np.array(MAIN_WEIGHTS) / np.sum(MAIN_WEIGHTS)
LINEAR_CONNECTIONS = [2, 3]
LINEAR_CONNECTION_WEIGHTS = [4, 1]
LINEAR_CONNECTION_PROBABILITIES = np.array(LINEAR_CONNECTION_WEIGHTS) / np.sum(LINEAR_CONNECTION_WEIGHTS)
ACTIVATIONS = ['activation_tanh', 'activation_sigm', 'activation_leaky_relu']
ACTIVATION_WEIGHTS = [1., 1., 1.]
ACTIVATION_PROBABILITIES = np.array(ACTIVATION_WEIGHTS) / np.sum(ACTIVATION_WEIGHTS)


class NasBenchNLPSearchSpace(Graph):
    """
    Represents the search space for NAS (Neural Architecture Search) in the context of NLP (Natural Language Processing).

    Note:
        Currently, building a NASLib object for nas-bench-nlp architectures is not supported.

    Attributes:
        QUERYABLE (bool): Specifies whether this class supports querying architectures.
    """

    QUERYABLE = True

    def __init__(self):
        """
        Initialize a new instance of the search space.
        """
        super().__init__()
        self.load_labeled = False
        self.max_epoch = 50
        self.max_nodes = 12
        self.accs = None

    def load_labeled_architecture(self, dataset_api=None, max_nodes=12):
        """
        Load a labeled architecture into the search space.

        Args:
            dataset_api : Dataset API containing information about available architectures.
            max_nodes (int): Maximum number of nodes for the architecture.

        Returns:
            None: The architecture is loaded into the instance.
        """
        while True:
            index = np.random.choice(len(dataset_api['nlp_arches']))
            compact = dataset_api['nlp_arches'][index]
            if len(compact[1]) <= max_nodes:
                break
        self.load_labeled = True
        self.set_compact(compact)

    def query(
        self,
        metric=None,
        dataset=None,
        path=None,
        epoch=-1,
        full_lc=False,
        dataset_api=None,
    ):
        """
        Query the architecture's performance metrics.

        Args:
            metric (Metric): Metric to query.
            dataset (str, optional): The dataset to query against.
            path (str, optional): File path for the saved architecture.
            epoch (int): The epoch at which to query the metric.
            full_lc (bool): Whether to query the full learning curve.
            dataset_api : The dataset API to be used for the query.

        Returns:
            Union[int, float, dict, list]: Query results.

        Raises:
            NotImplementedError: If querying metrics for extra training epochs is attempted.
        """
        if self.load_labeled:

            metric_to_nlp = {
                Metric.TRAIN_ACCURACY: "train_losses",
                Metric.VAL_ACCURACY: "val_losses",
                Metric.TEST_ACCURACY: "test_losses",
                Metric.TRAIN_TIME: "wall_times",
                Metric.TRAIN_LOSS: "train_losses",
            }

            assert self.load_labeled
            """
            If we loaded the architecture from the nas-bench-nlp data (using 
            load_labeled_architecture()), then self.compact will contain the architecture spec.
            """
            assert metric in [
                Metric.TRAIN_ACCURACY,
                Metric.TRAIN_LOSS,
                Metric.VAL_ACCURACY,
                Metric.TEST_ACCURACY,
                Metric.TRAIN_TIME,
            ]
            query_results = dataset_api["nlp_data"][self.compact]

            sign = 1
            if metric == Metric.TRAIN_LOSS:
                sign = -1

            if metric == Metric.TRAIN_TIME:
                return query_results[metric_to_nlp[metric]]
            elif metric == Metric.HP:
                # todo: compute flops/params/latency for each arch. These are placeholders
                return {"flops": 15, "params": 0.1, "latency": 0.01}
            elif full_lc and epoch == -1:
                return [
                    sign * (100 - loss) for loss in query_results[metric_to_nlp[metric]]
                ]
            elif full_lc and epoch != -1:
                return [
                    sign * (100 - loss)
                    for loss in query_results[metric_to_nlp[metric]][:epoch]
                ]
            else:
                # return the value of the metric only at the specified epoch
                return sign * (100 - query_results[metric_to_nlp[metric]][epoch])
        else:
            """
            If we did not load the architecture using load_labeled_architecture(), then we can
            query the learning curve by using the nas-bench-nlp surrogate.
            The surrogate outputs a learning curve of (100 - validation loss)
            """
            if self.accs is not None:
                NotImplementedError("Training with extra epochs not yet supported")

            arch = encode_nlp(self, encoding_type=EncodingType.ADJACENCY_MIX, max_nodes=12, accs=None)
            if metric == Metric.RAW:
                # TODO: add raw results
                return 0
            
            elif metric == Metric.TRAIN_TIME:
                # todo: right now it uses the average train time (in seconds)
                if epoch == -1:
                    return 9747
                else:
                    return int(9747 * epoch / self.max_epoch)

            lc = dataset_api['nlp_model'].predict(config=arch, 
                                                  representation='compact',
                                                  search_space='nlp')
            if full_lc and epoch == -1:
                return lc
            elif full_lc and epoch != -1:
                return lc[:epoch]
            else:
                # return the value of the metric only at the specified epoch
                return lc[epoch]

    def get_compact(self):
        """
        Get the compact representation of the architecture.

        Returns:
            Compact representation of the architecture.

        Raises:
            AssertionError: If the compact representation is not set.
        """
        assert self.compact is not None
        return self.compact
    
    def get_hash(self):
        """
        Get the hash based on the architecture's compact representation.

        Returns:
            Hash representation of the architecture.
        """
        return self.get_compact()

    def set_compact(self, compact):
        """
        Set the compact representation of the architecture.

        Args:
            compact : The compact representation of the architecture.

        Returns:
            None: The architecture is updated in-place.
        """
        self.compact = tuple(compact)

    def get_arch_iterator(self, dataset_api=None):
        """
        Get an iterator for iterating over the architectures in the dataset API.

        Args:
            dataset_api (dict): The dataset API containing information about architectures.

        Returns:
            np.array: An array of architecture representations to be iterated over.
        """
        # currently set up for nasbenchnlp data, not surrogate
        arch_list = np.array(dataset_api["nlp_arches"])
        random.shuffle(arch_list)
        return arch_list

    def set_spec(self, compact, dataset_api=None):
        """
        Set the architecture specification. This function exists to unify the interface across search spaces.

        Args:
            compact : The compact representation of the architecture.
            dataset_api : The dataset API.

        Returns:
            None: The architecture specification is set.
        """
        # this is just to unify the setters across search spaces
        # TODO: change it to set_spec on all search spaces
        self.set_compact(compact)

    def _generate_redundant_graph(self, recipe, base_nodes):
        """
        Generates a redundant graph based on the given recipe and base nodes.

        This function is adapted from NAS-Bench-NLP (https://arxiv.org/abs/2006.07116)
        and is responsible for generating a redundant graph according to specified
        probabilities and connections.

        Args:
            recipe : The architecture recipe.
            base_nodes : List of base nodes to consider.

        Returns:
            None: Modifies `recipe` in place.
        """
        i = 0
        activation_nodes = []
        while i < HIDDEN_TUPLE_SIZE + INTERMEDIATE_VERTICES:
            op = np.random.choice(MAIN_OPERATIONS, 1, p=MAIN_PROBABILITIES)[0]
            if op == 'linear':
                num_connections = np.random.choice(LINEAR_CONNECTIONS, 1, 
                                                   p=LINEAR_CONNECTION_PROBABILITIES)[0]
                connection_candidates = base_nodes + activation_nodes
                if num_connections > len(connection_candidates):
                    num_connections = len(connection_candidates)
                
                connections = np.random.choice(connection_candidates, num_connections, replace=False)
                recipe[f'node_{i}'] = {'op':op, 'input':connections}
                i += 1
                
                # after linear force add activation node tied to the new node, if possible (nodes budget)
                op = np.random.choice(ACTIVATIONS, 1, p=ACTIVATION_PROBABILITIES)[0]
                recipe[f'node_{i}'] = {'op':op, 'input':[f'node_{i - 1}']}
                activation_nodes.append(f'node_{i}')
                i += 1
                
            elif op in ['blend', 'elementwise_prod', 'elementwise_sum']:
                # inputs must exclude x
                if op == 'blend':
                    num_connections = 3
                else:
                    num_connections = 2
                connection_candidates = list(set(base_nodes) - set('x')) + list(recipe.keys())
                if num_connections <= len(connection_candidates):
                    connections = np.random.choice(connection_candidates, num_connections, replace=False)
                    recipe[f'node_{i}'] = {'op':op, 'input':connections}
                    i += 1

    def _create_hidden_nodes(self, recipe):
        """
        Creates hidden nodes based on the existing recipe.

        This function is adapted from NAS-Bench-NLP (https://arxiv.org/abs/2006.07116)
        and is responsible for creating hidden nodes in the architecture.

        Args:
            recipe (dict): The architecture recipe.

        Returns:
            None: Modifies `recipe` in place.
        """
        new_hiddens_map = {}
        for k in np.random.choice(list(recipe.keys()), HIDDEN_TUPLE_SIZE, replace=False):
            new_hiddens_map[k] = f'h_new_{len(new_hiddens_map)}'
            
        for k in new_hiddens_map:
            recipe[new_hiddens_map[k]] = recipe[k]
            del recipe[k]
            
        for k in recipe:
            recipe[k]['input'] = [new_hiddens_map.get(x, x) for x in recipe[k]['input']]

    def _remove_redundant_nodes(self, recipe):
        """
        Removes redundant nodes from the recipe.

        This function is adapted from NAS-Bench-NLP (https://arxiv.org/abs/2006.07116)
        and is responsible for removing any redundant nodes from the architecture.

        Args:
            recipe (dict): The architecture recipe.

        Returns:
            set: A set of visited nodes.
        """
        q = [f'h_new_{i}' for i in range(HIDDEN_TUPLE_SIZE)]
        visited = set(q)
        while len(q) > 0:
            if q[0] in recipe:
                for node in recipe[q[0]]['input']:
                    if node not in visited:
                        q.append(node)
                        visited.add(node)
            q = q[1:]

        for k in list(recipe.keys()):
            if k not in visited:
                del recipe[k]

        return visited

    def sample_random_architecture(self, dataset_api):
        """
        Samples a random architecture that satisfies the constraints.

        Args:
            dataset_api : The dataset API for querying architectures.

        Returns:
             The compact representation of the sampled architecture.
        """
        while True:
            prev_hidden_nodes = [f'h_prev_{i}' for i in range(HIDDEN_TUPLE_SIZE)]
            base_nodes = ['x'] + prev_hidden_nodes

            recipe = {}
            self._generate_redundant_graph(recipe, base_nodes)
            self._create_hidden_nodes(recipe)
            visited_nodes = self._remove_redundant_nodes(recipe)
            valid_recipe = True

            # check that all input nodes are in the graph
            for node in base_nodes:
                if node not in visited_nodes:
                    valid_recipe = False
                    break

            # constraint: prev hidden nodes are not connected directly to new hidden nodes
            for i in range(HIDDEN_TUPLE_SIZE):
                if len(set(recipe[f'h_new_{i}']['input']) & set(prev_hidden_nodes)) > 0:
                    valid_recipe = False
                    break

            if valid_recipe:
                compact = convert_recipe_to_compact(recipe)
                if len(compact[1]) > self.max_nodes:
                    continue
                self.set_compact(compact)
                return compact
            
    def mutate(self, parent, mutation_rate=1, dataset_api=None):
        """
        Mutates the architecture by altering edges or operations.

        Args:
            parent (object): Parent architecture.
            mutation_rate (int, optional): Number of mutations to perform. Defaults to 1.
            dataset_api (object, optional): The dataset API for querying architectures. Defaults to None.

        Returns:
            None: Modifies the architecture in place.

        Todo: mutate by adding/removing nodes.
        Todo: mutate the list of hidden nodes.
        Todo: edges between initial hidden nodes are not mutated.
        """

        parent_compact = parent.get_compact()
        parent_compact = make_compact_mutable(parent_compact)
        compact = copy.deepcopy(parent_compact)

        edges, ops, hiddens = compact
        max_node_idx = max([max(edge) for edge in edges])

        for _ in range(int(mutation_rate)):
            mutation_type = np.random.choice(2) + 1

            if mutation_type == 0:
                # change a hidden node. Note: currently not being used
                hiddens.pop(np.random.choice(len(hiddens)))
                choices = [i for i in range(4, max_node_idx) if i not in hiddens]
                hiddens.append(np.random.choice(choices))
                hiddens.sort()

            elif mutation_type == 1:
                # change an edge
                # Currently cannot change an edge to/from an h_prev node
                edge_choices = [i for i in range(len(edges)) if edges[i][0] >= 4]
                if len(edge_choices) > 0:
                    i = np.random.choice(edge_choices)
                    node_choices = [j for j in range(4, edges[i][1])]
                    if len(node_choices) > 0:
                        edges[i][0] = np.random.choice(node_choices)

            else:
                # change an op. Note: the first 4 nodes don't have ops
                idx_choices = [i for i in range(len(ops)) if ops[i] not in [0, 6, 7]]
                if len(idx_choices) > 0:
                    idx = np.random.choice(idx_choices)
                    num_inputs = len([edge for edge in edges if edge[1] == idx])

                    # each operation can have 1, 2, [2,3], or 3 inputs only
                    groups = [[0], [1, 2, 3], [4, 5]]
                    group = groups[num_inputs]
                    choices = [i for i in group if i != ops[idx]]
                    ops[idx] = np.random.choice(choices)

        compact = (edges, ops, hiddens)
        self.set_compact(compact)

    def get_nbhd(self, dataset_api=None):
        """
        Gets the neighborhood architectures based on the current architecture.

        Args:
            dataset_api (object, optional): The dataset API for querying architectures. Defaults to None.

        Returns:
            list: List of neighborhood architectures.

        Currently has the same todo's as in mutate()
        """

        compact = self.get_compact()
        compact = make_compact_mutable(compact)
        edges, ops, hiddens = compact
        nbhd = []

        def add_to_nbhd(new_compact, nbhd):
            nbr = NasBenchNLPSearchSpace()
            nbr.set_compact(new_compact)
            nbr_model = torch.nn.Module()
            nbr_model.arch = nbr
            nbhd.append(nbr_model)
            return nbhd

        # add op neighbors
        idx_choices = [i for i in range(len(ops)) if ops[i] not in [0, 6, 7]]
        for idx in idx_choices:
            num_inputs = len([edge for edge in edges if edge[1] == idx])
            groups = [[0], [1, 2, 3], [4, 5]]
            group = groups[num_inputs]
            choices = [i for i in group if i != ops[idx]]
            for choice in choices:
                new_ops = ops.copy()
                new_ops[idx] = choice
                nbhd = add_to_nbhd([copy.deepcopy(edges), new_ops, hiddens.copy()], nbhd)

        # add edge neighbors
        edge_choices = [i for i in range(len(edges)) if edges[i][0] >= 4]
        for i in edge_choices:
            node_choices = [j for j in range(4, edges[i][1])]
            for j in node_choices:
                new_edges = copy.deepcopy(edges)
                new_edges[i][0] = j
                nbhd = add_to_nbhd([new_edges, ops.copy(), hiddens.copy()], nbhd)

        random.shuffle(nbhd)
        return nbhd

    def get_type(self):
        """
        Gets the type of the search space.

        Returns:
            str: The type of the search space ("nlp").
        """
        return 'nlp'

    def get_max_epochs(self):
        """
        Gets the maximum number of epochs for training.

        Returns:
            int: The maximum number of epochs (49).
        """
        return 49

    def encode(self, encoding_type=EncodingType.ADJACENCY_ONE_HOT):
        """
        Encodes the architecture into a specific format.

        Args:
            encoding_type (EncodingType, optional): The type of encoding to use. Defaults to `EncodingType.ADJACENCY_ONE_HOT`.

        Returns:
            The encoded representation of the architecture.
        """
        return encode_nlp(self, encoding_type=encoding_type)
