import numpy as np
import copy
import random
import torch
import torch.nn.functional as F
from typing import *

from naslib.search_spaces.core.graph import Graph
from naslib.search_spaces.core.query_metrics import Metric
from naslib.search_spaces.nasbench101.conversions import convert_spec_to_model, convert_spec_to_tuple, \
    convert_tuple_to_spec
from naslib.search_spaces.nasbench101.encodings import encode_101, encode_101_spec
from naslib.utils.encodings import EncodingType
from naslib.utils import get_dataset_api

INPUT = "input"
OUTPUT = "output"
CONV3X3 = "conv3x3-bn-relu"
CONV1X1 = "conv1x1-bn-relu"
MAXPOOL3X3 = "maxpool3x3"
OPS = [CONV3X3, CONV1X1, MAXPOOL3X3]

NUM_VERTICES = 7
OP_SPOTS = NUM_VERTICES - 2
MAX_EDGES = 9


class NasBench101SearchSpace(Graph):
    """
    This class is a search space for NAS-Bench-101, a tabular benchmark for neural
    architecture search.

    Args:
        n_classes : int, default=10
            Number of classes for the classification task.

    Attributes:
        QUERYABLE : bool
            Whether this search space is queryable, always True for NAS-Bench-101.
        num_classes : int
            Number of classes for the classification task.
        space_name : str
            Name of the search space.
        spec : dict or None
            The spec is a dict with the adjacency matrix and operations. None by default.
        labeled_archs : list of str or None
            List of labeled architectures to sample from.
        instantiate_model : bool
            If true, a model will be instantiated when a new spec is set.
        sample_without_replacement : bool
            If true, a sampled architecture will be removed from the list of available
            architectures.
    """

    QUERYABLE = True

    def __init__(self, n_classes=10):
        super().__init__()
        self.num_classes = n_classes
        self.space_name = "nasbench101"
        self.spec = None
        self.labeled_archs = None
        self.instantiate_model = True
        self.sample_without_replacement = False

        self.add_edge(1, 2)

    def convert_to_cell(self, matrix: np.ndarray, ops: list) -> dict:
        """
        Converts matrix and operations into a dictionary of NAS-Bench-101 cell.

        Args:
            matrix : np.ndarray
                Adjacency matrix of the cell.
            ops : list
                List of operations in the cell.

        Returns:
            dict
                Dictionary representation of the NAS-Bench-101 cell.
        """
        if len(matrix) < 7:
            # the nasbench spec can have an adjacency matrix of n x n for n<7, 
            # but in the nasbench api, it is always 7x7 (possibly containing blank rows)
            # so this method will add a blank row/column

            new_matrix = np.zeros((7, 7), dtype='int8')
            new_ops = []
            n = matrix.shape[0]
            for i in range(7):
                for j in range(7):
                    if j < n - 1 and i < n:
                        new_matrix[i][j] = matrix[i][j]
                    elif j == n - 1 and i < n:
                        new_matrix[i][-1] = matrix[i][j]

            for i in range(7):
                if i < n - 1:
                    new_ops.append(ops[i])
                elif i < 6:
                    new_ops.append('conv3x3-bn-relu')
                else:
                    new_ops.append('output')
            return {
                'matrix': new_matrix,
                'ops': new_ops
            }

        else:
            return {
                'matrix': matrix,
                'ops': ops
            }

    def query(self,
              metric: Metric,
              dataset: str = "cifar10",
              path: str = None,
              epoch: int = -1,
              full_lc: bool = False,
              dataset_api: dict = None) -> Union[list, float]:
        """
        Query the performance metrics of the architecture from NAS-Bench-101.

        Args:
            metric : Metric
                Performance metric to query.
            dataset : str, default="cifar10"
                Dataset for which to query the metric. Only "cifar10" is currently supported.
            path : str or None, optional
                Path to the NAS-Bench-101 dataset.
            epoch : int, default=-1
                Epoch for which to query the metric. If -1, return metric for all available epochs.
            full_lc : bool, default=False
                Whether to return the full learning curve.
            dataset_api : dict or None, optional
                API of the NAS-Bench-101 dataset.

        Returns:
            list or float
                Query result from NAS-Bench-101.
        """
        assert isinstance(metric, Metric)
        assert dataset in ["cifar10", None], "Unknown dataset: {}".format(dataset)

        if metric in [Metric.ALL, Metric.HP]:
            raise NotImplementedError()
        if dataset_api is None:
            raise NotImplementedError("Must pass in dataset_api to query nasbench101")
        assert epoch in [
            -1,
            4,
            12,
            36,
            108,
            None,
        ], f"Metric is not available at epoch {epoch}. NAS-Bench-101 does not have full learning curve information. Available epochs are [4, 12, 36, and 108]."

        metric_to_nb101 = {
            Metric.TRAIN_ACCURACY: "train_accuracy",
            Metric.VAL_ACCURACY: "validation_accuracy",
            Metric.TEST_ACCURACY: "test_accuracy",
            Metric.TRAIN_TIME: "training_time",
            Metric.PARAMETERS: "trainable_parameters",
        }

        if self.get_spec() is None:
            raise NotImplementedError(
                "Cannot yet query directly from the naslib object"
            )
        api_spec = dataset_api["api"].ModelSpec(**self.spec)

        if not dataset_api["nb101_data"].is_valid(api_spec):
            return -1

        query_results = dataset_api["nb101_data"].query(api_spec)
        if full_lc:
            vals = [
                dataset_api["nb101_data"].query(api_spec, epochs=e)[
                    metric_to_nb101[metric]
                ]
                for e in [4, 12, 36, 108]
            ]
            # return a learning curve with unique values only at 4, 12, 36, 108
            nums = [4, 8, 20, 56]
            lc = [val for i, val in enumerate(vals) for _ in range(nums[i])]
            if epoch == -1:
                return lc
            else:
                return lc[:epoch]

        if metric == Metric.RAW:
            return query_results
        elif metric == Metric.TRAIN_TIME:
            return query_results[metric_to_nb101[metric]]
        else:
            return query_results[metric_to_nb101[metric]] * 100

    def get_spec(self) -> dict:
        """
        Get the spec of the architecture.

        Returns:
            dict
                The spec of the architecture.
        """
        return self.spec

    def get_hash(self) -> tuple:
        """
        Get the hash of the architecture.

        Returns:
            The hash of the architecture.
        """
        return convert_spec_to_tuple(self.get_spec())

    def set_spec(self, spec: Union[str, dict, tuple], dataset_api: dict = None) -> None:
        """
        Set the spec of the architecture.

        Args:
            spec : str, dict, or tuple
                The spec to set for the architecture.
            dataset_api : dict or None, optional
                API of the NAS-Bench-101 dataset.
        """
        # TODO: convert the naslib object to this spec
        # convert_spec_to_naslib(spec, self)
        # assert self.spec is None, f"An architecture has already been assigned to this instance of {self.__class__.__name__}. Instantiate a new instance to be able to sample a new model or set a new architecture."
        assert isinstance(spec, str) or isinstance(spec, tuple) or isinstance(spec,
                                                                              dict), "The spec has to be a string (hash of the architecture), a dict with the matrix and operations, or a tuple (NASLib representation)."

        if isinstance(spec, str):
            """
            TODO: I couldn't find a better solution here.
            We need the arch iterator to return strings because the matrix/ops
            representation is too large for 400k elements. But having the `spec' be 
            strings would require passing in dataset_api for all of this search 
            space's methods. So the solution is to optionally pass in the dataset 
            api in set_spec and check whether `spec' is a string or a dict.
            """
            assert dataset_api is not None, "To set the hash string as the spec, the NAS-Bench-101 API must be passed as the dataset_api argument"
            fix, comp = dataset_api["nb101_data"].get_metrics_from_hash(spec)
            spec = self.convert_to_cell(fix['module_adjacency'], fix['module_operations'])
        elif isinstance(spec, tuple):
            spec = convert_tuple_to_spec(spec)

        if self.instantiate_model:
            assert self.spec is None, f"An architecture has already been assigned to this instance of {self.__class__.__name__}. Instantiate a new instance to be able to sample a new model or set a new architecture."
            model = convert_spec_to_model(spec)
            self.edges[1, 2].set('op', model)

        self.spec = spec

    def get_arch_iterator(self, dataset_api: dict) -> Iterator:
        """
        Get an iterator over all architectures in NAS-Bench-101.

        Args:
            dataset_api : dict
                API of the NAS-Bench-101 dataset.

        Returns:
            Iterator over all architectures in NAS-Bench- and used 4 entries.
        """
        return dataset_api["nb101_data"].hash_iterator()

    def sample_random_labeled_architecture(self) -> None:
        """
        Sample a random labeled architecture from the NAS-Bench-101 dataset.
        After the architecture is chosen, it is removed from the pool if sampling
        is without replacement.
        The architecture indices are then set as the current specification.

        Raises:
            AssertionError
                If the labeled architectures are not provided.
        """
        assert self.labeled_archs is not None, "Labeled archs not provided to sample from"

        while True:
            op_indices = random.choice(self.labeled_archs)
            if len(op_indices) == 56:
                break

        if self.sample_without_replacement == True:
            self.labeled_archs.pop(self.labeled_archs.index(op_indices))

        self.set_spec(op_indices)

    def sample_random_architecture(self, dataset_api: dict, load_labeled: bool = False) -> None:
        """
        This will sample a random architecture and update the edges in the
        naslib object accordingly.
        From the NASBench repository:
        one-hot adjacency matrix
        draw [0,1] for each slot in the adjacency matrix
        """

        if load_labeled == True:
            return self.sample_random_labeled_architecture()

        while True:
            matrix = np.random.choice([0, 1], size=(NUM_VERTICES, NUM_VERTICES))
            matrix = np.triu(matrix, 1)
            ops = np.random.choice(OPS, size=NUM_VERTICES).tolist()
            ops[0] = INPUT
            ops[-1] = OUTPUT

            spec = dataset_api["api"].ModelSpec(matrix=matrix, ops=ops)
            if dataset_api["nb101_data"].is_valid(spec):
                break

        self.set_spec({"matrix": matrix, "ops": ops})

    def mutate(self, parent: Graph, dataset_api: dict, edits: int = 1) -> None:
        """
        Mutate the given parent architecture specification. The mutation process
        includes flipping of edges and changing of operations with a defined
        probability. The new architecture is then set as the current specification.

        Args:
            parent : Graph
                The parent graph from which to derive the mutated architecture.
            dataset_api : dict
                API of the NAS-Bench-101 dataset.
            edits : int, optional
                The number of mutations to apply. Default is 1.
        Code inspird by https://github.com/google-research/nasbench
        """

        parent_spec = parent.get_spec()
        spec = copy.deepcopy(parent_spec)
        matrix, ops = spec['matrix'], spec['ops']
        for _ in range(edits):
            while True:
                new_matrix = copy.deepcopy(matrix)
                new_ops = copy.deepcopy(ops)
                for src in range(0, NUM_VERTICES - 1):
                    for dst in range(src + 1, NUM_VERTICES):
                        if np.random.random() < 1 / NUM_VERTICES:
                            new_matrix[src][dst] = 1 - new_matrix[src][dst]
                for ind in range(1, NUM_VERTICES - 1):
                    if np.random.random() < 1 / len(OPS):
                        available = [op for op in OPS if op != new_ops[ind]]
                        new_ops[ind] = np.random.choice(available)
                new_spec = dataset_api['api'].ModelSpec(new_matrix, new_ops)
                if dataset_api['nb101_data'].is_valid(new_spec):
                    break

        self.set_spec({'matrix': new_matrix, 'ops': new_ops})

    def get_nbhd(self, dataset_api: dict) -> list:
        """
        Retrieve all the valid neighbors of the current architecture. Both operation
        and edge neighbors are considered.

        Args:
            dataset_api : dict
                API of the NAS-Bench-101 dataset.

        Returns:
            List of all valid neighboring architectures.
        """
        spec = self.get_spec()
        matrix, ops = spec["matrix"], spec["ops"]
        nbhd = []

        def add_to_nbhd(new_matrix: np.ndarray, new_ops: list, nbhd: list) -> list:
            new_spec = {"matrix": new_matrix, "ops": new_ops}
            model_spec = dataset_api["api"].ModelSpec(new_matrix, new_ops)
            if dataset_api["nb101_data"].is_valid(model_spec):
                nbr = NasBench101SearchSpace()
                nbr.set_spec(new_spec)
                nbr_model = torch.nn.Module()
                nbr_model.arch = nbr
                nbhd.append(nbr_model)
            return nbhd

        # add op neighbors
        for vertex in range(1, OP_SPOTS + 1):
            if is_valid_vertex(matrix, vertex):
                available = [op for op in OPS if op != ops[vertex]]
                for op in available:
                    new_matrix = copy.deepcopy(matrix)
                    new_ops = copy.deepcopy(ops)
                    new_ops[vertex] = op
                    nbhd = add_to_nbhd(new_matrix, new_ops, nbhd)

        # add edge neighbors
        for src in range(0, NUM_VERTICES - 1):
            for dst in range(src + 1, NUM_VERTICES):
                new_matrix = copy.deepcopy(matrix)
                new_ops = copy.deepcopy(ops)
                new_matrix[src][dst] = 1 - new_matrix[src][dst]
                new_spec = {"matrix": new_matrix, "ops": new_ops}

                if matrix[src][dst] and is_valid_edge(matrix, (src, dst)):
                    nbhd = add_to_nbhd(new_matrix, new_ops, nbhd)

                if not matrix[src][dst] and is_valid_edge(new_matrix, (src, dst)):
                    nbhd = add_to_nbhd(new_matrix, new_ops, nbhd)

        random.shuffle(nbhd)
        return nbhd

    def get_loss_fn(self) -> Callable:
        """
        Get the loss function to use during optimization.

        Returns:
            Callable cross entropy loss function.
        """
        return F.cross_entropy

    def get_type(self) -> str:
        """
        Retrieve the type of the search space.

        Returns:
            A string representing the type of the search space ('nasbench101' in this case).
        """
        return "nasbench101"

    def forward_before_global_avg_pool(self, x: torch.Tensor) -> list:
        """
        Apply the forward pass of the architecture up until the global average pooling
        layer. The intermediate output is saved and returned.

        Args:
            x : torch.Tensor
                The input tensor.

        Returns:
            The intermediate output of the forward pass.
        """
        outputs = []

        def hook_fn(module, input_t, output_t):
            # print(f'Input tensor shape: {input_t[0].shape}')
            # print(f'Output tensor shape: {output_t.shape}')
            outputs.append(output_t)

        model = self.edges[1, 2]['op'].model
        model.layers[-1].register_forward_hook(hook_fn)

        self.forward(x, None)

        assert len(outputs) == 1
        return outputs[0]

    def encode(self, encoding_type=EncodingType.ADJACENCY_ONE_HOT):
        """
        Encode the architecture using the specified encoding type.

        Args:
            encoding_type : EncodingType, optional
                The type of encoding to use. Default is adjacency one-hot.

        Returns:
            The encoded architecture.
        """
        return encode_101(arch=self, encoding_type=encoding_type)


def get_utilized(matrix):
    # return the sets of utilized edges and nodes
    # first, compute all paths
    n = np.shape(matrix)[0]
    sub_paths = []
    for j in range(0, n):
        sub_paths.append([[(0, j)]]) if matrix[0][j] else sub_paths.append([])

    # create paths sequentially
    for i in range(1, n - 1):
        for j in range(1, n):
            if matrix[i][j]:
                for sub_path in sub_paths[i]:
                    sub_paths[j].append([*sub_path, (i, j)])
    paths = sub_paths[-1]

    utilized_edges = []
    for path in paths:
        for edge in path:
            if edge not in utilized_edges:
                utilized_edges.append(edge)

    utilized_nodes = []
    for i in range(NUM_VERTICES):
        for edge in utilized_edges:
            if i in edge and i not in utilized_nodes:
                utilized_nodes.append(i)

    return utilized_edges, utilized_nodes


def is_valid_vertex(matrix: np.ndarray, vertex: int) -> bool:
    edges, nodes = get_utilized(matrix)
    return vertex in nodes


def is_valid_edge(matrix: np.ndarray, edge: tuple) -> bool:
    edges, nodes = get_utilized(matrix)
    return edge in edges


if __name__ == '__main__':
    dataset_api = get_dataset_api('nasbench101', None)
    search_space = NasBench101SearchSpace()

    for i in range(1):
        graph = search_space.clone()
        graph.sample_random_architecture(dataset_api=dataset_api)

        graph_hash = graph.get_hash()
        print(graph_hash)

        x = torch.randn(2, 3, 32, 32)
        result = graph(x)
        print(result)
