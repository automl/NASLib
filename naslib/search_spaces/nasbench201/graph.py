import numpy as np
import random
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import *

from naslib.search_spaces.core import primitives as ops
from naslib.search_spaces.core.graph import Graph
from naslib.search_spaces.core.query_metrics import Metric
from naslib.search_spaces.nasbench201.conversions import (
    convert_op_indices_to_naslib,
    convert_naslib_to_op_indices,
    convert_naslib_to_str,
    convert_op_indices_to_str,
)
from naslib.search_spaces.nasbench201.encodings import encode_201, encode_adjacency_one_hot_op_indices
from naslib.utils.encodings import EncodingType

from .primitives import ResNetBasicblock

NUM_EDGES = 6
NUM_OPS = 5

OP_NAMES = ["Identity", "Zero", "ReLUConvBN3x3", "ReLUConvBN1x1", "AvgPool1x1"]


class NasBench201SearchSpace(Graph):
    """
    Represents the NASBench201 search space.

    This class provides methods for querying and manipulating architectures
    within the search space, including methods for mutation and random sampling.

    Attributes:
        num_classes (int): Number of classes for classification tasks.
        in_channels (int): Number of input channels.
        max_epoch (int): Maximum number of epochs for training.
        space_name (str): The name of the search space.
        labeled_archs (list): A list of labeled architectures.
        instantiate_model (bool): Boolean indicating whether to instantiate the model during initialization.
        sample_without_replacement (bool): Boolean indicating whether to sample architectures without replacement.
        channels (list): Number of channels at different stages of the architecture.
        op_indices (list): Indices of the operations.
        OPTIMIZER_SCOPE (list): A list of the stages in the architecture, useful for scoping during optimization.
        QUERYABLE (bool): A boolean indicating whether the search space is queryable or not.
    """

    OPTIMIZER_SCOPE = [
        "stage_1",
        "stage_2",
        "stage_3",
    ]

    QUERYABLE = True

    def __init__(self, n_classes=10, in_channels=3):
        """
        Constructor method.

        This initializes the NasBench201SearchSpace object with provided number of classes and input channels.

        Args:
            n_classes (int, optional): The number of classes for the classification task. Defaults to 10.
            in_channels (int, optional): The number of input channels. Defaults to 3.
        """
        super().__init__()
        self.num_classes = n_classes
        self.op_indices = None

        self.max_epoch = 199
        self.in_channels = in_channels
        self.space_name = "nasbench201"
        self.labeled_archs = None
        self.instantiate_model = True
        self.sample_without_replacement = False

        #
        # Cell definition
        #
        cell = Graph()
        cell.name = "cell"  # Use the same name for all cells with shared attributes

        # Input node
        cell.add_node(1)

        # Intermediate nodes
        cell.add_node(2)
        cell.add_node(3)

        # Output node
        cell.add_node(4)

        # Edges
        cell.add_edges_densly()

        #
        # Makrograph definition
        #
        self.name = "makrograph"

        # Cell is on the edges
        # 1-2:               Preprocessing
        # 2-3, ..., 6-7:     cells stage 1
        # 7-8:               residual block stride 2
        # 8-9, ..., 12-13:   cells stage 2
        # 13-14:             residual block stride 2
        # 14-15, ..., 18-19: cells stage 3
        # 19-20:             post-processing

        total_num_nodes = 20
        self.add_nodes_from(range(1, total_num_nodes + 1))
        self.add_edges_from([(i, i + 1) for i in range(1, total_num_nodes)])

        self.channels = [16, 32, 64]

        #
        # operations at the edges
        #

        # preprocessing
        self.edges[1, 2].set("op", ops.Stem(C_in=self.in_channels,
                                            C_out=self.channels[0]))

        # stage 1
        for i in range(2, 7):
            self.edges[i, i + 1].set("op", cell.copy().set_scope("stage_1"))

        # stage 2
        self.edges[7, 8].set(
            "op", ResNetBasicblock(C_in=self.channels[0], C_out=self.channels[1], stride=2)
        )
        for i in range(8, 13):
            self.edges[i, i + 1].set("op", cell.copy().set_scope("stage_2"))

        # stage 3
        self.edges[13, 14].set(
            "op", ResNetBasicblock(C_in=self.channels[1], C_out=self.channels[2], stride=2)
        )
        for i in range(14, 19):
            self.edges[i, i + 1].set("op", cell.copy().set_scope("stage_3"))

        # post-processing
        self.edges[19, 20].set(
            "op",
            ops.Sequential(
                nn.BatchNorm2d(self.channels[-1]),
                nn.ReLU(inplace=False),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(self.channels[-1], self.num_classes),
            ),
        )

        self._set_cell_ops()

    def _set_cell_ops(self) -> None:
        """
        Sets the operations at the cells (channel dependent).
        The operations are set by calling the `update_edges` method with the provided update function for each optimizer scope.
        """
        for scope, c in zip(self.OPTIMIZER_SCOPE, self.channels):
            self.update_edges(
                update_func=lambda edge: _set_ops(edge, C=c),
                scope=scope,
                private_edge_data=True,
            )

    def query(
            self,
            metric: Metric,
            dataset: str,
            path: str = None,
            epoch: int = -1,
            full_lc: bool = False,
            dataset_api: dict = None) -> float:
        """
        Query results from the nasbench201 database based on the specified metric and dataset.

        Args:
            metric (Metric): The performance metric to query for.
            dataset (str): The dataset to query for.
            path (str, optional): The path to the nasbench201 database. Defaults to None.
            epoch (int, optional): The training epoch to query for. Defaults to -1, which means the last epoch.
            full_lc (bool, optional): If True, returns the full learning curve. Defaults to False.
            dataset_api (dict, optional): The api containing nasbench201 data. Defaults to None.

        Raises:
            NotImplementedError: If the `metric` is Metric.ALL or the `dataset_api` is not provided.

        Returns:
            float: The queried result.
        """
        assert isinstance(metric, Metric)
        if metric == Metric.ALL:
            raise NotImplementedError()
        if metric != Metric.RAW and metric != Metric.ALL:
            assert dataset in [
                "cifar10",
                "cifar100",
                "ImageNet16-120",
                "ninapro"
            ], "Unknown dataset: {}".format(dataset)
        if dataset_api is None:
            raise NotImplementedError("Must pass in dataset_api to query NAS-Bench-201")

        metric_to_nb201 = {
            Metric.TRAIN_ACCURACY: "train_acc1es",
            Metric.VAL_ACCURACY: "eval_acc1es",
            Metric.TEST_ACCURACY: "eval_acc1es",
            Metric.TRAIN_LOSS: "train_losses",
            Metric.VAL_LOSS: "eval_losses",
            Metric.TEST_LOSS: "eval_losses",
            Metric.TRAIN_TIME: "train_times",
            Metric.VAL_TIME: "eval_times",
            Metric.TEST_TIME: "eval_times",
            Metric.FLOPS: "flop",
            Metric.LATENCY: "latency",
            Metric.PARAMETERS: "params",
            Metric.EPOCH: "epochs",
        }

        if self.instantiate_model:
            arch_str = convert_naslib_to_str(self)
        else:
            arch_str = convert_op_indices_to_str(self.get_hash())

        if metric == Metric.RAW:
            # return all data
            return dataset_api["nb201_data"][arch_str]

        if dataset not in ["cifar10", "cifar10-valid", "cifar100", "ImageNet16-120", "ninapro"]:
            raise NotImplementedError("Invalid dataset")

        if dataset in ["cifar10", "cifar10-valid"]:
            # set correct cifar10 dataset
            dataset = "cifar10-valid"

        query_results = dataset_api["nb201_data"][arch_str]

        if metric == Metric.HP:
            # return hyperparameter info
            return query_results[dataset]["cost_info"]
        elif metric == Metric.TRAIN_TIME:
            return query_results[dataset]["cost_info"]["train_time"]

        if full_lc and epoch == -1:
            return query_results[dataset][metric_to_nb201[metric]]
        elif full_lc and epoch != -1:
            return query_results[dataset][metric_to_nb201[metric]][:epoch]
        else:
            # return the value of the metric only at the specified epoch
            return query_results[dataset][metric_to_nb201[metric]][epoch]

    def get_op_indices(self) -> list:
        """
        Gets the operation indices of the architecture.
        If they are not defined yet, it will convert the naslib object to operation indices and save them.

        Returns:
            list: The operation indices of the architecture.
        """
        if self.op_indices is None:
            self.op_indices = convert_naslib_to_op_indices(self)
        return self.op_indices

    def get_hash(self) -> tuple:
        """
        Gets a hash representation of the architecture.
        The hash is a tuple of the operation indices.

        Returns:
            tuple: The hash of the architecture.
        """
        return tuple(self.get_op_indices())

    def get_arch_iterator(self, dataset_api=None) -> Iterator:
        """
        Returns an iterator for all possible architectures in the search space.
        The iterator is a product of the number of operations for each edge in the graph.

        Args:
            dataset_api (optional): The dataset api. Defaults to None.

        Returns:
            Iterator: An iterator over all possible architectures.
        """
        return itertools.product(range(NUM_OPS), repeat=NUM_EDGES)

    def set_op_indices(self, op_indices: list) -> None:
        """
        Sets the operation indices for the current architecture.
        If the model should be instantiated, it will convert the operation indices to a naslib object.

        Args:
            op_indices (list): List of operation indices to set.
        """
        if self.instantiate_model == True:
            assert self.op_indices is None, f"An architecture has already been assigned to this instance of {self.__class__.__name__}. Instantiate a new instance to be able to sample a new model or set a new architecture."
            convert_op_indices_to_naslib(op_indices, self)

        self.op_indices = op_indices

    def set_spec(self, op_indices: list, dataset_api=None) -> None:
        """
        Sets the specifications of the architecture.

        Args:
            op_indices (list): List of operation indices to set.
            dataset_api (optional): The dataset api. Defaults to None.
        """
        self.set_op_indices(op_indices)

    def sample_random_labeled_architecture(self) -> None:
        """
        Samples a random labeled architecture and sets it as the current architecture.
        """
        assert self.labeled_archs is not None, "Labeled archs not provided to sample from"

        op_indices = random.choice(self.labeled_archs)

        if self.sample_without_replacement == True:
            self.labeled_archs.pop(self.labeled_archs.index(op_indices))

        self.set_spec(op_indices)

    def sample_random_architecture(self, dataset_api: dict = None, load_labeled: bool = False) -> None:
        """
        Samples a random architecture and sets it as the current architecture.

        Args:
            dataset_api (dict, optional): The api containing nasbench201 data. Defaults to None.
            load_labeled (bool, optional): If True, a random labeled architecture is sampled instead. Defaults to False.
        """

        if load_labeled == True:
            return self.sample_random_labeled_architecture()

        def is_valid_arch(op_indices: list) -> bool:
            return not ((op_indices[0] == op_indices[1] == op_indices[2] == 1) or
                        (op_indices[2] == op_indices[4] == op_indices[5] == 1))

        while True:
            op_indices = np.random.randint(NUM_OPS, size=(NUM_EDGES)).tolist()

            if not is_valid_arch(op_indices):
                continue

            self.set_op_indices(op_indices)
            break
        self.compact = self.get_op_indices()

    def mutate(self, parent: Graph, dataset_api: dict = None) -> None:
        """
        Mutates one operation from the parent operation indices and sets them as the operation indices of the current object.

        Args:
            parent (Graph): The parent Graph object from which to mutate.
            dataset_api (dict, optional): The api containing nasbench201 data. Defaults to None.
        """
        parent_op_indices = parent.get_op_indices()
        op_indices = list(parent_op_indices)

        edge = np.random.choice(len(parent_op_indices))
        available = [o for o in range(len(OP_NAMES)) if o != parent_op_indices[edge]]
        op_index = np.random.choice(available)
        op_indices[edge] = op_index
        self.set_op_indices(op_indices)

    def get_nbhd(self, dataset_api: dict = None) -> list:
        """
        Returns all neighbors of the architecture.

        Args:
            dataset_api (dict, optional): The api containing nasbench201 data. Defaults to None.

        Returns:
            list: List of neighbor models.
        """
        self.get_op_indices()
        nbrs = []
        for edge in range(len(self.op_indices)):
            available = [o for o in range(len(OP_NAMES)) if o != self.op_indices[edge]]

            for op_index in available:
                nbr_op_indices = list(self.op_indices).copy()
                nbr_op_indices[edge] = op_index
                nbr = NasBench201SearchSpace()
                nbr.set_op_indices(nbr_op_indices)
                nbr_model = torch.nn.Module()
                nbr_model.arch = nbr
                nbrs.append(nbr_model)

        random.shuffle(nbrs)
        return nbrs

    def get_type(self) -> str:
        """
        Returns the type of the search space.

        Returns:
            str: The type of the search space, "nasbench201" in this case.
        """
        return "nasbench201"

    def get_loss_fn(self) -> Callable:
        """
        Returns the loss function to be used for this architecture.

        Returns:
            Callable: A callable object (cross entropy loss function) that can be used as a loss function.
        """
        return F.cross_entropy

    def forward_before_global_avg_pool(self, x: torch.Tensor) -> list:
        """
        Performs a forward pass until the global average pooling layer and returns the outputs.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            list: List of outputs from the forward pass.
        """
        outputs = []

        def hook_fn(module, inputs, output_t):
            # print(f'Input tensor shape: {inputs[0].shape}')
            # print(f'Output tensor shape: {output_t.shape}')
            outputs.append(inputs[0])

        for m in self.modules():
            if isinstance(m, torch.nn.AdaptiveAvgPool2d):
                m.register_forward_hook(hook_fn)

        self.forward(x, None)

        assert len(outputs) == 1
        return outputs[0]

    def encode(self, encoding_type=EncodingType.ADJACENCY_ONE_HOT) -> Union[List, np.ndarray, dict]:
        """
        Encodes the current architecture based on a given encoding type.

        Args:
            encoding_type (EncodingType): The encoding type for the architecture.

        Returns:
            Any: The encoded architecture. The return type depends on the chosen encoding type.

        Raises:
            NotImplementedError: If the given encoding type is not yet supported as an architecture encoding for nb201.
        """
        return encode_201(self, encoding_type=encoding_type)


def _set_ops(edge, C: int) -> None:
    edge.data.set(
        "op",
        [
            ops.Identity(),
            ops.Zero(stride=1),
            ops.ReLUConvBN(C, C, kernel_size=3, affine=False, track_running_stats=False),
            ops.ReLUConvBN(C, C, kernel_size=1, affine=False, track_running_stats=False),
            ops.AvgPool1x1(kernel_size=3, stride=1, affine=False),
        ],
    )
