import os
import pickle
import numpy as np
import random
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F

from naslib.search_spaces.core import primitives as ops
from naslib.search_spaces.core.graph import Graph, EdgeData
from naslib.search_spaces.core.query_metrics import Metric
from naslib.search_spaces.nasbench201.conversions import (
    convert_op_indices_to_naslib,
    convert_naslib_to_op_indices,
    convert_naslib_to_str,
)

from naslib.utils.utils import get_project_root

from .primitives import ResNetBasicblock

NUM_EDGES = 6
NUM_OPS = 5

OP_NAMES = ["Identity", "Zero", "ReLUConvBN3x3", "ReLUConvBN1x1", "AvgPool1x1"]


class NasBench201SearchSpace(Graph):
    """
    Implementation of the nasbench 201 search space.
    It also has an interface to the tabular benchmark of nasbench 201.
    """

    OPTIMIZER_SCOPE = [
        "stage_1",
        "stage_2",
        "stage_3",
    ]

    QUERYABLE = True

    def __init__(self, n_classes=10, in_channels=3):
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

    def _set_cell_ops(self):
        # set the ops at the cells (channel dependent)
        for scope, c in zip(self.OPTIMIZER_SCOPE, self.channels):
            self.update_edges(
                update_func=lambda edge: NasBench201SearchSpace._set_ops(edge, C=c),
                scope=scope,
                private_edge_data=True,
            )

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
        Query results from nasbench 201
        """
        assert isinstance(metric, Metric)
        if metric == Metric.ALL:
            raise NotImplementedError()
        if metric != Metric.RAW and metric != Metric.ALL:
            assert dataset in [
                "cifar10",
                "cifar100",
                "ImageNet16-120",
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

        arch_str = convert_naslib_to_str(self)

        if metric == Metric.RAW:
            # return all data
            return dataset_api["nb201_data"][arch_str]

        if dataset in ["cifar10", "cifar10-valid"]:
            query_results = dataset_api["nb201_data"][arch_str]
            # set correct cifar10 dataset
            dataset = "cifar10-valid"
        elif dataset == "cifar100":
            query_results = dataset_api["nb201_data"][arch_str]
        elif dataset == "ImageNet16-120":
            query_results = dataset_api["nb201_data"][arch_str]
        else:
            raise NotImplementedError("Invalid dataset")

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

    @staticmethod
    def _set_ops(edge, C):
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

    def get_op_indices(self):
        if self.op_indices is None:
            self.op_indices = convert_naslib_to_op_indices(self)
        return self.op_indices

    def get_hash(self):
        return tuple(self.get_op_indices())

    def get_arch_iterator(self, dataset_api=None):
        return itertools.product(range(NUM_OPS), repeat=NUM_EDGES)

    def set_op_indices(self, op_indices):
        # This will update the edges in the naslib object to op_indices
        self.op_indices = op_indices

        if self.instantiate_model == True:
            convert_op_indices_to_naslib(op_indices, self)

    def set_spec(self, op_indices, dataset_api=None):
        self.set_op_indices(op_indices)

    def sample_random_labeled_architecture(self):
        assert self.labeled_archs is not None, "Labeled archs not provided to sample from"

        op_indices = random.choice(self.labeled_archs)

        if self.sample_without_replacement == True:
            self.labeled_archs.pop(self.labeled_archs.index(op_indices))

        self.set_spec(op_indices)

    def sample_random_architecture(self, dataset_api=None, load_labeled=False):
        """
        This will sample a random architecture and update the edges in the
        naslib object accordingly.
        """

        if load_labeled == True:
            return self.sample_random_labeled_architecture()

        def is_valid_arch(op_indices):
            return not ((op_indices[0] == op_indices[1] == op_indices[2] == 1) or \
                        (op_indices[2] == op_indices[4] == op_indices[5] == 1))

        while True:
            op_indices = np.random.randint(NUM_OPS, size=(NUM_EDGES)).tolist()

            if not is_valid_arch(op_indices):
                continue

            self.set_op_indices(op_indices)
            break
        self.compact = self.get_op_indices()

    def mutate(self, parent, dataset_api=None):
        """
        This will mutate one op from the parent op indices, and then
        update the naslib object and op_indices
        """
        parent_op_indices = parent.get_op_indices()
        op_indices = list(parent_op_indices)

        edge = np.random.choice(len(parent_op_indices))
        available = [o for o in range(len(OP_NAMES)) if o != parent_op_indices[edge]]
        op_index = np.random.choice(available)
        op_indices[edge] = op_index
        self.set_op_indices(op_indices)

    def get_type(self):
        return "nasbench201"

    def get_loss_fn(self):
        return F.cross_entropy

    def forward_before_global_avg_pool(self, x):
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
