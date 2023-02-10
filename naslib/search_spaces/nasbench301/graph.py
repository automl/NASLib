import os
import random
import torch
import logging
import numpy as np

from torch import nn
from copy import deepcopy
from ConfigSpace.read_and_write import json as config_space_json_r_w

from typing import *
from naslib.search_spaces.core import primitives as ops
from naslib.utils import get_project_root, AttrDict
from naslib.search_spaces.core.graph import Graph, EdgeData
from naslib.search_spaces.nasbench301.conversions import (
    convert_compact_to_genotype,
    convert_compact_to_naslib,
    convert_naslib_to_compact,
    convert_naslib_to_genotype,
    make_compact_mutable,
    make_compact_immutable,
)
from naslib.utils.encodings import EncodingType
from naslib.search_spaces.core.query_metrics import Metric
from naslib.search_spaces.nasbench301.encodings import encode_darts, encode_darts_compact
from .primitives import FactorizedReduce

import torch.nn.functional as F

logger = logging.getLogger(__name__)

NUM_VERTICES = 4
NUM_OPS = 7


class NasBench301SearchSpace(Graph):
    """
    The search space for CIFAR-10 as defined in

        Liu et al. 2019: DARTS: Differentiable Architecture Search

    It consists of a makrograph which is predefined and not optimized
    and two kinds of learnable cells: normal and reduction cells. At
    each edge are 8 primitive operations.
    """

    """
    Scope is used to target different instances of the same cell.
    Here we divide the cells in normal/reduction cell and stage.
    This is necessary to set the correct channels at each stage.
    The architecture optimizer should consider all of them equally.
    """
    OPTIMIZER_SCOPE = [
        "n_stage_1",
        "n_stage_2",
        "n_stage_3",
        "r_stage_1",
        "r_stage_2",
    ]

    QUERYABLE = True

    def __init__(self, n_classes=10, in_channels=3, auxiliary=True):
        """
        Initialize a new instance of the DARTS search space.
        Note:
            __init__ cannot take any parameters due to the way networkx is implemented.
            If we want to change the number of classes set a static attribute `NUM_CLASSES`
            before initializing the class. Default is 10 as for cifar-10.
        """
        super().__init__()

        self.channels = [16, 32, 64]
        self.compact = None
        self.load_labeled = None
        self.num_classes = n_classes
        self.max_epoch = 100
        self.in_channels = in_channels
        self.space_name = "nasbench301"
        self.auxiliary_output = auxiliary
        self.labeled_archs = None
        self.instantiate_model = True
        self.sample_without_replacement = False

        """
        Build the search space with the parameters specified in __init__.
        """
        #
        # Cell definition
        #

        # Normal cell first
        normal_cell = Graph()
        normal_cell.name = (
            "normal_cell"  # Use the same name for all cells with shared attributes
        )

        # Input nodes
        normal_cell.add_node(1)
        normal_cell.add_node(2)

        # Intermediate nodes
        normal_cell.add_node(3)
        normal_cell.add_node(4)
        normal_cell.add_node(5)
        normal_cell.add_node(6)

        # Output node
        normal_cell.add_node(7)

        # Edges
        normal_cell.add_edges_from([(1, i) for i in range(3, 7)])  # input 1
        normal_cell.add_edges_from([(2, i) for i in range(3, 7)])  # input 2
        normal_cell.add_edges_from([(3, 4), (3, 5), (3, 6)])
        normal_cell.add_edges_from([(4, 5), (4, 6)])
        normal_cell.add_edges_from([(5, 6)])

        # Edges connecting to the output are always the identity
        normal_cell.add_edges_from(
            [(i, 7, EdgeData().finalize()) for i in range(3, 7)]
        )  # output

        # Reduction cell has the same topology
        reduction_cell = deepcopy(normal_cell)
        reduction_cell.name = "reduction_cell"

        # set the cell name for all edges. This is necessary to convert a genotype to a naslib object
        for _, _, edge_data in normal_cell.edges.data():
            if not edge_data.is_final():
                edge_data.set("cell_name", "normal_cell")

        for _, _, edge_data in reduction_cell.edges.data():
            if not edge_data.is_final():
                edge_data.set("cell_name", "reduction_cell")

        #
        # Makrograph definition
        #
        self.name = "makrograph"

        self.add_node(1)  # input node
        self.add_node(2)  # preprocessing
        self.add_node(3)

        # cells
        self.add_node(4, subgraph=normal_cell.set_scope("n_stage_1").set_input([2, 3]))
        self.add_node(
            5, subgraph=normal_cell.copy().set_scope("n_stage_1").set_input([2, 4])
        )
        self.add_node(
            6, subgraph=reduction_cell.set_scope("r_stage_1").set_input([4, 5])
        )
        self.add_node(
            7, subgraph=normal_cell.copy().set_scope("n_stage_2").set_input([5, 6])
        )
        self.add_node(
            8, subgraph=normal_cell.copy().set_scope("n_stage_2").set_input([6, 7])
        )
        self.add_node(
            9, subgraph=reduction_cell.copy().set_scope("r_stage_2").set_input([7, 8])
        )
        self.add_node(
            10, subgraph=normal_cell.copy().set_scope("n_stage_3").set_input([8, 9])
        )
        self.add_node(
            11, subgraph=normal_cell.copy().set_scope("n_stage_3").set_input([9, 10])
        )

        # output
        self.add_node(12)

        # chain connections
        self.add_edges_from([(i, i + 1) for i in range(1, 11)])

        # skip connections
        self.add_edges_from([(i, i + 2) for i in range(4, 10)])
        self.add_edge(2, 4)
        self.add_edge(2, 5)

        if self.auxiliary_output:
            # node 12 becomes aux head
            self.add_node(13)

            # auxiliary
            self.add_edge(11, 12)

            # final output
            self.add_edge(11, 13)
        else:
            # final output
            self.add_edge(11, 12)

        #
        # Operations at the makrograph edges
        #
        self.num_in_edges = 4
        reduction_cell_indices = [6, 9]

        channel_map_from, channel_map_to = channel_maps(
            reduction_cell_indices, max_index=12
        )

        self._set_makrograph_ops(
            channel_map_from,
            channel_map_to,
            reduction_cell_indices,
            max_index=12,
            affine=True,
        )

        self._set_cell_ops(reduction_cell_indices)

    def _set_makrograph_ops(
            self,
            channel_map_from: dict,
            channel_map_to: dict,
            reduction_cell_indices: list,
            max_index: int,
            affine: bool = True,
    ) -> None:
        # pre-processing
        # In darts there is a hardcoded multiplier of 3 for the output of the stem
        stem_multiplier = 3
        self.edges[1, 2].set("op", ops.Stem(C_in=self.in_channels,
                                            # TODO_ARJUN: Make Stem use C_in. Currently, it is hardcoded to 3.
                                            C_out=self.channels[0] * stem_multiplier))

        # edges connecting cells
        for u, v, data in sorted(self.edges(data=True)):
            if u > 1 and v < max_index:
                if u == 3:
                    continue
                C_in = self.channels[channel_map_from[u]]
                C_out = self.channels[channel_map_to[v]]
                if C_in == C_out:
                    C_in = (
                        C_in * stem_multiplier if u == 2 else C_in * self.num_in_edges
                    )  # handle Stem
                    if v in reduction_cell_indices:
                        C_out *= 2
                    data.set(
                        "op", ops.ReLUConvBN(C_in, C_out, kernel_size=1, affine=affine)
                    )
                else:
                    data.set(
                        "op",
                        FactorizedReduce(
                            C_in * self.num_in_edges, C_out, affine=affine
                        ),
                    )

        # post-processing
        _, _, data = sorted(self.edges(data=True))[-1]
        data.set(
            "op",
            ops.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(self.channels[-1] * self.num_in_edges, self.num_classes),
            ),
        )

    def _set_cell_ops(self, reduction_cell_indices: list) -> None:
        # normal cells
        stages = ["n_stage_1", "n_stage_2", "n_stage_3"]

        for scope, c in zip(stages, self.channels):
            self.update_edges(
                update_func=lambda edge: _set_ops(edge, c, stride=1),
                scope=scope,
                private_edge_data=True,
            )

        # reduction cells
        # stride=2 is only for some edges, that's why we have to do it this way
        for n, c in zip(reduction_cell_indices, self.channels[1:]):
            reduction_cell = self.nodes[n]["subgraph"]
            for u, v, data in reduction_cell.edges.data():
                stride = 2 if u in (1, 2) else 1
                if not data.is_final():
                    edge = AttrDict(data=data)
                    _set_ops(edge, c, stride)

        #
        # Combining operations
        #
        for _, cell in sorted(self.nodes("subgraph")):
            if cell:
                cell.nodes[7]["comb_op"] = channel_concat

    def prepare_discretization(self) -> None:
        """
        In DARTS a node can have a maximum of two incoming edges.
        This is handled here.
        """

        self.update_nodes(
            _truncate_input_edges, scope=self.OPTIMIZER_SCOPE,
            single_instances=True
        )

    def prepare_evaluation(self) -> None:
        """
        In DARTS the evaluation model has 32 channels after the Stem
        and 3 normal cells at each stage.
        """

        # Taken from DARTS implementation
        # assuming input size 8x8
        if self.auxiliary_output:
            self.edges[11, 12].set(
                "op",
                ops.Sequential(
                    nn.ReLU(inplace=False),
                    nn.AvgPool2d(
                        5, stride=3, padding=0, count_include_pad=False
                    ),  # image size = 2 x 2
                    nn.Conv2d(self.channels[-1] * self.num_in_edges, 128, 1, bias=False),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=False),
                    nn.Conv2d(128, 768, 2, bias=False),
                    nn.BatchNorm2d(768),
                    nn.ReLU(inplace=False),
                    nn.Flatten(),
                    nn.Linear(768, self.num_classes),
                ),
            )

    def auxiliary_logits(self) -> torch.Tensor:
        return self.graph["out_from_12"]

    def load_labeled_architecture(self, dataset_api: dict = None) -> None:
        """
        This is meant to be called by a new NasBench301SearchSpace() object
        (one that has not already been discretized).
        It samples a random architecture from the nasbench301 training data,
        and updates the graph object to match the architecture.
        """
        index = np.random.choice(len(dataset_api["nb301_arches"]))
        compact = dataset_api["nb301_arches"][index]
        self.load_labeled = True
        self.set_compact(compact)

    def query(
            self,
            metric: Metric = None,
            dataset: str = None,
            path: str = None,
            epoch: int = -1,
            full_lc: bool = False,
            dataset_api: dict = None) -> Union[float, dict]:
        """
        Query results from nasbench 301
        """
        if dataset_api is None:
            raise NotImplementedError('Must pass in dataset_api to query NAS-Bench-301')

        assert dataset == 'cifar10' or dataset is None, "NAS-Bench-301 supports only CIFAR10 dataset"

        metric_to_nb301 = {
            Metric.TRAIN_LOSS: "train_losses",
            Metric.VAL_ACCURACY: "val_accuracies",
            Metric.TEST_ACCURACY: "val_accuracies",
            Metric.TRAIN_TIME: "runtime",
        }

        if self.load_labeled:
            """
            If we loaded the architecture from the nasbench301 training data (using
            load_labeled_architecture()), then self.compact will contain the architecture spec,
            and we can query the train loss or val accuracy at a specific epoch
            (also, querying will give 'real' answers, since these arches were actually trained)
            """
            assert metric in [
                Metric.VAL_ACCURACY,
                Metric.TEST_ACCURACY,
                Metric.TRAIN_LOSS,
                Metric.TRAIN_TIME,
                Metric.HP,
            ], "Only VAL_ACCURACY, TEST_ACCURACY, TRAIN_LOSS, TRAIN_TIME, and HP can be queried for the given model."
            query_results = dataset_api["nb301_data"][self.compact]

            if metric == Metric.TRAIN_TIME:
                return query_results[metric_to_nb301[metric]]
            elif metric == Metric.HP:
                # todo: compute flops/params/latency for each arch. These are placeholders
                return {"flops": 15, "params": 0.1, "latency": 0.01}
            elif full_lc and epoch == -1:
                return query_results[metric_to_nb301[metric]]
            elif full_lc and epoch != -1:
                return query_results[metric_to_nb301[metric]][:epoch]
            else:
                # return the value of the metric only at the specified epoch
                return query_results[metric_to_nb301[metric]][epoch]

        else:
            """
            If we did not load the architecture using load_labeled_architecture(), then we can
            only query the validation accuracy at epoch 100 by using nasbench301.
            """
            assert not epoch or epoch in [-1, 100]
            # assert metric in [Metric.VAL_ACCURACY, Metric.RAW]
            if self.instantiate_model == True:
                genotype = convert_naslib_to_genotype(self)
            else:
                genotype = convert_compact_to_genotype(self.compact)
            if metric == Metric.VAL_ACCURACY:
                val_acc = dataset_api["nb301_model"][0].predict(
                    config=genotype, representation="genotype"
                )
                return val_acc
            elif metric == Metric.TRAIN_TIME:
                runtime = dataset_api["nb301_model"][1].predict(
                    config=genotype, representation="genotype"
                )
                return runtime
            else:
                return -1

    def get_compact(self) -> tuple:
        if self.compact is None and self.instantiate_model == True:
            self.compact = convert_naslib_to_compact(self)
        return self.compact

    def get_hash(self) -> tuple:
        return self.get_compact()

    def get_arch_iterator(self, dataset_api: dict) -> Iterator:
        # currently set up for nasbench301 data, not surrogate
        arch_list = np.array(dataset_api["nb301_arches"])
        random.shuffle(arch_list)
        return arch_list

    def set_compact(self, compact: tuple) -> None:
        if self.instantiate_model == True:
            assert self.compact is None, f"An architecture has already been assigned to this instance of {self.__class__.__name__}. Instantiate a new instance to be able to sample a new model or set a new architecture."
            convert_compact_to_naslib(compact, self)

        self.compact = compact

    def set_spec(self, compact: tuple, dataset_api: dict = None):
        self.set_compact(make_compact_immutable(compact))

    def sample_random_labeled_architecture(self) -> None:
        assert self.labeled_archs is not None, "Labeled archs not provided to sample from"

        op_indices = random.choice(self.labeled_archs)

        if self.sample_without_replacement == True:
            self.labeled_archs.pop(self.labeled_archs.index(op_indices))

        self.set_spec(op_indices)

    def sample_random_architecture(self, dataset_api: dict = None, load_labeled: bool = False) -> None:
        """
        This will sample a random architecture and update the edges in the
        naslib object accordingly.
        """
        if load_labeled == True:
            assert dataset_api is not None, "NAS-Bench-301 API must be passed as argument to sample a trained model"
            self.load_labeled_architecture(dataset_api=dataset_api)
            return

        compact = [[], []]
        for i in range(NUM_VERTICES):
            ops = np.random.choice(range(NUM_OPS), NUM_VERTICES)

            nodes_in_normal = np.random.choice(range(i + 2), 2, replace=False)
            nodes_in_reduce = np.random.choice(range(i + 2), 2, replace=False)

            compact[0].extend(
                [(nodes_in_normal[0], ops[0]), (nodes_in_normal[1], ops[1])]
            )
            compact[1].extend(
                [(nodes_in_reduce[0], ops[2]), (nodes_in_reduce[1], ops[3])]
            )

        # convert the lists to tuples
        compact[0] = tuple(compact[0])
        compact[1] = tuple(compact[1])
        compact = tuple(compact)

        self.set_spec(compact)

    def mutate(self, parent: Graph, mutation_rate: int = 1, dataset_api: dict = None):
        """
        This will mutate one op from the parent op indices, and then
        update the naslib object and op_indices
        """
        parent_compact = parent.get_compact()
        parent_compact = make_compact_mutable(parent_compact)
        compact = parent_compact

        while True:
            for _ in range(int(mutation_rate)):
                cell = np.random.choice(2)
                pair = np.random.choice(8)
                num = np.random.choice(2)
                if num == 1:
                    compact[cell][pair][num] = np.random.choice(NUM_OPS)
                else:
                    inputs = pair // 2 + 2
                    choice = np.random.choice(inputs)
                    if pair % 2 == 0 and compact[cell][pair + 1][num] != choice:
                        compact[cell][pair][num] = choice
                    elif pair % 2 != 0 and compact[cell][pair - 1][num] != choice:
                        compact[cell][pair][num] = choice

            if make_compact_immutable(parent_compact) != make_compact_immutable(compact):
                break

            parent_compact = make_compact_mutable(parent_compact)
            compact = make_compact_mutable(compact)

        self.set_spec(compact)

    def get_nbhd(self, dataset_api: dict = None) -> list:
        # return all neighbors of the architecture
        self.get_compact()
        nbrs = []

        for i, cell in enumerate(self.compact):
            for j, pair in enumerate(cell):

                # mutate the op
                available = [op for op in range(NUM_OPS) if op != pair[1]]
                for op in available:
                    nbr_compact = make_compact_mutable(self.compact)
                    nbr_compact[i][j][1] = op
                    nbr = NasBench301SearchSpace()
                    nbr.set_compact(nbr_compact)
                    nbr_model = torch.nn.Module()
                    nbr_model.arch = nbr
                    nbrs.append(nbr_model)

                # mutate the edge
                other = j + 1 - 2 * (j % 2)
                available = [
                    edge
                    for edge in range(j // 2 + 2)
                    if edge not in [cell[other][0], pair[0]]
                ]

                for edge in available:
                    nbr_compact = make_compact_mutable(self.compact)
                    nbr_compact[i][j][0] = edge
                    nbr = NasBench301SearchSpace()
                    nbr.set_compact(nbr_compact)
                    nbr_model = torch.nn.Module()
                    nbr_model.arch = nbr
                    nbrs.append(nbr_model)

        random.shuffle(nbrs)
        return nbrs

    @staticmethod
    def get_configspace(
            path_to_configspace_obj=os.path.join(
                get_project_root(), "search_spaces/nasbench301/configspace.json"
            )
    ):
        """
        Returns the ConfigSpace object for the search space

        Args:
            path_to_configspace_obj: path to ConfigSpace json encoding

        Returns:
            ConfigSpace.ConfigutationSpace: a ConfigSpace object
        """
        with open(path_to_configspace_obj, "r") as fh:
            json_string = fh.read()
            config_space = config_space_json_r_w.read(json_string)
        return config_space

    def get_type(self) -> str:
        return "nasbench301"

    def get_loss_fn(self) -> Callable:
        return F.cross_entropy

    def forward_before_global_avg_pool(self, x: torch.Tensor) -> list:
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

    def encode(self, encoding_type=EncodingType.ADJACENCY_ONE_HOT):
        return encode_darts(self, encoding_type=encoding_type)


def _set_ops(edge, C: int, stride: int) -> None:
    """
    Replace the 'op' at the edges with the ones defined here.
    This function is called by the framework for every edge in
    the defined scope.
    Args:
        current_egde_data (EdgeData): The data that currently sits
            at the edge.
        C (int): convolutional channels
        stride (int): stride for the operation

    Returns:
        EdgeData: the updated EdgeData object.
    """
    edge.data.set(
        "op",
        [
            ops.Identity()
            if stride == 1
            else FactorizedReduce(C, C, stride, affine=False),
            ops.Zero(stride=stride),
            ops.MaxPool(C, 3, stride, use_bn=True),
            ops.AvgPool(C, 3, stride, use_bn=True),
            ops.SepConv(C, C, kernel_size=3, stride=stride, padding=1, affine=False),
            ops.SepConv(C, C, kernel_size=5, stride=stride, padding=2, affine=False),
            ops.DilConv(
                C, C, kernel_size=3, stride=stride, padding=2, dilation=2, affine=False
            ),
            ops.DilConv(
                C, C, kernel_size=5, stride=stride, padding=4, dilation=2, affine=False
            ),
        ],
    )


def _truncate_input_edges(node: tuple, in_edges: list, out_edges: list) -> None:
    """
    Removes input edges if there are more than k.
    """

    def _largest_post_softmax_weight(edge) -> int:
        _, edge_data = edge

        alpha = edge_data.alpha.detach()
        # The zero operation has its value set to -inf to ensure it never gets selected
        # This hack just ensures that it is the weakest softmax activation, since softmax can't
        # take inf as input
        alpha[1] = torch.min(alpha) - 0.001
        alpha_softmax = F.softmax(alpha)

        return torch.max(alpha_softmax)

    k = 2
    if len(in_edges) >= k:
        if any(e.has("alpha") or (e.has("final") and e.final) for _, e in in_edges):
            # We are in the one-shot case
            for _, data in in_edges:
                if data.has("final") and data.final:
                    return  # We are looking at an out node
                data.alpha.data[1] = -float("Inf")
            sorted_edge_ids = sorted(in_edges, key=_largest_post_softmax_weight, reverse=True)
            keep_edges, _ = zip(*sorted_edge_ids[:k])
            for edge_id, edge_data in in_edges:
                if edge_id not in keep_edges:
                    edge_data.delete()
        else:
            # We are in the discrete case (e.g. random search)
            for _, data in in_edges:
                if isinstance(data.op, list) and data.op[1].get_op_name == "Zero":
                    data.op.pop(1)
            if any(e.has("final") and e.final for _, e in in_edges):
                return  # TODO: how about mixed final and non-final?
            else:
                for _ in range(len(in_edges) - k):
                    in_edges[random.randint(0, len(in_edges) - 1)][1].delete()


def channel_concat(tensors):
    return torch.cat(tensors, dim=1)


def channel_maps(reduction_cell_indices: list, max_index: int) -> List[dict]:
    # calculate the mapping from edge indices to the respective channel

    assert len(reduction_cell_indices) == 2
    r_1, r_2 = reduction_cell_indices
    channel_map_from = {}
    channel_map_from.update({i: 0 for i in range(2, r_1)})
    channel_map_from.update({i: 1 for i in range(r_1, r_2)})
    channel_map_from.update({i: 2 for i in range(r_2, max_index)})

    channel_map_to = {}
    channel_map_to.update({i: 0 for i in range(3, r_1 + 1)})
    channel_map_to.update({i: 1 for i in range(r_1 + 1, r_2 + 1)})
    channel_map_to.update({i: 2 for i in range(r_2 + 1, max_index)})

    return channel_map_from, channel_map_to
