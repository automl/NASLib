import random
import torch
import logging
import numpy as np
import networkx as nx
from naslib.search_spaces.core import primitives as ops

from torch import nn
from copy import deepcopy

from naslib.utils.utils import get_project_root
from naslib.search_spaces.core.graph import Graph, EdgeData
from .primitives import FactorizedReduce

logger = logging.getLogger(__name__)


class DartsSearchSpace(Graph):
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

    def __init__(self):
        """
        Initialize a new instance of the DARTS search space.

        Note:
            __init__ cannot take any parameters due to the way networkx is implemented.
            If we want to change the number of classes set a static attribute `NUM_CLASSES`
            before initializing the class. Default is 10 as for cifar-10.
        """
        super().__init__()

        self.channels = [16, 32, 64]

        self.num_classes = self.NUM_CLASSES if hasattr(self, 'NUM_CLASSES') else 10
        
        """
        Build the search space with the parameters specified in __init__.
        """
        #
        # Cell definition
        #

        # Normal cell first
        normal_cell = Graph()
        normal_cell.name = "normal_cell"    # Use the same name for all cells with shared attributes

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
        normal_cell.add_edges_from([(1, i) for i in range(3, 7)])   # input 1
        normal_cell.add_edges_from([(2, i) for i in range(3, 7)])   # input 2
        normal_cell.add_edges_from([(3, 4), (3, 5), (3, 6)])
        normal_cell.add_edges_from([(4, 5), (4, 6)])
        normal_cell.add_edges_from([(5, 6)])

        # Edges connecting to the output are always the identity
        normal_cell.add_edges_from([(i, 7, EdgeData().finalize()) for i in range(3, 7)])   # output
        
        # Reduction cell has the same topology
        reduction_cell = deepcopy(normal_cell)
        reduction_cell.name = "reduction_cell"

        #
        # Makrograph definition
        #
        self.name = "makrograph"

        self.add_node(1)    # input node
        self.add_node(2)    # preprocessing
        self.add_node(3, subgraph=normal_cell.set_scope("n_stage_1").set_input([2, 2]))
        self.add_node(4, subgraph=normal_cell.copy().set_scope("n_stage_1").set_input([2, 3]))
        self.add_node(5, subgraph=reduction_cell.set_scope("r_stage_1").set_input([3, 4]))
        self.add_node(6, subgraph=normal_cell.copy().set_scope("n_stage_2").set_input([4, 5]))
        self.add_node(7, subgraph=normal_cell.copy().set_scope("n_stage_2").set_input([5, 6]))
        self.add_node(8, subgraph=reduction_cell.copy().set_scope("r_stage_2").set_input([6, 7]))
        self.add_node(9, subgraph=normal_cell.copy().set_scope("n_stage_3").set_input([7, 8]))
        self.add_node(10, subgraph=normal_cell.copy().set_scope("n_stage_3").set_input([8, 9]))
        self.add_node(11)   # output

        self.add_edges_from([(i, i+1) for i in range(1, 11)])
        self.add_edges_from([(i, i+2) for i in range(2, 9)])
 
        #
        # Operations at the makrograph edges
        #
        self.num_in_edges = 4
        reduction_cell_indices = [5, 8]

        channel_map_from, channel_map_to = channel_maps(reduction_cell_indices, max_index=11)

        self._set_makrograph_ops(channel_map_from, channel_map_to, max_index=11, affine=False)

        self._set_cell_ops(reduction_cell_indices)


    def _set_makrograph_ops(self, channel_map_from, channel_map_to, max_index, affine=True):
        # pre-processing
        self.edges[1, 2].set('op', ops.Stem(self.channels[0]))

        # edges connecting cells
        for u, v, data in sorted(self.edges(data=True)):
            if u > 1 and v < max_index:
                C_in = self.channels[channel_map_from[u]] 
                C_out = self.channels[channel_map_to[v]]
                if C_in == C_out:
                    C_in = C_in if u == 2 else C_in * self.num_in_edges     # handle Stem
                    data.set('op', ops.ReLUConvBN(C_in, C_out, kernel_size=1, affine=affine))
                else:
                    data.set('op', FactorizedReduce(C_in * self.num_in_edges, C_out, affine=affine))
        
        # post-processing
        _, _, data = sorted(self.edges(data=True))[-1]
        data.set('op', ops.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.channels[-1] * self.num_in_edges, self.num_classes))
        )


    def _set_cell_ops(self, reduction_cell_indices):
         # normal cells
        stages = ["n_stage_1", "n_stage_2", "n_stage_3"]

        for scope, c in zip(stages, self.channels):
            self.update_edges(
                update_func=lambda current_edge_data: _set_cell_ops(current_edge_data, c, stride=1),
                scope=scope,
                private_edge_data=True
            )

        # reduction cells
        # stride=2 is only for some edges, that's why we have to do it this way
        for n, c in zip(reduction_cell_indices, self.channels[1:]):
            reduction_cell = self.nodes[n]['subgraph']
            for u, v, data in reduction_cell.edges.data():
                stride = 2 if u in (1, 2) else 1
                if not data.is_final():
                    reduction_cell.edges[u, v].update(_set_cell_ops(data, c, stride))

        #
        # Combining operations
        #
        for _, cell in sorted(self.nodes('subgraph')):
            if cell:
                cell.nodes[7]['comb_op'] = channel_concat


    def prepare_discretization(self):
        """
        In DARTS a node can have a maximum of two incoming edges.
        This is handled here.
        """
        
        self.update_nodes(_truncate_input_edges, scope=self.OPTIMIZER_SCOPE, single_instances=True)
    
    
    def prepare_evaluation(self):
        """
        In DARTS the evaluation model has 32 channels after the Stem
        and 3 normal cells at each stage.
        """
        # this is called after the optimizer has discretized the graph
        self._expand()
        
        # Operations at the edges
        self.channels = [32, 64, 128]
        reduction_cell_indices = [9, 16]

        channel_map_from, channel_map_to = channel_maps(reduction_cell_indices, max_index=23)
        self._set_makrograph_ops(channel_map_from, channel_map_to, max_index=23, affine=True)

        # Taken from DARTS implementation
        # assuming input size 8x8
        self.edges[22, 23].set('op', ops.Sequential(
            nn.ReLU(inplace=True),
            nn.AvgPool2d(5, stride=3, padding=0, count_include_pad=False), # image size = 2 x 2
            nn.Conv2d(self.channels[-1] * self.num_in_edges, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 768, 2, bias=False),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(768, self.num_classes))
        )

        self.update_edges(
            update_func=_double_channels,
            scope=self.OPTIMIZER_SCOPE,
            private_edge_data=True
        )


    def _expand(self):
        # shift the node indices to make space for 4 more nodes at each stage
        # and the auxiliary logits
        mapping = {
            5: 9,
            6: 10,
            7: 11,
            8: 16,
            9: 17,
            10: 18,
            11: 24,     # 23 is auxiliary
        }
        nx.relabel_nodes(self, mapping, copy=False)
        
        # fix edges
        self.remove_edges_from(list(self.edges()))
        self.add_edges_from([(i, i+1) for i in range(1, 22)])
        self.add_edges_from([(i, i+2) for i in range(2, 21)])
        self.add_edge(22, 23)   # auxiliary output
        self.add_edge(22, 24)   # final output
        
        to_insert = [] + list(range(5, 9)) + list(range(12, 16)) + list(range(19, 23))
        for i in to_insert:
            normal_cell = self.nodes[i-1]['subgraph']
            self.add_node(i, subgraph=normal_cell.copy().set_scope(normal_cell.scope).set_input([i-2, i-1]))
        
        for i, cell in sorted(self.nodes(data='subgraph')):
            if cell:
                if i == 3:
                    cell.input_node_idxs = [2, 2]
                else:
                    cell.input_node_idxs = [i-2, i-1]


    def auxilary_logits(self):
        return self.graph['out_from_23']


    def query(self, metric=None, dataset=None, path=None):
        """
        Query results from nasbench 301. Currently we only provide the 
        genotype query as list but we will integrate nb301 in the future.
        """
        def convert(cell):
            """convert the naslib representation to nasbench301"""
            ops_to_nb301 = {
                'Identity': 'skip_connect',
                'FactorizedReduce': 'skip_connect',
                'SepConv3x3': 'sep_conv_3x3',
                'DilConv3x3': 'dil_conv_3x3',
                'SepConv5x5': 'sep_conv_5x5',
                'DilConv5x5': 'dil_conv_5x5',
                'AvgPool1x1': 'avg_pool_3x3',
                'MaxPool1x1': 'max_pool_3x3',
            }
            edge_op_dict = {
                (i, j): ops_to_nb301[cell.edges[i, j]['op'].get_op_name] for i, j in cell.edges
            }
            op_edge_list = [
                (edge_op_dict[(i, j)], i-1) for i, j in sorted(edge_op_dict, key=lambda x: x[1]) if j < 7
            ]
            return op_edge_list

        normal_cell = self.nodes[5]['subgraph']
        reduction_cell = self.nodes[9]['subgraph']

        logger.info("Until nasbench 301 is published as a pypi package please use these strings to query the result.")
        logger.info("normal={}".format(convert(normal_cell)))
        logger.info("reduce={}".format(convert(reduction_cell)))


def _set_cell_ops(current_edge_data, C, stride):
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
    if current_edge_data.has('final') and current_edge_data.final:
        return current_edge_data
    else:
        C_in = C if stride==1 else C//2
        current_edge_data.set('op', [
            ops.Identity() if stride==1 else FactorizedReduce(C_in, C, affine=False),
            ops.Zero(stride=stride),
            ops.MaxPool1x1(3, stride, C_in, C, affine=False),
            ops.AvgPool1x1(3, stride, C_in, C, affine=False),
            ops.SepConv(C_in, C, kernel_size=3, stride=stride, padding=1, affine=False),
            ops.SepConv(C_in, C, kernel_size=5, stride=stride, padding=2, affine=False),
            ops.DilConv(C_in, C, kernel_size=3, stride=stride, padding=2, dilation=2, affine=False),
            ops.DilConv(C_in, C, kernel_size=5, stride=stride, padding=4, dilation=2, affine=False),
        ])
    return current_edge_data


def _truncate_input_edges(node, in_edges, out_edges):
    """
    Removes input edges if there are more than k.
    """
    k = 2
    if len(in_edges) >= k:
        if any(e.has('alpha') or (e.has('final') and e.final) for _, e in in_edges):
            # We are in the one-shot case
            for _, data in in_edges:
                if data.has('final') and data.final:
                    return  # We are looking at an out node
                data.alpha[1] = -float("Inf")   # Zero op should never be max alpha
            sorted_edge_ids = sorted(in_edges, key=lambda x: max(x[1].alpha), reverse=True)
            keep_edges, _ = zip(*sorted_edge_ids[:k])
            for edge_id, edge_data in in_edges:
                if edge_id not in keep_edges:
                    edge_data.delete()
        else:
            # We are in the discrete case (e.g. random search)
            for _, data in in_edges:
                assert isinstance(data.op, list)
                data.op.pop(1)      # Remove the zero op
            if any(e.has('final') and e.final for _, e in in_edges):
                return  # TODO: how about mixed final and non-final?
            else:
                for _ in range(len(in_edges) - k):
                    in_edges[random.randint(0, len(in_edges)-1)][1].delete()


def _double_channels(current_edge_data):
    if current_edge_data.has('final') and current_edge_data.final:
        return current_edge_data
    else:
        init_params = current_edge_data.op.init_params
        if 'C_in' in init_params:
            init_params['C_in'] *= 2 
        if 'C_out' in init_params:
            init_params['C_out'] *= 2
        if 'affine' in init_params:
            init_params['affine'] = True
        current_edge_data.set('op', current_edge_data.op.__class__(**init_params))
    return current_edge_data


def channel_concat(tensors):
    return torch.cat(tensors, dim=1)


def channel_maps(reduction_cell_indices, max_index):
    # calculate the mapping from edge indices to the respective channel

    assert len(reduction_cell_indices) == 2
    r_1, r_2 = reduction_cell_indices
    channel_map_from = {}
    channel_map_from.update({i: 0 for i in range(2, r_1)})
    channel_map_from.update({i: 1 for i in range(r_1, r_2)})
    channel_map_from.update({i: 2 for i in range(r_2, max_index)})

    channel_map_to = {}
    channel_map_to.update({i: 0 for i in range(3, r_1+1)})
    channel_map_to.update({i: 1 for i in range(r_1+1, r_2+1)})
    channel_map_to.update({i: 2 for i in range(r_2+1, max_index)})

    return channel_map_from, channel_map_to
