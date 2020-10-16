import random
import torch
import logging
import numpy as np
import networkx as nx
from naslib.search_spaces.core import primitives as ops

from torch import nn
from copy import deepcopy

from naslib.utils.utils import get_project_root, AttrDict
from naslib.search_spaces.core.graph import Graph, EdgeData
from .primitives import FactorizedReduce

logger = logging.getLogger(__name__)


edge_attributes = {
    'op': [
            FactorizedReduce,   # classes of op, not instances
            ops.Zero, 
            ops.MaxPool,
            ops.AvgPool, 
            ops.SepConv, 
            ops.SepConv, 
            ops.DilConv,
            ops.DilConv,
        ],
    'stride': 1,
    'kernel_size': [None, None, 3, 3, 3, 5, 3, 5],      # if different for each op specify full list
    'padding': [None, None, None, None, 1, 2, 2, 4],
    'dilation': 2,
    'affine': False
}


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
        "stage_1",
        "stage_2", 
        "stage_3", 
    ]

    QUERYABLE = False

    def __init__(self, classes: int = 10, channels: list = [16, 32, 64]):
        """
        Initialize a new instance of the DARTS search space.

        Note:
            __init__ cannot take any parameters due to the way networkx is implemented.
            If we want to change the number of classes set a static attribute `NUM_CLASSES`
            before initializing the class. Default is 10 as for cifar-10.
        """
        assert len(channels) == len(self.OPTIMIZER_SCOPE), \
            "Expecting a channel for each scope. Expected {}, got {}.".format(len(self.OPTIMIZER_SCOPE), len(channels))
        super().__init__()

        self.num_classes = classes

        # Normal cell first
        normal_cell = Graph(name="normal_cell") # Use the same name for all cells with shared attributes
        normal_cell.add_nodes_from([1, 2, 3, 4, 5, 6, 7])
        
        # Edges
        normal_cell.add_edges_from([(1, i) for i in range(3, 7)])   # input 1
        normal_cell.add_edges_from([(2, i) for i in range(3, 7)])   # input 2
        normal_cell.add_edges_from([(3, 4), (3, 5), (3, 6)])
        normal_cell.add_edges_from([(4, 5), (4, 6)])
        normal_cell.add_edges_from([(5, 6)])

        # Edges connecting to the output are always the identity
        normal_cell.add_edges_from([(i, 7, EdgeData().finalize()) for i in range(3, 7)])   # output
        
         # set the parameters for the ops at all edges (that are not final)
        for k, v in edge_attributes.items():
            normal_cell.set_at_edges(k, v)

        normal_cell.nodes[7]['comb_op'] = channel_concat


        # Reduction cell has the same topology
        reduction_cell = normal_cell.clone()
        reduction_cell.name = "reduction_cell"
        reduction_cell.update_edges(
            lambda edge: edge.data.set('stride', 2) if edge.head in [1, 2] else None
        )

        
        # Macrograph definition
        self.build_macrograph(channels, cells=(normal_cell, reduction_cell))

        def set_channels(edge, C):
            edge.data.set('C_in', C)
            edge.data.set('C_out', C)

        for scope, c in zip(self.OPTIMIZER_SCOPE, channels):
            self.update_edges(lambda edge: set_channels(edge, c), scope, private_edge_data=True)


        self.compile()


    def build_macrograph(self, channels, cells, cells_per_stage=2, affine=False, auxiliary=False):

        normal, reduction = cells
        
        stem_multiplier = 3
        num_intermediate_nodes = 4

        self.name = "makrograph"
        self.add_node(1)    # input node
        self.add_node(2)    # preprocessing
        self.add_edge(1, 2, op=ops.Stem(channels[0] * stem_multiplier))     # pre-processing (stem)

        j = 3   # index of next node to add
        for scope, c in zip(self.OPTIMIZER_SCOPE, channels):
            
            # reduction cell (beginning of each stage but first)
            if j > 3:
                input = [j-2, j-1]
                edge_prev = (input[1], j)
                edge_prev_prev = (input[0], j)
                
                self.add_node(j, subgraph=reduction.copy().set_scope(scope).set_input(input))
                self.add_edges_from([edge_prev, edge_prev_prev])

                # with preprocessing we fix the number of input channels for each cell
                C_in = c_prev * num_intermediate_nodes
                self.edges[edge_prev].set('op', ops.ReLUConvBN(C_in, C_out=c, kernel_size=1, affine=affine))
                self.edges[edge_prev_prev].set('op', ops.ReLUConvBN(C_in, C_out=c, kernel_size=1, affine=affine))

                j += 1

            # normal cells
            for i in range(cells_per_stage):
                # single (copied) input if first cell after stem
                input = [j-1, j-1] if j == 3 else [j-2, j-1]
                edge_prev = (input[1], j)
                edge_prev_prev = (input[0], j)

                self.add_node(j, subgraph=normal.copy().set_scope(scope).set_input(input))
                self.add_edges_from([edge_prev, edge_prev_prev])

                # with preprocessing we fix the number of input channels for each cell
                C_in = c * stem_multiplier if j == 3 else c * num_intermediate_nodes
                self.edges[edge_prev].set('op', ops.ReLUConvBN(C_in, C_out=c, kernel_size=1, affine=affine))
                if i==0 and j>3:
                    # we have a connection that jumps over a reduction cell
                    self.edges[edge_prev_prev].set('op', FactorizedReduce(C_in//2, C_out=c, affine=affine)) 
                else:
                    if j <= 4:
                        C_in = c * stem_multiplier
                    self.edges[edge_prev_prev].set('op', ops.ReLUConvBN(C_in, C_out=c, kernel_size=1, affine=affine))
                    
                j += 1
            c_prev = c
        
        if auxiliary:
            # Auxiliary, taken from DARTS implementation
            # hardcoded, assuming input size 8x8
            self.add_node(j)
            self.add_edge(j-1, j, op=ops.Sequential(
                nn.ReLU(inplace=True),
                nn.AvgPool2d(5, stride=3, padding=0, count_include_pad=False), # image size = 2 x 2
                nn.Conv2d(channels[-1] * num_intermediate_nodes, 128, 1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 768, 2, bias=False),
                nn.BatchNorm2d(768),
                nn.ReLU(inplace=True),
                nn.Flatten(),
                nn.Linear(768, self.num_classes)
            ))
            j += 1
            self.add_node(j)    # output    
            self.add_edge(j-2, j, op=ops.Sequential(    # post-processing (pooling, classifier)
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(channels[-1] * num_intermediate_nodes, self.num_classes),
            ))
        else:
            self.add_node(j)    # output    
            self.add_edge(j-1, j, op=ops.Sequential(    # post-processing (pooling, classifier)
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(channels[-1] * num_intermediate_nodes, self.num_classes),
            ))


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

        normal, reduction = self._get_child_graphs(True)
        self.clear()

        # We need to reset the initialized ops because the channels will change
        def reset_op(edge):
            op_attr = edge.data.op.init_params
            edge.data.set('op', edge.data.op.__class__)
            edge.data.update(op_attr)
        
        normal.update_edges(reset_op)
        reduction.update_edges(reset_op)

        channels = [36, 72, 144]
        
        self.build_macrograph(channels, (normal, reduction), cells_per_stage=6, affine=True, auxiliary=True)

        def set_channels(edge, C):
            edge.data.set('C_in', C)
            edge.data.set('C_out', C)
            edge.data.set('affine', True)

        for scope, c in zip(self.OPTIMIZER_SCOPE, channels):
            self.update_edges(lambda edge: set_channels(edge, c), scope, private_edge_data=True)

        self.compile()
        

    def auxilary_logits(self):
        """
        Return the axiliary logits as a additional function which can or cannot be
        used during evaluation.
        """
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
                'AvgPool': 'avg_pool_3x3',
                'MaxPool': 'max_pool_3x3',
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
        return "normal={} | reduction={}".format(convert(normal_cell), convert(reduction_cell))


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


def channel_concat(tensors):
    return torch.cat(tensors, dim=1)
