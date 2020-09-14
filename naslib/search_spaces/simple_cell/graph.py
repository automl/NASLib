import numpy as np
import networkx as nx
from torch import nn
from copy import deepcopy

from naslib.search_spaces.core.graph import Graph, EdgeData
from naslib.search_spaces.core import primitives as ops

from ..darts.graph import _truncate_input_edges
from ..darts.primitives import FactorizedReduce


def _set_cell_ops(current_edge_data, C, stride):
    if current_edge_data.has('final') and current_edge_data.final:
        return current_edge_data
    else:
        C_in = C if stride==1 else C//2
        current_edge_data.set('op', [
            ops.Identity() if stride==1 else FactorizedReduce(C_in, C),    # TODO: what is this and why is it not in the paper?
            ops.Zero(stride=stride),
            ops.MaxPool1x1(3, stride, C_in, C),
            ops.SepConv(C_in, C, kernel_size=3, stride=stride, padding=1, affine=False),
            ops.DilConv(C_in, C, kernel_size=3, stride=stride, padding=2, dilation=2, affine=False),
        ])
    return current_edge_data


class SimpleCellSearchSpace(Graph):
    """
    A simplified version of the DARTS cell search space for playing around.
    """

    OPTIMIZER_SCOPE = [
        "n_stage_1",
        "n_stage_2",  
        "r_stage_1",
    ]

    def __init__(self):
        super().__init__()
        
        #
        # Cell definition
        #
        normal_cell = Graph()
        normal_cell.name = "normal_cell"    # Use the same name for all cells with shared attributes

        # Input nodes
        normal_cell.add_node(1)
        normal_cell.add_node(2)

        # Intermediate nodes
        normal_cell.add_node(3)
        normal_cell.add_node(4)

        # Output node
        normal_cell.add_node(5)
        
        # Edges
        normal_cell.add_edges_from([(1, i) for i in range(3, 5)])   # input 1
        normal_cell.add_edges_from([(2, i) for i in range(3, 5)])   # input 2
        normal_cell.add_edges_from([(3, 4)])
        normal_cell.add_edges_from([(i, 5, EdgeData({'final': True})) for i in range(3, 5)])   # output
        

        reduction_cell = deepcopy(normal_cell)
        reduction_cell.name = "reduction_cell"

        #
        # Makrograph definition
        #
        self.name = "makrograph"

        self.add_node(1)    # input node
        self.add_node(2)    # preprocessing
        self.add_node(3, subgraph=normal_cell.set_scope("n_stage_1").set_input([2, 2]))
        self.add_node(4, subgraph=reduction_cell.set_scope("r_stage_1").set_input([2, 3]))
        self.add_node(5, subgraph=normal_cell.copy().set_scope("n_stage_2").set_input([4, 4]))
        self.add_node(6)    # output

        self.add_edge(1, 2)     # pre-processing (stem)
        self.add_edges_from([(2, 3), (2, 4), (3, 4)])   # first stage
        self.add_edges_from([(4, 5)])   # second stage
        self.add_edge(5, 6)   # post-processing (pooling, classifier)

        #
        # Operations at the edges
        #

        # pre-processing
        self.edges[1, 2].set('op', ops.Stem(16))

        # normal cells
        channels = [16, 32]
        stages = ["n_stage_1", "n_stage_2"]

        for scope, c in zip(stages, channels):
            self.update_edges(
                update_func=lambda current_edge_data: _set_cell_ops(current_edge_data, c, stride=1),
                scope=scope,
                private_edge_data=True
            )

        # reduction cells
        nodes = [4]
        for n, c in zip(nodes, channels[1:]):
            reduction_cell = self.nodes[n]['subgraph']
            for u, v, data in reduction_cell.edges.data():
                stride = 2 if u in (1, 2) else 1
                reduction_cell.edges[u, v].update(_set_cell_ops(data, c, stride))
        
        # post-processing
        self.edges[5, 6].set('op', ops.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels[-1], 10))
        )

        #
        # Combining operations
        #
        self.nodes[3]['subgraph'].nodes[5]['comb_op'] = ops.Concat1x1(num_in_edges=2, C_out=16)
        self.nodes[4]['subgraph'].nodes[5]['comb_op'] = ops.Concat1x1(num_in_edges=2, C_out=32)
        self.nodes[5]['subgraph'].nodes[5]['comb_op'] = ops.Concat1x1(num_in_edges=2, C_out=32)


    def prepare_discretization(self):
        self.update_nodes(_truncate_input_edges, scope=self.OPTIMIZER_SCOPE, single_instances=True)