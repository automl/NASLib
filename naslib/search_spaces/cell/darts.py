import random
import numpy as np
import naslib.search_spaces.core.primitives as ops

from torch import nn
from copy import deepcopy

from naslib.search_spaces.core.graph import Graph, EdgeData


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
            ops.Identity() if stride==1 else ops.FactorizedReduce(C_in, C),    # TODO: what is this and why is it not in the paper?
            ops.Zero(stride=stride),
            ops.MaxPool1x1(3, stride, C_in, C),
            ops.AvgPool1x1(3, stride, C_in, C),
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
            
    else:
        print("no update for node")


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

    def __init__(self):
        """
        Initialize a new instance of the DARTS search space.
        """
        super().__init__()
        
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
        normal_cell.add_edges_from([(i, 7, EdgeData({'final': True})) for i in range(3, 7)])   # output
        
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
        # Operations at the edges
        #
        channels = [16, 32, 64]

        # pre-processing
        self.edges[1, 2].set('op', ops.Stem(channels[0]))

        # Replace Identity for normal cells after reductions cells to handle resolution
        self.edges[4, 6].set('op', ops.FactorizedReduce(channels[0], channels[1]))
        self.edges[7, 9].set('op', ops.FactorizedReduce(channels[1], channels[2]))

        # normal cells
        stages = ["n_stage_1", "n_stage_2", "n_stage_3"]

        for scope, c in zip(stages, channels):
            self.update_edges(
                update_func=lambda current_edge_data: _set_cell_ops(current_edge_data, c, stride=1),
                scope=scope,
                private_edge_data=True
            )

        # reduction cells
        # stride=2 is only for some edges, that's why we have to do it this way
        r_cell_nodes = [5, 8]
        for n, c in zip(r_cell_nodes, channels[1:]):
            reduction_cell = self.nodes[n]['subgraph']
            for u, v, data in reduction_cell.edges.data():
                stride = 2 if u in (1, 2) else 1
                reduction_cell.edges[u, v].update(_set_cell_ops(data, c, stride))
        
        # post-processing
        self.edges[10, 11].set('op', ops.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels[-1], 10))
        )

        #
        # Combining operations
        #
        for n, c in zip(range(3, 11), [16, 16, 32, 32, 32, 64, 64, 64]):
            self.nodes[n]['subgraph'].nodes[7]['comb_op'] = ops.Concat1x1(num_in_edges=4, channels=c)


    def prepare_discretization(self):
        """
        In DARTS a node can have a maximum of two incoming edges.
        This is handled here.
        """
        
        self.update_nodes(_truncate_input_edges, scope=self.OPTIMIZER_SCOPE, single_instances=True)