import numpy as np
import networkx as nx
from torch import nn
from copy import deepcopy

from naslib.search_spaces.core.graph import Graph, EdgeData
from naslib.search_spaces.core import primitives as ops

from ..nasbench301.graph import _truncate_input_edges
from ..nasbench301.primitives import FactorizedReduce


edge_attributes = {
    "op": [
        FactorizedReduce,  # classes of op, not instances
        ops.Zero1x1,
        ops.MaxPool1x1,
        ops.SepConv,
        ops.DilConv,
    ],
    "stride": 1,  # stride will be adaped later accordingly
    "kernel_size": 3,
    "padding": [None, None, None, 1, 2],  # if different for each op specify full list
    "dilation": 2
    # C_in and C_out will be specified later
}


class SimpleCellSearchSpace(Graph):
    """
    A simplified version of the DARTS cell search space for playing around.

    Differences:
    Two stages, smaller cells, no preprocessing for cells, no stem multiplier,
    input edges not cut to 2, same evaluation as search architecture.
    """

    OPTIMIZER_SCOPE = [
        "stage_1",
        "stage_2",
    ]

    def __init__(
        self,
        classes: int = 10,
        intermediate_nodes: int = 2,
        cells_per_stage: int = 1,
        channels: list = [16, 32],
    ):
        """
        Initializes the simple cell search space.

        Args:
            classes (int): Number of classes. Default: 10.
            intermediate_nodes (int): Number of intermediate nodes for normal and
                reduction cells. Default: 2.
            cells_per_stage (int): Number of normal cells at each stage. Default: 1.
            channels (list): Channels for each stage. Must have len 2. Default: [16, 32]
        """
        assert len(channels) == len(
            self.OPTIMIZER_SCOPE
        ), "Expecting a channel for each scope. Expected {}, got {}.".format(
            len(self.OPTIMIZER_SCOPE), len(channels)
        )
        super().__init__()

        # Cell definition
        normal_cell = Graph(
            name="normal_cell"
        )  # Use the same name for all cells with shared attributes

        # Nodes
        out_node_idx = intermediate_nodes + 3
        normal_cell.add_nodes_from(range(1, out_node_idx + 1))

        # Edges
        normal_cell.add_edges_from([(1, i) for i in range(3, out_node_idx)])  # input 1
        normal_cell.add_edges_from([(2, i) for i in range(3, out_node_idx)])  # input 2
        normal_cell.add_edges_from(
            [
                (u, v)
                for u, v in normal_cell.get_dense_edges()
                if u not in [1, 2] and v != out_node_idx
            ]
        )
        # Edges connecting to the output are always the identity
        normal_cell.add_edges_from(
            [(i, out_node_idx, EdgeData().finalize()) for i in range(3, out_node_idx)]
        )  # output

        # set the parameters for the ops at all edges (that are not final)
        for k, v in edge_attributes.items():
            normal_cell.set_at_edges(k, v)

        # Reduction cell
        reduction_cell = deepcopy(normal_cell)
        reduction_cell.name = "reduction_cell"

        reduction_cell.update_edges(
            update_func=lambda edge: edge.data.set("stride", 2)
            if edge.head in [1, 2]
            else None
        )

        # Makrograph definition
        self.name = "makrograph"

        self.add_node(1)  # input node
        self.add_node(2)  # preprocessing
        self.add_edge(1, 2)  # pre-processing (stem)

        j = 3  # index of next node to add
        for scope in self.OPTIMIZER_SCOPE:
            # reduction cell (beginning of each stage but first)
            if j > 3:
                input = [j - 2, j - 1]
                self.add_node(
                    j, subgraph=reduction_cell.copy().set_scope(scope).set_input(input)
                )
                self.add_edges_from([(input[0], j), (input[1], j)])
                j += 1

            # normal cells
            for _ in range(cells_per_stage):
                # single (copied) input if first node or every normal node after reduction cell
                input = (
                    [j - 1, j - 1]
                    if j == 3 or (j - 3) % (cells_per_stage + 1) == 0
                    else [j - 2, j - 1]
                )
                self.add_node(
                    j, subgraph=normal_cell.copy().set_scope(scope).set_input(input)
                )
                self.add_edges_from([(input[0], j), (input[1], j)])
                j += 1

        self.add_node(j)  # output
        self.add_edge(j - 1, j)  # post-processing (pooling, classifier)

        # Compile the ops
        self.edges[1, 2].set(
            "op", ops.Stem(channels[0])
        )  # we can also set a compiled op. Will be ignored by compile()

        def set_channels(edge, C):
            C_in = C if edge.data.stride == 1 else C // 2
            edge.data.set("C_in", C_in)
            edge.data.set("C_out", C)

        for scope, c in zip(self.OPTIMIZER_SCOPE, channels):
            self.update_edges(
                lambda edge: set_channels(edge, c), scope, private_edge_data=True
            )

        self.edges[j - 1, j].set(
            "op",
            ops.Sequential(
                nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(channels[-1], classes)
            ),
        )

        self.compile()

        # Combining operations are currently not considered by compile()
        def set_comb_op(node, in_edges, out_edges, C):
            if node[0] == out_node_idx:
                node[1]["comb_op"] = ops.Concat1x1(
                    num_in_edges=intermediate_nodes, C_out=C
                )

        for scope, c in zip(self.OPTIMIZER_SCOPE, channels):
            self.update_nodes(
                update_func=lambda node, in_edges, out_edges: set_comb_op(
                    node, in_edges, out_edges, c
                ),
                scope=scope,
                single_instances=False,
            )
