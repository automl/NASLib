import torch.nn as nn

from naslib.search_spaces.core import primitives as ops

from naslib.search_spaces.core.graph import Graph, EdgeData
from .primitives import ConvBNReLU, DepthwiseConv



def _set_cell_ops(current_edge_data, C, stride):
    if current_edge_data.has('final') and current_edge_data.final:
        return current_edge_data
    elif isinstance(current_edge_data.op, list) and all(isinstance(op, Graph) for op in current_edge_data.op):
        return current_edge_data    # We are at the edge of an motif
    elif isinstance(current_edge_data.op, ops.Identity):
        current_edge_data.set('op', [
            ops.Identity() if stride==1 else ops.FactorizedReduce(C, C),
            ops.Zero(stride=stride),
            ops.MaxPool1x1(3, stride),
            ops.AvgPool1x1(3, stride),
            ops.SepConv(C, C, kernel_size=3, stride=stride, padding=1, affine=False),
            DepthwiseConv(C, C, kernel_size=3, stride=stride, padding=1, affine=False),
            ConvBNReLU(C, C, kernel_size=1),
        ])
        return current_edge_data
    else:
        raise ValueError()


def _set_motifs(current_edge_data, ops):
    if current_edge_data.has('final') and current_edge_data.final:
        return current_edge_data
    else:
        # We have to set the op as shared because the archparameters
        # of lower level motivs are shared across cells.
        #current_edge_data.remove('op')
        current_edge_data.set('op', [m.copy() for m in ops])
    return current_edge_data


class SmallHierarchicalSearchSpace(Graph):
    """
    Hierarchical search space as defined in

        Liu et al.: Hierarchical Representations for Efficient Architecture Search
    """

    OPTIMIZER_SCOPE = [
        "stage_1",
        "stage_2",
        "stage_3",
    ]

    def __init__(self):
        super().__init__()

        # Define the motifs (6 level-2 motifs)
        level2_motifs = []
        for j in range(2):
            motif = Graph()
            motif.name = "motif{}".format(j)
            # motif.add_nodes_from([i for i in range(1, 5)])
            # motif.add_edges_from([(i, i+1) for i in range(1, 4)])
            # motif.add_edges_from([(i, i+2) for i in range(1, 3)])
            # motif.add_edge(1, 4)
            motif.add_edge(1, 2)
            motif.add_edge(2, 3)

            level2_motifs.append(motif)
        
        # cell (= one level-3 motif)
        cell = Graph()
        cell.name = "cell"
        # cell.add_nodes_from([i for i in range(1, 6)])
        # cell.add_edges_from([(i, i+1) for i in range(1, 5)])
        # cell.add_edges_from([(i, i+2) for i in range(1, 4)])
        # cell.add_edges_from([(i, i+3) for i in range(1, 3)])
        # cell.add_edge(1, 5)
        cell.add_edge(1, 2)
        cell.add_edge(2, 3)

        cells = []
        channels = [16, 32, 64]
        for scope, c in zip(SmallHierarchicalSearchSpace.OPTIMIZER_SCOPE, channels):
            cell_i = cell.copy().set_scope(scope)

            cell_i.update_edges(
                update_func=lambda current_edge_data: _set_motifs(current_edge_data, ops=level2_motifs),
                private_edge_data=True
            )

            cell_i.set_scope(scope)

            cell_i.update_edges(
                update_func=lambda current_edge_data: _set_cell_ops(current_edge_data, c, stride=1),
                scope=[scope],
                private_edge_data=True
            )
            cells.append(cell_i)


        self.name = "makrograph"

        self.add_nodes_from([i for i in range(1, 9)])
        self.add_edges_from([(i, i+1) for i in range(1, 8)])

        self.edges[1, 2].set('op', ops.Stem(16))
        self.edges[2, 3].set('op', cells[0])
        self.edges[3, 4].set('op', ops.SepConv(16, 32, kernel_size=3, stride=2, padding=1))
        self.edges[4, 5].set('op', cells[1])
        self.edges[5, 6].set('op', ops.SepConv(32, 64, kernel_size=3, stride=2, padding=1))
        self.edges[6, 7].set('op', cells[2])
        self.edges[7, 8].set('op', ops.Sequential(
            ops.SepConv(64, 64, kernel_size=3, stride=1, padding=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels[-1], 10))
        )
        
        
        