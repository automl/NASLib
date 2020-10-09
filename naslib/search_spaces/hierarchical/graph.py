import torch.nn as nn
import networkx as nx

from naslib.search_spaces.core import primitives as ops

from naslib.search_spaces.core.graph import Graph, EdgeData
from .primitives import ConvBNReLU, DepthwiseConv


class HierarchicalSearchSpace(Graph):
    """
    Hierarchical search space as defined in

        Liu et al.: Hierarchical Representations for Efficient Architecture Search
    
    The version which they search for Cifar-10.

    Note that we do not use channel-wise concat as merge operation for
    intermediate notes for simplicity. Only the output node uses channel-wise
    concat followed by a 1x1 convolution.
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
        for j in range(6):
            motif = Graph()
            motif.name = "motif{}".format(j)
            motif.add_nodes_from([i for i in range(1, 5)])
            motif.add_edges_from([(i, i+1) for i in range(1, 4)])
            motif.add_edges_from([(i, i+2) for i in range(1, 3)])
            motif.add_edge(1, 4)

            level2_motifs.append(motif)
        
        # cell (= one level-3 motif)
        cell = Graph()
        cell.name = "cell"
        cell.add_nodes_from([i for i in range(1, 6)])
        cell.add_edges_from([(i, i+1) for i in range(1, 5)])
        cell.add_edges_from([(i, i+2) for i in range(1, 4)])
        cell.add_edges_from([(i, i+3) for i in range(1, 3)])
        cell.add_edge(1, 5)

        cells = []
        channels = [16, 32, 64]
        for scope, c in zip(self.OPTIMIZER_SCOPE, channels):
            cell_i = cell.copy().set_scope(scope)

            cell_i.update_edges(
                update_func=lambda current_edge_data: _set_motifs(current_edge_data, motifs=level2_motifs, c=c),
                private_edge_data=True
            )
            

            cell_i.set_scope(scope)

            # set the level 1 motifs (i.e. primitives)
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

    # just change the channels but the same macro graph as in search.
    # this is just for some experiments and will likely be removed in the future
    # def prepare_evaluation(self):
    #     cells = [self.edges[2, 3].op, self.edges[4, 5].op, self.edges[6, 7].op]
    #     channels = [32, 64, 128]

    #     for cell, c in zip(cells, channels):
    #         for _, _, data in cell.edges.data():
    #             data.op.update_nodes(
    #                 lambda node, in_edges, out_edges: _set_comb_op_channels(node, in_edges, out_edges, c=c),
    #                 single_instances=False
    #             )

    #     self.edges[1, 2].set('op', ops.Stem(channels[0]))
    #     self.edges[2, 3].set('op', cells[0].copy())
    #     self.edges[3, 4].set('op', ops.SepConv(channels[0], channels[1], kernel_size=3, stride=2, padding=1))
    #     self.edges[4, 5].set('op', cells[1].copy())
    #     self.edges[5, 6].set('op', ops.SepConv(channels[1], channels[2], kernel_size=3, stride=2, padding=1))
    #     self.edges[6, 7].set('op', cells[2].copy())
    #     self.edges[7, 8].set('op', ops.Sequential(
    #         ops.SepConv(channels[2], channels[2], kernel_size=3, stride=1, padding=1),
    #         nn.AdaptiveAvgPool2d(1),
    #         nn.Flatten(),
    #         nn.Linear(channels[-1], 10))
    #     )

    #     self.update_edges(
    #         update_func=lambda current_edge_data: _increase_channels(current_edge_data, factor=2),
    #         scope=self.OPTIMIZER_SCOPE,
    #         private_edge_data=True
    #     )

        
    # The large makro graph is too large and contains too much parameters to be comparable
    def prepare_evaluation(self):
        """
        The evaluation model has N=2 cells at each stage and a sepconv with stride 1
        between them. Initial channels = 64, trained 512 epochs. Learning rate 0.1
        reduced by 10x after 40K, 60K, and 70K steps.
        """
        # this is called after the optimizer has discretized the graph
        cells = [self.edges[2, 3].op, self.edges[4, 5].op, self.edges[6, 7].op]

        self._expand()
        
        # channels = [64, 128, 256]
        # factor = 4
        channels = [32, 64, 128]
        factor = 2

        for cell, c in zip(cells, channels):
            for _, _, data in cell.edges.data():
                data.op.update_nodes(
                    lambda node, in_edges, out_edges: _set_comb_op_channels(node, in_edges, out_edges, c=c),
                    single_instances=False
                )

        self.edges[1, 2].set('op', ops.Stem(channels[0]))
        self.edges[2, 3].set('op', cells[0].copy())
        self.edges[3, 4].set('op', ops.SepConv(channels[0], channels[0], kernel_size=3, stride=1, padding=1))
        self.edges[4, 5].set('op', cells[0].copy())
        self.edges[5, 6].set('op', ops.SepConv(channels[0], channels[1], kernel_size=3, stride=2, padding=1))
        self.edges[6, 7].set('op', cells[1].copy())
        self.edges[7, 8].set('op', ops.SepConv(channels[1], channels[1], kernel_size=3, stride=1, padding=1))
        self.edges[8, 9].set('op', cells[1].copy())
        self.edges[9, 10].set('op', ops.SepConv(channels[1], channels[2], kernel_size=3, stride=2, padding=1))
        self.edges[10, 11].set('op', cells[2].copy())
        self.edges[11, 12].set('op', ops.SepConv(channels[2], channels[2], kernel_size=3, stride=1, padding=1))
        self.edges[12, 13].set('op', cells[2].copy())
        self.edges[13, 14].set('op', ops.Sequential(
            ops.SepConv(channels[-1], channels[-1], kernel_size=3, stride=1, padding=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels[-1], 10))
        )

        self.update_edges(
            update_func=lambda current_edge_data: _increase_channels(current_edge_data, factor),
            scope=self.OPTIMIZER_SCOPE,
            private_edge_data=True
        )


    def _expand(self):
        # shift the node indices to make space for 2 more edges at each stage
        mapping = {
            4: 6,
            5: 7,
            6: 10,
            7: 11,
            8: 14,
        }
        nx.relabel_nodes(self, mapping, copy=False)
        
        # fix edges
        self.remove_edges_from(list(self.edges()))
        self.add_edges_from([(i, i+1) for i in range(1, 14)])


def _set_comb_op_channels(node, in_edges, out_edges, c):
    index, n = node
    if index == 4:
        n['comb_op'] = ops.Concat1x1(num_in_edges=3, C_out=c)

        
def _set_cell_ops(current_edge_data, C, stride):
    """
    Set the primitives for the bottom level motif where we
    have actual ops at the edges.
    """
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


def _set_motifs(current_edge_data, motifs, c):
    """
    Set l-1 level motifs as ops at the edges for l level motifs
    """
    if current_edge_data.has('final') and current_edge_data.final:
        return current_edge_data
    else:
        op = []
        for motif in motifs:
            m = motif.copy()    # We need copies because they will be set at every edge
            m.nodes[4]['comb_op'] = ops.Concat1x1(num_in_edges=3, C_out=c)
            op.append(m)
        
        current_edge_data.set('op', op)

    return current_edge_data


def _increase_channels(current_edge_data, factor=4):
    if isinstance(current_edge_data.op, Graph):
        return current_edge_data
    else:
        init_params = current_edge_data.op.init_params
        if 'C_in' in init_params and init_params['C_in'] is not None:
            init_params['C_in'] *= factor 
        if 'C_out' in init_params and init_params['C_out'] is not None:
            init_params['C_out'] *= factor
        current_edge_data.set('op', current_edge_data.op.__class__(**init_params))
    return current_edge_data