import torch.nn as nn

import naslib.search_spaces.core.primitives as ops
from naslib.search_spaces.core.graph import Graph, EdgeData
from naslib.search_spaces.core.primitives import AbstractPrimitive


def _set_cell_ops(current_edge_data, C):
    current_edge_data.set('op', [
        ops.Identity(),
        ops.Zero(stride=1),
        ReLUConvBN(C, C, kernel_size=3),
        ReLUConvBN(C, C, kernel_size=1),
        ops.AvgPool1x1(kernel_size=3, stride=1),
    ])
    return current_edge_data


class NasBench201SeachSpace(Graph):
    """
    A simplified version of the DARTS cell search space for playing around.
    """

    OPTIMIZER_SCOPE = [
        "stage_1",
        "stage_2",  
        "stage_3",
    ]

    def __init__(self):
        super().__init__()
        
        #
        # Cell definition
        #
        cell = Graph()
        cell.name = "cell"    # Use the same name for all cells with shared attributes

        # Input node
        cell.add_node(1)

        # Intermediate nodes
        cell.add_node(2)
        cell.add_node(3)

        # Output node
        cell.add_node(4)

        # Edges
        cell.add_edges_from([(1, i) for i in range(2, 4)])
        cell.add_edges_from([(2, i) for i in range(3, 4)])
        cell.add_edges_from([(3, 4)])

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
        self.add_nodes_from(range(1, total_num_nodes+1))
        self.add_edges_from([(i, i+1) for i in range(1, total_num_nodes)])

        channels = [16, 32, 64]

        #
        # operations at the edges
        #

        # preprocessing
        self.edges[1, 2].set('op', ops.Stem(channels[0]))
        
        # stage 1
        for i in range(2, 7):
            self.edges[i, i+1].set('op', cell.copy().set_scope('stage_1'))
        
        # stage 2
        self.edges[7, 8].set('op', ResNetBasicblock(C_in=channels[0], C_out=channels[1], stride=2))
        for i in range(8, 13):
            self.edges[i, i+1].set('op', cell.copy().set_scope('stage_2'))

        # stage 3
        self.edges[13, 14].set('op', ResNetBasicblock(C_in=channels[1], C_out=channels[2], stride=2))
        for i in range(14, 19):
            self.edges[i, i+1].set('op', cell.copy().set_scope('stage_3'))

        # post-processing
        self.edges[19, 20].set('op', ops.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels[-1], 10)
        ))
        
        # set the ops at the cells (channel dependent)
        for c, scope in zip(channels, self.OPTIMIZER_SCOPE):
            self.update_edges(
                update_func=lambda current_edge_data: _set_cell_ops(current_edge_data, C=c),
                scope=scope,
                private_edge_data=True
            )



"""
Code below from NASBench-201 and slighly adapted
@inproceedings{dong2020nasbench201,
  title     = {NAS-Bench-201: Extending the Scope of Reproducible Neural Architecture Search},
  author    = {Dong, Xuanyi and Yang, Yi},
  booktitle = {International Conference on Learning Representations (ICLR)},
  url       = {https://openreview.net/forum?id=HJxyZkBKDr},
  year      = {2020}
}
"""
class ReLUConvBN(AbstractPrimitive):

    def __init__(self, C_in, C, kernel_size, stride=1, affine=False):
        super(ReLUConvBN, self).__init__()
        pad = 0 if stride == 1 and kernel_size == 1 else 1
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C, kernel_size, stride=stride, padding=pad, bias=False),
            nn.BatchNorm2d(C, affine=affine)
        )


    def forward(self, x, edge_data):
        return self.op(x)
    

    def get_embedded_ops(self):
        return None


class ResNetBasicblock(AbstractPrimitive):

    def __init__(self, C_in, C_out, stride, affine=True):
        super(ResNetBasicblock, self).__init__()
        assert stride == 1 or stride == 2, 'invalid stride {:}'.format(stride)
        self.conv_a = ReLUConvBN(C_in, C_out, 3, stride)
        self.conv_b = ReLUConvBN(C_out, C_out, 3)
        if stride == 2:
            self.downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
                nn.Conv2d(C_in, C_out, kernel_size=1, stride=1, padding=0, bias=False))
        #elif C_in != C_out: # why do we need this?
        #    self.downsample = ReLUConvBN(C_in, C_out, 1, 1, 0, 1, affine)
        else:
            self.downsample = None


    def forward(self, x, edge_data):
        basicblock = self.conv_a(x, None)
        basicblock = self.conv_b(basicblock, None)
        residual = self.downsample(x) if self.downsample is not None else x
        return residual + basicblock
    
    def get_embedded_ops(self):
        return None