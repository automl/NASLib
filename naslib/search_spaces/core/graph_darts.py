import numpy as np
import torch
import yaml
from torch import nn

from naslib.utils import AttrDict
from naslib.search_spaces.core.graphs import EdgeOpGraph, NodeOpGraph
from naslib.search_spaces.core.operations import CategoricalOp
from naslib.search_spaces.core.primitives import FactorizedReduce, ReLUConvBN, Identity

PRIMITIVES = [
    'none',
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5'
]


class DARTSCell(EdgeOpGraph):
    def __init__(self, C_prev_prev, C_prev, C, reduction_prev, *args, **kwargs):
        self.C_prev_prev = C_prev_prev
        self.C_prev = C_prev
        self.C = C
        self.reduction_prev = reduction_prev
        super().__init__(*args, **kwargs)

    def _build_graph(self):
        # Input Nodes: Previous / Previous-Previous cell
        preprocessing0 = FactorizedReduce(
            self.C_prev_prev, self.C, affine=False
        ) if self.reduction_prev else ReLUConvBN(self.C_prev_prev, self.C, 1,
                                                 1, 0, affine=False)
        preprocessing1 = ReLUConvBN(self.C_prev, self.C, 1, 1, 0, affine=False)

        self.add_node(0, type='input', preprocessing=preprocessing0, desc='previous-previous')
        self.add_node(1, type='input', preprocessing=preprocessing1, desc='previous')

        # 4 intermediate nodes
        self.add_node(2, type='inter', comb_op=sum)
        self.add_node(3, type='inter', comb_op=sum)
        self.add_node(4, type='inter', comb_op=sum)
        self.add_node(5, type='inter', comb_op=sum)

        # Output node
        self.add_node(6, type='output', comb_op=torch.cat)

        # Edges: input-inter and inter-inter
        for to_node in self.inter_nodes():
            for from_node in range(to_node):
                stride = 2 if self.graph['type'] == 'reduction' and from_node < 2 else 1
                self.add_edge(
                    from_node, to_node, op=CategoricalOp(
                        primitives=PRIMITIVES, C=self.C, stride=stride,
                        out_node_op=self.nodes[to_node]['comb_op']
                    )
                )

        # Edges: inter-output
        self.add_edge(2, 6, op=Identity())
        self.add_edge(3, 6, op=Identity())
        self.add_edge(4, 6, op=Identity())
        self.add_edge(5, 6, op=Identity())


class DARTSMacroGraph(NodeOpGraph):
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config

    def _build_graph(self):
        num_layers = self.config['layers']
        C = self.config['init_channels']
        C_curr = self.config['stem_multiplier'] * C

        stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr)
        )
        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C

        self.add_node(0, type='input')
        self.add_node(1, op=stem, type='stem')
        self.add_node('1b', op=stem, type='stem')

        # Normal and reduction cells
        reduction_prev = False
        for cell_num in range(2, num_layers + 2):

            if cell_num == np.floor(num_layers / 3) or cell_num == np.floor(2 * num_layers / 3):
                C_curr *= 2
                self.add_node(cell_num, op=DARTSCell(type='reduction', C_prev_prev=C_prev_prev, C_prev=C_prev, C=C_curr,
                                                     reduction=True, reduction_prev=reduction_prev), type='reduction')
                reduction_prev = True
            else:
                self.add_node(cell_num, op=DARTSCell(type='normal', C_prev_prev=C_prev_prev, C_prev=C_prev, C=C_curr,
                                                     reduction=False, reduction_prev=reduction_prev), type='normal')
                reduction_prev = False

            C_prev_prev, C_prev = C_prev, self.config['channel_multiplier'] * C_curr

        self.add_node(num_layers + 2, op=nn.AdaptiveAvgPool2d(1), type='pooling')
        self.add_node(num_layers + 3, op=nn.Linear(C_prev, self.config['num_classes']), type='classification')

        # Edges
        self.add_edge(0, 1)
        self.add_edge(0, '1b')

        # Parallel edge that's why MultiDiGraph
        self.add_edge(1, 2, type='input', desc='previous-previous')
        self.add_edge('1b', 2, type='input', desc='previous')

        for i in range(3, num_layers + 2):
            self.add_edge(i - 2, i, type='input', desc='previous-previous')
            self.add_edge(i - 1, i, type='input', desc='previous')

        # From output of normal-reduction cell to pooling layer
        self.add_edge(num_layers + 1, num_layers + 2)
        self.add_edge(num_layers + 2, num_layers + 3)


if __name__ == '__main__':
    with open('../../configs/default.yaml') as f:
        config = yaml.safe_load(f)
        config = AttrDict(config)

    graph = DARTSMacroGraph(config=config)
    # graph(*torch.zeros(size=[1], dtype=torch.float, requires_grad=False))
    graph = DARTSCell(C_prev_prev=4, C_prev=4, C=8, reduction_prev=False, type='reduction')
    graph([torch.zeros(size=[1, 4, 28, 28], dtype=torch.float, requires_grad=False) for _ in range(2)])
    pass
