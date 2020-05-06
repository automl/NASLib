import torch
from torch import nn

from naslib.optimizers.optimizer import DARTSOptimizer
from naslib.search_spaces.core import EdgeOpGraph, NodeOpGraph
from naslib.search_spaces.core.primitives import FactorizedReduce, ReLUConvBN, Stem, Identity
from naslib.utils import config_parser


class Cell(EdgeOpGraph):
    def __init__(self, primitives, cell_type, C_prev_prev, C_prev, C, reduction_prev, *args, **kwargs):
        self.primitives = primitives
        self.cell_type = cell_type
        self.C_prev_prev = C_prev_prev
        self.C_prev = C_prev
        self.C = C
        self.reduction_prev = reduction_prev
        super(Cell, self).__init__(*args, **kwargs)

    def _build_graph(self):
        # Input Nodes: Previous / Previous-Previous cell
        preprocessing0 = FactorizedReduce(self.C_prev_prev, self.C, affine=False) \
            if self.reduction_prev else ReLUConvBN(self.C_prev_prev, self.C, 1, 1, 0, affine=False)
        preprocessing1 = ReLUConvBN(self.C_prev, self.C, 1, 1, 0, affine=False)

        self.add_node(0, type='input', preprocessing=preprocessing0, desc='previous-previous')
        self.add_node(1, type='input', preprocessing=preprocessing1, desc='previous')

        # 4 intermediate nodes
        self.add_node(2, type='inter', comb_op='sum')
        self.add_node(3, type='inter', comb_op='sum')
        self.add_node(4, type='inter', comb_op='sum')
        self.add_node(5, type='inter', comb_op='sum')

        # Output node
        self.add_node(6, type='output', comb_op='cat_channels')

        # Edges: input-inter and inter-inter
        for to_node in self.inter_nodes():
            for from_node in range(to_node):
                stride = 2 if self.cell_type == 'reduction' and from_node < 2 else 1
                self.add_edge(
                    from_node, to_node, op=None, op_choices=self.primitives,
                    op_kwargs={'C': self.C, 'stride': stride, 'out_node_op':
                               'sum'},
                    to_node=to_node, from_node=from_node)

        # Edges: inter-output
        self.add_edge(2, 6, op=Identity())
        self.add_edge(3, 6, op=Identity())
        self.add_edge(4, 6, op=Identity())
        self.add_edge(5, 6, op=Identity())


class MacroGraph(NodeOpGraph):
    def __init__(self, config, primitives, *args, **kwargs):
        self.config = config
        self.primitives = primitives
        super(MacroGraph, self).__init__(*args, **kwargs)

    def _build_graph(self):
        num_layers = self.config['layers']
        C = self.config['init_channels']
        C_curr = self.config['stem_multiplier'] * C

        stem = Stem(C_curr=C_curr)
        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C

        #TODO: set the input edges to the first cell in a nicer way
        self.add_node(0, type='input')
        self.add_node(1, op=stem, type='stem')
        self.add_node('1b', op=stem, type='stem')

        # Normal and reduction cells
        reduction_prev = False
        for cell_num in range(num_layers):
            if cell_num in [num_layers // 3, 2 * num_layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False

            self.add_node(cell_num + 2,
                          op=Cell(primitives=self.primitives,
                                  C_prev_prev=C_prev_prev,
                                  C_prev=C_prev, C=C_curr,
                                  reduction_prev=reduction_prev,
                                  cell_type='reduction' if
                                  reduction else 'normal'),
                          type='reduction' if reduction else 'normal')
            reduction_prev = reduction
            C_prev_prev, C_prev = C_prev, self.config['channel_multiplier'] * C_curr

        pooling = nn.AdaptiveAvgPool2d(1)
        classifier = nn.Linear(C_prev, self.config['num_classes'])

        self.add_node(num_layers + 2, op=pooling,
                      transform=lambda x: x[0], type='pooling')
        self.add_node(num_layers + 3, op=classifier,
                      transform=lambda x: x[0].view(x[0].size(0), -1), type='output')

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
    from naslib.search_spaces.darts import PRIMITIVES

    one_shot_optimizer = DARTSOptimizer()
    search_space = MacroGraph.from_optimizer_op(
        one_shot_optimizer,
        config=config_parser('../../configs/default.yaml'),
        primitives=PRIMITIVES
    )

    # Attempt forward pass
    res = search_space(torch.randn(size=[1, 3, 32, 32], dtype=torch.float, requires_grad=False))
