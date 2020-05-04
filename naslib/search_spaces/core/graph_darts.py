from functools import partial

import torch
import yaml
from torch import nn

from naslib.search_spaces.core.graphs import EdgeOpGraph, NodeOpGraph
from naslib.search_spaces.core.primitives import FactorizedReduce, ReLUConvBN, Identity, Stem
from naslib.utils import AttrDict

PRIMITIVES = [
    # 'none',
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
        preprocessing0 = FactorizedReduce(self.C_prev_prev, self.C, affine=False) \
            if self.reduction_prev else ReLUConvBN(self.C_prev_prev, self.C, 1, 1, 0, affine=False)
        preprocessing1 = ReLUConvBN(self.C_prev, self.C, 1, 1, 0, affine=False)

        self.add_node(0, type='input', preprocessing=preprocessing0, desc='previous-previous')
        self.add_node(1, type='input', preprocessing=preprocessing1, desc='previous')

        # 4 intermediate nodes
        self.add_node(2, type='inter', comb_op=sum)
        self.add_node(3, type='inter', comb_op=sum)
        self.add_node(4, type='inter', comb_op=sum)
        self.add_node(5, type='inter', comb_op=sum)

        # Output node
        self.add_node(6, type='output', comb_op=partial(torch.cat, dim=1))

        # Edges: input-inter and inter-inter
        for to_node in self.inter_nodes():
            for from_node in range(to_node):
                stride = 2 if self.graph['type'] == 'reduction' and from_node < 2 else 1
                self.add_edge(
                    from_node, to_node, op=None, op_choices=PRIMITIVES, op_kwargs={'C': self.C, 'stride': stride})

        # Edges: inter-output
        self.add_edge(2, 6, op=Identity())
        self.add_edge(3, 6, op=Identity())
        self.add_edge(4, 6, op=Identity())
        self.add_edge(5, 6, op=Identity())


class DARTSMacroGraph(NodeOpGraph):
    def __init__(self, config, *args, **kwargs):
        self.config = config
        super().__init__(*args, **kwargs)

    def _build_graph(self):
        num_layers = self.config['layers']
        C = self.config['init_channels']
        C_curr = self.config['stem_multiplier'] * C

        stem = Stem(C_curr=C_curr)
        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C

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
                          op=DARTSCell(C_prev_prev=C_prev_prev, C_prev=C_prev, C=C_curr, reduction_prev=reduction_prev,
                                       type='reduction' if reduction else 'normal'))
            reduction_prev = reduction
            C_prev_prev, C_prev = C_prev, self.config['channel_multiplier'] * C_curr

        pooling = nn.AdaptiveAvgPool2d(1)
        classifier = nn.Linear(C_prev, self.config['num_classes'])

        self.add_node(num_layers + 2, op=lambda x: pooling(x[0]), type='pooling')
        self.add_node(num_layers + 3, op=lambda x: classifier(x[0].view(x[0].size(0), -1)), type='classification')

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


'''
class Optimizer:
    def __init__(self):
        self.architectural_weights = []

    def replace_function(self, graph_obj):
        if graph_obj == 'edge':
            if graph_obj.contains('op_choices'):
                # create Categorical op (PRIMITIVES, args)
                return mixed_op
            # Replace the op
            self.architectural_weights.append(3)
            modified_edge = None
            return modified_edge
        elif graph_obj == 'node':
            pass
        else:
            raise NotImplementedError()


class SearchSpace:
    def __init__(self):
        pass

    def parse(self, optimizer):
        for graph_obj in ['edge', 'node']:
            graph_obj['op'] = optimizer.replace_function(graph_obj)
'''

if __name__ == '__main__':
    # one_shot_optimizer = Optimizer()
    # search_space = SearchSpace()
    # search_space.parse(one_shot_optimizer)
    # oneshot search space

    with open('../../configs/default.yaml') as f:
        config = yaml.safe_load(f)
        config = AttrDict(config)

    graph = DARTSMacroGraph(config=config)
    res = graph(torch.randn(size=[1, 3, 32, 32], dtype=torch.float, requires_grad=False))
    graph = DARTSCell(C_prev_prev=4, C_prev=4, C=8, reduction_prev=False, type='reduction')
    res = graph([torch.zeros(size=[1, 4, 28, 28], dtype=torch.float, requires_grad=False) for _ in range(2)])
    pass
