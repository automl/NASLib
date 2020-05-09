import torch
import yaml
from torch import nn

from naslib.optimizers.optimizer import OneShotOptimizer
from naslib.search_spaces.core import EdgeOpGraph, NodeOpGraph
from naslib.search_spaces.core.primitives import Stem
from naslib.search_spaces.nasbench_201.primitives import OPS as NASBENCH_201_OPS
from naslib.search_spaces.nasbench_201.primitives import ResNetBasicblock
from naslib.search_spaces.nasbench_201.primitives import Stem as NASBENCH_201_Stem
from naslib.utils import config_parser


class Cell(EdgeOpGraph):
    def __init__(self, primitives, C_prev, C, stride, ops_dict, *args, **kwargs):
        self.primitives = primitives
        self.C_prev = C_prev
        self.C = C
        self.stride = stride
        self.ops_dict = ops_dict
        super(Cell, self).__init__(*args, **kwargs)

    def _build_graph(self):
        # 4 intermediate nodes
        self.add_node(0, type='input', desc='previous')

        self.add_node(1, type='inter', comb_op='sum')
        self.add_node(2, type='inter', comb_op='sum')

        # Output node
        self.add_node(3, type='output', comb_op='sum')

        # Edges: input-inter, inter-inter, inter-outputs
        for to_node in range(4):
            for from_node in range(to_node):
                self.add_edge(
                    from_node, to_node, op=None, op_choices=self.primitives,
                    op_kwargs={'C_in': self.C_prev, 'C_out': self.C, 'stride': self.stride, 'affine': True,
                               'track_running_stats': True, 'ops_dict': self.ops_dict, 'out_node_op': 'sum'},
                    to_node=to_node, from_node=from_node)

    @classmethod
    def from_config(cls, graph_dict, primitives, cell_type, C_prev_prev,
                    C_prev, C, reduction_prev, *args, **kwargs):
        graph = cls(primitives, cell_type, C_prev_prev, C_prev, C,
                    reduction_prev, *args, **kwargs)

        graph.clear()
        # Input Nodes: Previous / Previous-Previous cell
        for node, attr in graph_dict['nodes'].items():
            if 'preprocessing' in attr:
                if attr['preprocessing'] == 'FactorizedReduce':
                    input_args = {'C_in': graph.C_prev_prev, 'C_out': graph.C,
                                  'affine': False}
                else:
                    input_args = {'C_in': graph.C_prev_prev, 'C_out': graph.C,
                                  'kernel_size': 1, 'stride': 1, 'padding': 0,
                                  'affine': False}

                preprocessing = eval(attr['preprocessing'])(**input_args)

                graph.add_node(node, type=attr['type'],
                               preprocessing=preprocessing)
            else:
                graph.add_nodes_from([(node, attr)])

        for edge, attr in graph_dict['edges'].items():
            from_node, to_node = eval(edge)
            graph.add_edge(*eval(edge), **{k: eval(v) for k, v in attr.items() if k
                                           != 'op'})
            graph[from_node][to_node]['op'] = None if attr['op'] != 'Identity' else eval(attr['op'])()
            print(graph[from_node][to_node])

        return graph


class MacroGraph(NodeOpGraph):
    def __init__(self, config, primitives, ops_dict, *args, **kwargs):
        self.config = config
        self.primitives = primitives
        self.ops_dict = ops_dict
        super(MacroGraph, self).__init__(*args, **kwargs)

    def _build_graph(self):
        num_cells_per_stack = self.config['num_cells_per_stack']
        C = self.config['init_channels']
        layer_channels = [C] * num_cells_per_stack + [C * 2] + [C * 2] * num_cells_per_stack + [C * 4] + [
            C * 4] * num_cells_per_stack
        layer_reductions = [False] * num_cells_per_stack + [True] + [False] * num_cells_per_stack + [True] + [
            False] * num_cells_per_stack

        stem = NASBENCH_201_Stem(C=C)
        self.add_node(0, type='input')
        self.add_node(1, op=stem, type='stem')

        C_prev = C
        self.cells = nn.ModuleList()
        for cell_num, (C_curr, reduction) in enumerate(zip(layer_channels, layer_reductions)):
            if reduction:
                cell = ResNetBasicblock(C_prev, C_curr, 2, True)
                self.add_node(cell_num + 2, op=cell, primitives=self.primitives, transform=lambda x: x[0])
            else:
                cell = Cell(primitives=self.primitives, stride=1, C_prev=C_prev, C=C_curr,
                            ops_dict=self.ops_dict)
                self.add_node(cell_num + 2, op=cell, primitives=self.primitives)

            C_prev = C_curr

        lastact = nn.Sequential(nn.BatchNorm2d(C_prev), nn.ReLU(inplace=True))
        pooling = nn.AdaptiveAvgPool2d(1)
        classifier = nn.Linear(C_prev, self.config['num_classes'])

        self.add_node(cell_num + 3, op=lastact, transform=lambda x: x[0], type='postprocessing_nb201')
        self.add_node(cell_num + 4, op=pooling, transform=lambda x: x[0], type='pooling')
        self.add_node(cell_num + 5, op=classifier, transform=lambda x: x[0].view(x[0].size(0), -1),
                      type='output')

        # Edges
        for i in range(1, cell_num + 6):
            self.add_edge(i - 1, i, type='input', desc='previous')

    @classmethod
    def from_config(cls, config=None, filename=None):
        with open(filename, 'r') as f:
            graph_dict = yaml.safe_load(f)

        if config is None:
            raise ('No configuration provided')

        graph = cls(config, [])

        graph_type = graph_dict['type']
        edges = [(*eval(e), attr) for e, attr in graph_dict['edges'].items()]
        graph.clear()
        graph.add_edges_from(edges)

        C = config['init_channels']
        C_curr = config['stem_multiplier'] * C

        stem = Stem(C_curr=C_curr)
        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C

        for node, attr in graph_dict['nodes'].items():
            node_type = attr['type']
            if node_type == 'input':
                graph.add_node(node, type='input')
            elif node_type == 'stem':
                graph.add_node(node, op=stem, type='stem')
            elif node_type in ['normal', 'reduction']:
                assert attr['op']['type'] == 'Cell'
                graph.add_node(node,
                               op=Cell.from_config(attr['op'], primitives=attr['op']['primitives'],
                                                   C_prev_prev=C_prev_prev, C_prev=C_prev,
                                                   C=C_curr,
                                                   reduction_prev=graph_dict['nodes'][node - 1]['type'] == 'reduction',
                                                   cell_type=node_type),
                               type=node_type)
                C_prev_prev, C_prev = C_prev, config['channel_multiplier'] * C_curr
            elif node_type == 'pooling':
                pooling = nn.AdaptiveAvgPool2d(1)
                graph.add_node(node, op=pooling, transform=lambda x: x[0],
                               type='pooling')
            elif node_type == 'output':
                classifier = nn.Linear(C_prev, config['num_classes'])
                graph.add_node(node, op=classifier, transform=lambda x:
                x[0].view(x[0].size(0), -1), type='output')

        return graph


if __name__ == '__main__':
    from naslib.search_spaces.nasbench_201.primitives import NAS_BENCH_201 as PRIMITIVES

    one_shot_optimizer = OneShotOptimizer()
    search_space = MacroGraph.from_optimizer_op(
        one_shot_optimizer,
        config=config_parser('../../configs/search_spaces/nasbench_201.yaml'),
        primitives=PRIMITIVES,
        ops_dict=NASBENCH_201_OPS
    )

    # Attempt forward pass
    res = search_space(torch.randn(size=[1, 3, 32, 32], dtype=torch.float, requires_grad=False))
