import torch
import yaml
import numpy as np
from scipy.special import softmax
from torch import nn
from copy import deepcopy

from naslib.search_spaces.core.graphs import Graph, EdgeData, GraphWrapper
from naslib.search_spaces.core import EdgeOpGraph, NodeOpGraph
from naslib.search_spaces.core.primitives import FactorizedReduce, ReLUConvBN, Stem, Identity, Zero, SepConv, DilConv, Sequential, ModuleWrapper, MaxPool1x1, AvgPool1x1

def set_cell_ops(current_edge_data, C, stride):
    if current_edge_data.has('final') and current_edge_data.final:
        return current_edge_data
    else:
        C_in = C if stride==1 else C//2
        current_edge_data.set('op', [
            Identity() if stride==1 else FactorizedReduce(C_in, C),    # TODO: what is this and why is it not in the paper?
            Zero(stride=stride),
            MaxPool1x1(3, stride, C_in, C),
            AvgPool1x1(3, stride, C_in, C),
            SepConv(C_in, C, kernel_size=3, stride=stride, padding=1, affine=False),
            SepConv(C_in, C, kernel_size=5, stride=stride, padding=2, affine=False),
            DilConv(C_in, C, kernel_size=3, stride=stride, padding=2, dilation=2, affine=False),
            DilConv(C_in, C, kernel_size=5, stride=stride, padding=4, dilation=2, affine=False),
        ])
    return current_edge_data


class DartsSearchSpace(GraphWrapper):

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
        

        reduction_cell = deepcopy(normal_cell)
        reduction_cell.name = "reduction_cell"

        #
        # Makrograph definition
        #
        self.name = "makrograph"

        self.add_node(1)    # input node
        self.add_node(2)    # preprocessing
        self.add_node(3, subgraph=normal_cell.copy().set_scope("n_stage_1").set_input([2, 2]))
        self.add_node(4, subgraph=normal_cell.copy().set_scope("n_stage_1").set_input([2, 3]))
        self.add_node(5, subgraph=reduction_cell.copy().set_scope("r_stage_1").set_input([3, 4]))
        self.add_node(6, subgraph=normal_cell.copy().set_scope("n_stage_2").set_input([5, 5]))   # TODO: is this correct?
        self.add_node(7, subgraph=normal_cell.copy().set_scope("n_stage_2").set_input([5, 6]))
        self.add_node(8, subgraph=reduction_cell.copy().set_scope("r_stage_2").set_input([6, 7]))
        self.add_node(9, subgraph=normal_cell.copy().set_scope("n_stage_3").set_input([8, 8]))   # See above
        self.add_node(10, subgraph=normal_cell.copy().set_scope("n_stage_3").set_input([8, 9]))
        self.add_node(11)   # output

        self.add_edge(1, 2)     # pre-processing (stem)
        self.add_edges_from([(2, 3), (2, 4), (3, 4), (3, 5), (4, 5)])   # first stage
        self.add_edges_from([(5, 6), (5, 7), (6, 7), (6, 8), (7, 8)])   # second stage
        self.add_edges_from([(8, 9), (8, 10), (9, 10)])                 # third stage
        self.add_edge(10, 11)   # post-processing (pooling, classifier)

        #
        # Operations at the edges
        #

        # pre-processing
        self.edges[1, 2].set('op', Stem(16))

        # normal cells
        channels = [16, 32, 64]
        stages = ["n_stage_1", "n_stage_2", "n_stage_3"]

        for scope, c in zip(stages, channels):
            self.update_edges(
                update_func=lambda current_edge_data: set_cell_ops(current_edge_data, c, stride=1),
                scope=scope,
                private_edge_data=True
            )

        # reduction cells
        nodes = [5, 8]
        for n, c in zip(nodes, channels[1:]):
            reduction_cell = self.nodes[n]['subgraph']
            for u, v, data in reduction_cell.edges.data():
                stride = 2 if u in (1, 2) else 1
                reduction_cell.edges[u, v].update(set_cell_ops(data, c, stride))
        
        # post-processing
        self.edges[10, 11].set('op', Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels[-1], 10))
        )







if __name__ == '__main__':
    sspace = DartsSearchSpace()

    print()









class Cell(EdgeOpGraph):
    def __init__(self, primitives, cell_type, C_prev_prev, C_prev, C,
                 reduction_prev, ops_dict, *args, **kwargs):
        self.primitives = primitives
        self.cell_type = cell_type
        self.C_prev_prev = C_prev_prev
        self.C_prev = C_prev
        self.C = C
        self.reduction_prev = reduction_prev
        self.ops_dict = ops_dict
        self.drop_path_prob = 0
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
                    op_kwargs={'C': self.C, 'stride': stride, 'out_node_op': 'sum', 'ops_dict': self.ops_dict,
                               'affine': False},
                    to_node=to_node, from_node=from_node)

        # Edges: inter-output
        self.add_edge(2, 6, op=Identity())
        self.add_edge(3, 6, op=Identity())
        self.add_edge(4, 6, op=Identity())
        self.add_edge(5, 6, op=Identity())

    @classmethod
    def from_config(cls, graph_dict, primitives, cell_type, C_prev_prev,
                    C_prev, C, reduction_prev, ops_dict, load_kwargs, *args, **kwargs):
        graph = cls(primitives, cell_type, C_prev_prev, C_prev, C,
                    reduction_prev, ops_dict, *args, **kwargs)

        graph.clear()
        # Input Nodes: Previous / Previous-Previous cell
        for node, attr in graph_dict['nodes'].items():
            if 'preprocessing' in attr:
                # Input Nodes: Previous / Previous-Previous cell
                #TODO: find better way to do this
                if node == 0:
                    preprocessing = FactorizedReduce(C_prev_prev, C, affine=False) \
                        if reduction_prev else ReLUConvBN(C_prev_prev,
                                                          C, 1, 1, 0, affine=False)
                elif node == 1:
                    preprocessing = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
                """
                if attr['preprocessing'] == 'FactorizedReduce':
                    input_args = {'C_in': C_prev_prev, 'C_out': C,
                                  'affine': False}
                else:
                    in_channels = C_prev_prev if reduction_prev else C_prev
                    input_args = {'C_in': in_channels, 'C_out': C,
                                  'kernel_size': 1, 'stride': 1, 'padding': 0,
                                  'affine': False}

                preprocessing = eval(attr['preprocessing'])(**input_args)
                """

                graph.add_node(node, type=attr['type'],
                               preprocessing=preprocessing)
            else:
                graph.add_nodes_from([(node, attr)])

        for edge, attr in graph_dict['edges'].items():
            from_node, to_node = eval(edge)
            graph.add_edge(*eval(edge), **{k: eval(v) for k, v in attr.items() if k
                                           in ['from_node', 'to_node',
                                               'op_choices']})
            graph[from_node][to_node]['op'] = None if attr['op'] != 'Identity' else eval(attr['op'])()
            if 'arch_weight' in attr:
                arch_weight = attr['arch_weight']
                arch_weight_str = arch_weight[arch_weight.index('['):
                                              arch_weight.index(']')+1]
                graph[from_node][to_node]['arch_weight'] = np.array(eval(arch_weight_str))
            elif 'sampled_arch_weight' in attr:
                arch_weight = attr['sampled_arch_weight']
                arch_weight_str = arch_weight[arch_weight.index('['):
                                              arch_weight.index(']')+1]
                graph[from_node][to_node]['arch_weight'] = np.array(eval(arch_weight_str))
            #TODO: add this option later
            if load_kwargs and 'op_choices' in graph[from_node][to_node]:
                graph[from_node][to_node]['op_kwargs'] = eval(attr['op_kwargs'])

            if 'op_kwargs' in graph[from_node][to_node]:
                graph[from_node][to_node]['op_kwargs']['ops_dict'] = ops_dict
                if 'affine' not in graph[from_node][to_node]['op_kwargs']:
                    graph[from_node][to_node]['op_kwargs']['affine'] = False

        return graph


class MacroGraph(NodeOpGraph):
    def __init__(self, config, primitives, ops_dict, *args, **kwargs):
        self.config = config
        self.primitives = primitives
        self.ops_dict = ops_dict
        super(MacroGraph, self).__init__(*args, **kwargs)

    def _build_graph(self):
        num_layers = self.config['layers']
        C = self.config['init_channels']
        C_curr = self.config['stem_multiplier'] * C

        stem = Stem(C_curr=C_curr)
        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C

        # TODO: set the input edges to the first cell in a nicer way
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
                          op=Cell(primitives=self.primitives, C_prev_prev=C_prev_prev, C_prev=C_prev, C=C_curr,
                                  reduction_prev=reduction_prev, cell_type='reduction' if reduction else 'normal',
                                  ops_dict=self.ops_dict),
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


    def get_cells(self, cell_type):
        cells = list()
        for n in self.nodes:
            if 'type' in self.nodes[n] and self.nodes[n]['type'] == cell_type:
                cells.append(n)
        return cells


    #TODO: merge with sample method
    def discretize(self, config, n_ops_per_edge=1, n_input_edges=None):
        """
        n_ops_per_edge:
            1; number of sampled operations per edge in cell
        n_input_edges:
            None; list equal with length with number of intermediate
        nodes. Determines the number of predecesor nodes for each of them
        """
        # create a new graph that we will discretize
        new_graph = MacroGraph(config, self.primitives, self.ops_dict)
        normal_cell = self.get_node_op(self.get_cells('normal')[0])
        reduction_cell = self.get_node_op(self.get_cells('reduction')[0])

        for node in new_graph:
            #_cell = self.get_node_op(node)
            cell = new_graph.get_node_op(node)
            if not isinstance(cell, Cell):
                continue
            _cell = normal_cell if cell.cell_type == 'normal' else reduction_cell

            if n_input_edges is not None:
                for inter_node, k in zip(_cell.inter_nodes(), n_input_edges):
                    # in case the start node index is not 0
                    node_idx = list(_cell.nodes).index(inter_node)
                    prev_node_choices = list(_cell.nodes)[:node_idx]
                    assert k <= len(prev_node_choices), 'cannot sample more'
                    ' than number of predecesor nodes'

                    previous_argmax_alphas = {}
                    op_choices = {}
                    for i in prev_node_choices:
                        op_choices[i] = _cell.get_edge_op_choices(i,
                                                                  inter_node)
                        arch_weight_data = _cell.get_edge_arch_weights(i,
                                                                       inter_node)
                        if type(arch_weight_data) == torch.nn.parameter.Parameter:
                            alphas = softmax(
                                arch_weight_data.cpu().detach()
                            )
                        else:
                            alphas = softmax(arch_weight_data)
                        if type(alphas) == torch.nn.parameter.Parameter:
                            alphas = alphas.numpy()
                        previous_argmax_alphas[i] = alphas

                    try:
                        sampled_input_edges = sorted(prev_node_choices, key=lambda
                                                     x:
                                                     -max(previous_argmax_alphas[x][k]
                                                          for k in
                                                          range(len(previous_argmax_alphas[x]))
                                                          if k !=
                                                          op_choices[x].index('none')))[:k]
                    except ValueError:
                        sampled_input_edges = sorted(prev_node_choices, key=lambda
                                                     x:
                                                     -max(previous_argmax_alphas[x][k]
                                                          for k in
                                                          range(len(previous_argmax_alphas[x]))))[:k]

                    for i in set(prev_node_choices) - set(sampled_input_edges):
                        cell.remove_edge(i, inter_node)

            for edge in cell.edges:
                if bool(set(_cell.output_nodes()) & set(edge)):
                    continue
                op_choices = deepcopy(_cell.get_edge_op_choices(*edge))
                _alphas = _cell.get_edge_arch_weights(*edge)
                if type(_alphas) == torch.nn.parameter.Parameter:
                    alphas = deepcopy(list(_alphas.cpu().detach().numpy()))
                else:
                    alphas = deepcopy(list(_alphas))

                if 'none' in op_choices:
                    none_idx = op_choices.index('none')
                    del op_choices[none_idx]
                    del alphas[none_idx]

                sampled_op = np.array(op_choices)[np.argsort(alphas)[-n_ops_per_edge:]]
                cell[edge[0]][edge[1]]['op_choices'] = [*sampled_op]

        return new_graph


    def sample(self, same_cell_struct=True, n_ops_per_edge=1,
               n_input_edges=None, dist=None, seed=1):
        """
        same_cell_struct:
            True; if the sampled cell topology is the same or not
        n_ops_per_edge:
            1; number of sampled operations per edge in cell
        n_input_edges:
            None; list equal with length with number of intermediate
        nodes. Determines the number of predecesor nodes for each of them
        dist:
            None; distribution to sample operations in edges from
        seed:
            1; random seed
        """
        # create a new graph that we will discretize
        new_graph = MacroGraph(self.config, self.primitives, self.ops_dict)
        np.random.seed(seed)
        seeds = {'normal': seed+1, 'reduction': seed+2}

        for node in new_graph:
            cell = new_graph.get_node_op(node)
            if not isinstance(cell, Cell):
                continue

            if same_cell_struct:
                np.random.seed(seeds[new_graph.get_node_type(node)])

            for edge in cell.edges:
                if bool(set(cell.output_nodes()) & set(edge)):
                    continue
                op_choices = cell.get_edge_op_choices(*edge)
                sampled_op = np.random.choice(op_choices, n_ops_per_edge,
                                              False, p=dist)
                cell[edge[0]][edge[1]]['op_choices'] = [*sampled_op]

            if n_input_edges is not None:
                for inter_node, k in zip(cell.inter_nodes(), n_input_edges):
                    # in case the start node index is not 0
                    node_idx = list(cell.nodes).index(inter_node)
                    prev_node_choices = list(cell.nodes)[:node_idx]
                    assert k <= len(prev_node_choices), 'cannot sample more'
                    ' than number of predecesor nodes'

                    sampled_input_edges = np.random.choice(prev_node_choices,
                                                           k, False)
                    for i in set(prev_node_choices) - set(sampled_input_edges):
                        cell.remove_edge(i, inter_node)

        return new_graph


    @classmethod
    def from_config(cls, config=None, filename=None, load_kwargs=False, **kwargs):
        with open(filename, 'r') as f:
            graph_dict = yaml.safe_load(f)

        if config is None:
            raise ('No configuration provided')

        graph = cls(config, [], **kwargs)

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
                if node_type == 'reduction':
                    C_curr *= 2
                graph.add_node(node,
                               op=Cell.from_config(attr['op'], primitives=attr['op']['primitives'],
                                                   C_prev_prev=C_prev_prev, C_prev=C_prev,
                                                   C=C_curr,
                                                   reduction_prev=graph_dict['nodes'][node - 1]['type'] == 'reduction',
                                                   cell_type=node_type,
                                                   ops_dict=kwargs['ops_dict'],
                                                   load_kwargs=load_kwargs),
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

