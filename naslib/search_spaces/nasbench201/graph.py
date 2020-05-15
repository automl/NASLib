import torch
import yaml
from nas_201_api import NASBench201API as API
from torch import nn

from naslib.search_spaces.core import EdgeOpGraph, NodeOpGraph
from naslib.search_spaces.core.primitives import Stem
from naslib.search_spaces.nasbench201.primitives import ResNetBasicblock, PRIMITIVES
from naslib.search_spaces.nasbench201.primitives import Stem as NASBENCH_201_Stem


class Cell(EdgeOpGraph):
    def __init__(self, primitives, cell_type, C_prev, C, stride, ops_dict, *args, **kwargs):
        self.primitives = primitives
        self.cell_type = cell_type
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
                    op_kwargs={'C': self.C, 'stride': self.stride, 'affine': True,
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
        self.nasbench_api = API('/home/siemsj/nasbench_201.pth')
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
                            ops_dict=self.ops_dict, cell_type='normal')
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
        seeds = {'normal': seed + 1, 'reduction': seed + 2}

        for node in new_graph:
            cell = new_graph.get_node_op(node)
            if not isinstance(cell, Cell):
                continue

            if same_cell_struct:
                np.random.seed(seeds[new_graph.get_node_type(node)])

            for edge in cell.edges:
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

    @staticmethod
    def export_nasbench_201_results_to_dict(information):
        results_dict = {}
        dataset_names = information.get_dataset_names()
        results_dict['arch'] = information.arch_str
        results_dict['datasets'] = dataset_names

        for ida, dataset in enumerate(dataset_names):
            dataset_results = {}
            dataset_results['dataset'] = dataset

            metric = information.get_compute_costs(dataset)
            flop, param, latency, training_time = metric['flops'], metric['params'], metric['latency'], metric[
                'T-train@total']
            dataset_results['flop'] = flop
            dataset_results['params (MB)'] = param
            dataset_results['latency (ms)'] = latency * 1000 if latency is not None and latency > 0 else None
            dataset_results['training_time'] = training_time

            train_info = information.get_metrics(dataset, 'train')
            if dataset == 'cifar10-valid':
                valid_info = information.get_metrics(dataset, 'x-valid')
                dataset_results['train_loss'] = train_info['loss']
                dataset_results['train_accuracy'] = train_info['accuracy']

                dataset_results['valid_loss'] = valid_info['loss']
                dataset_results['valid_accuracy'] = valid_info['accuracy']

            elif dataset == 'cifar10':
                test__info = information.get_metrics(dataset, 'ori-test')
                dataset_results['train_loss'] = train_info['loss']
                dataset_results['train_accuracy'] = train_info['accuracy']

                dataset_results['test_loss'] = test__info['loss']
                dataset_results['test_accuracy'] = test__info['accuracy']
            else:
                valid_info = information.get_metrics(dataset, 'x-valid')
                test__info = information.get_metrics(dataset, 'x-test')
                dataset_results['train_loss'] = train_info['loss']
                dataset_results['train_accuracy'] = train_info['accuracy']

                dataset_results['valid_loss'] = valid_info['loss']
                dataset_results['valid_accuracy'] = valid_info['accuracy']

                dataset_results['test_loss'] = test__info['loss']
                dataset_results['test_accuracy'] = test__info['accuracy']
            results_dict[dataset] = dataset_results
        return results_dict

    def query_architecture(self, arch_weights):
        arch_weight_idx_to_parent = {0: 0, 1: 0, 2: 1, 3: 0, 4: 1, 5: 2}
        arch_strs = {
            'cell_normal_from_0_to_1': '',
            'cell_normal_from_0_to_2': '',
            'cell_normal_from_1_to_2': '',
            'cell_normal_from_0_to_3': '',
            'cell_normal_from_1_to_3': '',
            'cell_normal_from_2_to_3': '',
        }
        for arch_weight_idx, (edge_key, edge_weights) in enumerate(arch_weights.items()):
            edge_weights_norm = torch.softmax(edge_weights, dim=-1)
            selected_op_str = PRIMITIVES[edge_weights_norm.argmax()]
            arch_strs[edge_key] = '{}~{}'.format(selected_op_str, arch_weight_idx_to_parent[arch_weight_idx])

        arch_str = '|{}|+|{}|{}|+|{}|{}|{}|'.format(*arch_strs.values())
        if not hasattr(self, 'nasbench_api'):
            self.nasbench_api = API('/home/siemsj/nasbench_201.pth')
        index = self.nasbench_api.query_index_by_arch(arch_str)
        self.nasbench_api.show(index)
        info = self.nasbench_api.query_by_index(index)
        return self.export_nasbench_201_results_to_dict(info)
