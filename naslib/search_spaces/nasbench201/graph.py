import os
import pickle
import torch.nn as nn

from naslib.search_spaces.core import primitives as ops
from naslib.search_spaces.core.graph import Graph, EdgeData
from naslib.search_spaces.core.primitives import AbstractPrimitive

from .primitives import ResNetBasicblock

def _set_cell_ops(current_edge_data, C):
    current_edge_data.set('op', [
        ops.Identity(),
        ops.Zero(stride=1),
        ops.ReLUConvBN(C, C, kernel_size=3),
        ops.ReLUConvBN(C, C, kernel_size=1),
        ops.AvgPool1x1(kernel_size=3, stride=1),
    ])
    return current_edge_data


def remove_zero_alpha(current_edge_data):
    if current_edge_data.has('alpha'):
        current_edge_data.alpha[1] = -float("Inf")   # Zero op should never be max alpha
    return current_edge_data

def remove_zero_op(current_edge_data):
    if isinstance(current_edge_data.op, list):
        current_edge_data.op.pop(1)      # Remove the zero op
        return current_edge_data
    else:
        raise ValueError("Unknown format of the op: {}".format(current_edge_data.op))


class NasBench201SeachSpace(Graph):
    """
    Implementation of the nasbench 201 search space.
    """

    OPTIMIZER_SCOPE = [
        "stage_1",
        "stage_2",
        "stage_3",
    ]

    QUERYABLE = True

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
        cell.add_edges_from([(i, i+1) for i in range(1, 4)])
        cell.add_edges_from([(i, i+2) for i in range(1, 3)])
        cell.add_edges_from([(1, 4)])

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


    def prepare_discretization(self):
        if self.get_all_edge_data('alpha'):
            self.update_edges(remove_zero_alpha, scope=self.OPTIMIZER_SCOPE, private_edge_data=False)
        else:
            self.update_edges(remove_zero_op, scope=self.OPTIMIZER_SCOPE, private_edge_data=True)
        


    def query(self, metric='eval_acc1es', dataset='cifar10', path='../../data'):
        """
            Return e.g.: '|avg_pool_3x3~0|+|nor_conv_1x1~0|skip_connect~1|+|nor_conv_1x1~0|skip_connect~1|skip_connect~2|'
        """
        assert metric in [
                'train_acc1es', 'train_losses',
                'train_times', 'params', 'flop', 'epochs', 'latency',
                'eval_acc1es', 'eval_times', 'eval_losses'
             ], "Unknown metric: {}".format(metric)
        
        assert dataset in ['cifar10-valid', 'cifar10', 'cifar100', 'ImageNet16-120'], "Unknown dataset: {}".format(dataset)
        
        ops_to_nb201 = {
            'AvgPool1x1': 'avg_pool_3x3',
            'ReLUConvBN1x1': 'nor_conv_1x1',
            'ReLUConvBN3x3': 'nor_conv_3x3',
            'Identity': 'skip_connect',
            'Zero': 'none',
        }

        # convert the naslib representation to nasbench201
        cell = self._get_child_graphs(single_instances=True)[0]
        edge_op_dict = {
            (i, j): ops_to_nb201[cell.edges[i, j]['op'].get_op_name] for i, j in cell.edges
        }
        op_edge_list = [
            '{}~{}'.format(edge_op_dict[(i, j)], i-1) for i, j in sorted(edge_op_dict, key=lambda x: x[1])
        ]

        arch_str = '|{}|+|{}|{}|+|{}|{}|{}|'.format(*op_edge_list)

        # load the nasbench201 data and return the queried data
        with open(os.path.join(path, 'nb201_all.pickle'), 'rb') as f:
            nb201_data = pickle.load(f)
        query_results = nb201_data[arch_str]
        if metric == 'all':
            return query_results[dataset]
        else:
            return query_results[dataset][metric]
