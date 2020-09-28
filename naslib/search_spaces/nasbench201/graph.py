import os
import pickle
import torch.nn as nn

from naslib.search_spaces.core import primitives as ops
from naslib.search_spaces.core.graph import Graph, EdgeData
from naslib.search_spaces.core.primitives import AbstractPrimitive

from naslib.utils.utils import get_project_root

from .primitives import ResNetBasicblock

# load the nasbench201 data
with open(os.path.join(get_project_root(), 'data', 'nb201_all.pickle'), 'rb') as f:
    nb201_data = pickle.load(f)


class NasBench201SeachSpace(Graph):
    """
    Implementation of the nasbench 201 search space.
    It also has an interface to the tabular benchmark of nasbench 201.
    """

    OPTIMIZER_SCOPE = [
        "stage_1",
        "stage_2",
        "stage_3",
    ]

    QUERYABLE = True


    def __init__(self):
        super().__init__()
        self.num_classes = self.NUM_CLASSES if hasattr(self, 'NUM_CLASSES') else 10

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
        cell.add_edges_densly()

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
            nn.Linear(channels[-1], self.num_classes)
        ))
        
        # set the ops at the cells (channel dependent)
        for c, scope in zip(channels, self.OPTIMIZER_SCOPE):
            self.update_edges(
                update_func=lambda current_edge_data: _set_cell_ops(current_edge_data, C=c),
                scope=scope,
                private_edge_data=True
            )
        

    def query(self, metric=None, dataset=None, path=None):
        """
            Return e.g.: '|avg_pool_3x3~0|+|nor_conv_1x1~0|skip_connect~1|+|nor_conv_1x1~0|skip_connect~1|skip_connect~2|'
        """
        assert metric in [
                'train_acc1es', 'train_losses',
                'train_times', 'params', 'flop', 'epochs', 'latency',
                'eval_acc1es', 'eval_times', 'eval_losses', None
             ], "Unknown metric: {}".format(metric)
        
        assert dataset in ['cifar10-valid', 'cifar10', 'cifar100', 'ImageNet16-120', None], "Unknown dataset: {}".format(dataset)
        
        ops_to_nb201 = {
            'AvgPool1x1': 'avg_pool_3x3',
            'ReLUConvBN1x1': 'nor_conv_1x1',
            'ReLUConvBN3x3': 'nor_conv_3x3',
            'Identity': 'skip_connect',
            'Zero': 'none',
        }

        # convert the naslib representation to nasbench201
        cell = self.edges[2, 3].op
        edge_op_dict = {
            (i, j): ops_to_nb201[cell.edges[i, j]['op'].get_op_name] for i, j in cell.edges
        }
        op_edge_list = [
            '{}~{}'.format(edge_op_dict[(i, j)], i-1) for i, j in sorted(edge_op_dict, key=lambda x: x[1])
        ]

        arch_str = '|{}|+|{}|{}|+|{}|{}|{}|'.format(*op_edge_list)

        # query data from nb201
        query_results = nb201_data[arch_str]
        if dataset:
            if metric is None or metric == 'all':
                return query_results[dataset]
            else:
                return query_results[dataset][metric]
        else:
            return query_results


def _set_cell_ops(current_edge_data, C):
    current_edge_data.set('op', [
        ops.Identity(),
        ops.Zero(stride=1),
        ops.ReLUConvBN(C, C, kernel_size=3),
        ops.ReLUConvBN(C, C, kernel_size=1),
        ops.AvgPool1x1(kernel_size=3, stride=1),
    ])
