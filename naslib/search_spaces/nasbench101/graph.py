import os
import pickle
import torch.nn as nn

from naslib.search_spaces.core import primitives as ops
from naslib.search_spaces.core.graph import Graph, EdgeData
from naslib.search_spaces.core.primitives import AbstractPrimitive

from naslib.utils.utils import get_project_root

from .primitives import ResNetBasicblock, ReLUConvBN

def _truncate_input_edges(node, in_edges, out_edges):
    """
    Discretize the one-shot model.
    """
    if any(e.has('alpha') or (e.has('final') and e.final) for _, e in in_edges):
        # We are in the one-shot case
        for _, data in in_edges:
            if data.has('final') and data.final:
                return  # We are looking at an out node
            data.alpha[1] = -float("Inf")   # Zero op should never be max alpha
        sorted_edge_ids = sorted(in_edges, key=lambda x: max(x[1].alpha), reverse=True)
        keep_edges, _ = zip(*sorted_edge_ids[:])
        for edge_id, edge_data in in_edges:
            if edge_id not in keep_edges:
                edge_data.delete()
    else:
        # We are in the discrete case (e.g. random search)
        k = 2
        for _, data in in_edges:
            assert isinstance(data.op, list)
            data.op.pop(1)      # Remove the zero op
        if any(e.has('final') and e.final for _, e in in_edges):
            return  # TODO: how about mixed final and non-final?
        else:
            for _ in range(len(in_edges) - k): #TODO: this is not correct. Fix it later
                in_edges[random.randint(0, len(in_edges)-1)][1].delete()



# load the nasbench101 data
# for querying the nb101 (data in tfrecords) -> through the nasbench api (need to download their repository?)
from nasbench import api

nb101_datadir = os.path.join(get_project_root(), 'data', 'nasbench_full.tfrecord')

nasbench = api.NASBench(nb101_datadir)

# data = nasbench.query(cell)

class NasBench101SeachSpace(Graph):
    """
    ResNet and Inception like search space used in the NB101 paper
    It also has an interface to the tabular benchmark of nasbench 101.
    """

    OPTIMIZER_SCOPE = [
        "stack_1",
        "stack_2",
        "stack_3",
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
        cell.add_node(4)
        cell.add_node(5)
        cell.add_node(6)

        # Output node
        cell.add_node(7)

        # Edges
        cell.add_edges_densly()

        #
        # Makrograph definition
        #
        self.name = "makrograph"

        # Cell is on the edges
        # 1-2:                  preprocessing: stem (128c, 3x3 conv)   
        # 2-3, 3-4, 4-5:        cells stack 1 
        # 5-6:                  downsample (img (h, w) 0.5x, channels 2x)
        # 6-7, 7-8, 8-9:        cells stack 2
        # 9-10:                 downsample (img (h, w) 0.5x, channels 2x)
        # 10-11, 11-12, 12-13:  cells stack 3          
        # 13-14:                post-processing:  global avg pool + fc

        total_num_nodes = 14
        self.add_nodes_from(range(1, total_num_nodes+1))
        self.add_edges_from([(i, i+1) for i in range(1, total_num_nodes)])

        channels = [128, 256, 512]

        #
        # operations at the edges
        #

        # preprocessing
        self.edges[1, 2].set('op', ops.Stem(channels[0]))
        
        # stage 1
        for i in range(2, 5):
            self.edges[i, i+1].set('op', cell.copy().set_scope('stack_1'))
        
        # stage 2
        self.edges[5, 6].set('op', ops.MaxPool1x1(C_in=channels[0], C_out=channels[1], stride=2))
        for i in range(6, 9):
            self.edges[i, i+1].set('op', cell.copy().set_scope('stack_2'))

        # stage 3
        self.edges[9, 10].set('op', MaxPool1x1(C_in=channels[1], C_out=channels[2], stride=2))
        for i in range(10, 13):
            self.edges[i, i+1].set('op', cell.copy().set_scope('stack_3'))

        # post-processing
        self.edges[13, 14].set('op', ops.Sequential(
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


    def query(self, metric='test_acc', dataset='cifar10', path='../../data'):
        """
            Return e.g.: '|avg_pool_3x3~0|+|nor_conv_1x1~0|skip_connect~1|+|nor_conv_1x1~0|skip_connect~1|skip_connect~2|'
        """
        assert metric in [
                'train_acc1es', 'train_losses',
                'train_times', 'params', 'flop', 'epochs', 'latency',
                'eval_acc1es', 'eval_times', 'eval_losses'
             ], "Unknown metric: {}".format(metric)

        assert dataset in ['cifar10-valid', 'cifar10', 'cifar100', 'ImageNet16-120'], "Unknown dataset: {}".format(dataset)

        ops_to_nb101 = {
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

def _set_cell_ops(current_edge_data, C):
    current_edge_data.set('op', [
        ops.Identity(),
        ops.Zero(stride=1),
        ReLUConvBN(C, C, kernel_size=3),
        ReLUConvBN(C, C, kernel_size=1),
        ops.MaxPool1x1(kernel_size=3, stride=1),
    ])
