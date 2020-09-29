import os
import pickle
import torch.nn as nn

from naslib.search_spaces.core import primitives as ops
from naslib.search_spaces.core.graph import Graph, EdgeData
from naslib.search_spaces.core.primitives import AbstractPrimitive
from naslib.search_spaces.core.query_metrics import Metric

from naslib.utils.utils import get_project_root

from .primitives import ReLUConvBN


# load the nasbench101 data
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


    def query(self, metric=None, dataset='cifar10', path=None):
        """
            Return e.g.: '|avg_pool_3x3~0|+|nor_conv_1x1~0|skip_connect~1|+|nor_conv_1x1~0|skip_connect~1|skip_connect~2|'
        """
        assert isinstance(metric, Metric)
        assert dataset in ['cifar10', None], "Unknown dataset: {}".format(dataset)
    
        cell = self.edges[2, 3].op
        # convert the naslib representation to nasbench101
        nb101_cell = _convert_cell_to_nb101(cell)
        query_results = nasbench.query(nb101_cell)

        metric_to_nb101 = {
            Metric.TRAIN_ACCURACY: 'train_accuracy',
            Metric.VAL_ACCURACY: 'validation_accuracy',
            Metric.TEST_ACCURACY: 'test_accuracy',
            Metric.TRAIN_TIME: 'training_time',
            Metric.PARAMETERS: 'trainable_parameters',
        }

        matrix = np.random.choice(ALLOWED_EDGES, size=(NUM_VERTICES, NUM_VERTICES))
        matrix = np.triu(matrix, 1)
        ops = np.random.choice(ALLOWED_OPS, size=(NUM_VERTICES)).tolist()
        ops[0] = INPUT
        ops[-1] = OUTPUT
        spec = api.ModelSpec(matrix=matrix, ops=ops)
        if nasbench.is_valid(spec):
            return spec

        if metric == Metric.RAW:
            return query_results
            
        return query_results[metric_to_nb101[metric]]


def _set_cell_ops(current_edge_data, C):
    current_edge_data.set('op', [
        ReLUConvBN(C, C, kernel_size=1),
        ops.Zero(stride=1),
        ReLUConvBN(C, C, kernel_size=3),
        ops.MaxPool1x1(kernel_size=3, stride=1),
    ])


def _convert_cell_to_nb101(cell):
    
    ops_to_nb101 = {
            'MaxPool1x1': 'maxpool3x3',
            'ReLUConvBN1x1': 'conv1x1-bn-relu',
            'ReLUConvBN3x3': 'conv3x3-bn-relu',
            'Zero': 'none',
        }

    return