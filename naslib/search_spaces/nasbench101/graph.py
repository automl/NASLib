import os
import pickle
import torch.nn as nn
import numpy as np

from naslib.search_spaces.core import primitives as ops
from naslib.search_spaces.core.graph import Graph, EdgeData
from naslib.search_spaces.core.primitives import AbstractPrimitive
from naslib.search_spaces.core.query_metrics import Metric

from naslib.utils.utils import get_project_root

from .primitives import ReLUConvBN


# load the nasbench101 data -- requires TF 1.x
from nasbench import api

nb101_datadir = os.path.join(get_project_root(), 'data', 'nasbench_only108.tfrecord')
nasbench = api.NASBench(nb101_datadir)

# data = nasbench.query(cell)

class NasBench101SeachSpace(Graph):
    """
    Contains the interface to the tabular benchmark of nasbench 101.
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
        node_pair = Graph()
        node_pair.name = "node_pair"    # Use the same name for all cells with shared attributes

        # need to add subgraphs on the nodes, each subgraph has option for 3 ops
        # Input node
        node_pair.add_node(1)
        node_pair.add_node(2)
        node_pair.add_edges_from[(1,2)]

        cell = Graph()
        
        cell.add_node(1)    # input node
        #* question: what is set_input doing here, when we are defining the edges later?
        cell.add_node(2, subgraph=node_pair.set_scope("stack_1").set_input([1]))
        cell.add_node(3, subgraph=node_pair.copy().set_scope("stack_1"))
        cell.add_node(4, subgraph=node_pair.copy().set_scope("stack_1"))
        cell.add_node(5, subgraph=node_pair.copy().set_scope("stack_1"))
        cell.add_node(6, subgraph=node_pair.copy().set_scope("stack_1"))
        cell.add_node(7)    # output

        # Edges
        cell.add_edges_densly()

        #
        # dummy Makrograph definition for RE for benchmark queries
        #
        channels = [128, 256, 512]
        self.name = "makrograph"
        self.edges[1, 2].set('op', ops.Stem(channels[0]))
        self.edges[2, 3].set('op', cell.copy().set_scope('stage_1'))
        
        
        node_pair.update_edges(
            update_func=lambda current_edge_data: _set_node_ops(current_edge_data, C=channels[0]),
            scope="node",
            private_edge_data=True
        )
        
        cell.update_edges(
            update_func=lambda current_edge_data: _set_cell_ops(current_edge_data, C=channels[0]),
            scope="cell",
            private_edge_data=True
        )

    def query(self, metric=None, dataset='cifar10', path=None):
        """
            Return e.g.: '|avg_pool_3x3~0|+|nor_conv_1x1~0|skip_connect~1|+|nor_conv_1x1~0|skip_connect~1|skip_connect~2|'
        """
        assert isinstance(metric, Metric)
        assert dataset in ['cifar10', None], "Unknown dataset: {}".format(dataset)
    
        cell = self.edges[2, 3].op
    
        metric_to_nb101 = {
            Metric.TRAIN_ACCURACY: 'train_accuracy',
            Metric.VAL_ACCURACY: 'validation_accuracy',
            Metric.TEST_ACCURACY: 'test_accuracy',
            Metric.TRAIN_TIME: 'training_time',
            Metric.PARAMETERS: 'trainable_parameters',
        }

        # convert the naslib representation to nasbench101
        nb101_spec = _convert_cell_to_nb101_spec(cell)        
    
        if not nasbench.is_valid(nb101_spec):
            return 'invalid' # or some negative reward or none

        query_results = nasbench.query(nb101_spec)

        if metric == Metric.RAW:
            return query_results
            
        return query_results[metric_to_nb101[metric]]

def _convert_cell_to_nb101_spec(cell):
    
    matrix = np.triu(np.ones((7,7)), 1)

    ops_to_nb101 = {
            'MaxPool1x1': 'maxpool3x3',
            'ReLUConvBN1x1': 'conv1x1-bn-relu',
            'ReLUConvBN3x3': 'conv3x3-bn-relu',
        }

    ops_to_nb101_edges = {
        'Identity': 1,
        'Zero': 0,
    }

    num_vertices = 7
    ops = ['input'] * num_vertices
    ops[-1] = 'output'

    for i in range(1, 6):
        ops[i] = ops_to_nb101[cell.nodes[i+1]['subgraph'].edges[1, 2]['op'].get_op_name]
    
    matrix[i][j] = ops_to_nb101_edges[cell.edges[i, j]['op'].get_op_name] for i, j in cell.edges
    
    spec = api.ModelSpec(matrix=matrix, ops=ops)
    
    return spec


def _set_pair_ops(current_edge_data, C):
    current_edge_data.set('op', [
        ReLUConvBN(C, C, kernel_size=1),
        # ops.Zero(stride=1),    #! does the second operation here always needs to be zero?
        ReLUConvBN(C, C, kernel_size=3),
        ops.MaxPool1x1(kernel_size=3, stride=1),
    ])

def _set_cell_ops(current_edge_data, C):
    current_edge_data.set('op', [
        ops.Identity(),
        ops.Zero(stride=1), 
    ])