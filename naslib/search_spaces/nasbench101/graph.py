import os
import pickle
import torch.nn as nn
import numpy as np

from naslib.search_spaces.core import primitives as ops
from naslib.search_spaces.core.graph import Graph, EdgeData
from naslib.search_spaces.core.primitives import AbstractPrimitive
from naslib.search_spaces.core.query_metrics import Metric
from naslib.search_spaces.nasbench101.conversions import convert_naslib_to_spec

from naslib.utils.utils import get_project_root

from .primitives import ReLUConvBN


class NasBench101SearchSpace(Graph):
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
        node_pair.add_edges_from([(1,2)])

        cell = Graph()
        cell.name = 'cell'

        cell.add_node(1)    # input node
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

        total_num_nodes = 3
        self.add_nodes_from(range(1, total_num_nodes+1))
        self.add_edges_from([(i, i+1) for i in range(1, total_num_nodes)])

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

    def query(self, metric=None, dataset='cifar10', path=None, epoch=-1, full_lc=False, dataset_api=None):

        assert isinstance(metric, Metric)
        assert dataset in ['cifar10', None], "Unknown dataset: {}".format(dataset)
        if metric in [Metric.ALL, Metric.HP]:
            raise NotImplementedError()
        if dataset_api is None:
            raise NotImplementedError('Must pass in dataset_api to query nasbench101')
        assert epoch in [-1, 108, None] and not full_lc, 'nasbench101 does not have full learning curve information'
    
        metric_to_nb101 = {
            Metric.TRAIN_ACCURACY: 'train_accuracy',
            Metric.VAL_ACCURACY: 'validation_accuracy',
            Metric.TEST_ACCURACY: 'test_accuracy',
            Metric.TRAIN_TIME: 'training_time',
            Metric.PARAMETERS: 'trainable_parameters',
        }

        #cell = self.edges[2, 3].op
        matrix, ops = convert_naslib_to_spec(self)
        spec = dataset_api['api'].ModelSpec(matrix=self.matrix, ops=self.ops)
    
        if not nasbench.is_valid(nb101_spec):
            return -1
        
        query_results = dataset_api['nasbench'].query(nb101_spec)

        if metric == Metric.RAW:
            return query_results
            
        return query_results[metric_to_nb101[metric]]


def _set_node_ops(current_edge_data, C):
    current_edge_data.set('op', [
        ReLUConvBN(C, C, kernel_size=1),
        # ops.Zero(stride=1),    #! recheck about the hardcoded second operation
        ReLUConvBN(C, C, kernel_size=3),
        ops.MaxPool1x1(kernel_size=3, stride=1),
    ])

def _set_cell_ops(current_edge_data, C):
    current_edge_data.set('op', [
        ops.Identity(),
        ops.Zero(stride=1), 
    ])