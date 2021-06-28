import os
import pickle
import numpy as np
import copy
import random
import torch
import torch.nn as nn

from naslib.search_spaces.core import primitives as ops
from naslib.search_spaces.core.graph import Graph, EdgeData
from naslib.search_spaces.core.primitives import ReLUConvBN, MaxPool1x1
from naslib.search_spaces.core.query_metrics import Metric
from naslib.search_spaces.nasbench101.conversions import convert_naslib_to_spec, \
convert_spec_to_naslib, convert_spec_to_tuple

from naslib.utils.utils import get_project_root

from .primitives import ReLUConvBN

INPUT = 'input'
OUTPUT = 'output'
CONV3X3 = 'conv3x3-bn-relu'
CONV1X1 = 'conv1x1-bn-relu'
MAXPOOL3X3 = 'maxpool3x3'
OPS = [CONV3X3, CONV1X1, MAXPOOL3X3]

NUM_VERTICES = 7
OP_SPOTS = NUM_VERTICES - 2
MAX_EDGES = 9


class NasBench101SearchSpace(Graph):
    """
    Contains the interface to the tabular benchmark of nasbench 101.
    """

    OPTIMIZER_SCOPE = [
        "node",
        "cell",
    ]

    QUERYABLE = True

    def __init__(self):
        super().__init__()
        self.num_classes = self.NUM_CLASSES if hasattr(self, 'NUM_CLASSES') else 10
        
        # creating a dummy graph!
        channels = [128, 256, 512]
        #
        # Cell definition
        #
        node_pair = Graph()
        node_pair.name = "node_pair"    # Use the same name for all cells with shared attributes
        node_pair.set_scope("node")

        # need to add subgraphs on the nodes, each subgraph has option for 3 ops
        # Input node
        node_pair.add_node(1)
        node_pair.add_node(2)
        node_pair.add_edges_from([(1,2)])

        cell = Graph()
        cell.name = 'cell'

        node_pair.update_edges(
            update_func=lambda edge: _set_node_ops(edge, C=channels[0]),
            private_edge_data=True
        )

        cell.add_node(1)    # input node
        cell.add_node(2, subgraph=node_pair.set_input([1]))
        cell.add_node(3, subgraph=node_pair.copy())
        cell.add_node(4, subgraph=node_pair.copy())
        cell.add_node(5, subgraph=node_pair.copy())
        cell.add_node(6, subgraph=node_pair.copy())
        cell.add_node(7)    # output
        cell.set_scope('cell', recursively=False)

        # Edges
        cell.add_edges_densly()
        
        cell.update_edges(
            update_func=lambda edge: _set_cell_ops(edge, C=channels[0]),
            scope="cell",
            private_edge_data=True
        )
        #
        # dummy Makrograph definition for RE for benchmark queries
        #
        
        self.name = "makrograph"

        total_num_nodes = 3
        self.add_nodes_from(range(1, total_num_nodes+1))
        self.add_edges_from([(i, i+1) for i in range(1, total_num_nodes)])

        self.edges[1, 2].set('op', ops.Stem(channels[0]))
        self.edges[2, 3].set('op', cell.copy().set_scope('cell'))

    def query(self, metric=None, dataset='cifar10', path=None, epoch=-1, full_lc=False, dataset_api=None):
        """
        Query results from nasbench 101
        """
        assert isinstance(metric, Metric)
        assert dataset in ['cifar10', None], "Unknown dataset: {}".format(dataset)
        if metric in [Metric.ALL, Metric.HP]:
            raise NotImplementedError()
        if dataset_api is None:
            raise NotImplementedError('Must pass in dataset_api to query nasbench101')
        assert epoch in [-1, 4, 12, 36, 108, None], 'nasbench101 does not have full learning curve info'
    
        metric_to_nb101 = {
            Metric.TRAIN_ACCURACY: 'train_accuracy',
            Metric.VAL_ACCURACY: 'validation_accuracy',
            Metric.TEST_ACCURACY: 'test_accuracy',
            Metric.TRAIN_TIME: 'training_time',
            Metric.PARAMETERS: 'trainable_parameters',
        }

        if self.spec is None:
            raise NotImplementedError('Cannot yet query directly from the naslib object')
        api_spec = dataset_api['api'].ModelSpec(**self.spec)
    
        if not dataset_api['nb101_data'].is_valid(api_spec):
            return -1
        
        query_results = dataset_api['nb101_data'].query(api_spec)
        if full_lc:
            vals =  [dataset_api['nb101_data'].query(api_spec, epochs=e)[metric_to_nb101[metric]] for e in [4, 12, 36, 108]]
            # return a learning curve with unique values only at 4, 12, 36, 108
            nums = [4, 8, 20, 56]
            lc = [val for i, val in enumerate(vals) for _ in range(nums[i])]
            if epoch == -1:
                return lc
            else:
                return lc[:epoch]

        if metric == Metric.RAW:
            return query_results
        elif metric == Metric.TRAIN_TIME:
            return query_results[metric_to_nb101[metric]]
        else:
            return query_results[metric_to_nb101[metric]] * 100
    
    def get_spec(self):
        if self.spec is None:
            self.spec = convert_naslib_to_spec(self)
        return self.spec
    
    def get_hash(self):
        return convert_spec_to_tuple(self.get_spec())

    def set_spec(self, spec):
        # TODO: convert the naslib object to this spec
        # convert_spec_to_naslib(spec, self)
        self.spec = spec

    def sample_random_architecture(self, dataset_api):
        """
        This will sample a random architecture and update the edges in the
        naslib object accordingly.
        From the NASBench repository:
        one-hot adjacency matrix
        draw [0,1] for each slot in the adjacency matrix
        """
        while True:
            matrix = np.random.choice(
                [0, 1], size=(NUM_VERTICES, NUM_VERTICES))
            matrix = np.triu(matrix, 1)
            ops = np.random.choice(OPS, size=NUM_VERTICES).tolist()
            ops[0] = INPUT
            ops[-1] = OUTPUT
            spec = dataset_api['api'].ModelSpec(matrix=matrix, ops=ops)
            if dataset_api['nb101_data'].is_valid(spec):
                break
                
        self.set_spec({'matrix':matrix, 'ops':ops})

    def mutate(self, parent, dataset_api, edits=1):
        """
        This will mutate the parent architecture spec.
        Code inspird by https://github.com/google-research/nasbench
        """
        parent_spec = parent.get_spec()
        spec = copy.deepcopy(parent_spec)
        matrix, ops = spec['matrix'], spec['ops']
        
        for _ in range(edits):
            while True:
                if np.random.random() < 0.5:
                    for src in range(0, NUM_VERTICES - 1):
                        for dst in range(src+1, NUM_VERTICES):
                            matrix[src][dst] = 1 - matrix[src][dst]
                else:
                    for ind in range(1, NUM_VERTICES - 1):
                        available = [op for op in OPS if op != ops[ind]]
                        ops[ind] = np.random.choice(available)

                new_spec = dataset_api['api'].ModelSpec(matrix, ops)
                if dataset_api['nb101_data'].is_valid(new_spec):
                    break
        
        self.set_spec({'matrix':matrix, 'ops':ops})

    def get_nbhd(self, dataset_api=None):
        # return all neighbors of the architecture
        spec = self.get_spec()
        matrix, ops = spec['matrix'], spec['ops']
        nbhd = []
        
        def add_to_nbhd(new_matrix, new_ops, nbhd):
            new_spec = {'matrix':new_matrix, 'ops':new_ops}
            model_spec = dataset_api['api'].ModelSpec(new_matrix, new_ops)
            if dataset_api['nb101_data'].is_valid(model_spec):
                nbr = NasBench101SearchSpace()
                nbr.set_spec(new_spec)
                nbr_model = torch.nn.Module()
                nbr_model.arch = nbr
                nbhd.append(nbr_model)
            return nbhd
        
        # add op neighbors
        for vertex in range(1, OP_SPOTS + 1):
            if is_valid_vertex(matrix, vertex):
                available = [op for op in OPS if op != ops[vertex]]
                for op in available:
                    new_matrix = copy.deepcopy(matrix)
                    new_ops = copy.deepcopy(ops)
                    new_ops[vertex] = op
                    nbhd = add_to_nbhd(new_matrix, new_ops, nbhd)

        # add edge neighbors
        for src in range(0, NUM_VERTICES - 1):
            for dst in range(src+1, NUM_VERTICES):
                new_matrix = copy.deepcopy(matrix)
                new_ops = copy.deepcopy(ops)
                new_matrix[src][dst] = 1 - new_matrix[src][dst]
                new_spec = {'matrix':new_matrix, 'ops':new_ops}
            
                if matrix[src][dst] and is_valid_edge(matrix, (src, dst)):
                    nbhd = add_to_nbhd(new_matrix, new_ops, nbhd)

                if not matrix[src][dst] and is_valid_edge(new_matrix, (src, dst)):
                    nbhd = add_to_nbhd(new_matrix, new_ops, nbhd)

        random.shuffle(nbhd)
        return nbhd

    def get_type(self):
        return 'nasbench101'
    
def _set_node_ops(current_edge_data, C):
    ops = [
        ReLUConvBN(C, C, kernel_size=1),
        # ops.Zero(stride=1),    #! recheck about the hardcoded second operation
        ReLUConvBN(C, C, kernel_size=3),
        MaxPool1x1(kernel_size=3, stride=1),
    ]
    current_edge_data['op'] = ops

def _set_cell_ops(edge, C):
    edge.data.set('op', [
        ops.Identity(),
        ops.Zero(stride=1), 
    ])
    

def get_utilized(matrix):
    # return the sets of utilized edges and nodes
    # first, compute all paths
    n = np.shape(matrix)[0]
    sub_paths = []
    for j in range(0, n):
        sub_paths.append([[(0, j)]]) if matrix[0][j] else sub_paths.append([])
    
    # create paths sequentially
    for i in range(1, n - 1):
        for j in range(1, n):
            if matrix[i][j]:
                for sub_path in sub_paths[i]:
                    sub_paths[j].append([*sub_path, (i, j)])
    paths = sub_paths[-1]

    utilized_edges = []
    for path in paths:
        for edge in path:
            if edge not in utilized_edges:
                utilized_edges.append(edge)

    utilized_nodes = []
    for i in range(NUM_VERTICES):
        for edge in utilized_edges:
            if i in edge and i not in utilized_nodes:
                utilized_nodes.append(i)

    return utilized_edges, utilized_nodes

def num_edges_and_vertices(matrix):
    # return the true number of edges and vertices
    edges, nodes = self.get_utilized(matrix)
    return len(edges), len(nodes)

def is_valid_vertex(matrix, vertex):
    edges, nodes = get_utilized(matrix)
    return (vertex in nodes)

def is_valid_edge(matrix, edge):
    edges, nodes = get_utilized(matrix)
    return (edge in edges)