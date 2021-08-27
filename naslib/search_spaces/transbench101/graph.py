import os
import pickle
import numpy as np
import random
import torch
import torch.nn as nn

from naslib.search_spaces.core import primitives as ops
from naslib.search_spaces.core.graph import Graph, EdgeData
from naslib.search_spaces.core.primitives import AbstractPrimitive
from naslib.search_spaces.core.query_metrics import Metric
from naslib.search_spaces.transbench101.conversions import convert_op_indices_to_naslib, \
convert_naslib_to_op_indices, convert_naslib_to_str, convert_naslib_to_transbench101_micro, convert_naslib_to_transbench101_macro #, convert_naslib_to_tb101

from naslib.utils.utils import get_project_root

from .primitives import ResNetBasicblock


OP_NAMES = ['Identity', 'Zero', 'ReLUConvBN3x3', 'ReLUConvBN1x1']


class TransBench101SearchSpace(Graph):
    """
    Implementation of the transbench 101 search space.
    It also has an interface to the tabular benchmark of transbench 101.
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
        self.op_indices = None

        self.max_epoch = 199
        self.space_name = 'transbench101'
        self.space = 'micro'
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
                update_func=lambda edge: _set_cell_ops(edge, C=c),
                scope=scope,
                private_edge_data=True
            )
        
    def query(self, metric=None, dataset=None, path=None, epoch=-1, full_lc=False, dataset_api=None):
        """
        Query results from transbench 101
        """
        assert isinstance(metric, Metric)
        if metric == Metric.ALL:
            raise NotImplementedError()
#         if metric != Metric.RAW and metric != Metric.ALL:
#             assert dataset in ['cifar10', 'cifar100', 'ImageNet16-120'], "Unknown dataset: {}".format(dataset)
        if dataset_api is None:
            raise NotImplementedError('Must pass in dataset_api to query transbench101')
            
            
        if self.space=='micro':
            arch_str = convert_naslib_to_transbench101_micro(self.op_indices) 
        elif self.space=='macro':
            arch_str = convert_naslib_to_transbench101_macro(self.op_indices) 
          
        query_results = dataset_api['api']
        task = dataset_api['task']
                
        
        if task in ['class_scene', 'class_object', 'jigsaw']:
            
            metric_to_tb101 = {
                Metric.TRAIN_ACCURACY: 'train_top1',
                Metric.VAL_ACCURACY: 'valid_top1',
                Metric.TEST_ACCURACY: 'test_top1',
                Metric.TRAIN_LOSS: 'train_loss',
                Metric.VAL_LOSS: 'valid_loss',
                Metric.TEST_LOSS: 'test_loss',
                Metric.TRAIN_TIME: 'time_elapsed',
            }

        elif task == 'room_layout':
            
            metric_to_tb101 = {
                Metric.TRAIN_ACCURACY: 'train_neg_loss',
                Metric.VAL_ACCURACY: 'valid_neg_loss',
                Metric.TEST_ACCURACY: 'test_neg_loss',
                Metric.TRAIN_LOSS: 'train_loss',
                Metric.VAL_LOSS: 'valid_loss',
                Metric.TEST_LOSS: 'test_loss',
                Metric.TRAIN_TIME: 'time_elapsed',
            }
            
        elif task == 'segmentsemantic':
            
            metric_to_tb101 = {
                Metric.TRAIN_ACCURACY: 'train_acc',
                Metric.VAL_ACCURACY: 'valid_acc',
                Metric.TEST_ACCURACY: 'test_acc',
                Metric.TRAIN_LOSS: 'train_loss',
                Metric.VAL_LOSS: 'valid_loss',
                Metric.TEST_LOSS: 'test_loss',
                Metric.TRAIN_TIME: 'time_elapsed',
            }    
        
        else: # ['normal', 'autoencoder']
            
            metric_to_tb101 = {
                Metric.TRAIN_ACCURACY: 'train_ssim',
                Metric.VAL_ACCURACY: 'valid_ssim',
                Metric.TEST_ACCURACY: 'test_ssim',
                Metric.TRAIN_LOSS: 'train_loss',
                Metric.VAL_LOSS: 'valid_loss',
                Metric.TEST_LOSS: 'test_loss',
                Metric.TRAIN_TIME: 'time_elapsed',
            }
        

        
        if metric == Metric.RAW:
            # return all data
            return query_results.get_arch_result(arch_str).query_all_results()[task]


        if metric == Metric.HP:
            # return hyperparameter info
            return query_results[dataset]['cost_info']
        elif metric == Metric.TRAIN_TIME:
            return query_results.get_single_metric(arch_str, task, metric_to_tb101[metric])


        if full_lc and epoch == -1:
            return query_results[dataset][metric_to_tb101[metric]]
        elif full_lc and epoch != -1:
            return query_results[dataset][metric_to_tb101[metric]][:epoch]
        else:
            return query_results.get_single_metric(arch_str, task, metric_to_tb101[metric])

        
    def get_op_indices(self):
        if self.op_indices is None:
            self.op_indices = convert_naslib_to_op_indices(self)
        return self.op_indices
 

    def get_hash(self):
        return tuple(self.get_op_indices())

    
    def set_op_indices(self, op_indices):
        # This will update the edges in the naslib object to op_indices
        self.op_indices = op_indices
#         convert_op_indices_to_naslib(op_indices, self)


    def sample_random_architecture_micro(self, dataset_api=None):
        """
        This will sample a random architecture and update the edges in the
        naslib object accordingly.
        """
        op_indices = np.random.randint(4, size=(6))
        self.set_op_indices(op_indices)

        
    def sample_random_architecture_macro(self, dataset_api=None):
        """
        This will sample a random architecture and update the edges in the
        naslib object accordingly.
        """
        r = random.randint(0, 2)
        p = random.randint(1, 4)
        q = random.randint(1, 3)
        u = [2*int(i<p) for i in range(r+4)]
        v = [int(i<q) for i in range(r+4)]
        w = [1+sum(x) for x in zip(u, v)]
        op_indices = np.random.permutation(w)
        while len(op_indices)<6:
            op_indices = np.append(op_indices, 0)
        self.set_op_indices(op_indices)
    
    
    def sample_random_architecture(self, dataset_api=None):
        if self.space=='micro':
            self.sample_random_architecture_micro(dataset_api)
        elif self.space=='macro':
            self.sample_random_architecture_macro(dataset_api)        
        

    def mutate_micro(self, parent, dataset_api=None):
        """
        This will mutate one op from the parent op indices, and then
        update the naslib object and op_indices
        """
        parent_op_indices = parent.get_op_indices()
        op_indices = parent_op_indices

        edge = np.random.choice(len(parent_op_indices))
        available = [o for o in range(len(OP_NAMES)) if o != parent_op_indices[edge]]
        op_index = np.random.choice(available)
        op_indices[edge] = op_index
        self.set_op_indices(op_indices)



    def mutate_macro(self, parent, dataset_api=None):
        """
        This will mutate one op from the parent op indices, and then
        update the naslib object and op_indices
        """
        parent_op_indices = parent.get_op_indices()
        parent_op_ind = parent_op_indices[parent_op_indices!=0]
        
        def f(g):
            r = len(g)
            p = sum([int(i==4 or i==3) for i in g])
            q = sum([int(i==4 or i==2) for i in g])
            return r, p, q

        def g(r, p, q):
            u = [2*int(i<p) for i in range(r)]
            v = [int(i<q) for i in range(r)]
            w = [1+sum(x) for x in zip(u, v)]
            return np.random.permutation(w)

        a, b, c = f(parent_op_ind)

        a_available = [i for i in [4, 5, 6] if i!=a]
        b_available = [i for i in range(1, 5) if i!=b]
        c_available = [i for i in range(1, 4) if i!=c]
        
        dic1 = {1: a, 2: b, 3: c}
        dic2 = {1: a_available, 2: b_available, 3: c_available}
        
        numb = random.randint(1, 3)
        
        dic1[numb] = random.choice(dic2[numb])

       
        op_indices = g(dic1[1], dic1[2], dic1[3])
        while len(op_indices)<6:
            op_indices = np.append(op_indices, 0)
                                
        self.set_op_indices(op_indices)
    
    
    def mutate(self, parent, dataset_api=None):
        if self.space=='micro':
            self.mutate_micro(parent, dataset_api)
        elif self.space=='macro':
            self.mutate_macro(parent, dataset_api)
        


    def get_nbhd_micro(self, dataset_api=None):
        # return all neighbors of the architecture
        self.get_op_indices()
        nbrs = []
        for edge in range(len(self.op_indices)):
            available = [o for o in range(len(OP_NAMES)) if o != self.op_indices[edge]]
            
            for op_index in available:
                nbr_op_indices = self.op_indices.copy()
                nbr_op_indices[edge] = op_index
                nbr = TransBench101SearchSpace()
                nbr.set_op_indices(nbr_op_indices)
                nbr_model = torch.nn.Module()
                nbr_model.arch = nbr
                nbrs.append(nbr_model)
        
        random.shuffle(nbrs)
        return nbrs


    def get_nbhd_macro(self, dataset_api=None):
        # return all neighbors of the architecture
        self.get_op_indices()
        op_ind = self.op_indices[self.op_indices!=0]
        nbrs = []

        def f(g):
            r = len(g)
            p = sum([int(i==4 or i==3) for i in g])
            q = sum([int(i==4 or i==2) for i in g])
            return r, p, q

        def g(r, p, q):
            u = [2*int(i<p) for i in range(r)]
            v = [int(i<q) for i in range(r)]
            w = [1+sum(x) for x in zip(u, v)]
            return np.random.permutation(w)

        a, b, c = f(op_ind)

        a_available = [i for i in [4, 5, 6] if i!=a]
        b_available = [i for i in range(1, 5) if i!=b]
        c_available = [i for i in range(1, 4) if i!=c]

        for r in a_available:
            for p in b_available:
                for q in c_available:
                    nbr_op_indices = g(r, p, q)
                    while len(nbr_op_indices)<6:
                        nbr_op_indices = np.append(nbr_op_indices, 0)
                    nbr = TransBench101SearchSpace()
                    nbr.set_op_indices(nbr_op_indices)
                    nbr_model = torch.nn.Module()
                    nbr_model.arch = nbr
                    nbrs.append(nbr_model)

        random.shuffle(nbrs)
        return nbrs
    
    
    def get_nbhd(self, dataset_api=None):
        if self.space=='micro':
            self.get_nbhd_micro(dataset_api)
        elif self.space=='macro':
            self.get_nbhd_macro(dataset_api)
        
        


    def get_type(self):
#         return 'transbench101'
        return 'transbench101'

    
def _set_cell_ops(edge, C):
    edge.data.set('op', [
        ops.Identity(),
        ops.Zero(stride=1),
        ops.ReLUConvBN(C, C, kernel_size=3),
        ops.ReLUConvBN(C, C, kernel_size=1),
    ])
    
    

