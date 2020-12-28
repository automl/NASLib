import collections
import logging
import torch
import copy
import random
import numpy as np

from naslib.optimizers.core.metaclasses import MetaOptimizer
from naslib.optimizers.discrete.utils.utils import sample_random_architecture, update_ops

from naslib.search_spaces.core.query_metrics import Metric
from naslib.search_spaces.nasbench201.graph import NasBench201SearchSpace

from naslib.utils.utils import AttrDict, count_parameters_in_MB
from naslib.utils.logging import log_every_n_seconds

logger = logging.getLogger(__name__)


class LocalSearch(MetaOptimizer):
    
    # training the models is not implemented
    using_step_function = False
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.epochs = config.search.epochs
        
        self.performance_metric = Metric.VAL_ACCURACY
        self.dataset = config.dataset
        
        self.num_init = config.search.num_init
        self.nbhd = []
        self.chosen = None
        self.best_arch = None
        
        self.history = torch.nn.ModuleList()


    def adapt_search_space(self, search_space, scope=None):
        assert search_space.QUERYABLE, "Local search is currently only implemented for benchmarks."
        assert isinstance(search_space, NasBench201SearchSpace), "Local search is currently only \
        implemented for NasBench201SearchSpace"
        self.search_space = search_space.clone()
        self.scope = scope if scope else search_space.OPTIMIZER_SCOPE

    def get_nbhd(self, chosen):
        arch = chosen.arch
        nbrs = []
        
        cells = arch._get_child_graphs(single_instances=True)
        for i, cell in enumerate(cells):
            edges = [(u, v) for u, v, data in sorted(cell.edges(data=True)) if not data.is_final()]
            for edge in edges:

                # change op at edge
                data = cell.edges[edge]
                for op_index in range(len(data.primitives)):
                    nbr = arch.clone()
                    nbr_cells = nbr._get_child_graphs(single_instances=True)
                    nbr_data = nbr_cells[i].edges[edge]
                    nbr_data.set('op_index', op_index, shared=True)
                    nbr.update_edges(update_ops, nbr.OPTIMIZER_SCOPE, private_edge_data=True)
                    nbr_model = torch.nn.Module()
                    nbr_model.arch = nbr
                    nbrs.append(nbr_model)
        
        random.shuffle(nbrs)
        return nbrs
    
    def new_epoch(self, epoch):

        if epoch < self.num_init:
            logger.info("Start sampling architectures to fill the initial set")
            # If there is no scope defined, let's use the search space default one
            
            model = torch.nn.Module()   # hacky way to get arch and accuracy checkpointable
            model.arch = sample_random_architecture(self.search_space, self.scope)
            model.accuracy = model.arch.query(self.performance_metric, self.dataset)

            if not self.best_arch or model.accuracy > self.best_arch.accuracy:
                self.best_arch = model
            self._update_history(model)

        else:
            if len(self.nbhd) == 0 and self.chosen and self.best_arch.accuracy <= self.chosen.accuracy:
                logger.info('Reached local minimum. Starting from new random architecture.')
                model = torch.nn.Module()   # hacky way to get arch and accuracy checkpointable
                model.arch = sample_random_architecture(self.search_space, self.scope)
                model.accuracy = model.arch.query(self.performance_metric, self.dataset)
                self.chosen = model
                self.best_arch = model
                self.nbhd = self.get_nbhd(self.chosen)

            else:
                if len(self.nbhd) == 0:
                    logger.info('Start a new iteration. Pick the best architecture and evaluate its neighbors.')
                    self.chosen = self.best_arch
                    self.nbhd = self.get_nbhd(self.chosen)
                    
                model = self.nbhd.pop()
                model.accuracy = model.arch.query(self.performance_metric, self.dataset)

                if model.accuracy > self.best_arch.accuracy:
                    self.best_arch = model
                    logger.info('Found new best architecture.')
                self._update_history(model)           
                        
    def _update_history(self, child):
        if len(self.history) < 100:
            self.history.append(child)
        else:
            for i, p in enumerate(self.history):
                if child.accuracy > p.accuracy:
                    self.history[i] = child
                    break

    def train_statistics(self):
        best_arch = self.get_final_architecture()
        return (
            best_arch.query(Metric.TRAIN_ACCURACY, self.dataset), 
            best_arch.query(Metric.TRAIN_LOSS, self.dataset), 
            best_arch.query(Metric.VAL_ACCURACY, self.dataset), 
            best_arch.query(Metric.VAL_LOSS, self.dataset), 
            best_arch.query(Metric.TEST_ACCURACY, self.dataset), 
            best_arch.query(Metric.TEST_LOSS, self.dataset), 
        )

    def test_statistics(self):
        best_arch = self.get_final_architecture()
        return best_arch.query(Metric.RAW, self.dataset)


    def get_final_architecture(self):
        return max(self.history, key=lambda x: x.accuracy).arch
    
    
    def get_op_optimizer(self):
        raise NotImplementedError()

    
    def get_checkpointables(self):
        return {'model': self.history}
    

    def get_model_size(self):
        return count_parameters_in_MB(self.history)