import collections
import logging
import torch
import copy
import numpy as np

from naslib.optimizers.core.metaclasses import MetaOptimizer
from naslib.optimizers.discrete.rs.optimizer import sample_random_architecture, update_ops

from naslib.search_spaces.core.query_metrics import Metric

from naslib.utils.utils import AttrDict, count_parameters_in_MB
from naslib.utils.logging import log_every_n_seconds

logger = logging.getLogger(__name__)


class RegularizedEvolution(MetaOptimizer):
    
    # training the models is not implemented
    using_step_function = False
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.epochs = config.epochs
        self.sample_size = config.sample_size
        self.population_size = config.population_size

        self.performance_metric = Metric.VAL_ACCURACY
        self.dataset = config.dataset

        self.population = collections.deque(maxlen=self.population_size)
        self.history = torch.nn.ModuleList()


    def adapt_search_space(self, search_space, scope=None):
        assert search_space.QUERYABLE, "Regularized evolution is currently only implemented for benchmarks."
        
        # We sample as many architectures as we need
        logger.info("Start sampling architectures to fill the population")
        while len(self.population) < self.population_size:
            # If there is no scope defined, let's use the search space default one
            if not scope:
                scope = search_space.OPTIMIZER_SCOPE
            
            model = torch.nn.Module()   # hacky way to get arch and accuracy checkpointable
            
            model.arch = sample_random_architecture(search_space, scope)
            model.accuracy = model.arch.query(self.performance_metric, self.dataset)
            
            self.population.append(model)
            self._update_history(model)
            log_every_n_seconds(logging.INFO, "Population size {}".format(len(self.population)))


    def _mutate(self, parent_arch):
        child = parent_arch.clone()
        
        # sample which cell/motif we want to mutate
        cells = child._get_child_graphs(single_instances=True)
        cell = np.random.choice(cells) if len(cells) > 1 else cells[0]
        
        edges = [(u, v) for u, v, data in sorted(cell.edges(data=True)) if not data.is_final()]

        # sample if op or edge change
        if np.random.choice(a=[False, True]):
            # change op
            random_edge = edges[np.random.choice(len(edges))]
            data = cell.edges[random_edge]
            op_index = np.random.randint(len(data.primitives))
            data.set('op_index', op_index, shared=True)
        else:
            # change edge by setting it to zero
            random_edge = edges[np.random.choice(len(edges))]
            cell.edges[random_edge].set('op_index', 1, shared=True)     # this is search space dependent

            random_edge = edges[np.random.choice(len(edges))]
            data = cell.edges[random_edge]
            op_index = np.random.randint(len(data.primitives))
            cell.edges[random_edge].set('op_index', op_index, shared=True)

        child.update_edges(update_ops, child.OPTIMIZER_SCOPE, private_edge_data=True)
        return child
    
    
    def new_epoch(self, epoch):
        sample = []
        while len(sample) < self.sample_size:
            candidate = np.random.choice(list(self.population))
            sample.append(candidate)
        
        parent = max(sample, key=lambda x: x.accuracy)

        child = torch.nn.Module()   # hacky way to get arch and accuracy checkpointable
        child.arch = self._mutate(parent.arch)
        child.accuracy = child.arch.query(self.performance_metric, self.dataset)

        self.population.append(child)
        self._update_history(child)
        
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