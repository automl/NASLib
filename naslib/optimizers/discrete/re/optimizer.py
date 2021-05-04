import collections
import logging
import torch
import copy
import numpy as np

from naslib.optimizers.core.metaclasses import MetaOptimizer

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
        self.epochs = config.search.epochs
        self.sample_size = config.search.sample_size
        self.population_size = config.search.population_size

        self.performance_metric = Metric.VAL_ACCURACY
        self.dataset = config.dataset

        self.population = collections.deque(maxlen=self.population_size)
        self.history = torch.nn.ModuleList()


    def adapt_search_space(self, search_space, scope=None, dataset_api=None):
        assert search_space.QUERYABLE, "Regularized evolution is currently only implemented for benchmarks."
        self.search_space = search_space.clone()
        self.scope = scope if scope else search_space.OPTIMIZER_SCOPE
        self.dataset_api = dataset_api
        
    def new_epoch(self, epoch):
        # We sample as many architectures as we need 
        if epoch < self.population_size:
            logger.info("Start sampling architectures to fill the population")
            # If there is no scope defined, let's use the search space default one
            
            model = torch.nn.Module()   # hacky way to get arch and accuracy checkpointable
            model.arch = self.search_space.clone()
            model.arch.sample_random_architecture(dataset_api=self.dataset_api)        
            model.accuracy = model.arch.query(self.performance_metric, 
                                              self.dataset, 
                                              dataset_api=self.dataset_api)
            
            self.population.append(model)
            self._update_history(model)
            log_every_n_seconds(logging.INFO, "Population size {}".format(len(self.population)))
        else:
            sample = []
            while len(sample) < self.sample_size:
                candidate = np.random.choice(list(self.population))
                sample.append(candidate)
            
            parent = max(sample, key=lambda x: x.accuracy)

            child = torch.nn.Module()   # hacky way to get arch and accuracy checkpointable
            child.arch = self.search_space.clone()
            child.arch.mutate(parent.arch, dataset_api=self.dataset_api)
            child.accuracy = child.arch.query(self.performance_metric, 
                                              self.dataset, 
                                              dataset_api=self.dataset_api)

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
            best_arch.query(Metric.TRAIN_ACCURACY, self.dataset, dataset_api=self.dataset_api),
            best_arch.query(Metric.VAL_ACCURACY, self.dataset, dataset_api=self.dataset_api),
            best_arch.query(Metric.TEST_ACCURACY, self.dataset, dataset_api=self.dataset_api),
            best_arch.query(Metric.TRAIN_TIME, self.dataset, dataset_api=self.dataset_api),
        )

    def test_statistics(self):
        best_arch = self.get_final_architecture()
        return best_arch.query(Metric.RAW, self.dataset, dataset_api=self.dataset_api)


    def get_final_architecture(self):
        return max(self.history, key=lambda x: x.accuracy).arch
    
    
    def get_op_optimizer(self):
        raise NotImplementedError()

    
    def get_checkpointables(self):
        return {'model': self.history}
    

    def get_model_size(self):
        return count_parameters_in_MB(self.history)
