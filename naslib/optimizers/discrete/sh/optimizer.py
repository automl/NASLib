import collections
import logging
import math
import torch
import copy
import numpy as np

from naslib.optimizers.core.metaclasses import MetaOptimizer

from naslib.search_spaces.core.query_metrics import Metric

from naslib.utils.utils import AttrDict, count_parameters_in_MB
from naslib.utils.logging import log_every_n_seconds

logger = logging.getLogger(__name__)
    
        
import collections
import logging
import math
import torch
import copy
import numpy as np

from naslib.optimizers.core.metaclasses import MetaOptimizer

from naslib.search_spaces.core.query_metrics import Metric

from naslib.utils.utils import AttrDict, count_parameters_in_MB
from naslib.utils.logging import log_every_n_seconds

logger = logging.getLogger(__name__)
    
        
class  SuccessiveHalving(MetaOptimizer):
    
    # training the models is not implemented
    using_step_function = False
    
    def __init__(self, config):
        super().__init__()
        # Hyperband related stuff
        self.config = config
        self.round_sizes = []
        self.fidelities = []
        self.max_budget = self.config.search.max_budget
        self.min_budget = self.config.search.min_budget
        self.eta = self.config.search.eta 
        
        times_of_split  = math.floor(math.log(self.max_budget / self.min_budget, self.eta) )
        # set up round sizes, fidelities, and list of arches
        
        n = math.ceil((times_of_split + 1) * self.eta ** times_of_split / (times_of_split + 1)) # initial number of configurations
        r = self.max_budget / self.eta**times_of_split # initial number of iterations to run configurations for
        for i in range(times_of_split):
           
            # n= 2*3 ** 1/2  for i in range(s + 1):
            n = math.floor(n / self.eta)
            r = min(math.floor(r * self.eta), config.search.fidelity)
            self.round_sizes.append(n)
            self.fidelities.append(r)
        
        self.performance_metric = Metric.VAL_ACCURACY
        self.dataset = config.dataset
        self.history = torch.nn.ModuleList()

        self.epochs = self.compute_epochs()
        self.current_round = []
        self.next_round = []
        self.round_number = 0
        self.prev_round = 0
        self.process = 0

    def adapt_search_space(self, search_space, scope=None, dataset_api=None):
        assert search_space.QUERYABLE, "Hyperband_simple is currently only implemented for benchmarks."
        self.search_space = search_space.clone()
        self.scope = scope if scope else search_space.OPTIMIZER_SCOPE
        self.dataset_api = dataset_api
        self.max_training_epoch = self.search_space.get_max_epochs()

    def compute_epochs(self):
        return [self.round_sizes], [0]

    def new_epoch(self, epoch, round, i):
        if self.process < i: # re-init for each new process
            self.current_round = []
            self.next_round = []
            self.round_number = 0
            self.prev_round = 0
            self.process = i

        if self.prev_round < round:  # reset round_number for each new round
            self.prev_round = round
            self.round_number = 0

        if epoch < self.round_sizes[round]:
            # sample random architectures
            model = torch.nn.Module()   # hacky way to get arch and accuracy checkpointable
            model.arch = self.search_space.clone()
            model.arch.sample_random_architecture(dataset_api=self.dataset_api)   
            model.epoch = self.fidelities[round]
            model.accuracy = model.arch.query(self.performance_metric,
                                              self.dataset, 
                                              epoch=model.epoch, 
                                              dataset_api=self.dataset_api)
            self._update_history(model)
            self.next_round.append(model)

        else:
            # train the next architecture
            model = self.current_round.pop()
            """
            Note: technically we would just continue training this arch, but right now,
            just for simplicity, we treat it as if we start to train it again from scratch
            """
            model = copy.deepcopy(model)
            model.epoch = self.fidelities[round]
            model.accuracy = model.arch.query(self.performance_metric,
                                              self.dataset, 
                                              epoch=model.epoch, 
                                              dataset_api=self.dataset_api)
            self._update_history(model)
            self.next_round.append(model)

    def _update_history(self, child):
        self.history.append(child)

    def get_final_architecture(self):
        
        # Returns the sampled architecture with the lowest validation error.
        best_arch = max(self.history, key=lambda x: x.accuracy)
        return best_arch.arch, best_arch.epoch

    def get_latest_architecture(self):

        # Returns the architecture from the most recent epoch
        latest_arch = self.history[-1]
        return latest_arch.arch, latest_arch.epoch

    def train_statistics(self):
        best_arch, best_arch_epoch = self.get_final_architecture()
        latest_arch, latest_arch_epoch = self.get_latest_architecture()
        return (
            best_arch.query(Metric.TRAIN_ACCURACY, self.dataset, dataset_api=self.dataset_api, epoch=best_arch_epoch-1), 
            best_arch.query(Metric.VAL_ACCURACY, self.dataset, dataset_api=self.dataset_api, epoch=best_arch_epoch), 
            best_arch.query(Metric.TEST_ACCURACY, self.dataset, dataset_api=self.dataset_api, epoch=best_arch_epoch), 
            latest_arch.query(Metric.TRAIN_TIME, self.dataset, dataset_api=self.dataset_api, epoch=latest_arch_epoch), 
        )
    
    def test_statistics(self):
        best_arch, epoch = self.get_final_architecture()
        return best_arch.query(Metric.RAW, self.dataset, dataset_api=self.dataset_api, epoch=epoch)

    
    def get_op_optimizer(self):
        raise NotImplementedError()

    def get_checkpointables(self):
        return {'model': self.history}

    def get_model_size(self):
        return count_parameters_in_MB(self.history)
