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

from naslib.search_spaces.nasbench201.conversions import convert_naslib_to_str

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
        self._epsilon = float(self.config.search.epsilon) 
        
        times_of_split  = math.floor(math.log(self.max_budget / self.min_budget, self.eta)  + self._epsilon )
        # set up round sizes, fidelities, and list of arches
        
        n = math.ceil((times_of_split + 1) * self.eta ** times_of_split / (times_of_split + 1) + self._epsilon) # initial number of configurations
        r = int(self.max_budget / self.eta**times_of_split) # initial number of iterations to run configurations for
        for i in range(times_of_split + 1):
           
            self.round_sizes.append(n)
            self.fidelities.append(r)
            # n= 2*3 ** 1/2  for i in range(s + 1):
            n = math.floor(n / self.eta)
            # TODO: maybe this can be replaced by search space get_max_epoch()
            r = min(math.floor(r * self.eta), config.search.fidelity)
        
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
            del self.current_round
            del  self.next_round
            del self.round_number
            del self.prev_round
            del self.process
            self.clean_history()
            self.current_round = []
            self.next_round = []
            self.round_number = 0
            self.prev_round = 0
            self.process = i

        
        if epoch < self.round_sizes[self.round_number]:
            # sample random architectures
            model = torch.nn.Module()   # hacky way to get arch and accuracy checkpointable
            model.arch = self.search_space.clone()
            model.arch.sample_random_architecture(dataset_api=self.dataset_api)   
            model.epoch = self.fidelities[self.round_number]
            model.accuracy = model.arch.query(self.performance_metric,
                                              self.dataset, 
                                              epoch=model.epoch, 
                                              dataset_api=self.dataset_api)
            
            self._update_history(model)
            self.next_round.append(model)

        else:
            if len(self.current_round) == 0:
                # if we are at the end of a round of hyperband, continue training only the best 
                logger.info("Starting a new round: continuing to train the best arches")
                self.round_number += 1
                cutoff = self.round_sizes[self.round_number]
                self.current_round = sorted(self.next_round, key=lambda x: -x.accuracy)[:cutoff]
                del self.next_round
                self.next_round = []


            # train the next architecture
            model = self.current_round.pop()
            """
            Note: technically we would just continue training this arch, but right now,
            just for simplicity, we treat it as if we start to train it again from scratch
            """
            model = copy.deepcopy(model)
            model.epoch = self.fidelities[self.round_number]
            model.accuracy = model.arch.query(self.performance_metric,
                                              self.dataset, 
                                              epoch=model.epoch, 
                                              dataset_api=self.dataset_api)
            self._update_history(model)
            self.next_round.append(model)
        logger.info("fidelity: {}".format(self.fidelities[self.round_number]))

    def _update_history(self, child):
        self.history.append(child)

    def clean_history(self):
        best_arch = max(self.history, key=lambda x: x.accuracy)
        self.history = torch.nn.ModuleList()
        self.history.append(best_arch)

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
        train_time = latest_arch.query(Metric.TRAIN_TIME, self.dataset, dataset_api=self.dataset_api, epoch=latest_arch_epoch)
        previous_train_time = latest_arch.query(Metric.TRAIN_TIME, self.dataset, dataset_api=self.dataset_api, epoch=self.fidelities[self.round_number - 1]) if self.round_number > 0 else 0
        train_time = train_time - previous_train_time
        return (
            best_arch.query(Metric.TRAIN_ACCURACY, self.dataset, dataset_api=self.dataset_api, epoch=best_arch_epoch-1), 
            best_arch.query(Metric.VAL_ACCURACY, self.dataset, dataset_api=self.dataset_api, epoch=best_arch_epoch), 
            best_arch.query(Metric.TEST_ACCURACY, self.dataset, dataset_api=self.dataset_api, epoch=best_arch_epoch), 
            train_time, 
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