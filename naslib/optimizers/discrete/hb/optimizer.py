#from hpbandster.core.base_iteration import BaseIteration
import numpy as np
import numpy as np
import torch

import math
from naslib.optimizers import SuccessiveHalving as SH
from naslib.optimizers.core.metaclasses import MetaOptimizer
from naslib.search_spaces.core.query_metrics import Metric


class HyperBand(MetaOptimizer):
    """
    This is right now only a little bit more as pseudocode, so it not runnable what is currently working on
    # TODO: Write fancy comment 🌈
    """

    # training the models is not implemented
    using_step_function = False

    def __init__(
        self,
        config,
        weight_optimizer=torch.optim.SGD,
        loss_criteria=torch.nn.CrossEntropyLoss(),
        grad_clip=None,
    ):
        """
        Initialize a Successive Halving  optimizer.

        Args:
            config
            weight_optimizer (torch.optim.Optimizer): The optimizer to
                train the (convolutional) weights.
            loss_criteria (TODO): The loss
            grad_clip (float): Where to clip the gradients (default None).
        """
        super(HyperBand, self).__init__()
        self.weight_optimizer = weight_optimizer
        self.loss = loss_criteria
        self.grad_clip = grad_clip

        self.performance_metric = Metric.VAL_ACCURACY
        self.dataset = config.dataset

        #self.fidelit_min = config.search.min_fidelity
        self.budget_max = config.search.budget_max
        self.budget_min =  config.search.budget_min
        self.number_archs = config.search.number_archs
        self.eta = config.search.eta
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.budget_type = config.search.budget_type #is not for one query is overall
        self.sampled_archs = []
        self.history = torch.nn.ModuleList()
        self.s_max = math.floor(math.log(self.eta)*self.budget_max/self.budget_min)
        self.s =  self.s_max

    def adapt_search_space(self, search_space, scope=None, dataset_api=None):
        assert (
            search_space.QUERYABLE
        ), "Successsive Halving is currently only implemented for benchmarks."
        self.search_space = search_space.clone()
        self.scope = scope if scope else search_space.OPTIMIZER_SCOPE
        self.dataset_api = dataset_api

    def new_epoch(self, e):
        """
        Sample a new architecture to train.
        """
        if self.s > 0:
            if sh.end == True:   #if sh is finish go to something diffrent as initial budget 
                n = math.ceil((self.s_max +1 )/ (self.s +1 )* self.eta**self.s)
                sh = SH(config) #should be in config 
                self.s -= 1
            budget = sh.new_epoch()
        return budget


        
        
    def _update_history(self, child):
        if len(self.history) < 100:
            self.history.append(child)
        else:
            for i, p in enumerate(self.history):
                if child.accuracy > p.accuracy:
                    self.history[i] = child
                    break

    def get_final_architecture(self):
        """
        Returns the sampled architecture with the lowest validation error.
        """
        return max(self.sampled_archs, key=lambda x: x.accuracy).arch

    def train_statistics(self, report_incumbent=True):

        if report_incumbent:
            best_arch = self.get_final_architecture()
        else:
            best_arch = self.sampled_archs[-1].arch

        return (
            best_arch.query(
                Metric.TRAIN_ACCURACY, self.dataset, dataset_api=self.dataset_api
            ),
            best_arch.query(
                Metric.VAL_ACCURACY, self.dataset, dataset_api=self.dataset_api
            ),
            best_arch.query(
                Metric.TEST_ACCURACY, self.dataset, dataset_api=self.dataset_api
            ),
            best_arch.query(
                Metric.TRAIN_TIME, self.dataset, dataset_api=self.dataset_api
            ),

        )
    def train_model_statistics(self, report_incumbent=True):

        
        best_arch = self.sampled_archs[self.fidelity_counter -1].arch
        best_arch_hash = hash(self.sampled_archs[self.fidelity_counter -1])
        return (
            best_arch.query(
                Metric.TRAIN_ACCURACY, self.dataset, dataset_api=self.dataset_api
            ),
            best_arch.query(
                Metric.VAL_ACCURACY, self.dataset, dataset_api=self.dataset_api
            ),
            best_arch.query(
                Metric.TEST_ACCURACY, self.dataset, dataset_api=self.dataset_api
            ),
            best_arch.query(
                Metric.TRAIN_TIME, self.dataset, dataset_api=self.dataset_api
            ),
            self.fidelity,
            best_arch_hash,
        )

    def test_statistics(self):
        best_arch = self.get_final_architecture()
        return best_arch.query(Metric.RAW, self.dataset, dataset_api=self.dataset_api)

    def get_op_optimizer(self):
        return self.weight_optimizer

    def get_checkpointables(self):
        return {"model": self.history}

# #from naslib.optimizers.core.metaclasses import MetaOptimizer
# class SuccessiveHalving(MetaOptimizer):
#     #also import or simular in NASLib?

# 	def _advance_to_next_stage(self, config_ids, losses):
# 		"""
# 			#SuccessiveHalving simply continues the best based on the current loss.
# 		"""
# 		ranks = np.argsort(np.argsort(losses))
# 		return(ranks < self.num_configs[self.stage])
