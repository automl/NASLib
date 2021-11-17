#from hpbandster.core.base_iteration import BaseIteration
import numpy as np
import numpy as np
import torch

import math

from naslib.optimizers.core.metaclasses import MetaOptimizer
from naslib.search_spaces.core.query_metrics import Metric


class SuccessiveHalving(MetaOptimizer):
    """
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
        super(SuccessiveHalving, self).__init__()
        self.weight_optimizer = weight_optimizer
        self.loss = loss_criteria
        self.grad_clip = grad_clip

        self.performance_metric = Metric.VAL_ACCURACY
        self.dataset = config.dataset

        self.fidelity = config.search.min_fidelity
        self.number_archs = config.search.number_archs
        self.eta = config.search.eta
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.budget_type = config.search.budget_type #is not for one query is overall
        self.fidelity_counter = 0
        self.sampled_archs = []
        self.history = torch.nn.ModuleList()
        #self.end = False

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

        model = torch.nn.Module()  # hacky way to get arch and accuracy checkpointable
        model.arch = self.search_space.clone()
        if len(self.sampled_archs) < self.number_archs:
            model.arch.sample_random_architecture(dataset_api=self.dataset_api)
        else:
            model = self.sampled_archs[self.fidelity_counter]
       
# 		    return(ranks < self.num_configs[self.stage])


        # DONE: define fidelity in multi-fidelity setting
        model.accuracy = model.arch.query(
            self.performance_metric,
            self.dataset,
            epoch=self.fidelity, # DONE: adapt this
            dataset_api=self.dataset_api,
        )

        budget = 1
        # TODO: make query type secure
        if self.budget_type == 'time':
            # TODO: make dependent on performance_metric
            model.time = model.arch.query( # TODO: this is the time for training from screatch.
                Metric.TRAIN_TIME,
                self.dataset,
                epoch=self.fidelity, # DONE: adapt this
                dataset_api=self.dataset_api,
            )
            budget = model.time
        elif not(self.budget_type == "epoch"):
            raise NameError("budget time should be time or epoch")
        # TODO: make this more beautiful/more efficient
        # TODO: we may need to track of all ever sampled archs
        if len(self.sampled_archs) < self.number_archs:
            self.sampled_archs.append(model)
        else:
            self.sampled_archs[self.fidelity_counter] = model
        
        self.fidelity_counter += 1
        # TODO: fidelity is changed for new epoch, what make the wrong values in the dictonary
        self._update_history(model)
        if self.fidelity_counter == self.number_archs:
            self.fidelity = math.floor(self.eta*self.fidelity) #
            self.sampled_archs.sort(key = lambda model: model.accuracy, reverse= True)
            if(math.floor(self.number_archs/self.eta)) != 0:
                self.sampled_archs = self.sampled_archs[0:math.floor(self.number_archs/self.eta)] #DONE round
            else:
                #TODO: here maybe something back for hyperand 
                #self.end = True
                self.sampled_archs = [self.sampled_archs[0]]  #but maybe there maybe a different way
            self.number_archs = len(self.sampled_archs)
            self.fidelity_counter = 0
        return budget
        # required if we want to train the models and not only query.
        # architecture_i.parse()
        # architecture_i.train()
        # architecture_i = architecture_i.to(self.device)
        # self.sampled_archs.append(architecture_i)
        # self.weight_optimizers.append(self.weight_optimizer(architecture_i.parameters(), 0.01))

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
                Metric.TRAIN_ACCURACY, self.dataset, dataset_api=self.dataset_api, epoch=self.fidelity
            ),
            best_arch.query(
                Metric.VAL_ACCURACY, self.dataset, dataset_api=self.dataset_api, epoch=self.fidelity
            ),
            best_arch.query(
                Metric.TEST_ACCURACY, self.dataset, dataset_api=self.dataset_api, epoch=self.fidelity
            ),
            best_arch.query(
                Metric.TRAIN_TIME, self.dataset, dataset_api=self.dataset_api, epoch=self.fidelity
            ),
            self.fidelity,
            best_arch_hash,
        )

    def test_statistics(self):
        best_arch = self.get_final_architecture()
        return best_arch.query(Metric.RAW, self.dataset, dataset_api=self.dataset_api)

    def get_op_optimizer(self):
        return self.weight_optimizer
    #TODO discuss about this 
    #def get_end(self):
    #    return self.end

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
