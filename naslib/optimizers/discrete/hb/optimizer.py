import collections
import copy
import logging
import math
import numpy as np
import torch

from naslib.optimizers.core.metaclasses import MetaOptimizer
from naslib.search_spaces.core.query_metrics import Metric
from naslib.utils.logging import log_every_n_seconds
from naslib.utils.utils import AttrDict, count_parameters_in_MB

logger = logging.getLogger(__name__)


class HB(MetaOptimizer):
    """
    This implementation is mainly based on https://github.com/automl/nas-bench-x11. 
    
    Li, Lisha, et al. 
    Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization. 
    arXiv:1603.06560, arXiv, 18th June 2018. 
    arXiv.org, http://arxiv.org/abs/1603.06560.
    """
    # training the models is not implemented
    using_step_function = False

    def __init__(self, config):
        super().__init__()
        # Hyperband related stuff
        self.config = config
        self.rounds = []
        self.round_sizes = []
        self.fidelities = []
        self.max_budget = self.config.search.max_budget
        self.min_budget = self.config.search.min_budget
        self.eta = self.config.search.eta
        self._epsilon = float(self.config.search.epsilon)
        self.min_budget = min(self.min_budget, self.max_budget)
        s_max = math.floor(math.log(self.max_budget / self.min_budget, self.eta) + self._epsilon)
        # set up round sizes, fidelities, and list of arches
        for s in reversed(range(s_max + 1)):
            self.rounds.append(s)
            round_sizes = []
            fidelities = []
            n = math.ceil((s_max + 1) * self.eta ** s / (s + 1) - self._epsilon)  # initial number of configurations
            r = self.max_budget / self.eta ** s  # initial number of iterations to run configurations for
            for i in range(s + 1):
                n_i = math.floor(n / self.eta ** i + self._epsilon)
                r_i = min(math.floor(r * self.eta ** i + self._epsilon), config.search.fidelity)
                round_sizes.append(n_i)
                fidelities.append(r_i)
            self.round_sizes.append(round_sizes)
            self.fidelities.append(fidelities)

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
        return self.round_sizes, self.rounds[::-1]

    def new_epoch(self, epoch, round, i):
        # round - bracket
        # epoch - number of architectures in bracket so far
        if self.process < i:  # re-init for each new process
            del self.current_round
            del self.next_round
            del self.round_number
            del self.prev_round
            del self.process
            self.current_round = []
            self.next_round = []
            self.round_number = 0  # index to fidelity
            self.prev_round = 0
            self.process = i
            self.clean_history()

        if self.prev_round < round:  # reset round_number for each new round # check if new bracket
            self.prev_round = round
            self.round_number = 0

        if epoch < self.round_sizes[round][0]:  # check if first fidelity of bracket
            # sample random architectures
            model = torch.nn.Module()  # hacky way to get arch and accuracy checkpointable
            model.arch = self.search_space.clone()
            model.arch.sample_random_architecture(dataset_api=self.dataset_api)
            model.epoch = self.fidelities[round][0]
            model.accuracy = model.arch.query(self.performance_metric,
                                              self.dataset,
                                              epoch=model.epoch,
                                              dataset_api=self.dataset_api)
            self._update_history(model)
            self.next_round.append(model)

        else:
            if len(self.current_round) == 0:  # fidelity is full
                # if we are at the end of a round of hyperband, continue training only the best
                logger.info("Starting a new round: continuing to train the best arches")
                self.round_number += 1
                cutoff = self.round_sizes[round][self.round_number]
                self.current_round = sorted(self.next_round, key=lambda x: -x.accuracy)[:cutoff]
                self.next_round = []

            # train the next architecture
            model = self.current_round.pop()  # architecture to train
            """
            Note: technically we would just continue training this arch, but right now,
            just for simplicity, we treat it as if we start to train it again from scratch
            """
            model = copy.deepcopy(model)
            model.epoch = self.fidelities[round][self.round_number]
            model.accuracy = model.arch.query(self.performance_metric,
                                              self.dataset,
                                              epoch=model.epoch,
                                              dataset_api=self.dataset_api)
            self._update_history(model)
            self.next_round.append(model)

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
        train_time = latest_arch.query(Metric.TRAIN_TIME, self.dataset, dataset_api=self.dataset_api,
                                       epoch=latest_arch_epoch)
        previous_train_time = latest_arch.query(Metric.TRAIN_TIME, self.dataset, dataset_api=self.dataset_api,
                                                epoch=self.fidelities[self.prev_round][
                                                    self.round_number - 1]) if self.round_number > 0 else 0
        train_time = train_time - previous_train_time
        return (
            best_arch.query(Metric.TRAIN_ACCURACY, self.dataset, dataset_api=self.dataset_api,
                            epoch=best_arch_epoch - 1),
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
