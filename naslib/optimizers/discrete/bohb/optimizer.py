import collections
import os
import math
import logging
import torch
import copy
import numpy as np

from naslib.optimizers.core.metaclasses import MetaOptimizer

from naslib.search_spaces.core.query_metrics import Metric
import statsmodels.api as sm

from naslib.utils.utils import AttrDict, count_parameters_in_MB
from naslib.utils.logging import log_every_n_seconds

logger = logging.getLogger(__name__)
    
        
class BOHB(MetaOptimizer):
    
    # training the models is not implemented
    using_step_function = False
    
    def __init__(self, config):
        super().__init__()
        # Hyperband related stuff
        self.config = config
        self.rounds = []
        self.round_sizes = []
        self.fidelities = []
        self.min_bandwidth = self.config.search.min_bandwith
        self.kde_models = dict()
        self._epsilon = float(self.config.search.epsilon)
        self.min_budget = self.config.search.min_budget
        self.max_budget = self.config.search.max_budget
        self.eta = self.config.search.eta
        self.min_points_in_model = self.config.search.min_points_in_model
        self.top_n_percent = self.config.search.top_n_percent
        s_max = math.floor(math.log(self.max_budget / self.min_budget, self.eta) + self._epsilon)
        # set up round sizes, fidelities, and list of arches
        for s in reversed(range(s_max + 1)):
            self.rounds.append(s)
            round_sizes = []
            fidelities = []
            n = math.ceil((s_max + 1) * self.eta ** s / (s + 1) - self._epsilon) # initial number of configurations
            r = self.max_budget / self.eta**s # initial number of iterations to run configurations for
            for i in range(s + 1):
                n_i = math.floor(n / self.eta ** i + self._epsilon)
                r_i = min(math.floor(r * self.eta ** i + self._epsilon), config.search.fidelity)
                round_sizes.append(n_i)
                fidelities.append(r_i)
            self.round_sizes.append(round_sizes)
            self.fidelities.append(fidelities)
        for budget in self.fidelities[0][1:]:
            budget = min(budget, config.search.fidelity)
            self.kde_models[budget] = {}
            self.kde_models[budget]['good'] = collections.deque(maxlen=300)
            self.kde_models[budget]['bad'] = collections.deque(maxlen=300)
            self.kde_models[budget]['minimize_kde'] = None
        self.performance_metric = Metric.VAL_ACCURACY
        self.dataset = config.dataset
        self.history = torch.nn.ModuleList()

        self.epochs = self.compute_epochs()
        self.current_round = []
        self.current_round_ = []
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


    def impute_conditional_data(self, array):
        return_array = np.zeros(array.shape)
        for i in range(array.shape[0]):
            datum = np.copy(array[i])
            nan_indices = np.argwhere(np.isnan(datum)).flatten()
            while np.any(nan_indices):
                nan_idx = nan_indices[0]
                valid_indices = np.argwhere(np.isfinite(array[:, nan_idx])).flatten()
                if valid_indices:
                    # pick one of them at random and overwrite all NaN values
                    row_idx = np.random.choice(valid_indices)
                    datum[nan_indices] = array[row_idx, nan_indices]
                else:
                    # no good point in the data has this value activated, so fill it with a valid but random value
                    t = self.vartypes[nan_idx]
                    if t == 0:
                        datum[nan_idx] = np.random.rand()
                    else:
                        datum[nan_idx] = np.random.randint(t)
                nan_indices = np.argwhere(np.isnan(datum)).flatten()
            return_array[i, :] = datum
        return return_array


    def fit_kde(self, round):
        budget = self.fidelities[round][0]
        good_models = self.kde_models[budget]['good']
        bad_models = self.kde_models[budget]['bad']
        if self.config.search_space == 'nasbench101':
            from naslib.predictors.utils.encodings_nb101 import encode_101
            good_enc = np.array([encode_101(m.arch, encoding_type='adjacency_cat') for m in good_models])
            bad_enc = np.array([encode_101(m.arch, encoding_type='adjacency_cat') for m in bad_models])
            self.kde_vartypes = ""
            self.vartypes = []
            for _ in range(len(good_enc[0])-5): # adj encoding + one-hot ops list
                self.kde_vartypes += 'u'
                self.vartypes += [2]
            for _ in range(len(good_enc[0])-5, len(good_enc[0])):  # adj encoding + one-hot ops list
                self.kde_vartypes += 'u'
                self.vartypes += [3]
        elif self.config.search_space == "nasbench201":
            from naslib.search_spaces.nasbench201.conversions import convert_naslib_to_op_indices
            good_enc = np.array([convert_naslib_to_op_indices(m.arch) for m in good_models])
            bad_enc = np.array([convert_naslib_to_op_indices(m.arch) for m in bad_models])
            self.kde_vartypes = ""
            self.vartypes = []
            for _ in range(len(good_enc[0])):  # we use unordered discrete variable
                self.kde_vartypes += 'u'
                self.vartypes += [5]  # depend on the encoding of search spaces
        elif self.config.search_space == "darts":
            from naslib.search_spaces.darts.conversions import convert_naslib_to_compact, \
                make_compact_mutable, convert_mutable_to_vector
            good_enc = np.array([convert_mutable_to_vector(make_compact_mutable(convert_naslib_to_compact(m.arch))) for m in good_models])
            bad_enc = np.array([convert_mutable_to_vector(make_compact_mutable(convert_naslib_to_compact(m.arch))) for m in bad_models])
            self.kde_vartypes = ""
            self.vartypes = []
            for i in range(len(good_enc[0])):  # we use unordered discrete variable
                self.kde_vartypes += 'u'
                if i % 2 == 0:
                    self.vartypes += [5]  # depend on the encoding of search spaces
                else:
                    self.vartypes += [7]
        self.vartypes = np.array(self.vartypes, dtype=int)
        good_enc = self.impute_conditional_data(good_enc)
        bad_enc = self.impute_conditional_data(bad_enc)
        self.good_kde = sm.nonparametric.KDEMultivariate(data=good_enc, var_type=self.kde_vartypes,
                                                         bw='normal_reference')
        self.bad_kde = sm.nonparametric.KDEMultivariate(data=bad_enc, var_type=self.kde_vartypes,
                                                        bw='normal_reference')
        self.bad_kde.bw = np.clip(self.bad_kde.bw, self.min_bandwidth, None)
        self.good_kde.bw = np.clip(self.good_kde.bw, self.min_bandwidth, None)
        l = self.good_kde.pdf
        g = self.bad_kde.pdf
        self.minimize_me = lambda x: max(1e-32, g(x) / max(l(x), 1e-32))


    def new_epoch(self, epoch, round, i):
        if self.process < i: # re-init for each new process
            del self.current_round
            del self.next_round
            self.current_round_ = []
            self.current_round = []
            self.next_round = []
            self.round_number = 0
            self.prev_round = 0
            self.process = i
            self.clean_history()

        if self.prev_round < round:  # reset round_number for each new round
            self.prev_round = round
            self.round_number = 0

        if epoch < self.round_sizes[round][0]:
            # sample random architectures
            model = torch.nn.Module()   # hacky way to get arch and accuracy checkpointable
            model.arch = self.search_space.clone()
            budget = self.fidelities[round][0]
            if round == 0:
                model.arch.sample_random_architecture(dataset_api=self.dataset_api)
            else:
                print("budget: {}, the number of good enc: {}".format(budget, len(self.kde_models[budget]['good'])))
                print("budget: {}, the number of bad enc: {}".format(budget, len(self.kde_models[budget]['bad'])))
                if epoch == 0 and \
                        len(self.kde_models[budget]['good']) >= self.min_points_in_model and \
                        len(self.kde_models[budget]['bad']) >= self.min_points_in_model:
                    self.fit_kde(round)
                    self.kde_models[budget]['minimize_kde'] = True
                if not self.kde_models[budget]['minimize_kde']:
                    model.arch.sample_random_architecture(dataset_api=self.dataset_api)
                else:
                    model.arch.model_based_sample_architecture(dataset_api=self.dataset_api,
                                                               minimize_me=self.minimize_me,
                                                               good_kde=self.good_kde,
                                                               vartypes=self.vartypes
                                                               )

            model.epoch = min(self.fidelities[round][0], self.max_training_epoch)
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
                cutoff = math.ceil(self.round_sizes[round][self.round_number] * self.top_n_percent)
                self.current_round = sorted(self.next_round, key=lambda x: -x.accuracy)[:cutoff]
                self.current_round_ = sorted(self.next_round, key=lambda x: -x.accuracy)[cutoff:]
                self.round_number += 1
                self.round_number = min(self.round_number, len(self.fidelities[round]) - 1)
                while len(self.current_round_) > 0:
                    self.kde_models[self.fidelities[round][self.round_number]]['bad'].append(self.current_round_.pop())
                self.next_round = []
            # train the next architecture
            model = self.current_round.pop()
            """
            Note: technically we would just continue training this arch, but right now,
            just for simplicity, we treat it as if we start to train it again from scratch
            """
            print(self.fidelities[round])
            print(self.round_number)
            print(self.fidelities[round][self.round_number])
            model = copy.deepcopy(model)
            model.epoch = min(self.fidelities[round][self.round_number], self.max_training_epoch)
            model.accuracy = model.arch.query(self.performance_metric,
                                              self.dataset, 
                                              epoch=model.epoch, 
                                              dataset_api=self.dataset_api)
            self.kde_models[self.fidelities[round][self.round_number]]['good'].append(model)
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
        return (
            best_arch.query(Metric.TRAIN_ACCURACY, self.dataset, dataset_api=self.dataset_api, epoch=best_arch_epoch-1), 
            best_arch.query(Metric.VAL_ACCURACY, self.dataset, dataset_api=self.dataset_api, epoch=best_arch_epoch), 
            best_arch.query(Metric.TEST_ACCURACY, self.dataset, dataset_api=self.dataset_api, epoch=best_arch_epoch), 
            latest_arch.query(Metric.TRAIN_TIME, self.dataset, dataset_api=self.dataset_api, epoch=latest_arch_epoch) * latest_arch_epoch, # TODO: Maybe we have to solve this directly in benchmark API
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
