import logging
import torch
import numpy as np

from naslib.optimizers.core.metaclasses import MetaOptimizer
from naslib.optimizers.discrete.bananas.acquisition_functions import acquisition_function

from naslib.predictors.ensemble import Ensemble
from naslib.predictors import ZeroCost
from naslib.predictors.utils.encodings import encode_spec

from naslib.search_spaces.core.query_metrics import Metric

from naslib.utils.utils import count_parameters_in_MB, get_train_val_loaders


logger = logging.getLogger(__name__)


class Bananas(MetaOptimizer):

    # training the models is not implemented
    using_step_function = False

    def __init__(self, config, zc_api):
        super().__init__()
        self.config = config
        self.epochs = config.search.epochs

        self.performance_metric = Metric.VAL_ACCURACY
        self.dataset = config.dataset

        self.k = config.search.k
        self.num_init = config.search.num_init
        self.num_ensemble = config.search.num_ensemble
        self.predictor_type = config.search.predictor_type
        self.acq_fn_type = config.search.acq_fn_type
        self.acq_fn_optimization = config.search.acq_fn_optimization
        self.encoding_type = config.search.encoding_type  # currently not implemented
        self.num_arches_to_mutate = config.search.num_arches_to_mutate
        self.max_mutations = config.search.max_mutations
        self.num_candidates = config.search.num_candidates
        self.max_zerocost = 1000

        self.train_data = []
        self.next_batch = []
        self.history = torch.nn.ModuleList()

        self.zc = config.search.zc_ensemble
        self.zc_names = config.search.zc_names

        self.sample_from_zc_api = zc_api is not None
        self.zc_api = zc_api

    def adapt_search_space(self, search_space, scope=None, dataset_api=None):
        assert search_space.QUERYABLE, "Bananas is currently only implemented for benchmarks."

        self.search_space = search_space.clone()
        self.scope = scope if scope else search_space.OPTIMIZER_SCOPE
        self.dataset_api = dataset_api
        self.ss_type = self.search_space.get_type()
        if self.zc:
            self.train_loader, _, _, _, _ = get_train_val_loaders(self.config)

    def get_zero_cost_predictors(self):
        return [ZeroCost(method_type=zc_name) for zc_name in self.zc_names]

    def query_zc_scores(self, arch, predictors, zc_api):
        zc_scores = {}

        for predictor in predictors:
            score = zc_api[str(arch)][predictor]['score']

            if float("-inf") == score:
                score = -1e9
            elif float("inf") == score:
                score = 1e9

            zc_scores[predictor] = score

        return zc_scores

    def _sample_new_model(self):
        self.search_space.sample_random_architecture(dataset_api=self.dataset_api, load_labeled=self.sample_from_zc_api)

        model = torch.nn.Module()
        model.arch_hash = self.search_space.get_hash()
        model.arch = encode_spec(model.arch_hash, encoding_type='adjacency_one_hot', ss_type=self.search_space.get_type())
        model.accuracy = self.zc_api[str(model.arch_hash)]['val_accuracy']

        if self.zc:
            model.zc_scores = self.query_zc_scores(model.arch_hash, self.zc_names, self.zc_api)

        self.train_data.append(model)
        self._update_history(model)

    def _train_new_ensemble(self):
        """Trains a new ensemble"""
        xtrain = [m.arch for m in self.train_data]
        ytrain = [m.accuracy for m in self.train_data]

        ensemble = Ensemble(num_ensemble=self.num_ensemble,
                            ss_type=self.ss_type,
                            predictor_type=self.predictor_type,
                            zc=self.config.search.zc_ensemble,
                            zc_only=self.config.search.zc_only,
                            config=self.config)

        if self.zc and len(self.train_data) <= self.max_zerocost:
            # pass the zero-cost scores to the ensemble
            train_info = {'zero_cost_scores': [m.zc_scores for m in self.train_data]}
            ensemble.set_pre_computations(xtrain_zc_info=train_info)

        ensemble.fit(xtrain, ytrain)

        return ensemble, xtrain, ytrain

    def _get_new_candidates(self, ytrain):
        # optimize the acquisition function to output k new architectures
        candidates = []
        if self.acq_fn_optimization == 'random_sampling':

            for _ in range(self.num_candidates):
                self.search_space.sample_random_architecture(dataset_api=self.dataset_api, load_labeled=self.sample_from_zc_api)

                model = torch.nn.Module()
                model.arch_hash = self.search_space.get_hash()
                model.arch = encode_spec(model.arch_hash, encoding_type='adjacency_one_hot', ss_type=self.search_space.get_type())
                model.accuracy = self.zc_api[str(model.arch_hash)]['val_accuracy']

                candidates.append(model)

        elif self.acq_fn_optimization == 'mutation':
            # mutate the k best architectures by x
            best_arch_indices = np.argsort(ytrain)[-self.num_arches_to_mutate:]
            best_arches = [self.train_data[i].arch for i in best_arch_indices]
            candidates = []
            for arch in best_arches:
                for _ in range(int(self.num_candidates / len(best_arches) / self.max_mutations)):
                    candidate = arch.clone()
                    for edit in range(int(self.max_mutations)):
                        arch = self.search_space.clone()
                        arch.mutate(candidate, dataset_api=self.dataset_api)
                        candidate = arch
                    candidates.append(candidate)

        else:
            logger.info('{} is not yet supported as a acq fn optimizer'.format(self.encoding_type))
            raise NotImplementedError()

        return candidates

    def _get_best_candidates(self, candidates, acq_fn):
        if self.zc:
            for model in candidates:
                model.zc_scores = self.query_zc_scores(model.arch_hash, self.zc_names, self.zc_api)

            values = [acq_fn(model.arch, [{'zero_cost_scores' : model.zc_scores}]) for model in candidates]
        else:
            values = [acq_fn(model.arch) for model in candidates]

        sorted_indices = np.argsort(values)
        choices = [candidates[i] for i in sorted_indices[-self.k:]]

        return choices

    def new_epoch(self, epoch):

        if epoch < self.num_init:
            self._sample_new_model()
        else:
            if len(self.next_batch) == 0:
                # train a neural predictor
                ensemble, xtrain, ytrain = self._train_new_ensemble()

                # Get new candidates
                candidates = self._get_new_candidates(ytrain)

                # define an acquisition function
                acq_fn = acquisition_function(ensemble=ensemble, ytrain=ytrain, acq_fn_type=self.acq_fn_type)

                # Update the batch with the candidates that maximize the acquisition function
                self.next_batch = self._get_best_candidates(candidates, acq_fn)

            # train the next architecture chosen by the neural predictor
            # model = torch.nn.Module()
            model = self.next_batch.pop()
            model.accuracy = self.zc_api[str(model.arch_hash)]['val_accuracy']

            if self.zc and len(self.train_data) <= self.max_zerocost:
                model.zc_scores = self.query_zc_scores(model.arch_hash, self.zc_names, self.zc_api)

            self._update_history(model)
            self.train_data.append(model)

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
        # self.search_space.set_spec(best_arch)
        return (
            -1,
            self.zc_api[str(best_arch)]['val_accuracy'],
            -1,
            -1,
        )

    def test_statistics(self):
        # best_arch = self.get_final_architecture()
        # self.search_space.set_spec(best_arch)
        return {}

    def get_final_architecture(self):
        return max(self.history, key=lambda x: x.accuracy).arch_hash

    def get_op_optimizer(self):
        raise NotImplementedError()

    def get_checkpointables(self):
        return {'model': self.history}

    def get_model_size(self):
        return count_parameters_in_MB(self.history)

    def get_arch_as_string(self, arch):
        if self.search_space.get_type() == 'nasbench301':
            str_arch = str(list((list(arch[0]), list(arch[1]))))
        else:
            str_arch = str(arch)
        return str_arch
