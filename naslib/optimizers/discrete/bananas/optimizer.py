import collections
import logging
import torch
import copy
import numpy as np

from naslib.optimizers.core.metaclasses import MetaOptimizer
from naslib.optimizers.discrete.bananas.acquisition_functions import (
    acquisition_function,
)

from naslib.predictors.ensemble import Ensemble
from naslib.predictors.zerocost import ZeroCost
from naslib.predictors.utils.encodings import encode_spec

from naslib.search_spaces.core.query_metrics import Metric

from naslib.utils import AttrDict, count_parameters_in_MB, get_train_val_loaders
from naslib.utils.log import log_every_n_seconds


logger = logging.getLogger(__name__)


class Bananas(MetaOptimizer):

    # training the models is not implemented
    using_step_function = False

    def __init__(self, config, zc_api=None):
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

        self.zc = config.search.zc if hasattr(config.search, 'zc') else None
        self.semi = "semi" in self.predictor_type 
        self.zc_api = zc_api
        self.use_zc_api = config.search.use_zc_api if hasattr(
            config.search, 'use_zc_api') else False
        self.zc_names = config.search.zc_names if hasattr(
            config.search, 'zc_names') else None
        self.zc_only = config.search.zc_only if hasattr(
            config.search, 'zc_only') else False
        
        self.load_labeled = config.search.load_labeled if hasattr(
            config.search, 'load_labeled') else False

    def adapt_search_space(self, search_space, scope=None, dataset_api=None):
        assert (
            search_space.QUERYABLE
        ), "Bananas is currently only implemented for benchmarks."

        self.search_space = search_space.clone()
        self.scope = scope if scope else search_space.OPTIMIZER_SCOPE
        self.dataset_api = dataset_api
        self.ss_type = self.search_space.get_type()
        if self.zc:
            self.train_loader, _, _, _, _ = get_train_val_loaders(
                self.config, mode="train")
        if self.semi:
            self.unlabeled = []

    def get_zero_cost_predictors(self):
        return {zc_name: ZeroCost(method_type=zc_name) for zc_name in self.zc_names}

    def query_zc_scores(self, arch):
        zc_scores = {}
        zc_methods = self.get_zero_cost_predictors()
        arch_hash = arch.get_hash()
        for zc_name, zc_method in zc_methods.items():

            if self.use_zc_api and str(arch_hash) in self.zc_api:
                score = self.zc_api[str(arch_hash)][zc_name]['score']
            else:
                zc_method.train_loader = copy.deepcopy(self.train_loader)
                score = zc_method.query(arch, dataloader=zc_method.train_loader)

            if float("-inf") == score:
                score = -1e9
            elif float("inf") == score:
                score = 1e9

            zc_scores[zc_name] = score

        return zc_scores

    def _set_scores(self, model):

        if self.use_zc_api and str(model.arch_hash) in self.zc_api:
            model.accuracy = self.zc_api[str(model.arch_hash)]['val_accuracy']
        else:
            model.accuracy = model.arch.query(
                self.performance_metric, self.dataset, dataset_api=self.dataset_api
            )

        if self.zc and len(self.train_data) <= self.max_zerocost:
            model.zc_scores = self.query_zc_scores(model.arch)

        self.train_data.append(model)
        self._update_history(model)

    def _sample_new_model(self):
        model = torch.nn.Module()
        model.arch = self.search_space.clone()
        model.arch.sample_random_architecture(
            dataset_api=self.dataset_api, load_labeled=self.load_labeled)
        model.arch_hash = model.arch.get_hash()
        
        if self.search_space.instantiate_model == True:
            model.arch.parse()

        return model

    def _get_train(self):
        xtrain = [m.arch for m in self.train_data]
        ytrain = [m.accuracy for m in self.train_data]
        return xtrain, ytrain

    def _get_ensemble(self):
        ensemble = Ensemble(num_ensemble=self.num_ensemble,
                            ss_type=self.ss_type,
                            predictor_type=self.predictor_type,
                            zc=self.zc,
                            zc_only=self.zc_only,
                            config=self.config)

        return ensemble

    def _get_new_candidates(self, ytrain):
        # optimize the acquisition function to output k new architectures
        candidates = []
        if self.acq_fn_optimization == 'random_sampling':

            for _ in range(self.num_candidates):
                # self.search_space.sample_random_architecture(dataset_api=self.dataset_api, load_labeled=self.sample_from_zc_api) # FIXME extend to Zero Cost case
                model = self._sample_new_model()
                model.accuracy = model.arch.query(
                    self.performance_metric, self.dataset, dataset_api=self.dataset_api
                )
                candidates.append(model)

        elif self.acq_fn_optimization == 'mutation':
            # mutate the k best architectures by x
            best_arch_indices = np.argsort(ytrain)[-self.num_arches_to_mutate:]
            best_archs = [self.train_data[i].arch for i in best_arch_indices]
            candidates = []
            for arch in best_archs:
                for _ in range(int(self.num_candidates / len(best_archs) / self.max_mutations)):
                    candidate = arch.clone()
                    for __ in range(int(self.max_mutations)):
                        arch = self.search_space.clone()
                        arch.mutate(candidate, dataset_api=self.dataset_api)
                        if self.search_space.instantiate_model == True:
                            arch.parse()
                        candidate = arch

                    model = torch.nn.Module()
                    model.arch = candidate
                    model.arch_hash = candidate.get_hash()
                    candidates.append(model)

        else:
            logger.info('{} is not yet supported as a acq fn optimizer'.format(
                self.encoding_type))
            raise NotImplementedError()

        return candidates

    def new_epoch(self, epoch):

        if epoch < self.num_init:
            model = self._sample_new_model()
            self._set_scores(model)
        else:
            if len(self.next_batch) == 0:
                # train a neural predictor
                xtrain, ytrain = self._get_train()
                ensemble = self._get_ensemble()

                if self.semi:
                    # create unlabeled data and pass it to the predictor
                    while len(self.unlabeled) < len(xtrain):
                        model = self._sample_new_model()

                        if self.zc and len(self.train_data) <= self.max_zerocost:
                            model.zc_scores = self.query_zc_scores(model.arch)

                        self.unlabeled.append(model)

                    ensemble.set_pre_computations(
                        unlabeled=[m.arch for m in self.unlabeled]
                    )

                if self.zc and len(self.train_data) <= self.max_zerocost:
                    # pass the zero-cost scores to the predictor
                    train_info = {'zero_cost_scores': [
                        m.zc_scores for m in self.train_data]}
                    ensemble.set_pre_computations(xtrain_zc_info=train_info)

                    if self.semi:
                        unlabeled_zc_info = {'zero_cost_scores': [
                            m.zc_scores for m in self.unlabeled]}
                        ensemble.set_pre_computations(
                            unlabeled_zc_info=unlabeled_zc_info)

                ensemble.fit(xtrain, ytrain)

                # define an acquisition function
                acq_fn = acquisition_function(
                    ensemble=ensemble, ytrain=ytrain, acq_fn_type=self.acq_fn_type
                )

                # optimize the acquisition function to output k new architectures
                candidates = self._get_new_candidates(ytrain=ytrain)

                self.next_batch = self._get_best_candidates(candidates, acq_fn)

            # train the next architecture chosen by the neural predictor
            model = self.next_batch.pop()
            self._set_scores(model)

    def _get_best_candidates(self, candidates, acq_fn):
        if self.zc and len(self.train_data) <= self.max_zerocost:
            for model in candidates:
                model.zc_scores = self.query_zc_scores(model.arch)

            values = [acq_fn(model.arch, [{'zero_cost_scores': model.zc_scores}]) for model in candidates]
        else:
            values = [acq_fn(model.arch) for model in candidates]

        sorted_indices = np.argsort(values)
        choices = [candidates[i] for i in sorted_indices[-self.k:]]

        return choices

    def _update_history(self, child):
        if len(self.history) < 100:
            self.history.append(child)
        else:
            for i, p in enumerate(self.history):
                if child.accuracy > p.accuracy:
                    self.history[i] = child
                    break

    def train_statistics(self, report_incumbent=True):
        if report_incumbent:
            best_arch = self.get_final_architecture()
        else:
            best_arch = self.train_data[-1].arch
        
        if self.search_space.space_name != "nasbench301":
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
        else:
            return (
                -1, 
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

    def test_statistics(self):
        best_arch = self.get_final_architecture()
        if self.search_space.space_name != "nasbench301":
            return best_arch.query(Metric.RAW, self.dataset, dataset_api=self.dataset_api)
        else:
            return -1

    def get_final_architecture(self):
        return max(self.history, key=lambda x: x.accuracy).arch

    def get_op_optimizer(self):
        raise NotImplementedError()

    def get_checkpointables(self):
        return {"model": self.history}

    def get_model_size(self):
        return count_parameters_in_MB(self.history)

    def get_arch_as_string(self, arch):
        if self.search_space.get_type() == 'nasbench301':
            str_arch = str(list((list(arch[0]), list(arch[1]))))
        else:
            str_arch = str(arch)
        return str_arch
