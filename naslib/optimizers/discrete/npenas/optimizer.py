import collections
import logging
from re import M
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


class Npenas(MetaOptimizer):
    """
    Implements the Npenas optimizer.

    Npenas is a Neural Predictor Guided Evolution for Neural Architecture Search
    optimization algorithm which uses machine learning to guide the search process of
    neural architecture search (NAS). This algorithm employs predictive
    models, acquisition functions, and zero-cost proxies to speed up the
    search process.

    Attributes:
        config (obj): Configuration object containing setup parameters.
        zc_api (obj, optional): Zero cost API object to query zero-cost proxies.
    """

    # training the models is not implemented
    using_step_function = False

    def __init__(self, config, zc_api=None):
        """
        Constructor of Npenas optimizer.

        Args:
            config (obj): Configuration object containing setup parameters.
            zc_api (obj, optional): Zero cost API object to query zero-cost proxies.
        """
        super().__init__()
        self.config = config
        self.epochs = config.search.epochs

        self.performance_metric = Metric.VAL_ACCURACY
        self.dataset = config.dataset

        self.k = config.search.k
        self.num_init = config.search.num_init
        self.num_ensemble = config.search.num_ensemble
        self.predictor_type = config.search.predictor_type
        self.encoding_type = config.search.encoding_type  # currently not implemented
        self.num_arches_to_mutate = config.search.num_arches_to_mutate
        self.max_mutations = config.search.max_mutations
        self.num_candidates = config.search.num_candidates
        self.max_zerocost = 1000

        self.train_data = []
        self.next_batch = []
        self.history = torch.nn.ModuleList()

        self.zc = config.search.zc_ensemble if hasattr(config.search, 'zc_ensemble') else None
        self.semi = "semi" in self.predictor_type
        self.zc_api = zc_api
        self.use_zc_api = config.search.use_zc_api if hasattr(
            config.search, 'use_zc_api') else False
        self.zc_names = config.search.zc_names if hasattr(
            config.search, 'zc_names') else None
        self.zc_only = config.search.zc_only if hasattr(
            config.search, 'zc_only') else False

    def adapt_search_space(self, search_space, scope=None, dataset_api=None):
        """
        Adapts the search space for optimization.

        Args:
            search_space (obj): The search space object that contains all possible architectures.
            scope (str, optional): Defines the scope of the search space for the optimizer.
            dataset_api (obj, optional): API for querying dataset specific metrics.

        Raises:
            AssertionError: If search_space is not queryable.
        """
        assert (
            search_space.QUERYABLE
        ), "Npenas is currently only implemented for benchmarks."

        self.search_space = search_space.clone()
        self.scope = scope if scope else search_space.OPTIMIZER_SCOPE
        self.dataset_api = dataset_api
        self.ss_type = self.search_space.get_type()
        if self.zc:
            self.train_loader, _, _, _, _ = get_train_val_loaders(
                self.config, mode="train"
            )
        if self.semi:
            self.unlabeled = []

    def get_zero_cost_predictors(self):
        """
        Creates and returns a dictionary of zero cost predictors based on the `zc_names` configuration.

        Returns:
            dict: A dictionary where keys are zero cost predictor names and values are the predictor objects.
        """
        return {zc_name: ZeroCost(method_type=zc_name) for zc_name in self.zc_names}

    def query_zc_scores(self, arch):
        """
        Queries and returns the zero-cost scores for a given architecture.

        Args:
            arch (obj): The architecture to query.

        Returns:
            dict: A dictionary where keys are zero cost predictor names and values are the corresponding scores.
        """
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
        """
        Sets the accuracy and zero-cost scores for a given model.

        Args:
            model (obj): The model to set scores for.
        """

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
        """
        Samples and returns a new model from the search space.

        Returns:
            obj: The new sampled model.
        """
        model = torch.nn.Module()
        model.arch = self.search_space.clone()
        model.arch.sample_random_architecture(
            dataset_api=self.dataset_api, load_labeled=self.use_zc_api)
        model.arch_hash = model.arch.get_hash()

        if self.search_space.instantiate_model == True:
            model.arch.parse()

        return model

    def _get_train(self):
        """
        Retrieves the training data.

        Returns:
            list: The list of architectures used for training.
            list: The corresponding list of accuracies for each architecture.
        """
        xtrain = [m.arch for m in self.train_data]
        ytrain = [m.accuracy for m in self.train_data]
        return xtrain, ytrain

    def _get_ensemble(self):
        """
        Creates and returns an ensemble of predictors.

        Returns:
            obj: The ensemble object.
        """
        ensemble = Ensemble(num_ensemble=self.num_ensemble,
                            ss_type=self.ss_type,
                            predictor_type=self.predictor_type,
                            zc=self.zc,
                            zc_only=self.zc_only,
                            config=self.config)

        return ensemble

    def _get_new_candidates(self, ytrain):
        """
        Returns a list of candidate architectures by mutating the best architectures.

        Args:
            ytrain (list): The list of accuracies of the architectures used for training.

        Returns:
            list: A list of candidate architectures.
        """
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

        return candidates

    def new_epoch(self, epoch):
        """
        Conducts one epoch of the optimization process.

        Args:
            epoch (int): The current epoch number.
        """

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
                    train_info = {'zero_cost_scores': [m.zc_scores for m in self.train_data]}
                    ensemble.set_pre_computations(xtrain_zc_info=train_info)

                    if self.semi:
                        unlabeled_zc_info = {'zero_cost_scores': [m.zc_scores for m in self.unlabeled]}
                        ensemble.set_pre_computations(unlabeled_zc_info=unlabeled_zc_info)

                ensemble.fit(xtrain, ytrain)

                # define an acquisition function
                acq_fn = acquisition_function(
                    ensemble=ensemble, ytrain=None, acq_fn_type="exploit_only"
                )

                # output k best candidates
                candidates = self._get_new_candidates(ytrain=ytrain)

                self.next_batch = self._get_best_candidates(candidates, acq_fn)

            # train the next architecture chosen by the neural predictor
            model = self.next_batch.pop()
            self._set_scores(model)

    def _get_best_candidates(self, candidates, acq_fn):
        """
        Returns the best candidate architectures based on the acquisition function.

        Args:
            candidates (list): A list of candidate architectures.
            acq_fn (function): The acquisition function.

        Returns:
            list: A list of the best candidate architectures.
        """

        if self.zc and len(self.train_data) <= self.max_zerocost:
            for model in candidates:
                model.zc_scores = self.query_zc_scores(model.arch_hash, self.zc_names, self.zc_api)

            values = [acq_fn(model.arch, [{'zero_cost_scores': model.zc_scores}]) for model in candidates]
        else:
            values = [acq_fn(model.arch) for model in candidates]

        sorted_indices = np.argsort(values)
        choices = [candidates[i] for i in sorted_indices[-self.k:]]

        return choices

    def _update_history(self, child):
        """
        Updates the history of the optimizer by replacing the worst model in history with the given model
        if the given model's accuracy is better.

        Args:
            child (obj): The model to potentially add to the history.
        """
        if len(self.history) < 100:
            self.history.append(child)
        else:
            for i, p in enumerate(self.history):
                if child.accuracy > p.accuracy:
                    self.history[i] = child
                    break

    def train_statistics(self, report_incumbent=True):
        """
        Returns the training statistics of the best architecture or the last trained one.

        Args:
            report_incumbent (bool, optional): Whether to return the statistics of the best architecture. Defaults to True.

        Returns:
            tuple: A tuple containing the training accuracy, validation accuracy, test accuracy, and training time of the architecture.
        """
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
        """
        Returns the test statistics of the final architecture.

        Returns:
            int: The raw test metric of the final architecture.
        """
        best_arch = self.get_final_architecture()
        if self.search_space.space_name != "nasbench301":
            return best_arch.query(Metric.RAW, self.dataset, dataset_api=self.dataset_api)
        else:
            return -1

    def get_final_architecture(self):
        """
        Returns the final/best architecture from the history.

        Returns:
            obj: The final/best architecture.
        """
        return max(self.history, key=lambda x: x.accuracy).arch

    def get_op_optimizer(self):
        """
        Method to get the optimizer of the operation. Not implemented in this class.

        Raises:
            NotImplementedError: Always, as this method is not implemented.
        """
        raise NotImplementedError()

    def get_checkpointables(self):
        """
        Returns the models that should be checkpointed.

        Returns:
            dict: A dictionary with 'model' as the key and the history of models as the value.
        """
        return {"model": self.history}

    def get_model_size(self):
        """
        Returns the size of the model.

        Returns:
            float: The size of the model in MB.
        """
        return count_parameters_in_MB(self.history)

    def get_arch_as_string(self, arch):
        """
        Returns a string representation of the given architecture.

        Args:
            arch (obj): The architecture to convert to string.

        Returns:
            str: The string representation of the architecture.
        """
        if self.search_space.get_type() == 'nasbench301':
            str_arch = str(list((list(arch[0]), list(arch[1]))))
        else:
            str_arch = str(arch)
        return str_arch
