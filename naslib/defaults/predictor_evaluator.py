import codecs
import time
import json
import logging
import os
import numpy as np
import torch
from scipy import stats

from naslib.search_spaces.core.query_metrics import Metric
from naslib.utils import utils

logger = logging.getLogger(__name__)


class PredictorEvaluator(object):
    """
    Evaluate a chosen predictor.
    """

    def __init__(self, predictor, config=None):
        """
        Initializes the Evaluator.
        """
        self.predictor = predictor
        self.config = config
        self.experiment_type = config.experiment_type

        self.test_size = config.test_size

        self.train_size_single = config.train_size_single
        self.train_size_list = config.train_size_list
        
        self.fidelity_single = config.fidelity_single
        self.fidelity_list = config.fidelity_list

        self.dataset = config.dataset
        self.metric = Metric.VAL_ACCURACY
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.results = []

    def adapt_search_space(self, search_space, load_labeled, scope=None):
        self.search_space = search_space.clone()
        self.scope = scope if scope else search_space.OPTIMIZER_SCOPE
        self.predictor.set_ss_type(self.search_space.get_type())
        self.load_labeled = load_labeled

    def load_dataset(self, load_labeled=False, data_size=10):
        """
        There are two ways to load a dataset.
        load_labeled=False: sample random architectures and then query the architectures.
        This works on NAS benchmarks where we can query any architecture.
        load_labeled=True: load a dataset of architectures where we have the training info
        (for example, load the set of 5k DARTS architectures which have the full training info)
        """
        # Note: currently ydata consists of the val_accs at the final training epoch
        xdata = []
        ydata = []
        for _ in range(data_size):
            if not load_labeled:
                arch = self.search_space.clone()
                arch.sample_random_architecture()
            else:
                arch = self.search_space.clone()
                arch.load_labeled_architecture()
                
            accuracy = arch.query(metric=self.metric, dataset=self.dataset)
            xdata.append(arch)
            ydata.append(accuracy)
        return xdata, ydata

    def single_evaluate(self, xtrain, ytrain, xtest, ytest, fidelity):
        info = self.predictor.requires_partial_training(xtest, fidelity)

        logger.info("Fit the predictor")
        self.predictor.fit(xtrain, ytrain)

        # query each architecture in the test set
        test_pred = self.predictor.query(xtest, info)
        test_pred = np.squeeze(test_pred)

        # this if statement is because of ensembles. TODO: think of a better solution
        if len(test_pred.shape) > 1:
            test_pred = np.mean(test_pred, axis=0)

        logger.info("Compute evaluation metrics")
        return self.compare(ytest, test_pred)

    def evaluate(self):

        # pre-process the predictor
        self.predictor.pre_process()

        logger.info("Load the test set")
        xtest, ytest = self.load_dataset(load_labeled=self.load_labeled, data_size=self.test_size)

        if self.experiment_type == 'single':
            train_size = self.train_size_single
            fidelity = self.fidelity_single

            logger.info("Load the training set")
            xtrain, ytrain = self.load_dataset(load_labeled=self.load_labeled,
                                               data_size=train_size)

            metrics = self.single_evaluate(xtrain, ytrain, xtest, ytest, fidelity=fidelity)
            test_error, correlation, rank_correlation = metrics
            logger.info("test error: {}, correlation: {}, rank correlation {}"
                        .format(test_error, correlation, rank_correlation))
            self.results.append({'train_size': train_size,
                                 'fidelity': fidelity,
                                 'test_error': test_error,
                                 'correlation': correlation,
                                 'rank_correlation': rank_correlation})

        elif self.experiment_type == 'vary_train_size':

            logger.info("Load the training set")
            xtrain_full, ytrain_full = self.load_dataset(load_labeled=self.load_labeled,
                                                         data_size=self.train_size_list[-1])
            fidelity = self.fidelity_single

            for train_size in self.train_size_list:
                
                xtrain, ytrain = xtrain_full[:train_size], ytrain_full[:train_size]

                metrics = self.single_evaluate(xtrain, ytrain, xtest, ytest, fidelity=fidelity)
                test_error, correlation, rank_correlation = metrics
                logger.info("train_size: {}, test error: {}, correlation: {}, rank correlation {}"
                            .format(train_size, test_error, correlation, rank_correlation))
                self.results.append({'train_size': train_size,
                                     'fidelity': fidelity,
                                     'test_error': test_error,
                                     'correlation': correlation,
                                     'rank_correlation': rank_correlation})

        elif self.experiment_type == 'vary_fidelity':

            train_size = self.train_size_single

            logger.info("Load the training set")
            xtrain, ytrain = self.load_dataset(load_labeled=self.load_labeled,
                                               data_size=self.train_size_single)

            for fidelity in self.fidelity_list:

                metrics = self.single_evaluate(xtrain, ytrain, xtest, ytest, fidelity=fidelity)
                test_error, correlation, rank_correlation = metrics
                logger.info("fidelity: {}, test error: {}, correlation: {}, rank correlation {}"
                            .format(fidelity, test_error, correlation, rank_correlation))
                self.results.append({'train_size': train_size,
                                     'fidelity': fidelity,
                                     'test_error': test_error,
                                     'correlation': correlation,
                                     'rank_correlation': rank_correlation})
        else:
            raise NotImplementedError()


        """
        TODO: also return timing information
        (for preprocessing, training train set, and querying test set).
        start_time = time.time()
        """
        self._log_to_json()

    def compare(self, ytest, test_pred):
        test_error = np.mean(abs(test_pred-ytest))
        correlation = np.abs(np.corrcoef(np.array(ytest), np.array(test_pred))[1,0])
        rank_correlation, _ = stats.spearmanr(ytest, test_pred)
        return test_error, correlation, rank_correlation

    def _log_to_json(self):
        """log statistics to json file"""
        if not os.path.exists(self.config.save):
            os.makedirs(self.config.save)
        with codecs.open(os.path.join(self.config.save, 'errors.json'), 'w', encoding='utf-8') as file:
            json.dump(self.results, file, separators=(',', ':'))

