import codecs
import time
import json
import logging
import os
import numpy as np
import torch
from scipy import stats

from naslib.optimizers.discrete.utils.utils import sample_random_architecture

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
  
        self.train_size = config.train_size
        self.test_size = config.test_size
        self.dataset = config.dataset
        self.load_labeled = config.load_labeled

        self.metric = Metric.VAL_ACCURACY
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.results_dict = utils.AttrDict(
            {'test_error': [],
             'correlation': [],
             'rank_correlation': [],
             'runtime': []}
        )
        
    def adapt_search_space(self, search_space, scope=None):
        self.search_space = search_space.clone()
        self.scope = scope if scope else search_space.OPTIMIZER_SCOPE
        self.predictor.set_ss_type(self.search_space.get_type())

    def load_dataset(self, load_labeled=False):
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
        for _ in range(self.train_size):
            if not load_labeled:
                arch = sample_random_architecture(self.search_space, self.scope)
                accuracy = arch.query(metric=self.metric, dataset=self.dataset)[-1]
            else:
                arch = self.search_space.clone()
                accuracy = arch.load_labeled_architecture()
            xdata.append(arch)
            ydata.append(accuracy)
        return xdata, ydata
        
    def evaluate(self):

        # pre-process the predictor
        self.predictor.pre_process()
        
        logger.info("Load the training set")
        xtrain, ytrain = self.load_dataset(load_labeled=self.load_labeled)

        # fit the predictor (for model-based methods)
        # Note: currently we must pass in the search space type in order to
        # properly encode the training data
        logger.info("Fit the predictor")
        self.predictor.fit(xtrain, ytrain)
        
        logger.info("Load the test set")
        xtest, ytest = self.load_dataset(load_labeled=self.load_labeled)
        
        """
        Train the architectures in the test set partially
        if required by the predictor (e.g. learning curve extrapolators).
        Note that Predictors cannot train architectures themselves,
        so the partial training must be passed in as an argument.
        """

        info = self.predictor.requires_partial_training(xtest)

        # query each architecture in the test set
        test_pred = self.predictor.query(xtest, info)
        test_pred = np.squeeze(test_pred)
        
        # this if statement is because of ensembles. TODO: think of a better solution
        if len(test_pred.shape) > 1:
            test_pred = np.mean(test_pred, axis=0)
        
        logger.info("Compute evaluation metrics")
        test_error, correlation, rank_correlation = self.compare(ytest, test_pred)
        logger.info("test error: {}, correlation: {}, rank_correlation: {}".format(test_error, correlation, rank_correlation))
        self.results_dict['correlation'].append(correlation)
        self.results_dict['rank_correlation'].append(rank_correlation)
        self.results_dict['test_error'].append(test_error)
        self._log_to_json()
        """
        TODO: also return timing information
        (for preprocessing, training train set, and querying test set).
        start_time = time.time()
        """
        
    def compare(self, ytest, test_pred):
        
        test_error = np.mean(abs(test_pred-ytest))
        correlation = np.corrcoef(np.array(ytest), np.array(test_pred))[1,0]
        rank_correlation, _ = stats.spearmanr(ytest, test_pred)
        return test_error, correlation, rank_correlation
        
    def _log_to_json(self):
        """log statistics to json file"""
        if not os.path.exists(self.config.save):
            os.makedirs(self.config.save)
        with codecs.open(os.path.join(self.config.save, 'errors.json'), 'w', encoding='utf-8') as file:
            json.dump(self.results_dict, file, separators=(',', ':'))


    