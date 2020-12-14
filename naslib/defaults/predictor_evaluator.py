import codecs
import time
import json
import logging
import os
import numpy as np
import torch

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
  
        self.test_size = config.test_size
        self.train_size_increment = config.trainable.train_size_increment
        self.fidelity_increment = config.learning_curve.fidelity_increment

        if not config.single_run:

            self.train_size_start = config.trainable.train_size_start
            self.train_size_end = config.trainable.train_size_end
            
            self.fidelity_start = config.learning_curve.fidelity_start
            self.fidelity_end = config.learning_curve.fidelity_end
        else:
            self.train_size_start = config.trainable.train_size_single
            self.fidelity_start = config.learning_curve.fidelity_single
            self.train_size_end = config.trainable.train_size_single \
            + config.trainable.train_size_increment            
            self.fidelity_end = config.learning_curve.fidelity_start \
            + config.learning_curve.fidelity_increment
                
        self.dataset = config.dataset
        
        self.metric = Metric.VAL_LOSS
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.results_dict = []
        
    def adapt_search_space(self, search_space, load_labeled, scope=None):
        self.search_space = search_space.clone()
        self.scope = scope if scope else search_space.OPTIMIZER_SCOPE
        self.predictor.set_ss_type(self.search_space.get_type())
        self.load_labeled = load_labeled
        
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
        for _ in range(self.train_size_end):
            if not load_labeled:
                arch = sample_random_architecture(self.search_space, self.scope)
                accuracy = arch.query(metric=self.metric, dataset=self.dataset)
            else:
                arch = self.search_space.clone()
                accuracy = arch.load_labeled_architecture()
            xdata.append(arch)
            ydata.append(accuracy)
        return xdata, ydata
        
    def evaluate(self):

        # pre-process the predictor
        self.predictor.pre_process()
                
        logger.info("Load the test set")
        xtest, ytest = self.load_dataset(load_labeled=self.load_labeled)

        if self.predictor.get_type() == 'trainable':

            logger.info("Load the training set")
            xtrain, ytrain = self.load_dataset(load_labeled=self.load_labeled)
            
            for train_size in range(self.train_size_start, self.train_size_end, 
                                    self.train_size_increment):

                # fit the predictor (for model-based methods)
                # Note: currently we must pass in the search space type in order to
                # properly encode the training data
                logger.info("Fit the predictor")
                self.predictor.fit(xtrain[:train_size], ytrain[:train_size])        

                """
                Train the architectures in the test set partially
                if required by the predictor (e.g. learning curve extrapolators).
                Note that Predictors cannot train architectures themselves,
                so the partial training must be passed in as an argument.
                """
                if self.predictor.requires_partial_training():
                    logger.info("Perform partial training")
                    fidelity = self.predictor.get_fidelity()
                    metric = self.predictor.get_metric()
                    info = [arch.query(metric, self.dataset, epoch=fidelity) 
                            for arch in xtest]
                else:
                    info = None

                # query each architecture in the test set
                test_pred = self.predictor.query(xtest, info)
                test_pred = np.squeeze(test_pred)

                # this if statement is because of ensembles. TODO: think of a better solution
                if len(test_pred.shape) > 1:
                    test_pred = np.mean(test_pred, axis=0)

                logger.info("Compute evaluation metrics")
                test_error, correlation = self.compare(ytest, test_pred)
                logger.info("train size: {}, test error: {}, correlation: {}"
                            .format(train_size, test_error, correlation))
                self.results_dict.append({'train_size': train_size,
                                          'correlation': correlation,
                                          'test_error': test_error})

        else:
            for fidelity in range(self.fidelity_start, self.fidelity_end, 
                                    self.fidelity_increment):

                if self.predictor.requires_partial_training():
                    logger.info("Perform partial training")
                    metric = self.predictor.get_metric()
                    info = [arch.query(metric, self.dataset, epoch=fidelity) 
                            for arch in xtest]
                else:
                    info = None

                # query each architecture in the test set
                test_pred = self.predictor.query(xtest, info)
                test_pred = np.squeeze(test_pred)

                # this if statement is because of ensembles. TODO: think of a better solution
                if len(test_pred.shape) > 1:
                    test_pred = np.mean(test_pred, axis=0)

                logger.info("Compute evaluation metrics")
                test_error, correlation = self.compare(ytest, test_pred)
                logger.info("fidelity: {}, test error: {}, correlation: {}"
                            .format(fidelity, test_error, correlation))
                self.results_dict.append({'fidelity': fidelity,
                                          'correlation': correlation,
                                          'test_error': test_error})

        """
        TODO: also return timing information
        (for preprocessing, training train set, and querying test set).
        start_time = time.time()
        """
        self._log_to_json()

        
    def compare(self, ytest, test_pred):
        
        test_error = np.mean(abs(test_pred-ytest))
        correlation = np.abs(np.corrcoef(np.array(ytest), np.array(test_pred))[1,0])
        return test_error, correlation
        
    def _log_to_json(self):
        """log statistics to json file"""
        if not os.path.exists(self.config.save):
            os.makedirs(self.config.save)
        with codecs.open(os.path.join(self.config.save, 'errors.json'), 'w', encoding='utf-8') as file:
            json.dump(self.results_dict, file, separators=(',', ':'))


    