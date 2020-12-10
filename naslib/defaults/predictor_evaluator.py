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
  
        self.train_size = config.train_size
        self.test_size = config.test_size
        self.dataset = config.dataset
        
        self.metric = Metric.VAL_LOSS
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
    def adapt_search_space(self, search_space, scope=None):
        assert search_space.QUERYABLE, "PredictorEvaluator is currently only implemented for benchmarks."
        self.search_space = search_space.clone()
        self.scope = scope if scope else search_space.OPTIMIZER_SCOPE
        
    def load_dataset(self):
        
        xdata = []
        ydata = []
        for _ in range(self.train_size):
            arch = sample_random_architecture(self.search_space, self.scope)
            accuracy = arch.query(metric=self.metric, dataset=self.dataset)
            xdata.append(arch)
            ydata.append(accuracy)
        return xdata, ydata
        
    def evaluate(self):

        # pre-process the predictor
        self.predictor.pre_process()
        
        logger.info("Load the training set")
        xtrain, ytrain = self.load_dataset()

        # fit the predictor (for model-based methods)
        logger.info("Fit the predictor")

        self.predictor.fit(xtrain, ytrain)
        
        logger.info("Load the test set")
        xtest, ytest = self.load_dataset()
        
        """
        Train the architectures in the test set partially
        if required by the predictor (e.g. learning curve extrapolators).
        Note that Predictors cannot train architectures themselves,
        so the partial training must be passed in as an argument.
        """
        # if self.predictor.requires_partial_training():
        #     logger.info("Perform partial training")
        #     fidelity = self.predictor.get_fidelity()
        #     metric = self.predictor.get_metric()
        #     if self.predictor.name in 'SoLoss':
        #         info = [arch.query(metric, self.dataset, epoch=200)[:fidelity] for arch in xtest]
        #     elif self.predictor.name == 'LcSVR':
        #         val_acc_curve = []
        #         arch_params = []
        #         for arch in xtest:
        #             acc_metric = arch.query(metric, self.dataset, epoch=200)[:fidelity]
        #             arch_hp = [arch.query(Metric.RAW, self.dataset)['cifar10-valid'][metric_hp]
        #                         for metric_hp in ['flop', 'latency', 'params', 'epochs']]
        #             val_acc_curve.append(acc_metric)
        #             arch_params.append(arch_hp)
        #         info = {'val_acc': val_acc_curve, 'arch_param': arch_params}
        #     else:
        #         info = [arch.query(metric, self.dataset, epoch=fidelity) for arch in xtest]
        # else:
        #     info = None

        info = self.predictor.requires_partial_training(xtest, self.dataset)

        # query each architecture in the test set
        test_pred = self.predictor.query(xtest, info)
        test_pred = np.squeeze(test_pred)
        
        # this if statement is because of ensembles. TODO: think of a better solution
        if len(test_pred.shape) > 1:
            test_pred = np.mean(test_pred, axis=0)
        
        logger.info("Compute evaluation metrics")
        test_error, correlation = self.compare(ytest, test_pred)
        logger.info("test error: {}, correlation: {}".format(test_error, correlation))
        """
        TODO: also return timing information
        (for preprocessing, training train set, and querying test set).
        start_time = time.time()
        TODO: log results to json
        """
        
    def compare(self, ytest, test_pred):
        
        test_error = np.mean(abs(test_pred-ytest))
        correlation = np.corrcoef(np.array(ytest), np.array(test_pred))[1,0]
        #self.errors_dict.valid_acc.append(valid_acc)
        return test_error, correlation
        
    # TODO: log results
    def _log_to_json(self):
        """log statistics to json file"""
        if not os.path.exists(self.config.save):
            os.makedirs(self.config.save)
        with codecs.open(os.path.join(self.config.save, 'errors.json'), 'w', encoding='utf-8') as file:
            json.dump(self.errors_dict, file, separators=(',', ':'))


    