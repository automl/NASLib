import codecs
import json
import os
import logging
import torch
import numpy as np

from naslib.search_spaces.core.query_metrics import Metric

logger = logging.getLogger(__name__)


class StatisticsEvaluator(object):
    """
    This class will evaluate statistics for a
    given search space.
    """

    def __init__(self, config=None):

        self.config = config
        self.max_set_size = config.max_set_size
        self.bucket_sizes = config.bucket_sizes
        self.dataset = config.dataset
        self.run_acc_stats = config.run_acc_stats
        self.run_autocorrelation = config.run_autocorrelation

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.results = [config, {}]

    def adapt_search_space(
        self, search_space, load_labeled, scope=None, dataset_api=None
    ):
        self.search_space = search_space.clone()
        self.scope = scope if scope else search_space.OPTIMIZER_SCOPE
        self.load_labeled = load_labeled
        self.dataset_api = dataset_api

    # TODO: make this a classmethod in predictor_evaluator.py
    def get_full_arch_info(self, arch):
        """
        Given an arch, return the validation accuracy, 
        test accuracy, train_time, and other info
        """
        info_dict = {}
        val_acc = arch.query(
            metric=Metric.VAL_ACCURACY, dataset=self.dataset, dataset_api=self.dataset_api
        )
        test_acc = arch.query(
            metric=Metric.TEST_ACCURACY, dataset=self.dataset, dataset_api=self.dataset_api
        )
        train_time = arch.query(
            metric=Metric.TRAIN_TIME, dataset=self.dataset, dataset_api=self.dataset_api
        )
        return {'val_acc':val_acc, 'test_acc':test_acc, 'train_time':train_time}

    def evaluate_acc_stats(self):

        info_dicts = []
        i = 0
        for arch_spec in self.search_space.get_arch_iterator(dataset_api=self.dataset_api):
            arch = self.search_space.clone()
            arch.set_spec(arch_spec)
            info_dict = self.get_full_arch_info(arch)
            info_dicts.append(info_dict)
            i += 1
            if i % 1000 == 0:
                logger.info('at {}'.format(i))
            if i >= self.max_set_size:
                break

        val_accs = np.array([info_dict['val_acc'] for info_dict in info_dicts])
        test_accs = np.array([info_dict['test_acc'] for info_dict in info_dicts])

        mean = np.mean(val_accs)
        std = np.std(val_accs)
        minimum = np.min(val_accs)
        maximum = np.max(val_accs)
        # TODO: quartiles, buckets, val_test_corr

        logger.info(
            "mean: {}, std: {}, min: {}, max: {}".format(
                mean, std, minimum, maximum
            )
        )

        self.results[1]['mean'] = mean
        self.results[1]['std'] = std
        self.results[1]['minimum'] = minimum
        self.results[1]['maximum'] = maximum

    def evaluate_autocorrelation(self):
        return NotImplementedError()
        
    def evaluate(self):

        if self.run_acc_stats:
            logger.info('compute acc stats')
            self.evaluate_acc_stats()
            
        if self.run_autocorrelation:
            print('autocorrelation is not yet implemented')
            #self.evaluate_autocorrelation()

        self._log_to_json()
        return self.results

    # TODO: make this a classmethod in predictor_evaluator.py
    def _log_to_json(self):
        """log statistics to json file"""
        if not os.path.exists(self.config.save):
            os.makedirs(self.config.save)
        with codecs.open(
            os.path.join(self.config.save, "statistics.json"), "w", encoding="utf-8"
        ) as file:
            json.dump(self.results, file, separators=(",", ":"))
