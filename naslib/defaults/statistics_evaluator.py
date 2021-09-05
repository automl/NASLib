import codecs
import json
import os
import logging
import torch
import random
import collections
import numpy as np
from scipy import stats

from naslib.search_spaces.core.query_metrics import Metric

logger = logging.getLogger(__name__)


class StatisticsEvaluator(object):
    """
    This class will evaluate statistics for a
    given search space.
    """

    def __init__(self, config=None):

        self.config = config
        self.dataset = config.dataset

        self.run_acc_stats = config.run_acc_stats
        self.max_set_size = config.max_set_size
        self.bucket_sizes = config.bucket_sizes

        self.run_nbhd_size = config.run_nbhd_size
        self.max_nbhd_trials = config.max_nbhd_trials

        self.run_autocorr = config.run_autocorr
        self.max_autocorr_trials = config.max_autocorr_trials

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.results = [config, {}]

    def adapt_search_space(
        self, search_space, load_labeled, scope=None, dataset_api=None
    ):
        self.search_space = search_space.clone()
        self.scope = scope if scope else search_space.OPTIMIZER_SCOPE
        self.load_labeled = load_labeled
        self.dataset_api = dataset_api
        self.arch_iterator = self.search_space.get_arch_iterator(dataset_api=self.dataset_api)
        max_size = max(self.max_set_size, self.max_nbhd_trials, self.max_autocorr_trials)
        self.arch_iterator = list(self.arch_iterator)[:max_size]
        random.shuffle(self.arch_iterator)

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

        # loop through the full set of accuracies (or up to max_set_size)
        info_dicts = []
        i = 0
        for arch_spec in self.arch_iterator:
            i += 1
            if i % (self.max_set_size // 10) == 0:
                logger.info('acc stats trial {}'.format(i))

            arch = self.search_space.clone()
            arch.set_spec(arch_spec)
            arch.set_load_labeled()
            info_dict = self.get_full_arch_info(arch)
            info_dicts.append(info_dict)

        val_accs = np.array([info_dict['val_acc'] for info_dict in info_dicts])
        test_accs = np.array([info_dict['test_acc'] for info_dict in info_dicts])
        runtimes = np.array([info_dict['train_time'] for info_dict in info_dicts])

        # compute val accuracy statistics
        self.results[1]['mean'] = np.mean(val_accs)
        self.results[1]['std'] = np.std(val_accs)
        self.results[1]['minimum'] = np.min(val_accs)
        self.results[1]['maximum'] = np.max(val_accs)
        self.results[1]['median'] = np.median(val_accs)
        self.results[1]['mean_test'] = np.mean(test_accs)
        self.results[1]['std_test'] = np.std(test_accs)
        self.results[1]['mean_time'] = np.mean(runtimes)
        self.results[1]['std_time'] = np.std(runtimes)
        
        self.results[1]['boxplot'] = [np.percentile(val_accs, 25), 
                                      np.median(val_accs), 
                                      np.percentile(val_accs, 75)
                                     ]

        hist_20 = np.histogram(val_accs, 20)
        hist_30 = np.histogram(val_accs, 30)
        hist_50 = np.histogram(val_accs, 50)
        self.results[1]['hist_20'] = [hist_20[0].tolist(), hist_20[1].tolist()]
        self.results[1]['hist_30'] = [hist_30[0].tolist(), hist_30[1].tolist()]
        self.results[1]['hist_50'] = [hist_50[0].tolist(), hist_50[1].tolist()] 

        # compute val - test correlation
        self.results[1]['pearson'] = np.abs(np.corrcoef(val_accs, test_accs)[1, 0])
        self.results[1]['spearman'] = stats.spearmanr(val_accs, test_accs)[0]
        self.results[1]['kendalltau'] = stats.kendalltau(val_accs, test_accs)[0]
        
        # sanity check: return first 10 accs and length
        self.results[1]['val_accs_10'] = val_accs[:10].tolist()
        self.results[1]['test_accs_10'] = test_accs[:10].tolist()
        self.results[1]['size'] = len(val_accs)

        logger.info(
            "mean: {}, std: {}, min: {}, max: {}".format(
                self.results[1]['mean'], 
                self.results[1]['std'], 
                self.results[1]['minimum'], 
                self.results[1]['maximum']
            )
        )

    def evaluate_nbhd_sizes(self):
        nbhd_sizes = []
        i = 0
        for arch_spec in self.arch_iterator[:self.max_nbhd_trials]:
            i += 1
            if i % (self.max_nbhd_trials // 10) == 0:
                logger.info('nbhd trial {}'.format(i))

            arch = self.search_space.clone()
            arch.set_spec(arch_spec)
            arch.set_load_labeled()
            nbhd = arch.get_nbhd(dataset_api=self.dataset_api)
            nbhd_sizes.append(len(nbhd))

        self.results[1]['nbhd_size_mean'] = np.mean(nbhd_sizes)
        self.results[1]['nbhd_size_std'] = np.std(nbhd_sizes)
        logger.info("nbhd mean: {}, std: {}".format(
            self.results[1]['nbhd_size_mean'], 
            self.results[1]['nbhd_size_std'])) 

    def evaluate_autocorr(self, walks=100, size=36):

        corrs = []
        i = 0
        for arch_spec in self.arch_iterator[:self.max_autocorr_trials]:
            i += 1
            if i % (self.max_nbhd_trials // 10) == 0:
                logger.info('autocorr trial {}'.format(i))

            arch = self.search_space.clone()
            arch.set_spec(arch_spec)
            arch.set_load_labeled()
            val_acc = arch.query(metric=Metric.VAL_ACCURACY, 
                                 dataset=self.dataset, 
                                 dataset_api=self.dataset_api)

            # first create the initial window
            window = collections.deque([val_acc])
            for _ in range(size - 1):
                new_arch = self.search_space.clone()
                new_arch.mutate(arch)
                new_arch.set_load_labeled()
                val_acc = new_arch.query(metric=Metric.VAL_ACCURACY,
                                         dataset=self.dataset,
                                         dataset_api=self.dataset_api)
                window.append(val_acc)
                arch = new_arch

            # perform a random walk
            autocorrs = np.zeros((size, walks, 2))
            for t in range(walks):
                new_arch = self.search_space.clone()
                new_arch.mutate(arch)
                new_arch.set_load_labeled()
                val_acc = new_arch.query(metric=Metric.VAL_ACCURACY,
                                         dataset=self.dataset,
                                         dataset_api=self.dataset_api)
                window.append(val_acc)
                window.popleft()
                arch = new_arch
                autocorrs[:, t, 0] = np.array([window[-1]] * size)
                autocorrs[:, t, 1] = np.array(window)

            # compute autocorrelations
            corr = []
            for j in range(size):
                corr.append(np.corrcoef(autocorrs[j, :, 0], autocorrs[j, :, 1])[1,0])
            print(i, corr)
            corrs.append(corr)

        self.results[1]['autocorr_mean'] = np.mean(corrs, axis=0).tolist()
        self.results[1]['autocorr_std'] = np.std(corrs, axis=0).tolist()
        logger.info("autocorr mean: {}, std: {}".format(
            self.results[1]['autocorr_mean'],
            self.results[1]['autocorr_std']))
        self.results[1]['autocorr_x'] = [float(np.power(size - i - 1, 1/2)) for i in range(size)]
        
    def evaluate(self):

        if self.run_acc_stats:
            logger.info('compute acc stats')
            self.evaluate_acc_stats()

        if self.run_nbhd_size:
            logger.info('compute nbhd sizes')
            self.evaluate_nbhd_sizes()

        if self.run_autocorr:
            logger.info('compute autocorrelation')
            self.evaluate_autocorr()

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
