import codecs
import os
import json
import torch
import numpy as np
import logging
import timeit
from naslib.predictors.utils.encodings import encode
from naslib.predictors.zerocost import ZeroCost
from naslib.search_spaces.core.query_metrics import Metric


logger = logging.getLogger(__name__)

class ZCEnsembleEvaluator(object):
    def __init__(self, n_train, n_test, zc_names, zc_api=False):
        self.n_train = n_train
        self.n_test = n_test
        self.zc_names = zc_names
        self.performance_metric = Metric.VAL_ACCURACY
        self.zc_api = zc_api
        self.benchmarks = {}

    def _compute_zc_scores(self, encoding, predictors, train_loader):
        zc_scores = {}

        if self.zc_api is not None:
            zc_results = self.zc_api[str(encoding)]

        for idx, predictor in enumerate(predictors):
            zc_name = predictor.method_type
            if self.zc_api is not None and zc_name in zc_results:
                score = zc_results[zc_name]['score']
                total_time = zc_results[zc_name]['time']
            else:
                logger.info(predictor.method_type)
                graph = self.search_space.clone()
                graph.set_spec(encoding)
                graph.parse()
                start_time = timeit.default_timer()
                score = predictor.query(graph, train_loader)
                end_time = timeit.default_timer()
                total_time = end_time - start_time
                del graph

            zc_scores[zc_name] = {'score': score, 'time': total_time}

        return zc_scores

    def _sample_new_arch(self, train_loader):
        graph = self.search_space.clone()
        graph.sample_random_architecture(dataset_api=self.dataset_api)
        encoding = graph.get_hash()
        del graph

        return encoding

    def compute_scores(self, archs, zc_predictors, train_loader):
        for arch in archs:
            logger.info(f'Computing zero-cost scores for model {arch}')
            zc_scores = self._compute_zc_scores(arch, zc_predictors, train_loader)

            # Query validation accuracy from benchmark
            graph = self.search_space.clone()
            graph.set_spec(arch)
            graph.parse()
            accuracy = graph.query(self.performance_metric, self.dataset, dataset_api=self.dataset_api)

            zc_scores['val_accuracy'] = accuracy
            self.benchmarks[str(arch)] = zc_scores

            self._log_to_json([self.benchmarks], self.config.save, 'intermediate_benchmark.json')

    def _log_to_json(self, results, filepath, filename):
        """log statistics to json file"""
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        with codecs.open(
            os.path.join(filepath, filename), "w", encoding="utf-8"
        ) as file:
            for res in results:
                for key, value in res.items():
                    if type(value) == np.int32 or type(value) == np.int64:
                        res[key] = int(value)
                    if type(value) == np.float32 or type(value) == np.float64:
                        res[key] = float(value)

            json.dump(results, file, separators=(",", ":"))

    def adapt_search_space(self, search_space, dataset, dataset_api, config):
        self.search_space = search_space.clone()
        self.dataset = dataset
        self.dataset_api = dataset_api
        self.config = config

    def sample_random_archs(self, n, train_loader):
        archs = [self._sample_new_arch(train_loader) for _ in range(n)]
        return archs

    def evaluate(self, ensemble, train_loader):
        logger.info(f'Sampling {self.n_train} train models')

        # Load models to train
        archs = self.sample_random_archs(self.n_train, train_loader)
        zc_predictors = [ZeroCost(method_type=zc_name) for zc_name in self.zc_names]

        self.compute_scores(archs, zc_predictors, train_loader)
        archs_hash = hash(str(sorted(list(self.benchmarks.keys()))))
        self._log_to_json([self.benchmarks, {'hash': archs_hash}], self.config.save, 'benchmark.json')
        logger.info('Benchmark creation complete.')
