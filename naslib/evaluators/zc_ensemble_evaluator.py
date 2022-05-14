import codecs
import os
import json
import torch
import numpy as np
import logging
from naslib.predictors.zerocost import ZeroCost
from naslib.search_spaces.core.query_metrics import Metric


logger = logging.getLogger(__name__)

class ZCEnsembleEvaluator(object):
    def __init__(self, n_train, n_test, zc_names):
        self.n_train = n_train
        self.n_test = n_test
        self.zc_names = zc_names
        self.performance_metric = Metric.VAL_ACCURACY

        self.benchmarks = {}

    def _compute_zc_scores(self, model, predictors, train_loader):
        zc_scores = {}
        for predictor in predictors:
            score = predictor.query(model, train_loader)
            zc_scores[predictor.method_type] = score

        return zc_scores

    def _sample_new_model(self, train_loader):
        model = torch.nn.Module()
        model.arch = self.search_space.clone()
        model.arch.sample_random_architecture(dataset_api=self.dataset_api)
        model.arch.parse()
        model.accuracy = model.arch.query(self.performance_metric, self.dataset, dataset_api=self.dataset_api)

        zc_predictors = [ZeroCost(method_type=zc_name) for zc_name in self.zc_names]

        zc_scores = self._compute_zc_scores(model.arch, zc_predictors, train_loader)
        encoding = model.arch.get_hash()

        zc_scores['val_accuracy'] = model.accuracy
        self.benchmarks[str(encoding)] = zc_scores

        self._log_to_json([self.benchmarks], self.config.save, 'intermediate_benchmark.json')

        del(model)

        return None

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

    def sample_random_models(self, n, train_loader):
        models = [self._sample_new_model(train_loader) for _ in range(n)]
        return models

    def evaluate(self, ensemble, train_loader):
        # Load models to train
        self.sample_random_models(self.n_train, train_loader)
        self._log_to_json([self.benchmarks], self.config.save, 'benchmark.json')
        logger.info('Benchmark creation complete.')

