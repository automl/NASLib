import codecs
import os
import json
import torch
import numpy as np
import logging
from naslib.predictors.utils.encodings import encode
from naslib.predictors.zerocost import ZeroCost
from naslib.search_spaces.core.query_metrics import Metric

from naslib.utils import compute_scores

logger = logging.getLogger(__name__)

class ZCEnsembleEvaluator(object):
    def __init__(self, n_train, n_test, zc_names):
        self.n_train = n_train
        self.n_test = n_test
        self.zc_names = zc_names
        self.performance_metric = Metric.VAL_ACCURACY

    def _compute_zc_scores(self, encoding, predictors, train_loader):
        zc_scores = {}
        graph = self.search_space.clone()
        graph.set_spec(encoding)
        graph.parse()

        for predictor in predictors:
            score = predictor.query(graph, train_loader)
            zc_scores[predictor.method_type] = score

        del(graph)

        return zc_scores

    def _sample_new_model(self):
        model = torch.nn.Module()
        graph = self.search_space.clone()
        graph.sample_random_architecture(dataset_api=self.dataset_api)
        model.accuracy = graph.query(self.performance_metric, self.dataset, dataset_api=self.dataset_api)
        model.arch = graph.get_hash()

        del(graph)
        return model

    def _log_to_json(self, results, filepath):
        """log statistics to json file"""
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        with codecs.open(
            os.path.join(filepath, "scores.json"), "w", encoding="utf-8"
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

    def sample_random_models(self, n):
        models = [self._sample_new_model() for _ in range(n)]
        return models

    def compute_zc_scores(self, models, zc_predictors, train_loader):
        for model in models:
            model.zc_scores = self._compute_zc_scores(model.arch, zc_predictors, train_loader)

    def evaluate(self, ensemble, train_loader):
        # Load models to train
        train_models = self.sample_random_models(self.n_train)

        # Get their ZC scores
        zc_predictors = [ZeroCost(method_type=zc_name) for zc_name in self.zc_names]
        self.compute_zc_scores(train_models, zc_predictors, train_loader)

        # Set ZC results as precomputations and fit the ensemble
        train_info = {'zero_cost_scores': [m.zc_scores for m in train_models]}
        ensemble.set_pre_computations(xtrain_zc_info=train_info)

        xtrain = []
        for m in train_models:
            g = self.search_space.clone()
            g.set_spec(m.arch)
            g.parse()
            xtrain.append(encode(g, encoding_type='adjacency_one_hot', ss_type=g.get_type()))
            del(g)

        ytrain = [m.accuracy for m in train_models]

        ensemble.fit(xtrain, ytrain)

        # Get the feature importance
        # self.ensemble[0].feature_importance

        # Sample test models, query zc scores
        test_models = self.sample_random_models(self.n_test)
        self.compute_zc_scores(test_models, zc_predictors, train_loader)

        # Query the ensemble for the predicted accuracy
        x_test = []

        for m in test_models:
            g = self.search_space.clone()
            g.set_spec(m.arch)
            x_test.append(encode(g, encoding_type='one_hot_adjacency', ss_type=g.get_type()))
            del(g)

        test_info = [{'zero_cost_scores': m.zc_scores} for m in test_models]
        preds = np.mean(ensemble.query(x_test, test_info), axis=0)

        # Compute scores
        ground_truths = [m.accuracy for m in test_models]
        scores = compute_scores(ground_truths, preds)

        model = ensemble.ensemble[0].model
        feature_importances = model.get_fscore()
        feature_mapping = ensemble.ensemble[0].zc_to_features_map

        zc_feature_importances = {zc_name: 0 for zc_name in self.zc_names}
        for zc_name, feature_name in feature_mapping.items():
            if feature_name in feature_importances:
                zc_feature_importances[zc_name] = feature_importances[feature_name]

        scores['zc_feature_importances'] = zc_feature_importances
        scores['feature_importances'] = feature_importances

        logger.info(f'ZC feature importances: {zc_feature_importances}')
        self._log_to_json([self.config, scores], self.config.save)
