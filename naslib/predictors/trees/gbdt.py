# Author: Yang Liu @ Abacus.ai
# This is an implementation of GBDT predictor for NAS from the paper:
# Luo, Renqian, et al. "Neural architecture search with gbdt." arXiv preprint arXiv:2007.04785 (2020).

import numpy as np
import lightgbm as lgb

from naslib.predictors.trees import BaseTree

class GBDTPredictor(BaseTree):

    @property
    def parameters(self, params=None):
        if params is None:
            # default parameters used in Luo et al. 2020
            params = {
                'boosting_type': 'gbdt',
                'objective': 'regression',
                'metric': {'l2'},
                'min_data_in_leaf':5, # added by Colin White
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1
            }
        return params


    def get_dataset(self, encodings, labels=None):
        if labels is None:
            return encodings
        else:
            return lgb.Dataset(encodings, label=((labels-self.mean)/self.std))


    def train(self, train_data):
        feature_name = None #get_feature_name_nasbench201()
        return lgb.train(self.parameters, train_data,
                         feature_name=feature_name, num_boost_round=100)

    def predict(self, data):
        return self.model.predict(data,
                                  num_iteration=self.model.best_iteration)

    def fit(self, xtrain, ytrain, train_info=None, params=None, **kwargs):
        return super(GBDTPredictor, self).fit(xtrain, ytrain, params, **kwargs)

