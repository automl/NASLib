import xgboost as xgb
import numpy as np
import os
import json

from naslib.predictors.trees.ngb import loguniform
from naslib.predictors.trees import BaseTree


class XGBoost(BaseTree):
    @property
    def default_hyperparams(self):
        params = {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "booster": "gbtree",
            "max_depth": 6,
            "min_child_weight": 1,
            "colsample_bytree": 1,
            "learning_rate": 0.3,
            "colsample_bylevel": 1,
        }
        return params

    def set_random_hyperparams(self):
        if self.hyperparams is None:
            # evaluate the default config first during HPO
            params = self.default_hyperparams.copy()
        else:
            params = {
                "objective": "reg:squarederror",
                "eval_metric": "rmse",
                #'early_stopping_rounds': 100,
                "booster": "gbtree",
                "max_depth": int(np.random.choice(range(1, 15))),
                "min_child_weight": int(np.random.choice(range(1, 10))),
                "colsample_bytree": np.random.uniform(0.0, 1.0),
                "learning_rate": loguniform(0.001, 0.5),
                #'alpha': 0.24167936088332426,
                #'lambda': 31.393252465064943,
                "colsample_bylevel": np.random.uniform(0.0, 1.0),
            }
        self.hyperparams = params
        return params

    def get_dataset(self, encodings, labels=None):
        if labels is None:
            return xgb.DMatrix(encodings)
        else:
            return xgb.DMatrix(encodings, label=((labels - self.mean) / self.std))

    def train(self, train_data):
        return xgb.train(self.hyperparams, train_data, num_boost_round=500)

    def predict(self, data):
        return self.model.predict(self.get_dataset(data))

    def fit(self, xtrain, ytrain, train_info=None, params=None, **kwargs):
        if self.hparams_from_file and self.hparams_from_file not in ['False', 'None'] \
        and os.path.exists(self.hparams_from_file):
            self.hyperparams = json.load(open(self.hparams_from_file, 'rb'))['xgb']
            print('loaded hyperparams from', self.hparams_from_file)
        elif self.hyperparams is None:
            self.hyperparams = self.default_hyperparams.copy()
        return super(XGBoost, self).fit(xtrain, ytrain, train_info, params, **kwargs)
