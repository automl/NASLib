import xgboost as xgb
import numpy as np

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
            return xgb.DMatrix(encodings, label=labels)

    def train(self, train_data, num_boost_round=500):
        return xgb.train(self.hyperparams, train_data,
                         num_boost_round=num_boost_round)

    def predict(self, data):
        return self.model.predict(self.get_dataset(data))

    def fit(self, xtrain, ytrain, train_info=None, params=None, **kwargs):
        if self.hyperparams is None:
            self.hyperparams = self.default_hyperparams.copy()

        xtrain = np.array(xtrain)
        ytrain = np.array(ytrain)

        # convert to the right representation
        train_data = self.get_dataset(xtrain, ytrain)

        # fit to the training data
        self.model = self.train(train_data, **kwargs)

        # predict
        train_pred = np.squeeze(self.predict(xtrain))
        train_error = np.mean(abs(train_pred - ytrain))

        return train_error
