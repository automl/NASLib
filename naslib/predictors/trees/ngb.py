import numpy as np
from functools import wraps

from ngboost import NGBRegressor
from ngboost.distns import Normal
from ngboost.scores import LogScore
from sklearn.tree import DecisionTreeRegressor

from naslib.predictors.trees import BaseTree


def parse_params(params, identifier="base"):
    parsed_params = {}
    for k, v in params.items():
        if k.startswith(identifier):
            parsed_params[k.replace(identifier, "")] = v
    return parsed_params


def loguniform(low=0, high=1, size=None):
    return np.exp(np.random.uniform(np.log(low), np.log(high), size))


class NGBoost(BaseTree):
    @property
    def default_hyperparams(self):
        params = {
            "param:n_estimators": 505,
            "param:learning_rate": 0.08127053060223186,
            "base:max_depth": 6,
            "base:max_features": 0.7920456318712875,
            #'early_stopping_rounds': 100,
            #'verbose': -1
        }
        return params

    def set_random_hyperparams(self):
        if self.hyperparams is None:
            # evaluate the default config first during HPO
            params = self.default_hyperparams.copy()
        else:
            params = {
                "param:n_estimators": int(loguniform(128, 512)),
                "param:learning_rate": loguniform(0.001, 0.1),
                "base:max_depth": np.random.choice(24) + 1,
                "base:max_features": np.random.uniform(0.1, 1),
            }
        self.hyperparams = params
        return params

    def get_dataset(self, encodings, labels=None):
        if labels is None:
            return encodings
        else:
            return (encodings, (labels - self.mean) / self.std)

    def train(self, train_data):
        X_train, y_train = train_data
        # note: cross-validation will error unless these values are set:
        min_samples_leaf = 1
        min_samples_split = 2
        minibatch_frac = 0.5

        base_learner = DecisionTreeRegressor(
            criterion="friedman_mse",
            min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_split,
            random_state=None,
            splitter="best",
            **parse_params(self.hyperparams, identifier="base:")
        )
        model = NGBRegressor(
            Dist=Normal,
            Base=base_learner,
            Score=LogScore,
            minibatch_frac=minibatch_frac,
            verbose=True,
            **parse_params(self.hyperparams, identifier="param:")
        )

        return model.fit(X_train, y_train)

    def fit(self, xtrain, ytrain, train_info=None, params=None, **kwargs):
        if self.hyperparams is None:
            self.hyperparams = self.default_hyperparams.copy()
        return super(NGBoost, self).fit(xtrain, ytrain, train_info, params, **kwargs)
