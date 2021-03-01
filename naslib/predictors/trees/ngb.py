import numpy as np
from functools import wraps

from ngboost import NGBRegressor
from ngboost.distns import Normal
from ngboost.scores import LogScore
from sklearn.tree import DecisionTreeRegressor

from naslib.predictors.trees import BaseTree


def parse_params(params, identifier='base'):
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
            'param:n_estimators': 505,
            'param:learning_rate': 0.08127053060223186,
            'param:minibatch_frac': 0.5081694143793387,
            'base:max_depth': 6,
            'base:max_features': 0.7920456318712875,
            #'base:min_samples_leaf': 15,
            #'base:min_samples_split': 20,
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
            'param:n_estimators': int(loguniform(128, 512)),
            'param:learning_rate': loguniform(.001, .1),
            'param:minibatch_frac': np.random.uniform(.1, 1),
            'base:max_depth': np.random.choice(24) + 1,
            'base:max_features': np.random.uniform(.1, 1)
            }
        self.hyperparams = params
        return params

    def get_dataset(self, encodings, labels=None):
        if labels is None:
            return encodings
        else:
            return (encodings, (labels-self.mean)/self.std)


    def train(self, train_data):
        X_train, y_train = train_data
        min_samples_leaf = min(max(len(X_train)//2, 1), 15)
        min_samples_split = min(max(len(X_train)//2, 2), 20)
        
        base_learner = DecisionTreeRegressor(criterion='friedman_mse', 
                                             min_samples_leaf=min_samples_leaf, 
                                             min_samples_split=min_samples_split,
                                             random_state=None,
                                             splitter='best',
                                             **parse_params(self.hyperparams, identifier='base:'))
        model = NGBRegressor(Dist=Normal, Base=base_learner, Score=LogScore,
                             verbose=True, **parse_params(self.hyperparams, identifier='param:'))

        return model.fit(X_train, y_train)


    def fit(self, xtrain, ytrain, train_info=None, params=None, **kwargs):    
        if self.hyperparams is None:
            self.hyperparams = self.default_hyperparams.copy()
        return super(NGBoost, self).fit(xtrain, ytrain, train_info, params, **kwargs)