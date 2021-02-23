import numpy as np
from functools import wraps

from ngboost import NGBRegressor
from ngboost.distns import Normal
from ngboost.scores import LogScore
from sklearn.tree import DecisionTreeRegressor

from naslib.predictors.trees import BaseTree


def parse_params(func):
    @wraps(func)
    def wrapper(*args, identifier='base', **kwargs):
        params = dict()
        for k, v in func(*args, **kwargs).items():
            if k.startswith(identifier):
                params[k.replace(identifier, "")] = v
        return params
    return wrapper

def loguniform(low=0, high=1, size=None):
    return np.exp(np.random.uniform(np.log(low), np.log(high), size))


class NGBoost(BaseTree):

    @parse_params
    def parameters(self):
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
                                             **self.parameters(identifier='base:'))
        model = NGBRegressor(Dist=Normal, Base=base_learner, Score=LogScore,
                             verbose=True, **self.parameters(identifier='param:'))

        return model.fit(X_train, y_train)


    def fit(self, xtrain, ytrain, train_info=None, params=None, **kwargs):
        return super(NGBoost, self).fit(xtrain, ytrain, params, **kwargs)


