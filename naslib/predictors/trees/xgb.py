import xgboost as xgb
import numpy as np

from naslib.predictors.trees import BaseTree


def loguniform(low=0, high=1, size=None):
    return np.exp(np.random.uniform(np.log(low), np.log(high), size))


class XGBoost(BaseTree):

    @property
    def default_hyperparams(self):
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': "rmse",
            'booster': 'gbtree', 
            'max_depth': 6,
            'min_child_weight': 1,
            'colsample_bytree': 1,
            'learning_rate': .3,
            'colsample_bylevel': 1
        }
        return params

    def set_random_hyperparams(self):
        if self.hyperparams is None:
            # evaluate the default config first during HPO
            params = self.default_hyperparams.copy()
        else:
            params = {
                'objective': 'reg:squarederror',
                'eval_metric': "rmse",
                #'early_stopping_rounds': 100,
                'booster': 'gbtree',
                'max_depth': int(np.random.choice(range(1,15))),
                'min_child_weight': int(np.random.choice(range(1,10))),
                'colsample_bytree': np.random.uniform(.0, 1.0),
                'learning_rate': loguniform(.001, .5),
                #'alpha': 0.24167936088332426,
                #'lambda': 31.393252465064943,
                'colsample_bylevel': np.random.uniform(.0, 1.0),
            }
        self.hyperparams = params
        return params

    def get_dataset(self, encodings, labels=None):
        if labels is None:
            return xgb.DMatrix(encodings)
        else:
            return xgb.DMatrix(encodings, label=((labels-self.mean)/self.std))

    def train(self, train_data):
        return xgb.train(self.hyperparams, train_data, num_boost_round=500)

    def predict(self, data):
        return self.model.predict(self.get_dataset(data))

    def fit(self, xtrain, ytrain, train_info=None, params=None, **kwargs):
        if self.hyperparams is None:
            self.hyperparams = self.default_hyperparams.copy()
        return super(XGBoost, self).fit(xtrain, ytrain, train_info, params, **kwargs)

    def set_pre_computations(self, unlabeled=None, xtrain_zc_info=None, xtest_zc_info=None, unlabeled_zc_info=None):
        if xtrain_zc_info is not None:
            self.xtrain_zc_info = xtrain_zc_info
            self._verify_zc_info(xtrain_zc_info['zero_cost_scores'])
            self._set_zc_names(xtrain_zc_info['zero_cost_scores'])
            self.zc_features = self.create_zc_feature_vector(xtrain_zc_info['zero_cost_scores'])
        
    def _verify_zc_info(self, zero_cost_scores):
        zc_names = [set(zc_scores.keys()) for zc_scores in zero_cost_scores]
    
        assert len(zc_names) > 0, 'No ZC values found in zero_cost_scores'
        assert zc_names.count(zc_names[0]) == len(zc_names), 'All models do not have the same number of ZC values'

    def _set_zc_names(self, zero_cost_scores):
        zc_names = sorted(zero_cost_scores[0].keys())
        self.zc_names = zc_names

        
