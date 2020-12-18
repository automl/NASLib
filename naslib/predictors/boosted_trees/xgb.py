import xgboost as xgb
from naslib.predictors.boosted_trees import BaseTree


class XGBoost(BaseTree):

    @property
    def parameters(self):
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': "rmse",
            #'early_stopping_rounds': 100,
            'booster': 'gbtree',
            #NOTE: if using these hyperparameters XGB predicts the same
            # values always on NB201
            #'max_depth': 13,
            #'min_child_weight': 39,
            #'colsample_bytree': 0.2545374925231651,
            #'learning_rate': 0.008237525103357958,
            #'alpha': 0.24167936088332426,
            #'lambda': 31.393252465064943,
            #'colsample_bylevel': 0.6909224923784677,
            #'verbose': -1
        }
        return params


    def get_dataset(self, encodings, labels=None):
        if labels is None:
            return xgb.DMatrix(encodings)
        else:
            return xgb.DMatrix(encodings, label=((labels-self.mean)/self.std))


    def train(self, train_data):
        #NOTE: in nb301 num_boost_round=20000
        return xgb.train(self.parameters, train_data, num_boost_round=100)

    def predict(self, data):
        return self.model.predict(self.get_dataset(data))

    def fit(self, xtrain, ytrain, params=None, **kwargs):
        return super(XGBoost, self).fit(xtrain, ytrain, params, **kwargs)
