import numpy as np
import xgboost as xgb

from naslib.predictors.utils.encodings import encode
from naslib.predictors.predictor import Predictor


class XGBoost(Predictor):

    def __init__(self, encoding_type='adjacency_one_hot', ss_type='nasbench201'):
        self.encoding_type = encoding_type
        self.ss_type = ss_type

    def get_params(self, params=None):
        if params is None:
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

    def fit(self, xtrain, ytrain, params=None, **kwargs):

        # normalize accuracies
        self.mean = np.mean(ytrain)
        self.std = np.std(ytrain)

        xtrain = np.array([encode(arch, encoding_type=self.encoding_type,
                                  ss_type=self.ss_type) for arch in xtrain])
        ytrain = np.array(ytrain)

        # convert to xgb dataset
        xgb_train = xgb.DMatrix(xtrain, label=((ytrain-self.mean)/self.std))

        # get params
        params = self.get_params()
        self.model = xgb.train(params, xgb_train,
                               num_boost_round=100) #NOTE: in nb301 num_rounds=20000

        train_pred = np.squeeze(self.model.predict(xgb_train))
        train_error = np.mean(abs(train_pred-ytrain))
        return train_error

    def query(self, xtest, info=None):
        xtest = np.array([encode(arch, encoding_type=self.encoding_type,
                                 ss_type=self.ss_type) for arch in xtest])
        xgb_test = xgb.DMatrix(xtest)
        return np.squeeze(self.model.predict(xgb_test)) * self.std + self.mean

