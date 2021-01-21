import time
import numpy as np
import copy
import logging
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from ngboost import NGBRegressor
from ngboost.distns import Normal
from ngboost.scores import LogScore

from naslib.predictors.predictor import Predictor
from naslib.predictors.lcsvr import loguniform
from naslib.predictors.zerocost_estimators import ZeroCostEstimators
from naslib.predictors.utils.encodings import encode
from naslib.utils import utils
from naslib.search_spaces.core.query_metrics import Metric

logger = logging.getLogger(__name__)

def parse_params(params, identifier):
    to_return = {}
    for k, v in params.items():
        if k.startswith(identifier):
            to_return[k.replace(identifier, "")] = v

    return to_return


class OmniPredictor(Predictor):

    def __init__(self, zero_cost, lce, encoding_type, config, n_hypers=50):
        
        self.zero_cost = zero_cost
        self.lce = lce
        self.encoding_type = encoding_type
        self.config = config        
        self.n_hypers = n_hypers
        self.config = config

    def pre_compute(self, xtrain, xtest):
        """
        All of this computation could go into fit() and query(), but we do it
        here to save time, so that we don't have to re-compute Jacobian covariances
        for all train_sizes when running experiment_types that vary train size or fidelity.        
        """
        self.xtrain_zc_info = {}
        self.xtest_zc_info = {}

        if len(self.zero_cost) > 0:
            self.train_loader, _, _, _, _ = utils.get_train_val_loaders(self.config, mode='train')

            for method_name in self.zero_cost:
                zc_method = ZeroCostEstimators(self.config, batch_size=64, method_type=method_name)
                zc_method.train_loader = copy.deepcopy(self.train_loader)
                xtrain_zc_scores = zc_method.query(xtrain)
                xtest_zc_scores = zc_method.query(xtest)
                
                train_mean = np.mean(np.array(xtrain_zc_scores)) 
                train_std = np.std((np.array(xtrain_zc_scores)))
                
                normalized_train = (np.array(xtrain_zc_scores) - train_mean)/train_std
                normalized_test = (np.array(xtest_zc_scores) - train_mean)/train_std
                
                self.xtrain_zc_info[f'{method_name}_scores'] = normalized_train
                self.xtest_zc_info[f'{method_name}_scores'] = normalized_test

    def get_random_params(self):
        params = {
            'param:n_estimators': int(loguniform(128, 512)),
            'param:learning_rate': loguniform(.001, .1),
            'param:minibatch_frac': np.random.uniform(.1, 1),
            'base:max_depth': np.random.choice(24) + 1,
            'base:max_features': np.random.uniform(.1, 1),
            'base:min_samples_leaf': np.random.choice(18) + 2,
            'base:min_samples_split': np.random.choice(18) + 2,
        }
        return params   
    
    def run_hpo(self, xtrain, ytrain):
        min_score = 100000
        best_params = None
        for i in range(self.n_hypers):
            params = self.get_random_params()
            for key in ['base:min_samples_leaf', 'base:min_samples_split']:
                params[key] = max(2, min(params[key], int(len(xtrain)/3)-1))
            
            score = self.cross_validate(xtrain, ytrain, params)            
            if score < min_score:
                min_score = score
                best_params = params
                logger.info('{} new best {}, {}'.format(i, score, params))
        return best_params
        
    def cross_validate(self, xtrain, ytrain, params):

        base_learner = DecisionTreeRegressor(criterion='friedman_mse',
                                             random_state=None,
                                             splitter='best',
                                             **parse_params(params, 'base:'))
        model = NGBRegressor(Dist=Normal, Base=base_learner, Score=LogScore,
                             verbose=False, **parse_params(params, 'param:'))
        scores = cross_val_score(model, xtrain, ytrain, cv=3)
        return np.mean(scores)

    def prepare_features(self, xdata, train=True):
        # prepare training data features
        full_xdata = [[] for _ in range(len(xdata))]
        if len(self.zero_cost) > 0: 
            for key in self.xtrain_zc_info:
                if train:
                    full_xdata = [[*x, self.xtrain_zc_info[key][i]] for i, x in enumerate(full_xdata)]
                else:
                    full_xdata = [[*x, self.xtest_zc_info[key][i]] for i, x in enumerate(full_xdata)]

        if self.encoding_type is not None:
            xdata_encoded = np.array([encode(arch, encoding_type=self.encoding_type,
                                             ss_type=self.ss_type) for arch in xdata])            
            full_xdata = [[*x, *xdata_encoded[i]] for i, x in enumerate(full_xdata)]
        
        # todo:
        #if self.lce is not None:
        
        return full_xdata
        
    def fit(self, xtrain, ytrain, info, learn_hyper=True):
        
        # prepare training data labels
        self.mean = np.mean(ytrain)
        self.std = np.std(ytrain)
        ytrain = (np.array(ytrain)-self.mean)/self.std
        xtrain = self.prepare_features(xtrain, train=True)
        params = self.run_hpo(xtrain, ytrain)

        # todo: this code is repeated in cross_validate
        base_learner = DecisionTreeRegressor(criterion='friedman_mse',
                                             random_state=None,
                                             splitter='best',
                                             **parse_params(params, 'base:'))
        self.model = NGBRegressor(Dist=Normal, Base=base_learner, Score=LogScore,
                                  verbose=True, **parse_params(params, 'param:'))
        self.model.fit(xtrain, ytrain)

    def query(self, xtest, info):
        test_data = self.prepare_features(xtest, train=False)

        return np.squeeze(self.model.predict(test_data)) * self.std + self.mean
    
    def get_data_reqs_commented(self):
        """
        Returns a dictionary with info about whether the predictor needs
        extra info to train/query.
        """
        reqs = {'requires_partial_lc':True, 
                'metric':self.metric, 
                'requires_hyperparameters':True, 
                'hyperparams':['flops', 'latency', 'params']
               }
        return reqs