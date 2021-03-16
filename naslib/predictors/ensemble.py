import numpy as np
import copy

from naslib.predictors.predictor import Predictor
from naslib.predictors.feedforward import FeedforwardPredictor
from naslib.predictors.trees import GBDTPredictor, \
XGBoost, NGBoost, RandomForestPredictor
from naslib.predictors.gcn import GCNPredictor
from naslib.predictors.bonas import BonasPredictor
from naslib.predictors.bnn import DNGOPredictor, BOHAMIANN, \
BayesianLinearRegression
from naslib.predictors.seminas import SemiNASPredictor
from naslib.predictors.gp import GPPredictor, SparseGPPredictor, VarSparseGPPredictor
from naslib.predictors.omni import OmniPredictor
from naslib.predictors.omni_xgb import OmniXGBPredictor
from naslib.predictors.omni_seminas import OmniSemiNASPredictor

class Ensemble(Predictor):
    
    def __init__(self, 
                 encoding_type=None,
                 num_ensemble=3, 
                 predictor_type=None,
                 ss_type=None, 
                 hpo_wrapper=True, 
                 config=None):
        self.num_ensemble = num_ensemble
        self.predictor_type = predictor_type
        self.encoding_type = encoding_type
        self.ss_type = ss_type
        self.hpo_wrapper = hpo_wrapper
        self.config = config
        self.hyperparams = None
        self.ensemble = None

    def get_ensemble(self):
        # TODO: if encoding_type is not None, set the encoding type

        trainable_predictors = {
            'bananas': FeedforwardPredictor(ss_type=self.ss_type,
                                            encoding_type='path'),
            'feedforward': FeedforwardPredictor(ss_type=self.ss_type,
                                                encoding_type='adjacency_one_hot'),
            'gbdt': GBDTPredictor(ss_type=self.ss_type,
                                  encoding_type='adjacency_one_hot'),
            'gcn': GCNPredictor(ss_type=self.ss_type,
                                encoding_type='gcn'),
            'bonas': BonasPredictor(ss_type=self.ss_type,
                                    encoding_type='bonas'),
            'xgb': XGBoost(ss_type=self.ss_type, zc=False,
                           encoding_type='adjacency_one_hot'),
            'ngb': NGBoost(ss_type=self.ss_type,
                           encoding_type='adjacency_one_hot'),
            'rf': RandomForestPredictor(ss_type=self.ss_type,
                                        encoding_type='adjacency_one_hot'),
            'dngo': DNGOPredictor(ss_type=self.ss_type,
                                  encoding_type='adjacency_one_hot'),
            'bohamiann': BOHAMIANN(ss_type=self.ss_type,
                                   encoding_type='adjacency_one_hot'),
            'bayes_lin_reg': BayesianLinearRegression(ss_type=self.ss_type,
                                                      encoding_type='adjacency_one_hot'),
            'seminas': SemiNASPredictor(ss_type=self.ss_type, semi=True,
                                        encoding_type='seminas'),
            'nao': SemiNASPredictor(ss_type=self.ss_type, semi=False,
                                    encoding_type='seminas'),
            'gp': GPPredictor(ss_type=self.ss_type,
                              encoding_type='adjacency_one_hot'),
            'sparse_gp': SparseGPPredictor(ss_type=self.ss_type,
                                           encoding_type='adjacency_one_hot',
                                           optimize_gp_hyper=True,
                                           num_steps=100),
            'var_sparse_gp': VarSparseGPPredictor(ss_type=self.ss_type,
                                                  encoding_type='adjacency_one_hot',
                                                  optimize_gp_hyper=True, 
                                                  num_steps=200, zc=False),
            'omni': OmniPredictor(zero_cost=['jacov'], lce=[], encoding_type='adjacency_one_hot', 
                                  ss_type=self.ss_type, run_pre_compute=False, n_hypers=25, 
                                  min_train_size=0, max_zerocost=100),
            'ngb_hp': OmniPredictor(zero_cost=[], lce=[], encoding_type='adjacency_one_hot', 
                                    ss_type=self.ss_type, run_pre_compute=False, n_hypers=25, 
                                    min_train_size=0, max_zerocost=0),
            'omni_xgb': OmniXGBPredictor(zero_cost=['jacov'], lce=[], encoding_type='adjacency_one_hot', 
                                         ss_type=self.ss_type, run_pre_compute=False, n_hypers=0, 
                                         min_train_size=0, max_zerocost=1000),
            'omni_seminas': OmniSemiNASPredictor(zero_cost=['jacov'], encoding_type='seminas', 
                                                 ss_type=self.ss_type, run_pre_compute=False, semi=True,
                                                 max_zerocost=1000, config=self.config),
        }

        return [copy.deepcopy(trainable_predictors[self.predictor_type]) for _ in range(self.num_ensemble)]

    def fit(self, xtrain, ytrain, train_info=None):
        if self.ensemble is None:
            self.ensemble = self.get_ensemble()

        if self.hyperparams is None and hasattr(self.ensemble[0], 'default_hyperparams'):
            # todo: ideally should implement get_default_hyperparams() for all predictors
            self.hyperparams = self.ensemble[0].default_hyperparams.copy()
            
        self.set_hyperparams(self.hyperparams)

        train_errors = []
        for i in range(self.num_ensemble):
            train_error = self.ensemble[i].fit(xtrain, ytrain, train_info)
            train_errors.append(train_error)
        
        return train_errors

    def query(self, xtest, info=None):
        predictions = []
        for i in range(self.num_ensemble):
            prediction = self.ensemble[i].query(xtest, info)
            predictions.append(prediction)
            
        return np.array(predictions)
    
    def set_hyperparams(self, params):
        if self.ensemble is None:
            self.ensemble = self.get_ensemble()

        for model in self.ensemble:
            model.set_hyperparams(params)

        self.hyperparams = params        

    def set_random_hyperparams(self):
        if self.ensemble is None:
            self.ensemble = self.get_ensemble()

        if self.hyperparams is None and hasattr(self.ensemble[0], 'default_hyperparams'):
            # todo: ideally should implement get_default_hyperparams() for all predictors
            params = self.ensemble[0].default_hyperparams.copy()
            
        elif self.hyperparams is None:
            params = None
        else:
            params = self.ensemble[0].set_random_hyperparams()

        self.set_hyperparams(params)
        return params
    
    def set_pre_computations(self, unlabeled=None, xtrain_zc_info=None, 
                             xtest_zc_info=None, unlabeled_zc_info=None):
        """
        Some predictors have pre_computation steps that are performed outside the
        predictor. E.g., omni needs zerocost metrics computed, and unlabeled data
        generated. In the case of an ensemble, this method relays that info to
        the predictor.
        """
        if self.ensemble is None:
            self.ensemble = self.get_ensemble()

        for model in self.ensemble:
            assert hasattr(model, 'set_pre_computations'), \
            'set_pre_computations() not implemented'
            model.set_pre_computations(unlabeled=unlabeled, 
                                       xtrain_zc_info=xtrain_zc_info, 
                                       xtest_zc_info=xtest_zc_info, 
                                       unlabeled_zc_info=unlabeled_zc_info)
