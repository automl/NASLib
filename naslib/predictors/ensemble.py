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


class Ensemble(Predictor):
    
    def __init__(self, 
                 encoding_type=None,
                 num_ensemble=3, 
                 predictor_type='feedforward',
                 ss_type='nasbench201'):
        self.num_ensemble = num_ensemble
        self.predictor_type = predictor_type
        self.encoding_type = encoding_type
        self.ss_type = ss_type
    
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
            'xgb': XGBoost(ss_type=self.ss_type,
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
            'seminas': SemiNASPredictor(ss_type=self.ss_type,
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
                                                  num_steps=200),
        }

        return [copy.deepcopy(trainable_predictors[self.predictor_type]) for _ in range(self.num_ensemble)]

    def fit(self, xtrain, ytrain, train_info=None):
        
        self.ensemble = self.get_ensemble()
        train_errors = []
        for i in range(self.num_ensemble):
            train_error = self.ensemble[i].fit(xtrain, ytrain)
            train_errors.append(train_error)
        
        return train_errors

    def query(self, xtest, info=None):
        predictions = []
        for i in range(self.num_ensemble):
            prediction = self.ensemble[i].query(xtest)
            predictions.append(prediction)
            
        return np.array(predictions)
