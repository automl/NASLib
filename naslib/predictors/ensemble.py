import numpy as np

from naslib.predictors.feedforward import FeedforwardPredictor
from naslib.predictors.gbdt import GBDTPredictor
from naslib.predictors.predictor import Predictor


class Ensemble(Predictor):
    
    def __init__(self, 
                 encoding_type='adjacency_one_hot',
                 num_ensemble=3, 
                 predictor_type='feedforward'):
        self.num_ensemble = num_ensemble
        self.predictor_type = predictor_type
        self.encoding_type = encoding_type
    
    def get_ensemble(self):
        ensemble = []
        for _ in range(self.num_ensemble):
            if self.predictor_type == 'feedforward':
                predictor = FeedforwardPredictor()
            elif self.predictor_type == 'gbdt':
                predictor = GBDTPredictor()
            else:
                print('{} predictor not implemented'.format(self.predictor_type))
                raise NotImplementedError()
            ensemble.append(predictor)
        return ensemble

    def fit(self, xtrain, ytrain):
        
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
