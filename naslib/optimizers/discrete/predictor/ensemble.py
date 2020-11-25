import numpy as np

from naslib.optimizers.discrete.predictor.feedforward import FeedforwardPredictor
from naslib.optimizers.discrete.predictor.gbdt import GBDTPredictor


class Ensemble:
    
    def __init__(self, 
                 num_ensemble=1, 
                 predictor_type='feedforward'):
        self.num_ensemble = num_ensemble
        self.predictor_type = predictor_type
    
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

    def predict(self, xtest):
        predictions = []
        for i in range(self.num_ensemble):
            prediction = self.ensemble[i].predict(xtest)
            predictions.append(prediction)
            
        return np.array(predictions)
    
    
    