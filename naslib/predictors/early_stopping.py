import numpy as np

from naslib.predictors.predictor import Predictor


class EarlyStopping(Predictor):

    def __init__(self, metric):
        
        self.metric = metric
    
    def query(self, xtest, info):
        """
        info: a list of dictionaries which include the learning curve of the
        corresponding architecture.
        Return the final value on the learning curve
        """
        return np.array([inf['lc'][-1] for inf in info])
        
    def get_metric(self):
        return self.metric
    
    def get_data_reqs(self):
        """
        Returns a dictionary with info about whether the predictor needs
        extra info to train/query.
        """
        reqs = {'requires_partial_lc':True, 
                'metric':self.metric, 
                'requires_hyperparameters':False, 
                'hyperparams':None
               }
        return reqs