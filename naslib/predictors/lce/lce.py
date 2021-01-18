import numpy as np


from naslib.predictors.predictor import Predictor
from naslib.predictors.lce.parametric_model import model_name_list, model_config, construct_parametric_model
from naslib.predictors.lce.parametric_ensemble import ParametricEnsemble


class LCEPredictor(Predictor):
    
    def __init__(self, metric=None):
        self.metric = metric

    def query(self, xtest, info, nan_replacement=80.0):
        
        ensemble = ParametricEnsemble([construct_parametric_model(model_config, name) for name in model_name_list])
        
        learning_curves = np.array([np.array(inf['lc']) / 100 for inf in info])
        trained_epochs = len(info[0]['lc'])

        predictions = []
        for i in range(len(xtest)):
            ensemble.mcmc(learning_curves[i, :], N=100)
            prediction_epochs = epochs=list(range(93, 94))
            prediction = ensemble.mcmc_sample_predict(prediction_epochs) * 100
            
            if np.isnan(prediction):
                prediction = nan_replacement + np.random.rand()
            predictions.append(prediction)
            
        predictions = np.squeeze(np.array(predictions))
        return predictions

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