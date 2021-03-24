# This code is mostly from https://github.com/automl/pylearningcurvepredictor
# pylearningcurvepredictor author: Tobias Domhan, tdomhan

import numpy as np

from naslib.predictors.predictor import Predictor
from naslib.predictors.lce.parametric_model import model_name_list, model_config, construct_parametric_model
from naslib.predictors.lce.parametric_ensemble import ParametricEnsemble


class LCEPredictor(Predictor):
    
    def __init__(self, metric=None):
        self.metric = metric

    def query(self, xtest, info):
        
        ensemble = ParametricEnsemble([construct_parametric_model(model_config, name) for name in model_name_list])
        
        learning_curves = np.array([np.array(inf['lc']) / 100 for inf in info])
        trained_epochs = len(info[0]['lc'])
        
        if self.ss_type == 'nasbench201':
            final_epoch = 200
            default_guess = 85.0
            N = 300
        elif self.ss_type == 'darts':
            final_epoch = 98
            default_guess = 93.0
            N = 1000
        elif self.ss_type == 'nlp':
            final_epoch = 50
            default_guess = 94.83
            N = 1000
        else:
            raise NotImplementedError()

        predictions = []
        for i in range(len(xtest)):
            ensemble.mcmc(learning_curves[i, :], N=N)
            prediction = ensemble.mcmc_sample_predict([final_epoch])
            prediction = np.squeeze(prediction) * 100
            
            if np.isnan(prediction) or not np.isfinite(prediction):
                print('nan or finite')
                prediction = default_guess + np.random.rand()
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
                'hyperparams':None,
                'unlabeled':False, 
                'unlabeled_factor':0
               }
        return reqs