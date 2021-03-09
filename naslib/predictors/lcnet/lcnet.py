# This code is mostly from https://github.com/automl/pybnn
# pybnn authors: Aaron Klein, Moritz Freidank

import numpy as np

from naslib.predictors.predictor import Predictor
from naslib.predictors.lcnet.learning_curves import MCMCCurveModelCombination


class LCNetPredictor(Predictor):
    
    def __init__(self, metric=None):
        self.metric = metric

    def query(self, xtest, info):

        learning_curves = np.array([np.array(inf['lc']) / 100 for inf in info])
        trained_epochs = len(info[0]['lc'])
        t_idx = np.arange(1, trained_epochs+1)

        if self.ss_type == 'nasbench201':
            final_epoch = 200
            default_guess = 85.0
        elif self.ss_type == 'darts':
            final_epoch = 98
            default_guess = 93.0
        else:
            raise NotImplementedError()

        model = MCMCCurveModelCombination(final_epoch + 1,
                                          nwalkers=50,
                                          nsamples=800,
                                          burn_in=500,
                                          recency_weighting=False,
                                          soft_monotonicity_constraint=False,
                                          monotonicity_constraint=True,
                                          initial_model_weight_ml_estimate=True)

        predictions = []
        for i in range(len(xtest)):

            model.fit(t_idx, learning_curves[i])
            try:
                p = model.predictive_distribution(final_epoch)
                prediction = np.mean(p) * 100
            except AssertionError:
                # catch AssertionError in _split_theta method
                print('caught AssertionError running model')
                prediction = np.nan

            if np.isnan(prediction) or not np.isfinite(prediction):
                print('nan or finite')
                prediction = default_guess + np.random.rand()
            predictions.append(prediction)

        predictions = np.array(predictions)
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