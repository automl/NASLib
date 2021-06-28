# This is an implementation of the BOHAMIANN predictor from the paper:
# Springenberg et al., 2016. Bayesian Optimization with Robust Bayesian Neural
# Networks

import torch.nn as nn
from pybnn.bohamiann import Bohamiann, nll, get_default_network

from naslib.predictors.bnn.bnn_base import BNN


class BOHAMIANN(BNN):

    def get_model(self, **kwargs):
        predictor = Bohamiann(
            get_network=get_default_network,
            sampling_method="adaptive_sghmc",
            use_double_precision=True,
            metrics=(nn.MSELoss,),
            likelihood_function=nll,
            print_every_n_steps=10,
            normalize_input=False,
            normalize_output=True
        )
        return predictor

    def train_model(self, xtrain, ytrain):
        self.model.train(xtrain, ytrain,
                         num_steps=100,
                         num_burn_in_steps=10,
                         keep_every=5,
                         lr=1e-2,
                         verbose=True)

