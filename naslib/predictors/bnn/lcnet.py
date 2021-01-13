# This is an implementation of the LCNet predictor from the paper:
# Klein et al., 2017. LEARNING CURVE PREDICTION WITH BAYESIAN NEURAL NETWORKS
import torch.nn as nn
from pybnn.lcnet import LCNet
from pybnn.bohamiann import nll

from naslib.predictors.bnn.bnn_base import BNN


class LCNetPredictor(BNN):

    def get_model(self, **kwargs):
        predictor = LCNet(
            sampling_method="adaptive_sghmc",
            use_double_precision=True,
            metrics=(nn.MSELoss,),
            likelihood_function=nll,
            print_every_n_steps=100,
        )
        return predictor

    def train_model(self, xtrain, ytrain):
        self.model.train(xtrain, ytrain,
                         num_steps=1000,
                         num_burn_in_steps=100,
                         keep_every=50,
                         lr=1e-2,
                         verbose=True)

