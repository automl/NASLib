import torch
import pyro
import pyro.contrib.gp as gp
import numpy as np

from naslib.predictors.gp import GPPredictor


class SparseGPPredictor(GPPredictor):

    def train(self, train_data, optimize_gp_hyper=False):
        X_train, y_train = train_data
        # initialize the kernel and model
        pyro.clear_param_store()
        kernel = self.kernel(input_dim=X_train.shape[1])
        Xu = torch.arange(20.) / 4.0
        self.gpr = gp.models.SparseGPRegression(X_train, y_train, kernel,
                                                Xu=Xu, jitter=1.0e-5)
        return self.gpr

