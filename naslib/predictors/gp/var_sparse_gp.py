import torch
import pyro.contrib.gp as gp
import pyro.distributions as dist
import pyro
import numpy as np

from naslib.predictors.gp import GPPredictor


class VarSparseGPPredictor(GPPredictor):

    def train(self, train_data, optimize_gp_hyper=True):
        X_train, y_train = train_data
        # initialize the kernel and model
        pyro.clear_param_store()
        kernel = self.kernel(input_dim=X_train.shape[1])
        Xu = torch.arange(10.) / 2.0
        Xu.unsqueeze_(-1)
        Xu = Xu.expand(10, X_train.shape[1]).double()
        likelihood = gp.likelihoods.Gaussian()
        self.gpr = gp.models.VariationalSparseGP(X_train, y_train, kernel,
                                                 Xu=Xu, likelihood=likelihood,
                                                 whiten=True)

        return self.gpr

