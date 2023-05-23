from functools import partial
import torch
import pyro
import pyro.contrib.gp as gp
import pyro.distributions as dist
import numpy as np
import os
import json

from naslib.utils.encodings import EncodingType
from naslib.predictors.gp import BaseGPModel


class GPPredictor(BaseGPModel):
    def __init__(
        self,
        encoding_type=EncodingType.ADJACENCY_ONE_HOT,
        ss_type="nasbench201",
        kernel_type="RBF",
        optimize_gp_hyper=True,
        zc=False,
        hparams_from_file=None,
    ):
        """
        Params:
            kernel_type (str): determines the kernel type. Can be RBF,
            RationalQuadratic, Exponential, Matern32, Matern52, Cosine, Periodic
        """
        super(GPPredictor, self).__init__(
            encoding_type, ss_type, kernel_type, optimize_gp_hyper
        )
        self.kernel = partial(
            eval("gp.kernels." + kernel_type),
            variance=torch.tensor(5.0).double(),
            lengthscale=torch.tensor(10.0).double(),
        )
        self.hparams_from_file = hparams_from_file

    def get_model(self, train_data, **kwargs):
        X_train, y_train = train_data
        # initialize the kernel and model
        pyro.clear_param_store()
        kernel = self.kernel(input_dim=X_train.shape[1])
        gpr = gp.models.GPRegression(
            X_train, y_train, kernel, noise=torch.tensor(1.0).double()
        )

        # optional: fit the model using MAP
        gpr.kernel.lengthscale = pyro.nn.PyroSample(dist.LogNormal(0.0, 1.0))
        gpr.kernel.variance = pyro.nn.PyroSample(dist.LogNormal(0.0, 1.0))

        return gpr

    def predict(self, input_data, **kwargs):
        with torch.no_grad():
            if type(self.model) == gp.models.VariationalSparseGP:
                mean, cov = self.model(input_data, full_cov=True)
            else:
                mean, cov = self.model(input_data, full_cov=True, noiseless=False)
        return mean.numpy()

    def fit(self, xtrain, ytrain, train_info=None, params=None, **kwargs):
        if self.hparams_from_file and self.hparams_from_file not in ['False', 'None'] \
        and os.path.exists(self.hparams_from_file):
            self.num_steps = json.load(open(self.hparams_from_file, 'rb'))['gp']['num_steps']
            print('loaded hyperparams from', self.hparams_from_file)
        else:
            self.num_steps = 200
        return super(GPPredictor, self).fit(xtrain, ytrain, train_info, params, **kwargs)