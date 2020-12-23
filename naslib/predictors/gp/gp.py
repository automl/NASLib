from functools import partial
import torch
import pyro
import pyro.contrib.gp as gp
import pyro.distributions as dist
import numpy as np

from naslib.predictors.gp import BaseGPModel


class GPPredictor(BaseGPModel):

    def __init__(self, encoding_type='adjacency_one_hot',
                 ss_type='nasbench201', kernel_type='RBF'):
        """
        Params:
            kernel_type (str): determines the kernel type. Can be RBF,
            RationalQuadratic, Exponential, Matern32, Matern52, Cosine, Periodic
        """
        super(GPPredictor, self).__init__(encoding_type, ss_type, kernel_type)
        self.kernel = partial(eval('gp.kernels.'+kernel_type),
                              variance=torch.tensor(5.).double(),
                              lengthscale=torch.tensor(10.).double())

    def train(self, train_data, optimize_gp_hyper=False):
        X_train, y_train = train_data
        # initialize the kernel and model
        pyro.clear_param_store()
        kernel = self.kernel(input_dim=X_train.shape[1])
        self.gpr = gp.models.GPRegression(X_train, y_train, kernel,
                                          noise=torch.tensor(1.).double())

        # optional: fit the model using MAP
        self.gpr.kernel.lengthscale = pyro.nn.PyroSample(dist.LogNormal(0.0, 1.0))
        self.gpr.kernel.variance = pyro.nn.PyroSample(dist.LogNormal(0.0, 1.0))

        return self.gpr

    def predict(self, input_data, **kwargs):
        with torch.no_grad():
            if type(self.gpr) == gp.models.VariationalSparseGP:
                mean, cov = self.gpr(input_data, full_cov=True)
            else:
                mean, cov = self.gpr(input_data, full_cov=True, noiseless=False)
        return mean.numpy()

    def fit(self, xtrain, ytrain, params=None, **kwargs):
        return super(GPPredictor, self).fit(xtrain, ytrain, params, **kwargs)


if __name__ == '__main__':
    train_dataset = (torch.tensor(np.random.randn(10,30)),
                     torch.tensor(np.random.randn(10,)))
    model = GPPredictor(kernel_type='RBF')
    model.train(train_dataset)
