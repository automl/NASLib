from functools import partial
import torch
import pyro
import pyro.contrib.gp as gp
import pyro.distributions as dist
import numpy as np

from naslib.predictors.gp import BaseGPModel


class GPPredictor(BaseGPModel):
    def __init__(
        self,
        encoding_type="adjacency_one_hot",
        ss_type="nasbench201",
        kernel_type="RBF",
        optimize_gp_hyper=False,
        num_steps=200,
        zc=False,
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


if __name__ == "__main__":
    train_dataset = (
        torch.tensor(np.random.randn(10, 30)),
        torch.tensor(
            np.random.randn(
                10,
            )
        ),
    )
    model = GPPredictor(kernel_type="RBF")
    model.train(train_dataset)
