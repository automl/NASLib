import torch
import pyro
import pyro.contrib.gp as gp
import pyro.distributions as dist
import numpy as np

from naslib.predictors.gp import BaseGPModel
from naslib.utils import AverageMeterGroup, TensorDatasetWithTrans

device = torch.device('cpu') #NOTE: faster on CPU

# TODO

class DeepVarSparseGP(pyro.nn.PyroModule):
    def __init__(self, X, y, Xu, mean_fn):
        super(DeepVarSparseGP, self).__init__()
        self.layer1 = gp.models.VariationalSparseGP(
            X,
            None,
            gp.kernels.RBF(X.shape[1], variance=torch.tensor(5.).double(),
                           lengthscale=torch.tensor(10.).double()),
            Xu=Xu,
            likelihood=None,
            mean_function=mean_fn,
            latent_shape=torch.Size([10]))

        h = mean_fn(X).t()
        hu = mean_fn(Xu).t()
        self.layer2 = gp.models.VariationalSparseGP(
            h,
            y,
            gp.kernels.RBF(10, variance=torch.tensor(5.).double(),
                           lengthscale=torch.tensor(10.).double()),
            Xu=hu,
            likelihood=gp.likelihoods.Gaussian(),
            latent_shape=torch.Size([1]))

    def model(self, X, y):
        self.layer1.set_data(X, None)
        h_loc, h_var = self.layer1.model()
        # approximate with MC sample
        h = dist.Normal(h_loc, h_var.sqrt())()
        self.layer2.set_data(h.t(), y)
        self.layer2.model()

    def guide(self, X, y):
        self.layer1.guide()
        self.layer2.guide()

    # make predictions
    def forward(self, X_new):
        # because prediction is stochastic (due to Monte Carlo sample of hidden layer),
        # we make 100 prediction and take the most common one
        pred = []
        for _ in range(100):
            h_loc, h_var = self.layer1(X_new)
            h = dist.Normal(h_loc, h_var.sqrt())()
            f_loc, f_var = self.layer2(h.t())
            pred.append(f_loc.argmax(dim=0))
        return torch.stack(pred).mode(dim=0)[0]


class DeepVarSparseGPPredictor(BaseGPModel):

    def get_dataset(self, encodings, labels=None):
        if labels is None:
            return torch.tensor(encodings).double()
        else:
            return (torch.tensor(encodings).double(),
                    torch.tensor((labels-self.mean)/self.std).double())
        X_tensor = torch.FloatTensor(_xtrain).to(device)
        y_tensor = torch.FloatTensor(_ytrain).to(device)

        train_data = TensorDataset(X_tensor, y_tensor)

    def get_model(self, train_data, **kwargs):
        # deepgp = DeepVarSparseGP)
        pass

    def train(self, train_data, optimize_gp_hyper=False):
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

