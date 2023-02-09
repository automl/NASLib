import numpy as np
import torch
import pyro
import pyro.contrib.gp as gp

from naslib.utils.encodings import EncodingType
from naslib.predictors.predictor import Predictor


class BaseGPModel(Predictor):
    def __init__(
        self,
        encoding_type=EncodingType.ADJACENCY_ONE_HOT,
        ss_type="nasbench201",
        kernel_type=None,
        optimize_gp_hyper=False,
        zc=False,
    ):
        super(Predictor, self).__init__()
        self.encoding_type = encoding_type
        self.ss_type = ss_type
        self.kernel_type = kernel_type
        self.optimize_gp_hyper = optimize_gp_hyper
        self.zc = zc

    def get_dataset(self, encodings, labels=None):
        if labels is None:
            return torch.tensor(encodings).double()
        else:
            return (
                torch.tensor(encodings).double(),
                torch.tensor((labels - self.mean) / self.std).double(),
            )

    def get_model(self, train_data, **kwargs):
        return NotImplementedError

    def train(self, train_data, **kwargs):
        pass

    def predict(self, input_data, **kwargs):
        return NotImplementedError

    def optimize_GP_hyperparameters(self, gp_model):
        if type(gp_model) == gp.models.GPRegression:
            optimizer = torch.optim.Adam(gp_model.parameters(), lr=0.005)
            loss_fn = pyro.infer.Trace_ELBO().differentiable_loss
        else:
            optimizer = torch.optim.Adam(gp_model.parameters(), lr=0.01)
            loss_fn = pyro.infer.TraceMeanField_ELBO().differentiable_loss
        losses = gp.util.train(gp_model, num_steps=self.num_steps)
        return losses

    def fit(self, xtrain, ytrain, train_info=None, params=None, **kwargs):

        # normalize accuracies
        self.mean = np.mean(ytrain)
        self.std = np.std(ytrain)
        if self.encoding_type is not None:
            xtrain = np.array(
                [
                    arch.encode(encoding_type=self.encoding_type)
                    for arch in xtrain
                ]
            )
        if self.zc:
            mean, std = -10000000.0, 150000000.0
            xtrain = [[*x, (train_info[i] - mean) / std] for i, x in enumerate(xtrain)]
        xtrain = np.array(xtrain)
        ytrain = np.array(ytrain)

        # convert to the right representation
        train_data = self.get_dataset(xtrain, ytrain)

        # instantiate model and fit to the training data
        self.model = self.get_model(train_data, **kwargs)
        self.train(train_data, **kwargs)
        print("Finished fitting GP")

        if self.optimize_gp_hyper:
            losses = self.optimize_GP_hyperparameters(self.model)
            print("Finished tuning GP hyperparameters")

        # predict
        train_pred = np.squeeze(self.predict(train_data[0]))
        train_error = np.mean(abs(train_pred - ytrain))

        return train_error

    def query(self, xtest, info=None):
        if self.encoding_type is not None:
            xtest = np.array(
                [
                    arch.encode(encoding_type=self.encoding_type)
                    for arch in xtest
                ]
            )

        if self.zc:
            mean, std = -10000000.0, 150000000.0
            xtest = [[*x, (info[i] - mean) / std] for i, x in enumerate(xtest)]
        xtest = np.array(xtest)

        test_data = self.get_dataset(xtest)
        return np.squeeze(self.predict(test_data)) * self.std + self.mean
