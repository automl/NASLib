import numpy as np
import torch
import pyro
import pyro.contrib.gp as gp

from naslib.predictors.utils.encodings import encode
from naslib.predictors.predictor import Predictor


class BaseGPModel(Predictor):

    def __init__(self, encoding_type='adjacency_one_hot',
                 ss_type='nasbench201', kernel_type=None):
        super(Predictor, self).__init__()
        self.encoding_type = encoding_type
        self.ss_type = ss_type
        self.kernel_type = kernel_type

    def get_dataset(self, encodings, labels=None):
        if labels is None:
            return torch.tensor(encodings).double()
        else:
            return (torch.tensor(encodings).double(),
                    torch.tensor((labels-self.mean)/self.std).double())

    def train(self, train_data, **kwargs):
        return NotImplementedError

    def predict(self, input_data, **kwargs):
        return NotImplementedError

    def optimize_GP_hyperparameters(self, gp_model):
        if type(self.gpr) == gp.models.GPRegression:
            optimizer = torch.optim.Adam(gp_model.parameters(), lr=0.005)
            loss_fn = pyro.infer.Trace_ELBO().differentiable_loss
        else:
            optimizer = torch.optim.Adam(gp_model.parameters(), lr=0.01)
            loss_fn = pyro.infer.TraceMeanField_ELBO().differentiable_loss
        losses = gp.util.train(gp_model, num_steps=1000)
        return losses

    def fit(self, xtrain, ytrain, params=None,
            optimize_gp_hyper=False, **kwargs):

        # normalize accuracies
        self.mean = np.mean(ytrain)
        self.std = np.std(ytrain)

        xtrain = np.array([encode(arch, encoding_type=self.encoding_type,
                                  ss_type=self.ss_type) for arch in xtrain])
        ytrain = np.array(ytrain)

        # convert to the right representation
        train_data = self.get_dataset(xtrain, ytrain)

        # fit to the training data
        self.model = self.train(train_data, optimize_gp_hyper=False)
        print('Finished fitting GP')

        if optimize_gp_hyper:
            losses = self.optimize_GP_hyperparameters(self.model)
            print('Finished tuning GP hyperparameters')

        # predict
        train_pred = np.squeeze(self.predict(train_data[0]))
        train_error = np.mean(abs(train_pred-ytrain))

        return train_error

    def query(self, xtest, info=None):
        xtest = np.array([encode(arch, encoding_type=self.encoding_type,
                                 ss_type=self.ss_type) for arch in xtest])
        test_data = self.get_dataset(xtest)
        return np.squeeze(self.predict(test_data)) * self.std + self.mean


