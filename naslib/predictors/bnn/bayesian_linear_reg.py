# This is an implementation of Bayesian Linear Regression

from pybnn.bayesian_linear_regression import BayesianLinearRegression as BLR
from pybnn.bayesian_linear_regression import linear_basis_func, quadratic_basis_func

from naslib.predictors.bnn.bnn_base import BNN


class BayesianLinearRegression(BNN):

    def get_model(self, **kwargs):
        predictor = BLR(
            alpha=1.0,
            beta=100,
            basis_func=linear_basis_func,
            prior=None,
            do_mcmc=False, # turn this off for better sample efficiency
            n_hypers=20,
            chain_length=100,
            burnin_steps=100,
        )
        return predictor

    def train_model(self, xtrain, ytrain):
        self.model.train(xtrain, ytrain, do_optimize=True)

