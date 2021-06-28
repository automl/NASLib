# This is an implementation of the DNGO predictor from the paper:
# Snoek et al., 2015. Scalable Bayesian Optimization using DNNs

from pybnn.dngo import DNGO

from naslib.predictors.bnn.bnn_base import BNN


class DNGOPredictor(BNN):

    def get_model(self, **kwargs):
        predictor = DNGO(
            batch_size=10,
            num_epochs=500,
            learning_rate=0.01,
            adapt_epoch=5000,
            n_units_1=50,
            n_units_2=50,
            n_units_3=50,
            alpha=1.0,
            beta=1000,
            prior=None,
            do_mcmc=True, # turn this off for better sample efficiency
            n_hypers=20,
            chain_length=2000,
            burnin_steps=2000,
            normalize_input=False,
            normalize_output=True
        )
        return predictor

    def train_model(self, xtrain, ytrain):
        try:
            self.model.train(xtrain, ytrain, do_optimize=True)
        except ValueError:
            self.model.train(xtrain, ytrain, do_optimize=False)

