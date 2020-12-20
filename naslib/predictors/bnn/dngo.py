# This is an implementation of the DNGO predictor from the paper:
# Snoek et al., 2015. Scalable Bayesian Optimization using DNNs
import numpy as np

from naslib.predictors.bnn.dngo_model import DNGO
from naslib.predictors.utils.encodings import encode
from naslib.predictors.predictor import Predictor


class DNGOPredictor(Predictor):
    def __init__(self, encoding_type='adjacency_one_hot', ss_type='nasbench201'):
        self.encoding_type = encoding_type
        self.ss_type = ss_type

    def get_model(self, **kwargs):
        predictor = DNGO(**kwargs)
        return predictor

    def fit(self, xtrain, ytrain):
        _xtrain = np.array([encode(arch, encoding_type=self.encoding_type,
                                  ss_type=self.ss_type) for arch in xtrain])
        _ytrain = np.array(ytrain)

        self.model = self.get_model(
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
        self.model.train(_xtrain, _ytrain, do_optimize=True)

        train_pred = self.query(xtrain)
        train_error = np.mean(abs(train_pred - _ytrain))
        return train_error

    def query(self, xtest, info=None):
        test_data = np.array([encode(arch,encoding_type=self.encoding_type,
                              ss_type=self.ss_type) for arch in xtest])

        m, v = self.model.predict(test_data)
        return np.squeeze(m)
