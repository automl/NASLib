# This is an implementation of the BOHAMIANN predictor from the paper:
# Springenberg et al., 2016. Bayesian Optimization with Robust Bayesian Neural
# Networks
import numpy as np
import torch.nn as nn
from pybnn.bohamiann import Bohamiann, nll, get_default_network

from naslib.predictors.utils.encodings import encode
from naslib.predictors.predictor import Predictor


class BOHAMIANN(Predictor):
    def __init__(self, encoding_type='adjacency_one_hot', ss_type='nasbench201'):
        self.encoding_type = encoding_type
        self.ss_type = ss_type

    def get_model(self, **kwargs):
        predictor = Bohamiann(**kwargs)
        return predictor

    def fit(self, xtrain, ytrain):
        _xtrain = np.array([encode(arch, encoding_type=self.encoding_type,
                                  ss_type=self.ss_type) for arch in xtrain])
        _ytrain = np.array(ytrain)

        self.model = self.get_model(
            get_network=get_default_network,
            sampling_method="adaptive_sghmc",
            use_double_precision=True,
            metrics=(nn.MSELoss,),
            likelihood_function=nll,
            print_every_n_steps=10,
            normalize_input=False,
            normalize_output=True
        )
        self.model.train(_xtrain, _ytrain,
                         num_steps=100,
                         num_burn_in_steps=10,
                         keep_every=5,
                         lr=1e-2,
                         verbose=True)

        train_pred = self.query(xtrain)
        train_error = np.mean(abs(train_pred - _ytrain))
        return train_error

    def query(self, xtest, info=None):
        test_data = np.array([encode(arch,encoding_type=self.encoding_type,
                              ss_type=self.ss_type) for arch in xtest])

        m, v = self.model.predict(test_data)
        return np.squeeze(m)
