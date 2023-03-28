import numpy as np
import os
import json

from naslib.utils.encodings import EncodingType
from naslib.predictors.predictor import Predictor


class BNN(Predictor):
    def __init__(self, encoding_type=EncodingType.ADJACENCY_ONE_HOT,
                 ss_type="nasbench201", hparams_from_file=None):
        self.encoding_type = encoding_type
        self.ss_type = ss_type
        self.hparams_from_file=hparams_from_file

    def get_model(self, **kwargs):
        return NotImplementedError("Model needs to be defined.")

    def train_model(self, xtrain, ytrain):
        return NotImplementedError("Training method not defined.")

    def fit(self, xtrain, ytrain, train_info=None, **kwargs):
        if self.encoding_type is not None:
            _xtrain = np.array(
                [
                    arch.encode(encoding_type=self.encoding_type)
                    for arch in xtrain
                ]
            )
        else:
            _xtrain = xtrain
        _ytrain = np.array(ytrain)

        self.model = self.get_model(**kwargs)
        if self.hparams_from_file and self.hparams_from_file not in ['False', 'None'] \
        and os.path.exists(self.hparams_from_file):
            self.num_steps = json.load(open(self.hparams_from_file, 'rb'))['bohamiann']['num_steps']
            print('loaded hyperparams from', self.hparams_from_file)
        else:
            self.num_steps = 100
        self.train_model(_xtrain, _ytrain)

        train_pred = self.query(xtrain)
        train_error = np.mean(abs(train_pred - _ytrain))
        return train_error

    def query(self, xtest, info=None):
        if self.encoding_type is not None:
            test_data = np.array(
                [
                    arch.encode(encoding_type=self.encoding_type)
                    for arch in xtest
                ]
            )
        else:
            test_data = xtest

        m, v = self.model.predict(test_data)
        return np.squeeze(m)
