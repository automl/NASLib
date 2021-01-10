import numpy as np

from naslib.predictors.utils.encodings import encode
from naslib.predictors.predictor import Predictor


class BNN(Predictor):
    def __init__(self, encoding_type='adjacency_one_hot', ss_type='nasbench201'):
        self.encoding_type = encoding_type
        self.ss_type = ss_type

    def get_model(self, **kwargs):
        return NotImplementedError('Model needs to be defined.')

    def train_model(self, xtrain, ytrain):
        return NotImplementedError('Training method not defined.')

    def fit(self, xtrain, ytrain, train_info=None, **kwargs):
        _xtrain = np.array([encode(arch, encoding_type=self.encoding_type,
                                  ss_type=self.ss_type) for arch in xtrain])
        _ytrain = np.array(ytrain)

        self.model = self.get_model(**kwargs)
        self.train_model(_xtrain, _ytrain)

        train_pred = self.query(xtrain)
        train_error = np.mean(abs(train_pred - _ytrain))
        return train_error

    def query(self, xtest, info=None):
        test_data = np.array([encode(arch,encoding_type=self.encoding_type,
                              ss_type=self.ss_type) for arch in xtest])

        m, v = self.model.predict(test_data)
        return np.squeeze(m)
