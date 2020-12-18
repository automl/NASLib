import numpy as np

from naslib.predictors.utils.encodings import encode
from naslib.predictors.predictor import Predictor


class BaseTree(Predictor):

    def __init__(self, encoding_type='adjacency_one_hot', ss_type='nasbench201'):
        super(Predictor, self).__init__()
        self.encoding_type = encoding_type
        self.ss_type = ss_type

    @property
    def parameters(self, params=None):
        return {}

    def get_dataset(self, encodings, labels=None):
        return NotImplementedError('Tree cannot process the numpy data without \
                                   converting to the proper representation')

    def train(self, train_data, **kwargs):
        return NotImplementedError('Train method not implemented')

    def predict(self, data, **kwargs):
        return self.model.predict(data, **kwargs)

    def fit(self, xtrain, ytrain, params=None, **kwargs):

        # normalize accuracies
        self.mean = np.mean(ytrain)
        self.std = np.std(ytrain)

        xtrain = np.array([encode(arch, encoding_type=self.encoding_type,
                                  ss_type=self.ss_type) for arch in xtrain])
        ytrain = np.array(ytrain)

        # convert to the right representation
        train_data = self.get_dataset(xtrain, ytrain)

        # fit to the training data
        self.model = self.train(train_data)

        # predict
        train_pred = np.squeeze(self.predict(xtrain))
        train_error = np.mean(abs(train_pred-ytrain))

        return train_error

    def query(self, xtest, info=None):
        xtest = np.array([encode(arch, encoding_type=self.encoding_type,
                                 ss_type=self.ss_type) for arch in xtest])
        test_data = self.get_dataset(xtest)
        return np.squeeze(self.model.predict(test_data)) * self.std + self.mean


