import numpy as np

from naslib.predictors.utils.encodings import encode
from naslib.predictors.predictor import Predictor


class BaseTree(Predictor):

    def __init__(self, encoding_type='adjacency_one_hot', ss_type='nasbench201', zc=False, hpo_wrapper=False):
        super(Predictor, self).__init__()
        self.encoding_type = encoding_type
        self.ss_type = ss_type
        self.zc = zc
        self.hyperparams = None
        self.hpo_wrapper = hpo_wrapper

    @property
    def default_hyperparams(self):
        return {}

    def get_dataset(self, encodings, labels=None):
        return NotImplementedError('Tree cannot process the numpy data without \
                                   converting to the proper representation')

    def train(self, train_data, **kwargs):
        return NotImplementedError('Train method not implemented')

    def predict(self, data, **kwargs):
        return self.model.predict(data, **kwargs)

    def fit(self, xtrain, ytrain, train_info=None, params=None, **kwargs):

        # normalize accuracies
        self.mean = np.mean(ytrain)
        self.std = np.std(ytrain)

        if type(xtrain) is list:
            # when used in itself, we use
            xtrain = np.array([encode(arch, encoding_type=self.encoding_type,
                                      ss_type=self.ss_type) for arch in xtrain])

            if self.zc:
                mean, std = -10000000.0, 150000000.0
                xtrain = [[*x, (train_info[i]-mean)/std] for i, x in enumerate(xtrain)]
            xtrain = np.array(xtrain)
            ytrain = np.array(ytrain)

        else:
            # when used in aug_lcsvr we feed in ndarray directly
            xtrain = xtrain
            ytrain = ytrain


        # convert to the right representation
        train_data = self.get_dataset(xtrain, ytrain)

        # fit to the training data
        self.model = self.train(train_data)

        # predict
        train_pred = np.squeeze(self.predict(xtrain))
        train_error = np.mean(abs(train_pred-ytrain))

        return train_error

    def query(self, xtest, info=None):

        if type(xtest) is list:
            #  when used in itself, we use
            xtest = np.array([encode(arch, encoding_type=self.encoding_type,
                                 ss_type=self.ss_type) for arch in xtest])
            if self.zc:
                mean, std = -10000000.0, 150000000.0
                xtest = [[*x, (info[i]-mean)/std] for i, x in enumerate(xtest)]
            xtest = np.array(xtest)

        else:
            # when used in aug_lcsvr we feed in ndarray directly
            xtest = xtest

        test_data = self.get_dataset(xtest)
        return np.squeeze(self.model.predict(test_data)) * self.std + self.mean

    def get_random_hyperparams(self):
        pass
