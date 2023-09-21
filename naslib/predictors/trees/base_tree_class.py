from typing import Dict, List, Union
import numpy as np

import torch.nn as nn
from naslib.predictors.predictor import Predictor
from naslib.utils.encodings import EncodingType


class BaseTree(Predictor):

    def __init__(self, encoding_type=EncodingType.ADJACENCY_ONE_HOT, ss_type='nasbench201', zc=False, zc_only=False,
                 hpo_wrapper=False, hparams_from_file=None):
        super(Predictor, self).__init__()
        self.encoding_type = encoding_type
        self.ss_type = ss_type
        self.zc = zc
        self.zc_names = None
        self.zc_only = zc_only
        self.hyperparams = None
        self.hpo_wrapper = hpo_wrapper
        self.hparams_from_file = hparams_from_file

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

            # TODO: Fix. Hacky way to make XGBoost accept both encodings as well as NASLib Graphs as xtrain
            if isinstance(xtrain[0], nn.Module):
                xtrain = np.array([arch.encode(encoding_type=self.encoding_type) for arch in xtrain])

            if self.zc:
                # mean, std = -10000000.0, 150000000.0
                # xtrain = [[*x, (train_info[i]-mean)/std] for i, x in enumerate(xtrain)]
                if self.zc_only:
                    xtrain = self.zc_features
                else:
                    xtrain = [[*x, *zc_scores] for x, zc_scores in zip (xtrain, self.zc_features)]
            xtrain = np.array(xtrain)
            ytrain = np.array(ytrain)

        else:
            # when used in aug_lcsvr we feed in ndarray directly
            xtrain = xtrain
            ytrain = ytrain

        if self.zc:
            self.zc_to_features_map = self._get_zc_to_feature_mapping(self.zc_names, xtrain)

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

            # TODO: Fix. Hacky way to make XGBoost accept both encodings as well as NASLib Graphs as xtrain
            if isinstance(xtest[0], nn.Module):
                xtest = np.array([arch.encode(encoding_type=self.encoding_type) for arch in xtest])
            if self.zc:
                # mean, std = -10000000.0, 150000000.0
                zc_scores = [self.create_zc_feature_vector(data['zero_cost_scores']) for data in info]
                if self.zc_only:
                    xtest = zc_scores
                else:
                    xtest = [[*x, *zc] for x, zc in zip(xtest, zc_scores)]
            xtest = np.array(xtest)

        else:
            # when used in aug_lcsvr we feed in ndarray directly
            xtest = xtest

        test_data = self.get_dataset(xtest)
        return np.squeeze(self.model.predict(test_data)) * self.std + self.mean

    def get_random_hyperparams(self):
        pass

    def create_zc_feature_vector(self, zero_cost_scores: Union[List[Dict], Dict]) -> Union[List[List], List]:
        zc_features = []

        def _make_features(zc_scores):
            zc_values = []
            for zc_name in self.zc_names:
                zc_values.append(zc_scores[zc_name])

            zc_features.append(zc_values)

        if isinstance(zero_cost_scores, list):
            for zc_scores in zero_cost_scores:
                _make_features(zc_scores)
        elif isinstance(zero_cost_scores, dict):
            _make_features(zero_cost_scores)
            zc_features = zc_features[0]

        return zc_features

    def set_hyperparams(self, params):
        self.hyperparams = params

    def _get_zc_to_feature_mapping(self, zc_names, xtrain):
        x = xtrain[0] # Consider one datapoint

        n_zc = len(zc_names)
        n_arch_features = len(x[:-n_zc])
        mapping = {}

        for zc_name, feature_index in zip(zc_names, range(n_arch_features, n_arch_features+n_zc)):
            mapping[zc_name] = f'f{feature_index}'

        return mapping

    def set_pre_computations(self, unlabeled=None, xtrain_zc_info=None, xtest_zc_info=None, unlabeled_zc_info=None):
        if xtrain_zc_info is not None:
            self.xtrain_zc_info = xtrain_zc_info
            self._verify_zc_info(xtrain_zc_info['zero_cost_scores'])
            self._set_zc_names(xtrain_zc_info['zero_cost_scores'])
            self.zc_features = self.create_zc_feature_vector(xtrain_zc_info['zero_cost_scores'])
        
    def _verify_zc_info(self, zero_cost_scores):
        zc_names = [set(zc_scores.keys()) for zc_scores in zero_cost_scores]
    
        assert len(zc_names) > 0, 'No ZC values found in zero_cost_scores'
        assert zc_names.count(zc_names[0]) == len(zc_names), 'All models do not have the same number of ZC values'

    def _set_zc_names(self, zero_cost_scores):
        zc_names = sorted(zero_cost_scores[0].keys())
        self.zc_names = zc_names

        
