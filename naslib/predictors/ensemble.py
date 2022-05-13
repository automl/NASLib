import logging
import numpy as np
import copy

from naslib.predictors import Predictor
from naslib.predictors.trees import XGBoost

logger = logging.getLogger(__name__)

class Ensemble(Predictor):

    def __init__(self,
                 encoding_type=None,
                 num_ensemble=3,
                 predictor_type='xgb',
                 zc=True,
                 ss_type=None,
                 hpo_wrapper=True,
                 zc_only=False,
                 config=None):
        self.num_ensemble = num_ensemble
        self.predictor_type = predictor_type
        self.encoding_type = encoding_type
        self.ss_type = ss_type
        self.hpo_wrapper = hpo_wrapper
        self.config = config
        self.hyperparams = None
        self.ensemble = None
        self.zc = zc
        self.zc_only = zc_only

    def get_ensemble(self):

        trainable_predictors = {
            "xgb": XGBoost(
                ss_type=self.ss_type, zc=self.zc, encoding_type="adjacency_one_hot", zc_only=self.zc_only
            )
        }

        return [
            copy.deepcopy(trainable_predictors[self.predictor_type])
            for _ in range(self.num_ensemble)
        ]

    def fit(self, xtrain, ytrain, train_info=None):
        if self.ensemble is None:
            self.ensemble = self.get_ensemble()

        if self.hyperparams is None and hasattr(self.ensemble[0], 'default_hyperparams'):
            # todo: ideally should implement get_default_hyperparams() for all predictors
            self.hyperparams = self.ensemble[0].default_hyperparams.copy()

        self.set_hyperparams(self.hyperparams)

        train_errors = []
        for i in range(self.num_ensemble):
            logger.info(f'Training ensemble model {i+1} of {self.num_ensemble} ({self.ensemble[i]}) with {len(ytrain)} datapoints')
            train_error = self.ensemble[i].fit(xtrain, ytrain, train_info)
            train_errors.append(train_error)

        return train_errors

    def query(self, xtest, info=None):
        predictions = []
        for i in range(self.num_ensemble):
            prediction = self.ensemble[i].query(xtest, info)
            predictions.append(prediction)

        return np.array(predictions)

    def set_hyperparams(self, params):
        if self.ensemble is None:
            self.ensemble = self.get_ensemble()

        for model in self.ensemble:
            model.set_hyperparams(params)

        self.hyperparams = params

    def set_random_hyperparams(self):
        if self.ensemble is None:
            self.ensemble = self.get_ensemble()

        if self.hyperparams is None and hasattr(self.ensemble[0], 'default_hyperparams'):
            # todo: ideally should implement get_default_hyperparams() for all predictors
            params = self.ensemble[0].default_hyperparams.copy()

        elif self.hyperparams is None:
            params = None
        else:
            params = self.ensemble[0].set_random_hyperparams()

        self.set_hyperparams(params)
        return params

    def set_pre_computations(self, unlabeled=None, xtrain_zc_info=None,
                             xtest_zc_info=None, unlabeled_zc_info=None):
        """
        Some predictors have pre_computation steps that are performed outside the
        predictor. E.g., omni needs zerocost metrics computed, and unlabeled data
        generated. In the case of an ensemble, this method relays that info to
        the predictor.
        """
        if self.ensemble is None:
            self.ensemble = self.get_ensemble()

        for model in self.ensemble:
            assert hasattr(model, 'set_pre_computations'), \
                'set_pre_computations() not implemented'
            model.set_pre_computations(unlabeled=unlabeled,
                                       xtrain_zc_info=xtrain_zc_info,
                                       xtest_zc_info=xtest_zc_info,
                                       unlabeled_zc_info=unlabeled_zc_info)
