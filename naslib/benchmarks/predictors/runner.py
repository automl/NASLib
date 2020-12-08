import logging
import sys
import naslib as nl

from naslib.defaults.predictor_evaluator import PredictorEvaluator

import os

from naslib.predictors import Ensemble, FeedforwardPredictor, GBDTPredictor, EarlyStopping, GCNPredictor, BonasGCNPredictor, BonasMLPPredictor, BonasLSTMPredictor

from naslib.search_spaces import NasBench201SearchSpace
from naslib.search_spaces.core.query_metrics import Metric

from naslib.utils import utils, setup_logger
from naslib.utils.utils import get_project_root

from fvcore.common.config import CfgNode


config = utils.get_config_from_args(config_type='predictor')

utils.set_seed(config.seed)
logger = setup_logger(config.save + "/log.log")
logger.setLevel(logging.INFO)

utils.log_args(config)

supported_predictors = {
    'bananas': Ensemble(encoding_type='path',
                        predictor_type='feedforward'),
    'feedforward': FeedforwardPredictor(encoding_type='adjacency_one_hot'),
    'gbdt': GBDTPredictor(encoding_type='adjacency_one_hot'),
    'gcn': GCNPredictor(encoding_type='gcn'),
    'bonas_gcn': BonasGCNPredictor(encoding_type='bonas_gcn'),
    'bonas_mlp': BonasMLPPredictor(encoding_type='bonas_mlp'),
    'bonas_lstm': BonasLSTMPredictor(encoding_type='bonas_lstm'),
    'sovl_50': EarlyStopping(fidelity=50, metric=Metric.VAL_LOSS),
    'sotl_50': EarlyStopping(fidelity=50, metric=Metric.TRAIN_LOSS),
    'oracle': EarlyStopping(fidelity=200, metric=Metric.VAL_ACCURACY)
}

# set up the search space
search_space = NasBench201SearchSpace()

# choose a predictor
predictor = supported_predictors[config.predictor]
predictor_evaluator = PredictorEvaluator(predictor, config=config)
predictor_evaluator.adapt_search_space(search_space)

# evaluate the predictor
predictor_evaluator.evaluate()
