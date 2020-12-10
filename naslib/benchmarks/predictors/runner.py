import logging
import sys
import naslib as nl

from naslib.defaults.predictor_evaluator import PredictorEvaluator

import os

from naslib.predictors import Ensemble, FeedforwardPredictor, GBDTPredictor, EarlyStopping, GCNPredictor
from naslib.search_spaces import NasBench201SearchSpace
from naslib.search_spaces.core.query_metrics import Metric

from naslib.utils import utils, setup_logger
from naslib.utils.utils import get_project_root

from fvcore.common.config import CfgNode

# TODO: pass in a config to the seed, predictors, and PredictionEvaluator

# load the default base
with open(os.path.join(get_project_root(), 'benchmarks/predictors/', 'predictor_config.yaml')) as f:
    config = CfgNode.load_cfg(f)

config.save = '{}/{}/{}/{}'.format(config.out_dir, config.dataset, config.predictor, config.seed)
utils.set_seed(0)
logger = setup_logger(config.save + "/log.log")
logger.setLevel(logging.INFO)

supported_predictors = {
    'bananas': Ensemble(encoding_type='path',
                        predictor_type='feedforward'),
    'feedforward': FeedforwardPredictor(encoding_type='adjacency_one_hot'),
    'gbdt': GBDTPredictor(encoding_type='adjacency_one_hot'),
    'gcn': GCNPredictor(encoding_type='gcn'),
    'sovl_50': EarlyStopping(fidelity=50, metric=Metric.VAL_LOSS),
    'sotl_50': EarlyStopping(fidelity=50, metric=Metric.TRAIN_LOSS)    
}

# set up the search space
search_space = NasBench201SearchSpace()

# choose a predictor
predictor = supported_predictors[config.predictor]
predictor_evaluator = PredictorEvaluator(predictor, config=config)
predictor_evaluator.adapt_search_space(search_space)

# evaluate the predictor
predictor_evaluator.evaluate()
