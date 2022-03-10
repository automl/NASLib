import logging

from naslib.defaults.predictor_evaluator import PredictorEvaluator

from naslib.predictors import (
    ZeroCostV1,
    ZeroCostV2,
)

from naslib.search_spaces.core.query_metrics import Metric
from naslib.search_spaces import (
    NasBench101SearchSpace,
    NasBench201SearchSpace,
    DartsSearchSpace,
    NasBenchNLPSearchSpace,
    TransBench101SearchSpaceMicro,
    TransBench101SearchSpaceMacro,
    NasBenchASRSearchSpace,
)

from naslib.utils import utils, setup_logger, get_dataset_api

config = utils.get_config_from_args(config_type="predictor")
utils.set_seed(config.seed)
logger = setup_logger(config.save + "/log.log")
logger.setLevel(logging.INFO)
utils.log_args(config)

supported_predictors = {
    "fisher": ZeroCostV2(config, batch_size=64, method_type="fisher"),
    "flops": ZeroCostV2(config, batch_size=64, method_type="flops"),
    "grad_norm": ZeroCostV2(config, batch_size=64, method_type="grad_norm"),
    "grasp": ZeroCostV2(config, batch_size=64, method_type="grasp"),
    "jacov": ZeroCostV1(config, batch_size=64, method_type="jacov"),
    "jacov2": ZeroCostV2(config, batch_size=64, method_type="jacov"),
    "params": ZeroCostV2(config, batch_size=64, method_type="params"),
    "snip": ZeroCostV2(config, batch_size=64, method_type="snip"),
    "synflow": ZeroCostV2(config, batch_size=64, method_type="synflow"),
}

supported_search_spaces = {
    "nasbench101": NasBench101SearchSpace(),
    "nasbench201": NasBench201SearchSpace(),
    "darts": DartsSearchSpace(),
    "nlp": NasBenchNLPSearchSpace(),
    'transbench101_micro': TransBench101SearchSpaceMicro(config.dataset),
    'transbench101_macro': TransBench101SearchSpaceMacro(),
    "asr": NasBenchASRSearchSpace(),
}

"""
If the API did not evaluate *all* architectures in the search space, 
set load_labeled=True
"""
load_labeled = True if config.search_space in ["darts", "nlp"] else False
dataset_api = get_dataset_api(config.search_space, config.dataset)

# initialize the search space and predictor
utils.set_seed(config.seed)
predictor = supported_predictors[config.predictor]
search_space = supported_search_spaces[config.search_space]

# initialize the PredictorEvaluator class
predictor_evaluator = PredictorEvaluator(predictor, config=config)
predictor_evaluator.adapt_search_space(
    search_space, load_labeled=load_labeled, dataset_api=dataset_api
)

# evaluate the predictor
predictor_evaluator.evaluate()
