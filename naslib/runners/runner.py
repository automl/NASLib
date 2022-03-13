import logging

from naslib.evaluators.zc_evaluator import PredictorEvaluator
from naslib.predictors import ZeroCost
from naslib.search_spaces import get_search_space
from naslib.utils import utils, setup_logger, get_dataset_api


config = utils.get_config_from_args()
utils.set_seed(config.seed)
logger = setup_logger(config.save + "/log.log")
logger.setLevel(logging.INFO)
utils.log_args(config)

supported_predictors = {
    "fisher": ZeroCost(config, batch_size=64, method_type="fisher"),
    "flops": ZeroCost(config, batch_size=64, method_type="flops"),
    "grad_norm": ZeroCost(config, batch_size=64, method_type="grad_norm"),
    "grasp": ZeroCost(config, batch_size=64, method_type="grasp"),
    "jacov": ZeroCost(config, batch_size=64, method_type="jacov"),
    "params": ZeroCost(config, batch_size=64, method_type="params"),
    "snip": ZeroCost(config, batch_size=64, method_type="snip"),
    "synflow": ZeroCost(config, batch_size=64, method_type="synflow"),
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
search_space = get_search_space(name=config.search_space, dataset=config.dataset)

# initialize the PredictorEvaluator class
predictor_evaluator = PredictorEvaluator(predictor, config=config)
predictor_evaluator.adapt_search_space(
    search_space, load_labeled=load_labeled, dataset_api=dataset_api
)

# evaluate the predictor
predictor_evaluator.evaluate()
