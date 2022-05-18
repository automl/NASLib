""" Evaluates a ZeroCost predictor for a search space and dataset/task"""
import logging

from naslib.evaluators.zc_evaluator import ZeroCostPredictorEvaluator
from naslib.predictors import ZeroCost
from naslib.search_spaces import get_search_space
from naslib.utils import utils, setup_logger, get_dataset_api

# Get the configs from naslib/configs/predictor_config.yaml and the command line arguments
# The configs include the zero-cost method to use, the search space and dataset/task to use,
# amongst others.
config = utils.get_config_from_args()
utils.set_seed(config.seed)
logger = setup_logger(config.save + "/log.log")
logger.setLevel(logging.INFO)
utils.log_args(config)

# Get the benchmark API for this search space and dataset
dataset_api = get_dataset_api(config.search_space, config.dataset)

# Initialize the search space and predictor
# Method type can be "fisher", "grasp", "grad_norm", "jacov", "snip", "synflow", "flops" or "params"
predictor = ZeroCost(method_type=config.predictor)
search_space = get_search_space(name=config.search_space, dataset=config.dataset)

# Initialize the ZeroCostPredictorEvaluator class
predictor_evaluator = ZeroCostPredictorEvaluator(predictor, config=config)
predictor_evaluator.adapt_search_space(search_space, dataset_api=dataset_api)

# Evaluate the predictor
predictor_evaluator.evaluate()

logger.info('Correlation experiment complete.')
