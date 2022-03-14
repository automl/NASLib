import logging

from naslib.evaluators.zc_evaluator import PredictorEvaluator
from naslib.search_spaces import get_search_space
from naslib.utils import utils, setup_logger
from predictor import ZeroCostPredictor

config = utils.get_config_from_args()
utils.set_seed(config.seed)
logger = setup_logger(config.save + "/log.log")
logger.setLevel(logging.INFO)
utils.log_args(config)


# initialize the search space and predictor
predictor = ZeroCostPredictor()
search_space = get_search_space(name=config.search_space, dataset=config.dataset)

# initialize the PredictorEvaluator class
predictor_evaluator = PredictorEvaluator(predictor, config=config)
predictor_evaluator.adapt_search_space(search_space, load_labeled=False)

# evaluate the predictor
predictor_evaluator.evaluate()
