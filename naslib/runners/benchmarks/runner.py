import logging
from naslib.evaluators.zc_ensemble_evaluator import ZCEnsembleEvaluator
from naslib.search_spaces import get_search_space
from naslib.utils.get_dataset_api import get_dataset_api
from naslib.utils.logging import setup_logger
from naslib.utils import utils

config = utils.get_config_from_args()

logger = setup_logger(config.save + "/log.log")
logger.setLevel(logging.INFO)

utils.log_args(config)

search_space = get_search_space(config.search_space, config.dataset)
dataset_api = get_dataset_api(config.search_space, config.dataset)

utils.set_seed(config.seed)

evaluator = ZCEnsembleEvaluator(
    n_train=config.train_size,
    n_test=0,
    zc_names=[config.predictor],
    zc_api=None
)

evaluator.adapt_search_space(search_space, config.dataset, dataset_api, config)

train_loader, _, _, _, _ = utils.get_train_val_loaders(config)
evaluator.evaluate(None, train_loader)

logger.info('Done.')
