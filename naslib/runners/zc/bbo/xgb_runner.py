import logging
from naslib.evaluators.zc_ensemble_evaluator import ZCEnsembleEvaluator
from naslib.predictors.ensemble import Ensemble
from naslib.search_spaces import get_search_space
from naslib.utils.get_dataset_api import get_dataset_api, get_zc_benchmark_api
from naslib.utils.log import setup_logger
from naslib import utils

config = utils.get_config_from_args(config_type="zc")

logger = setup_logger(config.save + "/log.log")
logger.setLevel(logging.INFO)

utils.log_args(config)

search_space = get_search_space(config.search_space, config.dataset)
# dataset_api = None #get_dataset_api(config.search_space, config.dataset)
dataset_api = get_dataset_api(config.search_space, config.dataset)
zc_api = get_zc_benchmark_api(config.search_space, config.dataset)
search_space.instantiate_model = False
search_space.sample_without_replacement = True
search_space.labeled_archs = [eval(arch) for arch in zc_api.keys()]

utils.set_seed(config.seed)

evaluator = ZCEnsembleEvaluator(
    n_train=config.train_size,
    n_test=config.test_size,
    zc_names=config.zc_names,
    zc_api=zc_api
)

evaluator.adapt_search_space(search_space, config.dataset, dataset_api, config)

train_loader, _, _, _, _ = utils.get_train_val_loaders(config)

ensemble = Ensemble(num_ensemble=1,
                    ss_type=search_space.get_type(),
                    predictor_type='xgb',
                    zc=config.zc_ensemble,
                    zc_only=config.zc_only,
                    config=config)

evaluator.evaluate(ensemble, train_loader)

logger.info('Done.')
