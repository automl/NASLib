import logging
import sys
import naslib as nl

from naslib.defaults.predictor_evaluator import PredictorEvaluator
from naslib.defaults.trainer import Trainer
from naslib.optimizers import Bananas, OneShotNASOptimizer, RandomNASOptimizer
from naslib.predictors import OneShotPredictor

from naslib.search_spaces import (
    NasBench101SearchSpace,
    NasBench201SearchSpace,
    NasBench301SearchSpace,
)
from naslib import utils
from naslib.utils import setup_logger, get_dataset_api
from naslib.utils import get_project_root


config = utils.get_config_from_args(config_type="oneshot")

logger = setup_logger(config.save + "/log.log")
logger.setLevel(logging.INFO)

utils.log_args(config)

supported_optimizers = {
    "bananas": Bananas(config),
    "oneshot": OneShotNASOptimizer(config),
    "rsws": RandomNASOptimizer(config),
}

supported_search_spaces = {
    "nasbench101": NasBench101SearchSpace(),
    "nasbench201": NasBench201SearchSpace(),
    "nasbench301": NasBench301SearchSpace(),
}


# load_labeled = (True if config.search_space == 'nasbench301' else False)
load_labeled = False
dataset_api = get_dataset_api(config.search_space, config.dataset)
utils.set_seed(config.seed)

search_space = supported_search_spaces[config.search_space]

optimizer = supported_optimizers[config.optimizer]
optimizer.adapt_search_space(search_space, dataset_api=dataset_api)

trainer = Trainer(optimizer, config, lightweight_output=True)

if config.optimizer == "bananas":
    trainer.search(resume_from="")
    trainer.evaluate(resume_from="", dataset_api=dataset_api)
elif config.optimizer in ["oneshot", "rsws"]:
    predictor = OneShotPredictor(config, trainer, model_path=config.model_path)

    predictor_evaluator = PredictorEvaluator(predictor, config=config)
    predictor_evaluator.adapt_search_space(
        search_space, load_labeled=load_labeled, dataset_api=dataset_api
    )

    # evaluate the predictor
    predictor_evaluator.evaluate()
