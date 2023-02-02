import logging

from naslib.defaults.trainer import Trainer
from naslib.optimizers import (
    DrNASOptimizer,
)

from naslib.search_spaces import (
    NasBench201SearchSpace,
    NasBench301SearchSpace,
)

from naslib.utils import utils, setup_logger, get_dataset_api


config = utils.get_config_from_args(config_type='nas')

logger = setup_logger(config.save + "/log.log")
logger.setLevel(logging.INFO)

utils.log_args(config)

supported_optimizers = {
    'drnas': DrNASOptimizer(config),
}

supported_search_spaces = {
    'nasbench201': NasBench201SearchSpace(n_classes=config.n_classes),
    'nasbench301': NasBench301SearchSpace(n_classes=config.n_classes, auxiliary=False),
}

dataset_api = get_dataset_api(config.search_space, config.dataset)
utils.set_seed(config.seed)

search_space = supported_search_spaces[config.search_space]

optimizer = supported_optimizers[config.optimizer]
optimizer.adapt_search_space(search_space, dataset_api=dataset_api)

trainer = Trainer(optimizer, config, lightweight_output=True)

trainer.search(resume_from="")
if config.search_space == 'nasbench301':
    trainer.evaluate(resume_from="", retrain=True, dataset_api=dataset_api)
else:
    trainer.evaluate(resume_from="", retrain=False, dataset_api=dataset_api)
