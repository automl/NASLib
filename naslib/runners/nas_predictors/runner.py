import logging

from naslib.defaults.trainer import Trainer
from naslib.optimizers import Bananas

from naslib.search_spaces import (
    NasBench101SearchSpace,
    NasBench201SearchSpace,
)
from naslib.utils import utils, setup_logger, get_dataset_api

config = utils.get_config_from_args()

logger = setup_logger(config.save + "/log.log")
logger.setLevel(logging.INFO)

utils.log_args(config)

supported_optimizers = {
    "bananas": Bananas(config)
}

supported_search_spaces = {
    "nasbench101": NasBench101SearchSpace(),
    "nasbench201": NasBench201SearchSpace(),
}


load_labeled = False
dataset_api = get_dataset_api(config.search_space, config.dataset)
utils.set_seed(config.seed)

search_space = supported_search_spaces[config.search_space]

optimizer = supported_optimizers[config.optimizer]
optimizer.adapt_search_space(search_space, dataset_api=dataset_api)

trainer = Trainer(optimizer, config, lightweight_output=True)

if config.optimizer in ["bananas", "npenas"]:
    trainer.search(resume_from="")
    trainer.evaluate(resume_from="", dataset_api=dataset_api)
