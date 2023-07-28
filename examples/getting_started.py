import logging

from naslib.defaults.trainer import Trainer
from naslib.optimizers import (
    RegularizedEvolution,
    DARTSOptimizer,
)

from naslib.search_spaces import (
    NasBench201SearchSpace,
    NasBench301SearchSpace,
)
from naslib import utils
from naslib.utils import setup_logger, get_dataset_api

config = utils.get_config_from_args(config_type='nas')

logger = setup_logger(config.save + "/log.log")
logger.setLevel(logging.INFO)

utils.log_args(config)

supported_optimizers = {
    're': RegularizedEvolution(config),
    'darts': DARTSOptimizer(**config),
}

supported_search_spaces = {
    'nasbench201': NasBench201SearchSpace(),
    'nasbench301': NasBench301SearchSpace(),
}

dataset_api = get_dataset_api(config.search_space, config.dataset)
utils.set_seed(config.seed)

search_space = supported_search_spaces[config.search_space]

optimizer = supported_optimizers[config.optimizer]
optimizer.adapt_search_space(search_space, dataset=config.dataset, dataset_api=dataset_api)

trainer = Trainer(optimizer, config, lightweight_output=True)

trainer.search(resume_from="")
trainer.evaluate(resume_from="", dataset_api=dataset_api)
