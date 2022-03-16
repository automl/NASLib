import logging
import sys
import naslib as nl

from naslib.defaults.trainer import Trainer
from naslib.optimizers import (
    DARTSOptimizer,
    GDASOptimizer,
    OneShotNASOptimizer,
    RandomNASOptimizer,
    RandomSearch,
    RegularizedEvolution,
    LocalSearch,
    Bananas,
    BasePredictor,
    GSparseOptimizer
)

from naslib.search_spaces import NasBench201SearchSpace, DartsSearchSpace
from naslib.utils import utils, setup_logger, get_dataset_api

config = utils.get_config_from_args()
utils.set_seed(config.seed)

logger = setup_logger(config.save + "/log.log")
logger.setLevel(logging.INFO)  # default DEBUG is too verbose

utils.log_args(config)

supported_optimizers = {
    "darts": DARTSOptimizer(config),
    "gdas": GDASOptimizer(config),
    "oneshot": OneShotNASOptimizer(config),
    "rsws": RandomNASOptimizer(config),
    "re": RegularizedEvolution(config),
    "rs": RandomSearch(config),
    "ls": RandomSearch(config),
    "bananas": Bananas(config),
    "bp": BasePredictor(config),
    "gsparsity": GSparseOptimizer(config)
}

supported_search_space ={
    "nasbench201" : NasBench201SearchSpace(),
    "darts" : DartsSearchSpace()
}

#search_space = NasBench201SearchSpace()
search_space = supported_search_space[config.search_space]
#dataset_api = get_dataset_api("nasbench201", config.dataset)
print(search_space)
dataset_api = get_dataset_api(config.search_space, config.dataset)

optimizer = supported_optimizers[config.optimizer]
optimizer.adapt_search_space(search_space)

trainer = Trainer(optimizer, config, lightweight_output=True)
trainer.search()

# if not config.eval_only:
#    checkpoint = utils.get_last_checkpoint(config) if config.resume else ""
#    trainer.search(resume_from=checkpoint)

#checkpoint = utils.get_last_checkpoint(config, search_model=True) if config.resume else ""
#trainer.evaluate(resume_from=checkpoint, dataset_api=dataset_api)
trainer.evaluate(dataset_api=dataset_api)
