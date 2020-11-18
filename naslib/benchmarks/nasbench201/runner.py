import logging
import sys
import naslib as nl

from naslib.defaults.trainer import Trainer
from naslib.optimizers import DARTSOptimizer, GDASOptimizer, RegularizedEvolution
from naslib.optimizers.discrete.rs.optimizer import RandomSearch

from naslib.search_spaces import NasBench201SearchSpace
from naslib.utils import utils, setup_logger

config = utils.get_config_from_args()
utils.set_seed(config.seed)

logger = setup_logger(config.save + "/log.log")
logger.setLevel(logging.INFO)   # default DEBUG is too verbose

utils.log_args(config)

supported_optimizers = {
    'darts': DARTSOptimizer(config),
    'gdas': GDASOptimizer(config),
    're': RegularizedEvolution(config),
    'rs': RandomSearch(config),
}

search_space = NasBench201SearchSpace()

optimizer = supported_optimizers[config.optimizer]
optimizer.adapt_search_space(search_space)
    
trainer = Trainer(optimizer, config)

if not config.eval_only:
    checkpoint = utils.get_last_checkpoint(config) if config.resume else ""
    trainer.search(resume_from=checkpoint)

checkpoint = utils.get_last_checkpoint(config, search=False) if config.resume else ""
trainer.evaluate(resume_from=checkpoint)
