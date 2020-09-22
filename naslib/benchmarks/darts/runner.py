import logging
import sys
import naslib as nl

from naslib.defaults.trainer import Trainer
from naslib.optimizers import DARTSOptimizer, GDASOptimizer, RandomSearch
from naslib.optimizers.discrete.re.optimizer import RegularizedEvolution

from naslib.search_spaces import DartsSearchSpace
from naslib.utils import utils, setup_logger

from naslib.search_spaces.hierarchical.graph import SmallHierarchicalSearchSpace

config = utils.get_config_from_args()
utils.set_seed(config.seed)

logger = setup_logger(config.save + "/log.log")
logger.setLevel(logging.INFO)   # default DEBUG is too verbose

utils.log_args(config)

supported_optimizers = {
    'darts': DARTSOptimizer(config.search),
    'gdas': GDASOptimizer(config.search),
    'random': RandomSearch(sample_size=1),
}

search_space = DartsSearchSpace()

optimizer = supported_optimizers[config.optimizer]
optimizer.adapt_search_space(search_space)
    
trainer = Trainer(optimizer, config)

if config.eval_only:
    trainer.evaluate(resume_from=utils.get_last_checkpoint(config, search=False) if config.resume else "")
else:
    trainer.search(resume_from=utils.get_last_checkpoint(config) if config.resume else "")
    trainer.evaluate(resume_from=utils.get_last_checkpoint(config, search=False) if config.resume else "")
