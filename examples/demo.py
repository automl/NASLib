import logging
import sys

from naslib.defaults.trainer import Trainer
from naslib.optimizers import DARTSOptimizer, GDASOptimizer, RandomSearch
from naslib.optimizers.discrete.re.optimizer import RegularizedEvolution

from naslib.search_spaces import (
    DartsSearchSpace, 
    SimpleCellSearchSpace, 
    NasBench201SeachSpace, 
    SmallHierarchicalSearchSpace,
)

from naslib.utils import utils, setup_logger

# Read args and config, setup logger
config = utils.get_config_from_args()
utils.set_seed(config.seed)

logger = setup_logger(config.save + "/log.log")
logger.setLevel(logging.INFO)   # default DEBUG is too verbose

utils.log_args(config)

supported_optimizers = {
    'darts': DARTSOptimizer(config.search),
    'gdas': GDASOptimizer(config.search),
    'random': RandomSearch(sample_size=1),
    're': RegularizedEvolution(config.search),
}

# Changing the search space is one line of code

# search_space = SimpleCellSearchSpace()
search_space = NasBench201SeachSpace()
# search_space = SmallHierarchicalSearchSpace()
# search_space = DartsSearchSpace()

# Changing the optimizer is one line of code

optimizer = supported_optimizers[config.optimizer]

optimizer.adapt_search_space(search_space)

# Start the seach and evaluation
trainer = Trainer(optimizer, config)

if not config.eval_only:
    checkpoint = utils.get_last_checkpoint(config) if config.resume else ""
    trainer.search(resume_from=checkpoint)

checkpoint = utils.get_last_checkpoint(config, search=False) if config.resume else ""
trainer.evaluate(resume_from=checkpoint)
