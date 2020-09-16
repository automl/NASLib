import logging
import sys
import naslib as nl

from naslib.optimizers.core.evaluator import Trainer
from naslib.optimizers import DARTSOptimizer, GDASOptimizer, RandomSearch

from naslib.search_spaces import DartsSearchSpace, SimpleCellSearchSpace, NasBench201SeachSpace
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

# search_space = DartsSearchSpace()
# search_space = SimpleCellSearchSpace()
search_space = NasBench201SeachSpace()
# search_space = SmallHierarchicalSearchSpace()

optimizer = supported_optimizers[config.optimizer]
optimizer.adapt_search_space(search_space)
    
trainer = Trainer(optimizer, 'cifar10', config)

if config.eval_only:
    trainer.evaluate(from_file='run/cifar10/10/model_0.pth')
else:
    trainer.search()
    trainer.evaluate()
