import logging
import sys

from naslib.optimizers.core import Trainer
from naslib.optimizers.oneshot.darts import DARTSOptimizer
from naslib.optimizers.oneshot.gdas import GDASOptimizer
from naslib.optimizers.discrete.rs import RandomSearch

from naslib.search_spaces.darts import DartsSearchSpace, SimpleCellSearchSpace
from naslib.utils import utils
from naslib.utils.logging import setup_logger

logger = setup_logger("test.log")
logger.setLevel(logging.INFO)


if __name__ == '__main__':
    
    config = utils.get_config_from_args()
    utils.set_seed(config.seed)

    # search_space = DartsSearchSpace()
    search_space = SimpleCellSearchSpace()

    # optimizer = RandomSearch(sample_size=1)
    optimizer = DARTSOptimizer()
    # optimizer = GDASOptimizer(config.epochs)

    optimizer.adapt_search_space(search_space)
    
    trainer = Trainer(optimizer, 'cifar10', config)
    trainer.train()
    trainer.evaluate()
