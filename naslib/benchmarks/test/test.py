import logging
import sys
import naslib as nl

from naslib.optimizers.core.evaluator import Trainer
from naslib.optimizers import DARTSOptimizer, GDASOptimizer, RandomSearch

from naslib.search_spaces import DartsSearchSpace, SimpleCellSearchSpace, NasBench201SeachSpace
from naslib.utils import setup_logger, set_seed, get_config_from_args

from naslib.search_spaces.hierarchical.graph import SmallHierarchicalSearchSpace

logger = setup_logger("test.log")
logger.setLevel(logging.INFO)

if __name__ == '__main__':
    
    config = get_config_from_args()
    set_seed(config.seed)

    # search_space = DartsSearchSpace()
    # search_space = SimpleCellSearchSpace()
    # search_space = NasBench201SeachSpace()
    search_space = SmallHierarchicalSearchSpace()

    # optimizer = RandomSearch(sample_size=1)
    optimizer = DARTSOptimizer(config.search)
    # optimizer = GDASOptimizer(config.search)

    optimizer.adapt_search_space(search_space)
    
    trainer = Trainer(optimizer, 'cifar10', config)

    trainer.search()
    trainer.evaluate()

    # trainer.evaluate(from_file='run/cifar10/model_0.pt')