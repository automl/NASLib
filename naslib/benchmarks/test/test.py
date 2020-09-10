import logging
import sys
import naslib as nl

from naslib.optimizers.core.evaluator import Trainer
from naslib.optimizers.oneshot.darts.optimizer import DARTSOptimizer
from naslib.optimizers.oneshot.gdas.optimizer import GDASOptimizer
from naslib.optimizers.discrete.rs.optimizer import RandomSearch

from naslib.search_spaces.cell.darts import DartsSearchSpace
from naslib.search_spaces.cell.simple import SimpleCellSearchSpace
from naslib.search_spaces.nasbench201.nasbench201 import NasBench201SeachSpace
from naslib.utils import utils
from naslib.utils.logging import setup_logger

logger = setup_logger("test.log")
logger.setLevel(logging.INFO)


import types
from naslib.search_spaces.core.primitives import *

import networkx as nx

if __name__ == '__main__':
    
    config = utils.get_config_from_args()
    utils.set_seed(config.seed)

    # search_space = DartsSearchSpace()
    # search_space = SimpleCellSearchSpace()
    search_space = NasBench201SeachSpace()

    

    # optimizer = RandomSearch(sample_size=1)
    optimizer = DARTSOptimizer(config)
    # optimizer = GDASOptimizer(config.epochs)

    optimizer.adapt_search_space(search_space)
    
    trainer = Trainer(optimizer, 'cifar10', config)

    trainer.search()
    trainer.evaluate()

    # trainer.evaluate(from_file='run/cifar10/model_0.pt')