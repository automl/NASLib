import logging
import sys
import naslib as nl

from naslib.optimizers.core.evaluator import Trainer
from naslib.optimizers import DARTSOptimizer, GDASOptimizer, RandomSearch

from naslib.search_spaces import DartsSearchSpace, SimpleCellSearchSpace, NasBench201SeachSpace
from naslib.utils import setup_logger, set_seed, get_config_from_args

logger = setup_logger("test.log")
logger.setLevel(logging.INFO)

if __name__ == '__main__':
    
    config = get_config_from_args()
    set_seed(config.seed)

    search_space = DartsSearchSpace()
    # search_space = SimpleCellSearchSpace()

    # optimizer = RandomSearch(sample_size=1)
    optimizer = DARTSOptimizer(config)
    # optimizer = GDASOptimizer(config)

    optimizer.adapt_search_space(search_space)
    
    trainer = Trainer(optimizer, 'cifar10', config)

    trainer.search()
    trainer.evaluate()

    # trainer.evaluate(from_file='run/cifar10/model_0.pt')









# import argparse
# import logging
# import os
# import sys

# from naslib.optimizers.discrete.rs import RandomSearch
# from naslib.optimizers.core import NASOptimizer, Evaluator
# from naslib.optimizers.oneshot.darts import Searcher, DARTSOptimizer
# from naslib.optimizers.oneshot.gdas import GDASOptimizer
# from naslib.optimizers.oneshot.pc_darts import PCDARTSOptimizer
# from naslib.search_spaces.darts import MacroGraph, PRIMITIVES, OPS, DartsSearchSpace, SimpleCellSearchSpace
# from naslib.utils import config_parser
# from naslib.utils.parser import Parser
# from naslib.utils.utils import create_exp_dir

# opt_list = [DARTSOptimizer, GDASOptimizer, PCDARTSOptimizer]

# log_format = '%(asctime)s %(message)s'
# logging.basicConfig(stream=sys.stdout, level=logging.INFO,
#                     format=log_format, datefmt='%m/%d %I:%M:%S %p')

# parser = argparse.ArgumentParser('nasbench201')
# parser.add_argument('--optimizer', type=str, default='RandomSearch')    # 'DARTSOptimizer'
# parser.add_argument('--seed', type=int, default=1)
# parser.add_argument('--dataset', type=str, default='cifar10')
# args = parser.parse_args()