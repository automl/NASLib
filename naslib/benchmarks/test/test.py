import argparse
import logging
import os
import sys
import torch

from naslib.optimizers.discrete.rs import RandomSearch
from naslib.optimizers.core import NASOptimizer, Evaluator, Trainer
from naslib.optimizers.oneshot.darts import Searcher, DARTSOptimizer
from naslib.optimizers.oneshot.gdas import GDASOptimizer
from naslib.optimizers.oneshot.pc_darts import PCDARTSOptimizer
from naslib.search_spaces.darts import MacroGraph, PRIMITIVES, OPS, DartsSearchSpace, SimpleCellSearchSpace
from naslib.utils import config_parser
from naslib.utils.parser import Parser
from naslib.utils.utils import create_exp_dir

opt_list = [DARTSOptimizer, GDASOptimizer, PCDARTSOptimizer]

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')

parser = argparse.ArgumentParser('nasbench201')
parser.add_argument('--optimizer', type=str, default='RandomSearch')    # 'DARTSOptimizer'
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--dataset', type=str, default='cifar10')
args = parser.parse_args()

if __name__ == '__main__':
    config = config_parser('../../configs/default.yaml')
    parser = Parser('../../configs/default.yaml')
    config.seed = parser.config.seed = args.seed
    config.dataset = parser.config.dataset = args.dataset
    parser.config.save += '/{}/{}'.format(args.optimizer, args.dataset)
    create_exp_dir(parser.config.save)

    fh = logging.FileHandler(os.path.join(parser.config.save,
                                          'log_{}.txt'.format(config.seed)))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)






    search_space = SimpleCellSearchSpace()

    #optimizer = RandomSearch(sample_size=2)
    optimizer = DARTSOptimizer()

    optimizer.adapt_search_space(search_space)
    
    trainer = Trainer(optimizer, 'cifar10', config, parser)
    trainer.train()
    trainer.evaluate()
