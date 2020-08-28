import argparse
import logging
import os
import sys
import torch
import numpy as np
import random

from naslib.optimizers.discrete.rs import RandomSearch
from naslib.optimizers.core import NASOptimizer, Evaluator, Trainer
from naslib.optimizers.oneshot.darts import Searcher, DARTSOptimizer
from naslib.optimizers.oneshot.gdas import GDASOptimizer
from naslib.optimizers.oneshot.pc_darts import PCDARTSOptimizer
from naslib.search_spaces.darts import MacroGraph, PRIMITIVES, OPS, DartsSearchSpace, SimpleCellSearchSpace
from naslib.utils import config_parser, set_seed
from naslib.utils.parser import Parser
from naslib.utils.utils import create_exp_dir

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')



if __name__ == '__main__':
    config = config_parser('../../configs/default.yaml')
    parser = Parser('../../configs/default.yaml')

    set_seed(config.seed)

    search_space = SimpleCellSearchSpace()

    #optimizer = RandomSearch(sample_size=1)
    optimizer = DARTSOptimizer()

    optimizer.adapt_search_space(search_space)
    
    trainer = Trainer(optimizer, 'cifar10', config, parser)
    trainer.train()
    trainer.evaluate()
