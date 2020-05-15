import argparse
import logging
import os
import sys

import numpy as np

from naslib.optimizers.discrete.re import RegularizedEvolution as RE
from naslib.optimizers.discrete.re import Searcher as RESearcher
from naslib.optimizers.discrete.rs import RandomSearch as RS
from naslib.optimizers.discrete.rs import Searcher as RSSearcher
from naslib.optimizers.discrete.tpe import Searcher as TPESearcher
from naslib.optimizers.discrete.tpe import TPE
from naslib.optimizers.oneshot.darts import Searcher, DARTSOptimizer
from naslib.optimizers.oneshot.gdas import GDASOptimizer
from naslib.optimizers.oneshot.pc_darts import PCDARTSOptimizer
from naslib.search_spaces.nasbench201 import MacroGraph, PRIMITIVES, OPS
from naslib.utils import config_parser
from naslib.utils.parser import Parser
from naslib.utils.utils import create_exp_dir

opt_list = [DARTSOptimizer, GDASOptimizer, PCDARTSOptimizer, RE, TPE, RS]

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')

parser = argparse.ArgumentParser('nasbench201')
parser.add_argument('--optimizer', type=str, default='RE')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--dataset', type=str, default='cifar10')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--n_evals', type=int, default=200, help='num of function evaluations')
args = parser.parse_args()

if __name__ == '__main__':
    config = config_parser('../../configs/nasbench_201.yaml')
    parser = Parser('../../configs/nasbench_201.yaml')
    np.random.seed(args.seed)

    config.seed = parser.config.seed = args.seed
    config.epochs = parser.config.epochs = args.epochs
    parser.config.save += '/{}'.format(args.optimizer)
    create_exp_dir(parser.config.save)

    fh = logging.FileHandler(os.path.join(parser.config.save,
                                          'log_{}.txt'.format(config.seed)))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    one_shot_optimizer = eval(args.optimizer).from_config(**config)
    search_space = MacroGraph.from_optimizer_op(
        one_shot_optimizer,
        config=config,
        primitives=PRIMITIVES,
        ops_dict=OPS
    )
    one_shot_optimizer.init()
    if hasattr(one_shot_optimizer, 'fill_space'):
        one_shot_optimizer.fill_space()

    if args.optimizer == 'RS':
        _searcher = RSSearcher
    elif args.optimizer == 'RE':
        _searcher = RESearcher
    elif args.optimizer == 'TPE':
        _searcher = TPESearcher
    else:
        _searcher = Searcher

    searcher = _searcher(search_space, parser, arch_optimizer=one_shot_optimizer)
    searcher.run(n_evaluations=args.n_evals)
