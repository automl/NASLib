import argparse
import logging
import os
import sys

from naslib.search_spaces.darts import MacroGraph, PRIMITIVES, OPS
from naslib.optimizers.core import NASOptimizer, Evaluator
from naslib.optimizers.oneshot.darts import DARTSOptimizer, Searcher
from naslib.utils import config_parser
from naslib.utils.parser import Parser

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
format=log_format, datefmt='%m/%d %I:%M:%S %p')


parser = argparse.ArgumentParser('nasbench201')
parser.add_argument('--optimizer', type=str, default='DARTSOptimizer')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--dataset', type=str, default='cifar10')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
args = parser.parse_args()


if __name__ == '__main__':
    config = config_parser('../../configs/default.yaml')
    parser = Parser('../../configs/default.yaml')
    config.seed = parser.config.seed = args.seed
    config.epochs = parser.config.epochs = args.epochs
    parser.config.save += '/{}'.format(args.optimizer)

    fh = logging.FileHandler(os.path.join(parser.config.save,
                                      'log_{}.txt'.format(config.seed)))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    one_shot_optimizer = eval(args.optimizer).from_config(**config)
    search_space = MacroGraph.from_config(
        config=config,
        filename='../../configs/search_spaces/robust_darts/s1.yaml',
        ops_dict=OPS
    )
    search_space.parse(one_shot_optimizer)
    one_shot_optimizer.init()

    searcher = Searcher(search_space, parser, arch_optimizer=one_shot_optimizer)
    searcher.run()

    # discretize
    final_arch = search_space.discretize(n_input_edges=[2 for _ in search_space.inter_nodes()])
    del search_space, one_shot_optimizer, searcher

    # run final network from scratch
    opt = NASOptimizer()
    final_arch.parse(opt)
    evaluator = Evaluator(final_arch, parser)
    evaluator.run()

