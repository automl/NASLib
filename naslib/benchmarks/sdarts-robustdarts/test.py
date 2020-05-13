import argparse
import logging
import os
import sys
import time

from naslib.search_spaces.darts import MacroGraph, OPS
from naslib.optimizers.core import NASOptimizer, Evaluator
from naslib.optimizers.oneshot.gdas import GDASOptimizer
from naslib.optimizers.oneshot.darts import DARTSOptimizer
from naslib.optimizers.oneshot.pc_darts import PCDARTSOptimizer
from naslib.optimizers.oneshot.sdarts import Searcher
from naslib.utils import config_parser
from naslib.utils.utils import create_exp_dir
from naslib.utils.parser import Parser

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
format=log_format, datefmt='%m/%d %I:%M:%S %p')

opt_list = [DARTSOptimizer, GDASOptimizer]

parser = argparse.ArgumentParser('robustdarts')
parser.add_argument('--optimizer', type=str, default='DARTSOptimizer')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--dataset', type=str, default='cifar10')
parser.add_argument('--config', type=str, default='final_eval')
parser.add_argument('--space', type=str, default='s1')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
args = parser.parse_args()


robust_darts_primitives = {
    's2': ['skip_connect', 'sep_conv_3x3'],
    's3': ['none', 'skip_connect', 'sep_conv_3x3'],
    's4': ['noise', 'sep_conv_3x3']
}


if __name__ == '__main__':
    config = config_parser('../../configs/default.yaml')
    parser = Parser('../../configs/default.yaml')
    config.seed = parser.config.seed = args.seed
    config.dataset = parser.config.dataset = args.dataset
    parser.config.save += '/{}/{}/{}'.format(args.optimizer, args.dataset, args.space)

    search_space = MacroGraph.from_config(
        config=config,
        filename=os.path.join(parser.config.save,
                              'graph.yaml'),
        ops_dict=OPS
    )

    # discretize
    config = config_parser('../../configs/{}.yaml'.format(args.config))
    parser = Parser('../../configs/{}.yaml'.format(args.config))
    config.seed = parser.config.seed = args.seed
    config.dataset = parser.config.dataset = args.dataset
    parser.config.save += '/{}/{}/{}'.format(args.optimizer, args.dataset, args.space)
    create_exp_dir(parser.config.save)

    fh = logging.FileHandler(os.path.join(parser.config.save,
                                      'log_{}.txt'.format(config.seed)))

    final_arch = search_space.discretize(config, n_input_edges=[2 for _ in
                                                                search_space.get_node_op(2).inter_nodes()])
    del search_space

    # run final network from scratch
    opt = NASOptimizer()
    final_arch.parse(opt)
    evaluator = Evaluator(final_arch, parser)
    evaluator.run()

