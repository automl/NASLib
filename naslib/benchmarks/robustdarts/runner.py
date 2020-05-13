import argparse
import logging
import os
import sys

from naslib.search_spaces.darts import MacroGraph, OPS
from naslib.optimizers.core import NASOptimizer, Evaluator
from naslib.optimizers.oneshot.gdas import GDASOptimizer
from naslib.optimizers.oneshot.darts import DARTSOptimizer, Searcher
from naslib.optimizers.oneshot.pc_darts import PCDARTSOptimizer
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
    create_exp_dir(parser.config.save)

    fh = logging.FileHandler(os.path.join(parser.config.save,
                                      'log_{}.txt'.format(config.seed)))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    one_shot_optimizer = eval(args.optimizer).from_config(**config)
    if args.space == 's1':
        search_space = MacroGraph.from_config(
            config=config,
            filename='../../configs/search_spaces/robust_darts/s1.yaml',
            ops_dict=OPS,
            load_kwargs=True
        )
        search_space.parse(one_shot_optimizer)
    else:
        PRIMITIVES = robust_darts_primitives[args.space]
        search_space = MacroGraph.from_optimizer_op(
            one_shot_optimizer,
            config=config,
            primitives=PRIMITIVES,
            ops_dict=OPS
        )

    one_shot_optimizer.init()

    searcher = Searcher(search_space, parser, arch_optimizer=one_shot_optimizer)
    searcher.run()
    search_space.save_graph(filename=os.path.join(parser.config.save,
                                                  'graph.yaml'),
                            save_arch_weights=True)

    # discretize
    config = config_parser('../../configs/final_eval_2.yaml')
    parser = Parser('../../configs/final_eval_2.yaml')
    config.seed = parser.config.seed = args.seed
    config.dataset = parser.config.dataset = args.dataset
    parser.config.save += '/{}/{}/{}'.format(args.optimizer, args.dataset, args.space)
    create_exp_dir(parser.config.save)

    fh = logging.FileHandler(os.path.join(parser.config.save,
                                      'log_{}.txt'.format(config.seed)))

    final_arch = search_space.discretize(config,
                                         n_input_edges=[2 for _ in search_space.get_node_op(2).inter_nodes()])
    del search_space, one_shot_optimizer, searcher

    # run final network from scratch
    opt = NASOptimizer()
    final_arch.parse(opt)
    evaluator = Evaluator(final_arch, parser)
    evaluator.run()

