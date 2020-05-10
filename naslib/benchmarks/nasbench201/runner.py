import logging
import os
import sys

from naslib.optimizers.oneshot import Searcher
from naslib.optimizers.oneshot.darts import DARTSOptimizer
from naslib.search_spaces.nasbench201 import MacroGraph, PRIMITIVES, OPS
from naslib.utils import config_parser
from naslib.utils.parser import Parser

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
format=log_format, datefmt='%m/%d %I:%M:%S %p')


if __name__ == '__main__':
    config = config_parser('../../configs/nasbench_201.yaml')
    parser = Parser('../../configs/nasbench_201.yaml')

    fh = logging.FileHandler(os.path.join(parser.config.save,
                                      'log_{}.txt'.format(config.seed)))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    one_shot_optimizer = DARTSOptimizer.from_config(**config)
    search_space = MacroGraph.from_optimizer_op(
        one_shot_optimizer,
        config=config,
        primitives=PRIMITIVES,
        ops_dict=OPS
    )
    one_shot_optimizer.init()

    searcher = Searcher(search_space, arch_optimizer=one_shot_optimizer)
    searcher.run()

    # discretize
    final_arch = search_space.discretize(n_input_edges=[2 for _ in search_space.inter_nodes()])
    del search_space, one_shot_optimizer, searcher

    # run final network from scratch
    opt = NASOptimizer()
    final_arch.parse(opt)
    evaluator = Evaluator(final_arch, parser)
    evaluator.run()


