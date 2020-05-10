from naslib.search_spaces.darts import MacroGraph, PRIMITIVES, OPS
from naslib.optimizers.oneshot.darts import DARTSOptimizer, Searcher
from naslib.utils import config_parser


if __name__ == '__main__':
    config = config_parser('../../configs/default.yaml')

    one_shot_optimizer = DARTSOptimizer.from_config(**config)
    search_space = MacroGraph.from_config(
        config=config,
        filename='../../configs/search_spaces/robust_darts/s1.yaml',
        ops_dict=OPS
    )
    search_space.parse(one_shot_optimizer)
    one_shot_optimizer.init()

    searcher = Searcher(search_space, arch_optimizer=one_shot_optimizer)
    searcher.run()

