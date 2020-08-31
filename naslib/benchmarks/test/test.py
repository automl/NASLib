import logging

from naslib.optimizers.core import Trainer
from naslib.optimizers.oneshot.darts import DARTSOptimizer
from naslib.optimizers.oneshot.gdas import GDASOptimizer
from naslib.optimizers.discrete.rs import RandomSearch

from naslib.search_spaces.darts import DartsSearchSpace, SimpleCellSearchSpace
from naslib.utils import config_parser, set_seed
from naslib.utils.parser import Parser


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
