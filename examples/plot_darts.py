import os
import logging
from naslib.defaults.trainer import Trainer
from naslib.optimizers import DARTSOptimizer, GDASOptimizer, RandomSearch
from naslib.search_spaces import NasBench301SearchSpace, SimpleCellSearchSpace

from naslib.utils import set_seed, setup_logger, get_config_from_args

config = get_config_from_args()  # use --help so see the options
# config.search.batch_size = 128
config.search.epochs = 1
config.save_arch_weights = True
config.plot_arch_weights = True
config.save_arch_weights_path = f"{config.save}/save_arch"
set_seed(config.seed)

logger = setup_logger(config.save + "/log.log")
logger.setLevel(logging.INFO)  # default DEBUG is very verbose

search_space = NasBench301SearchSpace() #SimpleCellSearchSpace()   # use SimpleCellSearchSpace() for less heavy search

optimizer = DARTSOptimizer(config)
optimizer.adapt_search_space(search_space)

trainer = Trainer(optimizer, config)
trainer.search() 