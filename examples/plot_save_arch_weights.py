import os
import logging
from naslib.defaults.trainer import Trainer
from naslib.optimizers import DARTSOptimizer
from naslib.search_spaces import NasBench201SearchSpace

from naslib.utils import set_seed, setup_logger, get_config_from_args
from naslib.utils.vis import plot_architectural_weights

config = get_config_from_args()  # use --help so see the options
config.search.epochs = 50
config.save_arch_weights = True
config.plot_arch_weights = True

set_seed(config.seed)

logger = setup_logger(config.save + "/log.log")
logger.setLevel(logging.INFO)  # default DEBUG is very verbose

search_space = NasBench201SearchSpace()

optimizer = DARTSOptimizer(config)
optimizer.adapt_search_space(search_space)

trainer = Trainer(optimizer, config)
# trainer.search() 

plot_architectural_weights(config, optimizer)

