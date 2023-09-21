import os
import logging
from naslib.defaults.trainer import Trainer
from naslib.optimizers import DARTSOptimizer, GDASOptimizer, DrNASOptimizer
from naslib.search_spaces import NasBench101SearchSpace, NasBench201SearchSpace, NasBench301SearchSpace

from naslib.utils import set_seed, setup_logger, get_config_from_args, create_exp_dir
from naslib.utils.vis import plot_architectural_weights

config = get_config_from_args() # use --help so see the options
config.search.epochs = 50
config.save_arch_weights = True
config.plot_arch_weights = True
config.optimizer = 'gdas'
config.search_space = 'nasbench301'
config.save = "{}/{}/{}/{}/{}".format(
    config.out_dir, config.search_space, config.dataset, config.optimizer, config.seed
)
create_exp_dir(config.save)
create_exp_dir(config.save + "/search")  # required for the checkpoints
create_exp_dir(config.save + "/eval")

optimizers = {
    'gdas': GDASOptimizer(config),
    'darts': DARTSOptimizer(config),
    'drnas': DrNASOptimizer(config),
}

search_spaces = {
    'nasbench101': NasBench101SearchSpace(),
    'nasbench201': NasBench201SearchSpace(),
    'nasbench301': NasBench301SearchSpace(),
}

set_seed(config.seed)

logger = setup_logger(config.save + "/log.log")
logger.setLevel(logging.INFO)  # default DEBUG is very verbose

search_space = search_spaces[config.search_space]

optimizer = optimizers[config.optimizer]
optimizer.adapt_search_space(search_space)

trainer = Trainer(optimizer, config)
trainer.search() 

plot_architectural_weights(config, optimizer)
