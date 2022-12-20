import logging
from naslib.defaults.trainer import Trainer
from naslib.optimizers import DARTSOptimizer, GDASOptimizer, RandomSearch
from naslib.search_spaces import DartsSearchSpace, SimpleCellSearchSpace

from naslib.utils import set_seed, setup_logger, get_config_from_args
from naslib.utils.vis import plot_architectural_weights

config = get_config_from_args()  # use --help so see the options
config.save_arch_weights = True
# config.search.batch_size = 32
config.search.epochs = 1
set_seed(config.seed)

logger = setup_logger(config.save + "/log.log")
logger.setLevel(logging.INFO)  # default DEBUG is very verbose

search_space = SimpleCellSearchSpace()

optimizer = DARTSOptimizer(config)
optimizer.adapt_search_space(search_space)

trainer = Trainer(optimizer, config)
trainer.search()

for u,v in trainer.optimizer.graph.edges:
    print(trainer.optimizer.graph.edges[u,v].op)

# for u,v in optimizer.graph.edges:
#     print(optimizer.graph.edges[u,v].op)

# trainer.evaluate()

# plot_architectural_weights(config)


