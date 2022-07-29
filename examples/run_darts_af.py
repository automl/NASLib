import logging
from naslib.defaults.trainer import Trainer
from naslib.optimizers import DARTSOptimizer, GDASOptimizer, RandomSearch
from naslib.search_spaces import DartsSearchSpace, SimpleCellSearchSpace, AutoformerSearchSpace, NasBench201SearchSpace

from naslib.utils import set_seed, setup_logger, get_config_from_args
import torch
config = get_config_from_args()  # use --help so see the options
set_seed(config.seed)
#print(config)
logger = setup_logger(config.save + "/log.log")
logger.setLevel(logging.DEBUG)  # default DEBUG is very verbose

search_space = AutoformerSearchSpace(
)  # use SimpleCellSearchSpace() for less heavy search
#print(config)
optimizer = DARTSOptimizer(config,op_optimizer=torch.optim.Adam,arch_optimizer=torch.optim.Adam)
optimizer.adapt_search_space(search_space)
#print(optimizer.get_final_architecture().modules_str())


trainer = Trainer(optimizer, config)
trainer.search()  # Search for an architecture
trainer.evaluate()  # Evaluate the best architecture
