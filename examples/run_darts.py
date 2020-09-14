import logging

from naslib.optimizers.core.evaluator import Trainer
from naslib.optimizers import DARTSOptimizer, GDASOptimizer, RandomSearch
from naslib.search_spaces import DartsSearchSpace, SimpleCellSearchSpace

from naslib.utils import set_seed, setup_logger, get_config_from_args

logger = setup_logger("example.log")

#
# Run as `python run_darts.py --config-file ../naslib/configs/default.yaml data ./`
#

config = get_config_from_args()     # use --help so see the options
set_seed(config.seed)

search_space = SimpleCellSearchSpace()

optimizer = DARTSOptimizer(config)
optimizer.adapt_search_space(search_space)

trainer = Trainer(optimizer, 'cifar10', config)
trainer.search()        # Search for an architecture
trainer.evaluate()      # Evaluate the best architecture

