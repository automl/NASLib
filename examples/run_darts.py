import logging
import sys

from naslib.optimizers.core.evaluator import Trainer
from naslib.optimizers.oneshot.darts.optimizer import DARTSOptimizer

from naslib.search_spaces.cell.simple import SimpleCellSearchSpace
from naslib.utils import utils
from naslib.utils.logging import setup_logger

logger = setup_logger("example.log")

#
# Run as `python run_darts.py --config-file ../naslib/configs/default.yaml data ./`
#

config = utils.get_config_from_args()
utils.set_seed(config.seed)

search_space = SimpleCellSearchSpace()

optimizer = DARTSOptimizer(config)
optimizer.adapt_search_space(search_space)

trainer = Trainer(optimizer, 'cifar10', config)
trainer.search()
trainer.evaluate()

