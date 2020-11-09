import logging
import sys

from naslib.defaults.trainer import Trainer
from naslib.optimizers import DARTSOptimizer, GDASOptimizer, RandomSearch
from naslib.utils import utils, setup_logger

from naslib.search_spaces.hierarchical.graph import HierarchicalSearchSpace, LiuFinalArch
from naslib.optimizers.discrete.rs.optimizer import sample_random_architecture

config = utils.get_config_from_args()
utils.set_seed(config.seed)

logger = setup_logger(config.save + "/log.log")
logger.setLevel(logging.INFO)   # default DEBUG is too verbose

utils.log_args(config)

search_space = HierarchicalSearchSpace()

best_arch = None
if config.optimizer == 'darts':
    optimizer = DARTSOptimizer(config)
elif config.optimizer == 'gdas':
    optimizer = GDASOptimizer(config)
elif config.optimizer == 'liu_et_al':
    optimizer = DARTSOptimizer(config)      # hack to instanciate the trainer (is ignored during eval)
    best_arch = LiuFinalArch()
elif config.optimizer == 'random':
    optimizer = DARTSOptimizer(config)      # hack to instanciate the trainer (is ignored during eval)
    best_arch = sample_random_architecture(search_space, search_space.OPTIMIZER_SCOPE)
else:
    raise ValueError("Unknown optimizer : {}".format(config.optimizer))

optimizer.adapt_search_space(search_space)
trainer = Trainer(optimizer, config)


if config.eval_only and not best_arch:
    trainer.evaluate(resume_from=utils.get_last_checkpoint(config, search=False) if config.resume else "")
elif best_arch:
    best_arch.parse()
    trainer.evaluate(best_arch=best_arch)
else:
    trainer.search(resume_from=utils.get_last_checkpoint(config) if config.resume else "")
    trainer.evaluate(resume_from=utils.get_last_checkpoint(config, search=False) if config.resume else "")
