import logging
import sys
import naslib as nl

from naslib.defaults.trainer import Trainer
from naslib.optimizers import (
    DARTSOptimizer,
    GDASOptimizer,
    OneShotNASOptimizer,
    RandomNASOptimizer,
    RandomSearch,
    RegularizedEvolution,
    LocalSearch,
    Bananas,
    BasePredictor,
    GSparseOptimizer
)

from naslib.search_spaces import DartsSearchSpace, NasBench101SearchSpace
from naslib.utils import utils, setup_logger

config = utils.get_config_from_args()
utils.set_seed(config.seed)

logger = setup_logger(config.save + "/log.log")
logger.setLevel(logging.INFO)  # default DEBUG is too verbose

utils.log_args(config)

supported_optimizers = {
    "darts": DARTSOptimizer(config),
    "gdas": GDASOptimizer(config),
    "oneshot": OneShotNASOptimizer(config),
    "rsws": RandomNASOptimizer(config),
    "re": RegularizedEvolution(config),
    "rs": RandomSearch(config),
    "ls": RandomSearch(config),
    "bananas": Bananas(config),
    "bp": BasePredictor(config),
    "gsparsity": GSparseOptimizer(config)
}

if config.dataset == "cifar100":
    DartsSearchSpace.NUM_CLASSES = 100
search_space = DartsSearchSpace()
#search_space = NasBench101SearchSpace()

optimizer = supported_optimizers[config.optimizer]
optimizer.adapt_search_space(search_space)

trainer = Trainer(optimizer, config)
# trainer.search(resume_from=utils.get_last_checkpoint(config) if config.resume else "")

# if config.eval_only:
# trainer.evaluate(resume_from=utils.get_last_checkpoint(config, search=False) if config.resume else "")
# else:
#trainer.search(resume_from=utils.get_last_checkpoint(config) if config.resume else "")
#trainer.evaluate(resume_from=utils.get_last_checkpoint(config, search=False) if config.resume else "")
#trainer.evaluate(search_model="/work/dlclarge2/agnihotr-ml/NASLib/naslib/benchmarks/darts/run/darts/cifar10/gsparsity/2252403/search/model_final.pth", best_arch="/work/dlclarge2/agnihotr-ml/NASLib/naslib/benchmarks/darts/run/darts/cifar10/gsparsity/2252403/search/model_final.pth")


trainer = Trainer(optimizer, config, lightweight_output=True)
trainer.search()

# if not config.eval_only:
#    checkpoint = utils.get_last_checkpoint(config) if config.resume else ""
#    trainer.search(resume_from=checkpoint)

#checkpoint = utils.get_last_checkpoint(config, search_model=True) if config.resume else ""
#trainer.evaluate(resume_from=checkpoint, dataset_api=dataset_api)
trainer.evaluate(dataset_api=dataset_api)

