import logging
import sys

from naslib.defaults.trainer import Trainer
from naslib.optimizers import (
    DARTSOptimizer,
    GDASOptimizer,
    DrNASOptimizer,
    RandomSearch,
    RegularizedEvolution,
    LocalSearch,
    Bananas,
    BasePredictor,
    ConfigurableOptimizer,
)
from naslib.optimizers.oneshot.configurable.components import AbstractArchitectureSampler, AbstractCombOpModifier, AbstractEdgeOpModifier, NoCombOpModifier, NoEdgeOpModifer, OptimizationStrategy, PartialChannelConnection, DARTSSampler, RandomWeightPertubations

from naslib.search_spaces import (
    DartsSearchSpace,
    SimpleCellSearchSpace,
    NasBench201SearchSpace,
    HierarchicalSearchSpace,
)
from naslib.utils import get_dataset_api, utils
# from naslib.search_spaces.nasbench101 import graph

from naslib.utils import utils, setup_logger

# Read args and config, setup logger
config = utils.get_config_from_args()
utils.set_seed(config.seed)

logger = setup_logger(config.save + "/log.log")
# logger.setLevel(logging.INFO)   # default DEBUG is very verbose

utils.log_args(config)
config.dataset = "cifar10"
config.evaluation.learning_rate = 0.025
config.search.learning_rate = 0.025
config.search.batch_size = 64
config.evaluation.batch_size = 96
utils.log_args(config)
supported_optimizers = {
    "darts":
    DARTSOptimizer(config),
    "gdas":
    GDASOptimizer(config),
    "drnas":
    DrNASOptimizer(config),
    "rs":
    RandomSearch(config),
    "re":
    RegularizedEvolution(config),
    "ls":
    LocalSearch(config),
    "bananas":
    Bananas(config),
    "bp":
    BasePredictor(config),
    "pcoptimizer":
    ConfigurableOptimizer(config,
                          edge_op_modifier=PartialChannelConnection(4),
                          arch_sampler=DARTSSampler()),
    "sdarts":
    ConfigurableOptimizer(config,
                          arch_sampler=DARTSSampler(
                              arch_weights_modifier=RandomWeightPertubations(
                                  epsilon=1e-4, epochs=100))),
}
print(config)
# Changing the search space is one line of code
#search_space = SimpleCellSearchSpace()
# search_space = graph.NasBench101SearchSpace()
# search_space = HierarchicalSearchSpace()
search_space = DartsSearchSpace()
#search_space = NasBench201SearchSpace()

# Changing the optimizer is one line of code
# optimizer = supported_optimizers[config.optimizer]
optimizer = supported_optimizers["darts"]
#dataset_api=get_dataset_api(search_space='nasbench201', dataset="cifar10")
optimizer.adapt_search_space(search_space)  #, dataset_api=dataset_api)

# Start the search and evaluation
trainer = Trainer(optimizer, config)

if not config.eval_only:
    checkpoint = utils.get_last_checkpoint(config) if config.resume else ""
    trainer.search(resume_from=checkpoint)

checkpoint = utils.get_last_checkpoint(config,
                                       search=False) if config.resume else ""
trainer.evaluate(resume_from=checkpoint)  #, dataset_api=dataset_api )
