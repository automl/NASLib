import logging
import sys
import glob
import os

from fvcore.common.checkpoint import Checkpointer

from naslib.defaults.trainer import Trainer
from naslib.optimizers import DARTSOptimizer, GDASOptimizer, RandomSearch
from naslib.optimizers.discrete.re.optimizer import RegularizedEvolution

from naslib.search_spaces import (
    DartsSearchSpace, 
    SimpleCellSearchSpace, 
    NasBench201SearchSpace, 
    HierarchicalSearchSpace,
)

from naslib.utils import utils, setup_logger

# Read args and config, setup logger
config = utils.get_config_from_args()
utils.set_seed(config.seed)

logger = setup_logger(config.save + "/log.log")
logger.setLevel(logging.INFO)   # default DEBUG is too verbose

utils.log_args(config)

supported_optimizers = {
    'darts': DARTSOptimizer(config.search),
    'gdas': GDASOptimizer(config.search),
    'random': RandomSearch(sample_size=1),
    #'re': RegularizedEvolution(config.search),
}


# search_space = SimpleCellSearchSpace()
search_space = NasBench201SearchSpace()
# search_space = HierarchicalSearchSpace()
# search_space = DartsSearchSpace()

assert search_space.QUERYABLE

optimizer = supported_optimizers[config.optimizer]

optimizer.adapt_search_space(search_space)

checkpoint_dir = '/home/moa/dev/python_projects/NASLib/naslib/benchmarks/nasbench201/run/cifar10/{}/4/search/'.format(config.optimizer)
checkpointables = optimizer.get_checkpointables()

checkpointer = Checkpointer(
    model=checkpointables.pop('model'),
    save_dir="/tmp/",
    **checkpointables
)

for checkpoint in sorted(glob.glob(os.path.join(checkpoint_dir, 'model_0*.pth'))):

    checkpoint = checkpointer.resume_or_load(checkpoint, resume=False)
    epoch = checkpoint.get("iteration", -1)
    
    print(optimizer.test_statistics())



trainer.evaluate(resume_from=checkpoint)
